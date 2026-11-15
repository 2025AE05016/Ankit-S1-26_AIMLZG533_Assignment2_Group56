"""Reproducible training, held-out evaluation, checkpointing and experiment orchestration."""
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from .models import BetaVAE, VQVAE, PixelCNN, Generator, Critic, initialize_gan, gradient_penalty

BETAS = (1, 2, 4, 10)
CODEBOOKS = (512, 256, 128)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def sync(device):
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)

def atomic_save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary); temporary.replace(path)

def config_signature(cfg):
    ignored = {'group', 'members', 'additional_members', 'lab_note', 'lab_confirmed', 'run_dir', 'data_dir', 'workers'}
    payload = {k: v for k, v in cfg.items() if k not in ignored}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def load_config(path='config.json', smoke=False, require_training_device=True):
    cfg = json.loads(Path(path).read_text())
    if smoke:
        cfg.update(smoke=True, run_dir='runs/smoke', vae_epochs=1, vq_epochs=1, prior_epochs=1,
                   gan_epochs=1, train_examples=256, validation_examples=128, test_examples=128,
                   workers=0, batch_size=32, fid_samples=0, n_critic=1)
    if not cfg['smoke']:
        assert cfg['lab_confirmed'], 'The brief mandates BITS Prayogshala: confirm that environment, then set lab_confirmed=true.'
        if require_training_device:
            assert cfg['device'] == 'cuda' and torch.cuda.is_available(), 'Use the provisioned BITS GPU for the full run.'
        assert cfg['fid_samples'] == 10000 and cfg['test_examples'] == 10000, 'Use the same 10,000 test and generated images for all final FIDs.'
    elif cfg['device'] == 'cuda' and not torch.cuda.is_available():
        cfg['device'] = 'cpu'
    assert cfg['train_examples'] > 0 and cfg['validation_examples'] > 0
    assert cfg['train_examples'] + cfg['validation_examples'] <= 50000
    assert cfg['batch_size'] > 1 and cfg['n_critic'] >= 1
    assert all(cfg[name] > 0 for name in ['vae_epochs', 'vq_epochs', 'prior_epochs', 'gan_epochs'])
    return cfg

def prepare_data(cfg):
    # ToTensor scales the original uint8 RGB data to [0,1]. No [-1,1] normalization or augmentation.
    dataset = datasets.CIFAR10(cfg['data_dir'], train=True, download=True, transform=transforms.ToTensor())
    test = datasets.CIFAR10(cfg['data_dir'], train=False, download=True, transform=transforms.ToTensor())
    indices = torch.randperm(50000, generator=torch.Generator().manual_seed(cfg['seed'])).tolist()
    nt, nv = cfg['train_examples'], cfg['validation_examples']
    train_indices, val_indices = indices[:nt], indices[nt:nt + nv]
    root = Path(cfg['run_dir']); root.mkdir(parents=True, exist_ok=True)
    split = {'train': train_indices, 'validation': val_indices, 'test': list(range(cfg['test_examples']))}
    split_path = root / 'split_indices.json'
    if split_path.exists(): assert json.loads(split_path.read_text()) == split, 'Data split changed: use a different run_dir.'
    else: atomic_json(split_path, split)
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(test, split['test'])

def loader(dataset, cfg, shuffle=False, batch_size=None):
    generator = torch.Generator().manual_seed(cfg['seed'])
    return DataLoader(dataset, batch_size=batch_size or cfg['batch_size'], shuffle=shuffle,
                      num_workers=cfg['workers'], pin_memory=cfg['device'] == 'cuda',
                      generator=generator, drop_last=False)

def record_environment(cfg):
    root = Path(cfg['run_dir']); root.mkdir(parents=True, exist_ok=True)
    signature = config_signature(cfg)
    marker = root / 'config_used.json'
    if marker.exists():
        assert config_signature(json.loads(marker.read_text())) == signature, 'Configuration changed. Use a new run_dir; do not mix results.'
    else: atomic_json(marker, cfg)
    source_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                     for name in ['models.py', 'pipeline.py']}
    source_marker = root / 'training_source_hashes.json'
    if source_marker.exists():
        assert json.loads(source_marker.read_text()) == source_hashes, 'Training implementation changed: use a new run_dir.'
    else: atomic_json(source_marker, source_hashes)
    versions = {}
    for name in ['torch', 'torchvision', 'numpy', 'torchmetrics', 'torch-fidelity', 'reportlab']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'not installed'
    environment = {'python': platform.python_version(), 'system': platform.platform(), 'packages': versions,
                   'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   'lab_confirmed_by_user': cfg['lab_confirmed'], 'smoke_only': cfg['smoke'],
                   'note': 'This machine-readable log does not replace the mandatory BITS usage screenshot.'}
    atomic_json(root / 'environment.json', environment)
    print(json.dumps(environment, indent=2))
    return environment

def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda'] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])

def checkpoint(path, models, optimizers, epoch, history, cfg, elapsed):
    atomic_save(path, {'models': {k: m.state_dict() for k, m in models.items()},
                      'optimizers': {k: o.state_dict() for k, o in optimizers.items()},
                      'epoch': epoch, 'history': history, 'rng': rng_state(),
                      'signature': config_signature(cfg), 'elapsed': elapsed})

def resume(path, models, optimizers, cfg):
    if not Path(path).exists(): return 0, [], 0.0
    state = torch.load(path, map_location=cfg['device'], weights_only=False)  # own, trusted run checkpoint only
    assert state['signature'] == config_signature(cfg), 'Checkpoint/configuration mismatch.'
    for k, model in models.items(): model.load_state_dict(state['models'][k])
    for k, optimizer in optimizers.items(): optimizer.load_state_dict(state['optimizers'][k])
    restore_rng(state['rng'])
    print(f'Resuming {path} after epoch {state["epoch"]}')
    return state['epoch'], state['history'], state['elapsed']

def psnr_per_image(x, reconstruction):
    mse = (x - reconstruction).square().flatten(1).mean(1)
    return -10 * torch.log10(mse.clamp_min(1e-12))

@torch.no_grad()
def reconstruction_metrics(model, data, cfg, codebook_size=None):
    model.eval(); mse_sum = psnr_sum = 0.0; seen = 0
    counts = torch.zeros(codebook_size or 1, dtype=torch.long)
    for x, _ in loader(data, cfg, batch_size=cfg['evaluation_batch_size']):
        x = x.to(cfg['device'])
        if isinstance(model, BetaVAE): reconstruction = model(x, deterministic=True)[0]
        else:
            reconstruction, codes, _ = model(x)
            counts += torch.bincount(codes.flatten().cpu(), minlength=codebook_size)
        mse_sum += float((x - reconstruction).square().flatten(1).mean(1).sum())
        psnr_sum += float(psnr_per_image(x, reconstruction).sum()); seen += len(x)
    result = {'mse': mse_sum / seen, 'psnr': psnr_sum / seen}
    if codebook_size:
        probabilities = counts.double() / counts.sum()
        nonzero = probabilities[probabilities > 0]
        result.update(codebook_perplexity=float(torch.exp(-(nonzero * nonzero.log()).sum())),
                      active_codes=int((counts > 0).sum()), utilization=float((counts > 0).double().mean()),
                      dead_codes=int((counts == 0).sum()), code_counts=counts.tolist())
    return result

def train_autoencoder(model, name, train, val, cfg, beta=None):
    model = model.to(cfg['device'])
    opt = torch.optim.Adam(model.parameters(), lr=cfg['ae_lr'])
    path = Path(cfg['run_dir']) / name; path.mkdir(parents=True, exist_ok=True)
    total_epochs = cfg['vae_epochs'] if beta is not None else cfg['vq_epochs']
    start, history, elapsed = resume(path / 'last.pt', {'model': model}, {'optimizer': opt}, cfg)
    train_loader = loader(train, cfg, shuffle=True)
    for epoch in range(start + 1, total_epochs + 1):
        train_loader.generator.manual_seed(cfg['seed'] + epoch)
        sync(cfg['device']); t0 = time.perf_counter(); model.train(); sums = np.zeros(3); count = 0
        beta_effective = beta * min(1.0, epoch / max(cfg['kl_warmup_epochs'], 1)) if beta is not None else None
        for x, _ in train_loader:
            x = x.to(cfg['device']); opt.zero_grad(set_to_none=True)
            if beta is not None:
                rec, mu, logvar = model(x)
                loss, rec_loss, regularizer = model.loss(x, rec, mu, logvar, beta_effective)
            else:
                rec, _, regularizer = model(x)
                rec_loss = F.mse_loss(rec, x); loss = rec_loss + regularizer
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss in {name}')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            sums += np.array([loss.item(), rec_loss.item(), regularizer.item()]) * len(x); count += len(x)
        sync(cfg['device']); seconds = time.perf_counter() - t0; elapsed += seconds
        valid = reconstruction_metrics(model, val, cfg, None if beta is not None else model.size)
        row = {'epoch': epoch, 'loss': sums[0] / count, 'reconstruction_loss': sums[1] / count,
               'regularizer': sums[2] / count, 'val_psnr': valid['psnr'], 'val_mse': valid['mse'],
               'epoch_training_seconds': seconds, 'effective_beta': beta_effective}
        if beta is None: row.update({k: valid[k] for k in ['codebook_perplexity', 'utilization', 'active_codes']})
        history.append(row); atomic_json(path / 'history.json', history)
        checkpoint(path / 'last.pt', {'model': model}, {'optimizer': opt}, epoch, history, cfg, elapsed)
        print(f'{name} {epoch}/{total_epochs}: loss={row["loss"]:.4f}, val PSNR={valid["psnr"]:.2f} dB, {seconds:.1f}s', flush=True)
    return model.eval(), history, elapsed

@torch.no_grad()
def extract_codes(model, data, cfg):
    model.eval()
    return torch.cat([model.codes(x.to(cfg['device'])).cpu() for x, _ in loader(data, cfg)], dim=0)

def train_prior(vq, name, train, val, cfg):
    path = Path(cfg['run_dir']) / name
    maps_path = path / 'latent_maps.pt'
    # Latent maps belong to this final, frozen VQ checkpoint and are saved for the assignment.
    if maps_path.exists(): maps = torch.load(maps_path, map_location='cpu', weights_only=True)
    else:
        maps = {'train': extract_codes(vq, train, cfg), 'validation': extract_codes(vq, val, cfg)}
        atomic_save(maps_path, maps)
    prior = PixelCNN(vq.size).to(cfg['device'])
    opt = torch.optim.Adam(prior.parameters(), lr=cfg['prior_lr'])
    start, history, elapsed = resume(path / 'prior_last.pt', {'prior': prior}, {'optimizer': opt}, cfg)
    train_loader = loader(TensorDataset(maps['train']), cfg, shuffle=True)
    val_loader = loader(TensorDataset(maps['validation']), cfg)
    for epoch in range(start + 1, cfg['prior_epochs'] + 1):
        train_loader.generator.manual_seed(cfg['seed'] + epoch)
        sync(cfg['device']); t0 = time.perf_counter(); prior.train(); total = tokens = 0
        for (codes,) in train_loader:
            codes = codes.to(cfg['device']); opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(prior(codes), codes)
            loss.backward(); nn.utils.clip_grad_norm_(prior.parameters(), 5.0); opt.step()
            total += loss.item() * codes.numel(); tokens += codes.numel()
        sync(cfg['device']); seconds = time.perf_counter() - t0; elapsed += seconds
        prior.eval(); valid_total = valid_tokens = 0
        with torch.no_grad():
            for (codes,) in val_loader:
                codes = codes.to(cfg['device']); value = F.cross_entropy(prior(codes), codes, reduction='sum')
                valid_total += value.item(); valid_tokens += codes.numel()
        row = {'epoch': epoch, 'train_nats_per_code': total / tokens,
               'val_nats_per_code': valid_total / valid_tokens, 'val_bits_per_code': valid_total / valid_tokens / math.log(2),
               'epoch_training_seconds': seconds}
        history.append(row); atomic_json(path / 'prior_history.json', history)
        checkpoint(path / 'prior_last.pt', {'prior': prior}, {'optimizer': opt}, epoch, history, cfg, elapsed)
        print(f'{name} prior {epoch}/{cfg["prior_epochs"]}: val={row["val_bits_per_code"]:.3f} bits/code', flush=True)
    return prior.eval(), history, elapsed

def next_images(iterator, data_loader, device):
    try: batch = next(iterator)
    except StopIteration: iterator = iter(data_loader); batch = next(iterator)
    return batch[0].to(device), iterator

def train_gan(kind, train, cfg):
    assert kind in ('wgan', 'wgan_gp')
    seed_all(cfg['seed'])
    generator, critic = Generator(cfg['latent_dim']).to(cfg['device']), Critic().to(cfg['device'])
    generator.apply(initialize_gan); critic.apply(initialize_gan)
    if kind == 'wgan':
        go = torch.optim.RMSprop(generator.parameters(), lr=cfg['wgan_lr'])
        co = torch.optim.RMSprop(critic.parameters(), lr=cfg['wgan_lr'])
        with torch.no_grad():
            for p in critic.parameters(): p.clamp_(-cfg['weight_clip'], cfg['weight_clip'])
    else:
        go = torch.optim.Adam(generator.parameters(), lr=cfg['gp_lr'], betas=(0.0, 0.9))
        co = torch.optim.Adam(critic.parameters(), lr=cfg['gp_lr'], betas=(0.0, 0.9))
    path = Path(cfg['run_dir']) / kind; path.mkdir(parents=True, exist_ok=True)
    start, history, elapsed = resume(path / 'last.pt', {'generator': generator, 'critic': critic}, {'g': go, 'c': co}, cfg)
    dl = loader(train, cfg, shuffle=True)
    # One reported GAN epoch = len(dl) generator updates AND n_critic * len(dl) critic updates.
    # This is n_critic passes over the data, unlike an autoencoder epoch; report update counts explicitly.
    for epoch in range(start + 1, cfg['gan_epochs'] + 1):
        dl.generator.manual_seed(cfg['seed'] + epoch); iterator = iter(dl)
        sync(cfg['device']); t0 = time.perf_counter(); generator.train(); critic.train()
        values = []; gp_values = []; distances = []; g_values = []
        for _ in range(len(dl)):
            for p in critic.parameters(): p.requires_grad_(True)
            for _ in range(cfg['n_critic']):
                real, iterator = next_images(iterator, dl, cfg['device'])
                # Freeze generator BN statistics during critic-only updates.
                generator.eval()
                with torch.no_grad(): fake = generator.sample(len(real))
                co.zero_grad(set_to_none=True)
                distance = critic(real).mean() - critic(fake).mean()
                penalty = gradient_penalty(critic, real, fake)[0] if kind == 'wgan_gp' else torch.zeros((), device=cfg['device'])
                critic_loss = -distance + cfg['gp_lambda'] * penalty
                if not torch.isfinite(critic_loss): raise FloatingPointError(f'Nonfinite critic loss in {kind}')
                critic_loss.backward(); co.step()
                if kind == 'wgan':
                    with torch.no_grad():
                        for p in critic.parameters(): p.clamp_(-cfg['weight_clip'], cfg['weight_clip'])
                values.append(critic_loss.item()); gp_values.append(penalty.item()); distances.append(distance.item())
            for p in critic.parameters(): p.requires_grad_(False)
            generator.train(); go.zero_grad(set_to_none=True)
            generator_loss = -critic(generator.sample(len(real))).mean()
            if not torch.isfinite(generator_loss): raise FloatingPointError(f'Nonfinite generator loss in {kind}')
            generator_loss.backward(); go.step(); g_values.append(generator_loss.item())
        sync(cfg['device']); seconds = time.perf_counter() - t0; elapsed += seconds
        row = {'epoch': epoch, 'critic_loss': float(np.mean(values)), 'generator_loss': float(np.mean(g_values)),
               'wasserstein_estimate': float(np.mean(distances)), 'gradient_penalty': float(np.mean(gp_values)),
               'epoch_training_seconds': seconds, 'generator_updates_cumulative': epoch * len(dl),
               'critic_updates_cumulative': epoch * len(dl) * cfg['n_critic']}
        history.append(row); atomic_json(path / 'history.json', history)
        checkpoint(path / 'last.pt', {'generator': generator, 'critic': critic}, {'g': go, 'c': co}, epoch, history, cfg, elapsed)
        generator.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()] if cfg['device'] == 'cuda' else []):
            torch.manual_seed(cfg['seed'] + 1000)
            save_image(generator.sample(100).cpu(), path / 'progress_100.png', nrow=10)
        print(f'{kind} {epoch}/{cfg["gan_epochs"]}: D={row["critic_loss"]:.4f}, G={row["generator_loss"]:.4f}, GP={row["gradient_penalty"]:.3f}, {seconds:.1f}s', flush=True)
    return generator.eval(), history, elapsed

@torch.no_grad()
def evaluate_generator(sample_fn, real_data, cfg, output_dir):
    path = Path(output_dir); path.mkdir(parents=True, exist_ok=True)
    seed_all(cfg['seed'] + 10000)
    examples = sample_fn(100).detach().clamp(0, 1).cpu()
    assert examples.shape == (100, 3, 32, 32)
    save_image(examples, path / 'samples_100.png', nrow=10, padding=2)
    atomic_save(path / 'samples_100.pt', examples)
    # Warm up outside timing and count all VQ prior autoregressive work inside timing.
    _ = sample_fn(10); sync(cfg['device']); t0 = time.perf_counter()
    timed = sample_fn(100); sync(cfg['device']); duration = time.perf_counter() - t0
    del timed
    result = {'sampling_seconds_100': duration, 'sampling_images_per_second': 100 / duration,
              'fid': None, 'fid_n_real': 0, 'fid_n_fake': 0, 'fid_feature_dim': 2048}
    if cfg['smoke']: return result
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(cfg['device'])
    # TorchMetrics 1.9 uses float64 feature sums/covariance statistics by default.
    # Keep Inception in float32: converting convolutions to float64 is slow on T4 GPUs.
    assert fid.real_features_sum.dtype == torch.float64 and fid.real_features_cov_sum.dtype == torch.float64
    # Official torch-fidelity Inception weights are downloaded on first use; no substitute classifier is used.
    for real, _ in loader(real_data, cfg, batch_size=cfg['evaluation_batch_size']):
        fid.update(real.to(cfg['device']), real=True)
    generated = 0; seed_all(cfg['seed'] + 20000); t0 = time.perf_counter()
    while generated < cfg['fid_samples']:
        count = min(cfg['evaluation_batch_size'], cfg['fid_samples'] - generated)
        fake = sample_fn(count)
        assert torch.isfinite(fake).all() and fake.min() >= 0 and fake.max() <= 1
        fid.update(fake, real=False); generated += count
        if generated % 1000 == 0: print(f'FID {path.name}: {generated}/{cfg["fid_samples"]} generated', flush=True)
    result.update(fid=float(fid.compute()), fid_n_real=len(real_data), fid_n_fake=generated,
                  fid_generation_and_feature_seconds=time.perf_counter() - t0)
    del fid; gc.collect()
    return result

@torch.no_grad()
def save_reconstructions(model, test, cfg, path):
    x = next(iter(loader(test, cfg, batch_size=20)))[0].to(cfg['device'])
    reconstruction = model(x, deterministic=True)[0] if isinstance(model, BetaVAE) else model(x)[0]
    save_image(torch.cat([x[:10], reconstruction[:10], x[10:], reconstruction[10:]]).cpu(), path / 'reconstructions.png', nrow=10)

def write_result(name, row, cfg):
    row.update(experiment=name, smoke_only=cfg['smoke'], configuration_signature=config_signature(cfg),
               checkpoint_selection='final scheduled epoch; no selection on test FID', evaluation_split='held-out CIFAR-10 test')
    atomic_json(Path(cfg['run_dir']) / name / 'metrics.json', row)
    return row

def complete_result(name, cfg):
    path = Path(cfg['run_dir']) / name / 'metrics.json'
    if path.exists():
        value = json.loads(path.read_text()); assert value['configuration_signature'] == config_signature(cfg)
        print(f'{name}: using completed, matching run.'); return value
    return None

def run_part_a(cfg):
    record_environment(cfg); train, val, test = prepare_data(cfg); results = []
    for beta in BETAS:
        name = f'vae_beta{beta}'
        result = complete_result(name, cfg)
        if result is not None: results.append(result); continue
        seed_all(cfg['seed'])  # shared VAE initialization across beta values
        model, history, duration = train_autoencoder(BetaVAE(cfg['latent_dim']), name, train, val, cfg, beta)
        path = Path(cfg['run_dir']) / name
        reconstruction = reconstruction_metrics(model, test, cfg)
        save_reconstructions(model, test, cfg, path)
        row = dict(reconstruction, family='beta-VAE', beta=beta, codebook_size=None, training_seconds=duration,
                   epochs=cfg['vae_epochs'], prior_training_seconds=0,
                   trainable_parameters=sum(p.numel() for p in model.parameters()))
        row.update(evaluate_generator(model.sample, test, cfg, path)); results.append(write_result(name, row, cfg))
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    for size in CODEBOOKS:
        name = f'vqvae_k{size}'
        result = complete_result(name, cfg)
        if result is not None: results.append(result); continue
        seed_all(cfg['seed'])
        model, history, duration = train_autoencoder(VQVAE(size, cfg['embedding_dim'], cfg['commitment_cost']), name, train, val, cfg)
        prior, prior_history, prior_duration = train_prior(model, name, train, val, cfg)
        path = Path(cfg['run_dir']) / name
        row = dict(reconstruction_metrics(model, test, cfg, size), family='VQ-VAE + PixelCNN', beta=None,
                   codebook_size=size, training_seconds=duration + prior_duration, autoencoder_training_seconds=duration,
                   prior_training_seconds=prior_duration, epochs=cfg['vq_epochs'], prior_epochs=cfg['prior_epochs'],
                   trainable_parameters=sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in prior.parameters()),
                   prior_val_bits_per_code=prior_history[-1]['val_bits_per_code'])
        save_reconstructions(model, test, cfg, path)
        row.update(evaluate_generator(lambda count: model.decode_codes(prior.sample(count)), test, cfg, path))
        results.append(write_result(name, row, cfg)); del model, prior; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return results

def run_part_b(cfg):
    record_environment(cfg); train, _, test = prepare_data(cfg); results = []
    for kind in ['wgan', 'wgan_gp']:
        result = complete_result(kind, cfg)
        if result is not None: results.append(result); continue
        model, history, duration = train_gan(kind, train, cfg)
        row = {'family': kind.upper().replace('_', '-'), 'beta': None, 'codebook_size': None,
               'training_seconds': duration, 'prior_training_seconds': 0, 'epochs': cfg['gan_epochs'],
               'mse': None, 'psnr': None, 'trainable_parameters': sum(p.numel() for p in model.parameters()),
               'generator_updates': history[-1]['generator_updates_cumulative'],
               'critic_updates': history[-1]['critic_updates_cumulative']}
        row.update(evaluate_generator(model.sample, test, cfg, Path(cfg['run_dir']) / kind))
        results.append(write_result(kind, row, cfg)); del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return results
