"""Tables, charts, evidence checks and a Part C PDF based exclusively on recorded results."""
import json
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib
import sys
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from xml.sax.saxutils import escape
from .pipeline import BETAS, CODEBOOKS, config_signature

EXPERIMENTS = [f'vae_beta{b}' for b in BETAS] + [f'vqvae_k{k}' for k in CODEBOOKS] + ['wgan', 'wgan_gp']
SELECTED = [('vae_beta1', 'VAE, beta = 1'), ('vqvae_k256', 'VQ-VAE, K = 256, PixelCNN prior'),
            ('wgan', 'WGAN with weight clipping'), ('wgan_gp', 'WGAN-GP')]

def metrics_table(cfg, require_all=False):
    rows = []
    for name in EXPERIMENTS:
        path = Path(cfg['run_dir']) / name / 'metrics.json'
        if not path.exists():
            if require_all: raise FileNotFoundError(f'Complete the experiment first: {path}')
            continue
        result = json.loads(path.read_text())
        assert result['configuration_signature'] == config_signature(cfg), f'Inconsistent configuration: {name}'
        rows.append(result)
    df = pd.DataFrame(rows)
    if len(df): df['training_minutes'] = df['training_seconds'] / 60
    return df

def plot_results(cfg):
    root = Path(cfg['run_dir']); figures = root / 'figures'; figures.mkdir(exist_ok=True, parents=True)
    outputs = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for beta in BETAS:
        path = root / f'vae_beta{beta}' / 'history.json'
        if path.exists():
            h = json.loads(path.read_text()); ax.plot([r['epoch'] for r in h], [r['val_psnr'] for r in h], label=f'beta={beta}')
    ax.set(xlabel='Epoch', ylabel='Validation PSNR (dB)', title='Beta-VAE: PSNR versus epochs'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures / 'vae_psnr.png', dpi=180); plt.close(fig); outputs.append(figures / 'vae_psnr.png')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for size in CODEBOOKS:
        path = root / f'vqvae_k{size}' / 'history.json'
        if path.exists():
            h = json.loads(path.read_text()); epochs = [r['epoch'] for r in h]
            axes[0].plot(epochs, [r['val_psnr'] for r in h], label=f'K={size}')
            axes[1].plot(epochs, [r['utilization'] * 100 for r in h], label=f'K={size}')
    axes[0].set(xlabel='Epoch', ylabel='Validation PSNR (dB)', title='VQ-VAE reconstruction')
    axes[1].set(xlabel='Epoch', ylabel='Active codebook entries (%)', title='Codebook utilization')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures / 'vq_quality.png', dpi=180); plt.close(fig); outputs.append(figures / 'vq_quality.png')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ['wgan', 'wgan_gp']:
        path = root / name / 'history.json'
        if path.exists():
            h = json.loads(path.read_text()); epochs = [r['epoch'] for r in h]
            axes[0].plot(epochs, [r['wasserstein_estimate'] for r in h], label=name)
            axes[1].plot(epochs, [r['generator_loss'] for r in h], label=name)
    axes[0].set(xlabel='Reported GAN epoch', ylabel='E[D(real)] - E[D(fake)]', title='Critic estimate (different critic scales)')
    axes[1].set(xlabel='Reported GAN epoch', ylabel='-E[D(fake)]', title='Generator loss')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures / 'gan_losses.png', dpi=180); plt.close(fig); outputs.append(figures / 'gan_losses.png')
    table = metrics_table(cfg)
    if len(table) and table['fid'].notna().any():
        available = table.dropna(subset=['fid'])
        fig, ax = plt.subplots(figsize=(10, 4)); ax.bar(available['experiment'], available['fid'])
        ax.set(ylabel='FID (lower is better)', title='Held-out CIFAR-10 FID: 10,000 real / 10,000 generated'); ax.tick_params(axis='x', rotation=35)
        fig.tight_layout(); fig.savefig(figures / 'fid_comparison.png', dpi=180); plt.close(fig); outputs.append(figures / 'fid_comparison.png')
    return outputs

def review_flags(cfg):
    flags = []
    for name in EXPERIMENTS:
        path = Path(cfg['run_dir']) / name / 'metrics.json'
        if not path.exists(): flags.append(f'{name}: results missing'); continue
        r = json.loads(path.read_text())
        if r['fid'] is None: flags.append(f'{name}: final FID not computed')
        if r.get('utilization', 1) < .05: flags.append(f'{name}: fewer than 5% of codes used; inspect collapse and training')
        if r.get('psnr') is not None and r['psnr'] < 15: flags.append(f'{name}: low reconstruction PSNR; inspect training and images')
    return flags

def report_ready(cfg, observations):
    df = metrics_table(cfg, require_all=True)
    if cfg['smoke']: raise ValueError('Smoke results cannot produce a final submission PDF.')
    for _, row in df.iterrows():
        if row['smoke_only'] or not np.isfinite(row['fid']): raise ValueError('Missing full-run FID')
        if row['fid_n_real'] != 10000 or row['fid_n_fake'] != 10000: raise ValueError('Unequal/incomplete FID sample counts')
    for name, _ in SELECTED:
        if len(observations.get(name, '').split()) < 30:
            raise ValueError(f'Add your actual visual/perceptual assessment for {name} in observations.json (at least 30 words).')
    if len(observations.get('overall_discussion', '').split()) < 50:
        raise ValueError('Add the overall reconstruction/sampling/speed comparison in observations.json (at least 50 words).')
    screenshot = Path('BITS_lab_usage.jpg')
    if not screenshot.is_file(): raise FileNotFoundError('Capture and save the mandatory lab screenshot as BITS_lab_usage.jpg.')
    with PILImage.open(screenshot) as image:
        if image.format != 'JPEG': raise ValueError('The lab screenshot must be an actual JPG, not a renamed PNG.')
    return df

def build_report(cfg, final=False):
    root = Path(cfg['run_dir']); df = metrics_table(cfg, require_all=True)
    observations = json.loads(Path('observations.json').read_text())
    if final: report_ready(cfg, observations)
    target = root / ('PartC_Comparative_Analysis.pdf' if final else 'PartC_DRAFT.pdf')
    styles = getSampleStyleSheet(); story = []
    def paragraph(text, style='BodyText'):
        story.append(Paragraph(escape(str(text)), styles[style])); story.append(Spacer(1, 8))
    def picture(path, width=470):
        with PILImage.open(path) as image: w, h = image.size
        story.append(Image(str(path), width=width, height=width * h / w)); story.append(Spacer(1, 8))
    paragraph('UDL Assignment 2: Image generation on CIFAR-10', 'Title')
    paragraph(f'Group {cfg["group"]}: ' + '; '.join(f'{m["name"]} ({m["id"]})' for m in cfg['members']))
    if not final: paragraph('DRAFT: model results are recorded, but visual discussion and/or screenshot have not been approved for final submission.', 'Heading2')
    paragraph('Evaluation protocol', 'Heading1')
    paragraph(f"All input pixels are scaled to [0,1]. A fixed seed-{cfg['seed']} split uses {cfg['train_examples']:,} training and {cfg['validation_examples']:,} validation images; the {cfg['test_examples']:,} official test images are held out until final evaluation. Class labels are not used for training. Validation PSNR is recorded each epoch, and autoencoders and GANs use their final scheduled epoch, while PixelCNN uses the lowest validation cross-entropy checkpoint. No checkpoint is selected on test FID.")
    paragraph('FID uses the torch-fidelity Inception feature extractor through TorchMetrics, 2,048-dimensional features, float32 feature extraction and float64 statistics. Every final score uses the same 10,000 real test images and 10,000 independently generated images. VAE images come from a standard-normal prior; VQ-VAE images come from the trained PixelCNN, not uniform codes or reconstructions. A single-seed comparison has sampling and training uncertainty.')
    paragraph('PSNR is the mean of per-image 10 log10(1/MSE) values, using range 1. VAE reconstruction uses the posterior mean for repeatability. Unconditional GANs do not map a source image to a reconstruction, so paired PSNR/MSE is not applicable to WGANs.')
    paragraph('The beta-VAE objective is summed per-image squared reconstruction error plus beta times KL, averaged over images. A shared configurable KL warmup precedes the full target beta. VQ-VAE uses mean pixel MSE plus codebook loss and a configurable commitment term (0.25 in the standard configuration). Codebook perplexity and utilization are computed from the complete evaluation split.')
    env = json.loads((root / 'environment.json').read_text())
    paragraph(f'Hardware: {env.get("gpu")}; PyTorch {env["packages"].get("torch")}; CUDA {env.get("cuda")}. Reported training durations exclude FID, sampling and validation; VQ totals include both autoencoder and PixelCNN training.')
    paragraph(f"A GAN epoch contains one generator update per training minibatch and {cfg['n_critic']} critic updates per generator update. It is not the same computational workload as an autoencoder epoch. Sampling throughput uses a warmed-up batch of 100 and includes PixelCNN raster sampling for VQ-VAE. Compare wall-clock times on the recorded same hardware, not epoch counts alone.")
    story.append(PageBreak()); paragraph('Quantitative comparison', 'Heading1')
    headers = ['Experiment', 'FID ↓', 'PSNR dB ↑', 'MSE ↓', 'Train min', 'Samples/s ↑']
    table_rows = [headers]
    def number(value, precision=2): return 'N/A' if value is None or pd.isna(value) else f'{value:.{precision}f}'
    for _, row in df.iterrows():
        table_rows.append([row['experiment'], number(row['fid']), number(row.get('psnr')), number(row.get('mse'), 5),
                           number(row['training_minutes']), number(row['sampling_images_per_second'])])
    table = Table(table_rows, repeatRows=1, colWidths=[115, 48, 68, 60, 62, 70])
    table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#dbe8f4')), ('FONTSIZE', (0,0), (-1,-1), 8),
                               ('GRID', (0,0), (-1,-1), .3, colors.grey), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(table); story.append(Spacer(1, 14))
    valid = df.dropna(subset=['fid'])
    if len(valid):
        best = valid.loc[valid['fid'].idxmin()]
        paragraph(f'The lowest measured FID in this run is {best["fid"]:.2f}, obtained by {best["experiment"]}. This is an empirical result from one seed, not a universal ranking of architectures.')
    for size in CODEBOOKS:
        row = df[df.experiment == f'vqvae_k{size}'].iloc[0]
        paragraph(f'K={size}: codebook perplexity {row["codebook_perplexity"]:.2f}, {int(row["active_codes"])} active codes '
                  f'({row["utilization"] * 100:.1f}% utilization), {int(row["dead_codes"])} unused codes; validation prior cross-entropy '
                  f'{row["prior_val_bits_per_code"]:.3f} bits/code. Perplexity must be interpreted relative to K, not as FID.')
    for path in plot_results(cfg):
        story.append(PageBreak()); paragraph(path.stem.replace('_', ' ').title(), 'Heading1'); picture(path)
    for name, title in SELECTED:
        story.append(PageBreak()); paragraph(title + ': 100 generated examples', 'Heading1')
        paragraph('Exactly 100 unconditional outputs are shown in a 10 × 10 grid, with a fixed seed and no best-sample selection.')
        picture(root / name / 'samples_100.png', width=470)
        paragraph(observations.get(name) or 'REQUIRED: inspect this grid and add observations on sharpness, diversity, recognizable objects, repeated patterns and artifacts. Do not infer perceptual quality from architecture alone.')
    story.append(PageBreak()); paragraph('Comparative discussion and limitations', 'Heading1')
    paragraph(observations.get('overall_discussion') or 'REQUIRED: compare observed reconstruction error, prior sampling quality, codebook quality and measured speed. Explain deviations from expected tendencies and whether additional training is needed.')
    paragraph('Larger beta generally puts more pressure on posterior regularization; reconstruction and sampling tradeoffs must be checked from this run. VQ-VAE separates quantization/reconstruction from the learned prior, so a good PSNR does not guarantee a good sample FID. PixelCNN sampling is sequential. WGAN clipping may constrain critic capacity; WGAN-GP penalizes gradient norms and incurs extra backward-pass cost. These are mechanisms to interpret, not substitute for results.')
    flags = review_flags(cfg)
    paragraph('Run diagnostics: ' + ('; '.join(flags) if flags else 'No automatic missing/low-PSNR/code-utilization flags. Inspect images and learning curves before accepting the results.'))
    paragraph('Limitations: one random seed; finite-sample FID bias; resizing of small CIFAR-10 images inside the FID extractor; different loss scales; and no paired reconstruction metric for unconditional GANs. Low-quality outputs should prompt a documented new experiment rather than invented scores or cherry-picked samples.')
    paragraph('References', 'Heading2')
    for reference in [
        'Kingma and Welling, Auto-Encoding Variational Bayes: https://arxiv.org/abs/1312.6114',
        'Higgins et al., beta-VAE: https://openreview.net/forum?id=Sy2fzU9gl',
        'van den Oord et al., Neural Discrete Representation Learning: https://arxiv.org/abs/1711.00937',
        'van den Oord et al., Conditional Image Generation with PixelCNN Decoders: https://arxiv.org/abs/1606.05328',
        'Arjovsky et al., Wasserstein GAN: https://arxiv.org/abs/1701.07875',
        'Gulrajani et al., Improved Training of Wasserstein GANs: https://arxiv.org/abs/1704.00028',
        'Heusel et al., GANs Trained by a Two Time-Scale Update Rule: https://arxiv.org/abs/1706.08500',
        'TorchMetrics FID documentation: https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html']:
        paragraph(reference)
    def footer(canvas, document):
        canvas.setFont('Helvetica', 8); canvas.drawString(40, 24, f'Group {cfg["group"]} | ' + ('FINAL' if final else 'DRAFT — not submission ready'))
        canvas.drawRightString(A4[0]-40, 24, str(document.page))
    SimpleDocTemplate(str(target), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40).build(story, onFirstPage=footer, onLaterPages=footer)
    print('Written:', target)
    return target
