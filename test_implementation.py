"""CPU unit/integration checks; these are not assignment training results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from udl.models import BetaVAE, VQVAE, PixelCNN, Generator, Critic, gradient_penalty
from udl.pipeline import psnr_per_image, checkpoint, resume, seed_all, config_signature, train_autoencoder

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); seed_all(42)
    def test_vae_shape_range_and_objective(self):
        model = BetaVAE(8); x = torch.rand(2, 3, 32, 32)
        rec, mu, logvar = model(x)
        self.assertEqual(rec.shape, x.shape); self.assertTrue(((rec >= 0) & (rec <= 1)).all())
        loss, mse, kl = model.loss(x, rec, mu, logvar, 4)
        self.assertTrue(torch.allclose(loss, mse + 4 * kl)); loss.backward()
        self.assertIsNotNone(model.mu.weight.grad)
    def test_vae_zero_kl(self):
        x = torch.zeros(2, 3, 32, 32); mu = torch.zeros(2, 8)
        loss, mse, kl = BetaVAE.loss(x, x, mu, mu, 10)
        self.assertEqual(float(loss), 0); self.assertEqual(float(kl), 0)
    def test_quantizer_codes_and_gradients(self):
        model = VQVAE(16, 8); x = torch.rand(2, 3, 32, 32)
        reconstruction, codes, quantization_loss = model(x)
        self.assertEqual(codes.shape, (2, 8, 8)); self.assertTrue((codes >= 0).all() and (codes < 16).all())
        self.assertEqual(model.decode_codes(codes).shape, x.shape)
        (F.mse_loss(reconstruction, x) + quantization_loss).backward()
        self.assertGreater(float(model.quantizer.embedding.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.encoder[0].weight.grad.abs().sum()), 0)
    def test_pixelcnn_has_no_current_or_future_leakage(self):
        model = PixelCNN(8, channels=16, layers=2).eval()
        x = torch.randint(0, 8, (2, 8, 8)); modified = x.clone()
        # Change the current site and ALL later raster sites.
        modified.flatten(1)[:, 3 * 8 + 4:] = (modified.flatten(1)[:, 3 * 8 + 4:] + 1) % 8
        with torch.no_grad():
            a, b = model(x)[:, :, 3, 4], model(modified)[:, :, 3, 4]
        self.assertTrue(torch.allclose(a, b, atol=1e-7))
    def test_pixelcnn_sampling_and_training(self):
        model = PixelCNN(8, channels=16, layers=2)
        codes = torch.randint(0, 8, (2, 8, 8)); loss = F.cross_entropy(model(codes), codes)
        loss.backward(); self.assertTrue(torch.isfinite(loss))
        sample = model.eval().sample(2, height=2, width=2)
        self.assertEqual(sample.shape, (2, 2, 2)); self.assertTrue((sample < 8).all())
    def test_gan_shapes_and_unbounded_critic(self):
        generator, critic = Generator(8), Critic()
        fake = generator.sample(2); self.assertEqual(fake.shape, (2, 3, 32, 32))
        self.assertEqual(critic(fake).shape, (2,))
        self.assertFalse(any(isinstance(m, (nn.Sigmoid, nn.BatchNorm2d)) for m in critic.modules()))
        (-critic(fake).mean()).backward(); self.assertIsNotNone(generator.net[0].weight.grad)
    def test_gradient_penalty_uses_per_image_gradient_norm(self):
        class LinearCritic(nn.Module):
            def __init__(self, scale):
                super().__init__(); self.scale = nn.Parameter(torch.tensor(scale))
            def forward(self, x):
                return x.flatten(1).sum(1) * self.scale / (x[0].numel() ** .5)
        real, fake = torch.rand(3, 3, 32, 32), torch.rand(3, 3, 32, 32)
        gp, norm = gradient_penalty(LinearCritic(1.), real, fake)
        self.assertLess(float(gp), 1e-8)
        critic = LinearCritic(2.); gp, norm = gradient_penalty(critic, real, fake)
        self.assertAlmostEqual(float(gp), 1.0, places=4); gp.backward()
        self.assertGreater(float(critic.scale.grad), 0)
    def test_psnr_range_one(self):
        x = torch.zeros(2, 3, 4, 4); reconstruction = torch.ones_like(x) * .1
        self.assertTrue(torch.allclose(psnr_per_image(x, reconstruction), torch.tensor([20., 20.]), atol=1e-4))
    def test_checkpoint_restores_model_optimizer_and_rng(self):
        cfg = {'seed': 42, 'device': 'cpu', 'smoke': True}
        with tempfile.TemporaryDirectory() as tmp:
            model = nn.Linear(3, 2); opt = torch.optim.Adam(model.parameters())
            model(torch.ones(1, 3)).sum().backward(); opt.step()
            checkpoint(Path(tmp) / 'last.pt', {'m': model}, {'o': opt}, 3, [{'epoch': 3}], cfg, 12.0)
            expected_random = torch.rand(3)
            expected_weights = copy.deepcopy(model.state_dict())
            replacement = nn.Linear(3, 2); new_opt = torch.optim.Adam(replacement.parameters())
            epoch, history, seconds = resume(Path(tmp) / 'last.pt', {'m': replacement}, {'o': new_opt}, cfg)
            self.assertEqual(epoch, 3); self.assertEqual(seconds, 12.0)
            self.assertTrue(torch.equal(torch.rand(3), expected_random))
            for name, value in replacement.state_dict().items(): self.assertTrue(torch.equal(value, expected_weights[name]))
            self.assertTrue(new_opt.state_dict()['state'])
    def test_actual_small_autoencoder_training_and_resume(self):
        cfg = json.loads(Path('config.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            cfg.update(device='cpu', run_dir=tmp, workers=0, smoke=True, batch_size=2,
                       evaluation_batch_size=2, vae_epochs=1)
            dataset = TensorDataset(torch.rand(4, 3, 32, 32), torch.zeros(4))
            model, history, seconds = train_autoencoder(BetaVAE(8), 'unit_vae', dataset, dataset, cfg, beta=1)
            self.assertEqual(len(history), 1); self.assertTrue(torch.isfinite(torch.tensor(history[0]['loss'])))
            _, history2, seconds2 = train_autoencoder(BetaVAE(8), 'unit_vae', dataset, dataset, cfg, beta=1)
            self.assertEqual(history, history2); self.assertEqual(seconds, seconds2)

if __name__ == '__main__': unittest.main(verbosity=2)
