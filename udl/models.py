"""CIFAR-10 beta-VAE, VQ-VAE, causal PixelCNN, and Wasserstein GAN models."""
import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1),
                                 nn.ReLU(), nn.Conv2d(channels, channels, 1))
    def forward(self, x):
        return x + self.net(x)

class BetaVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), nn.Flatten())
        self.mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.logvar = nn.Linear(128 * 4 * 4, latent_dim)
        self.project = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(),
            Residual(128), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid())
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(-20, 20)
    def decode(self, z):
        return self.decoder(self.project(z).view(-1, 128, 4, 4))
    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        z = mu if deterministic else mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.decode(z), mu, logvar
    @staticmethod
    def loss(x, reconstruction, mu, logvar, beta):
        # Fixed-variance Gaussian reconstruction term, sum over pixels / mean over images.
        reconstruction_loss = (reconstruction - x).square().flatten(1).sum(1).mean()
        kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1)).mean()
        return reconstruction_loss + beta * kl, reconstruction_loss, kl
    def sample(self, n):
        return self.decode(torch.randn(n, self.latent_dim, device=next(self.parameters()).device))

class VectorQuantizer(nn.Module):
    def __init__(self, size=256, dim=64, commitment=0.25):
        super().__init__()
        self.size, self.dim, self.commitment = size, dim, commitment
        self.embedding = nn.Embedding(size, dim)
        nn.init.uniform_(self.embedding.weight, -1 / size, 1 / size)
    def forward(self, z):
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.dim)
        distances = flat.square().sum(1, keepdim=True) + self.embedding.weight.square().sum(1)[None] - 2 * flat @ self.embedding.weight.t()
        codes = distances.argmin(1).view(z.shape[0], z.shape[2], z.shape[3])
        quantized = self.lookup(codes)
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        loss = codebook_loss + self.commitment * commitment_loss
        straight_through = z + (quantized - z).detach()
        return straight_through, codes, loss
    def lookup(self, codes):
        return self.embedding(codes).permute(0, 3, 1, 2).contiguous()

class VQVAE(nn.Module):
    def __init__(self, size=256, dim=64, commitment=0.25):
        super().__init__()
        self.size, self.dim = size, dim
        self.encoder = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), Residual(128), Residual(128), nn.Conv2d(128, dim, 1))
        self.quantizer = VectorQuantizer(size, dim, commitment)
        self.decoder = nn.Sequential(nn.Conv2d(dim, 128, 3, padding=1), Residual(128), Residual(128),
            nn.ReLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid())
    def forward(self, x):
        quantized, codes, loss = self.quantizer(self.encoder(x))
        return self.decoder(quantized), codes, loss
    def codes(self, x):
        return self.quantizer(self.encoder(x))[1]
    def decode_codes(self, codes):
        return self.decoder(self.quantizer.lookup(codes))

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ('A', 'B')
        mask = torch.ones_like(self.weight)
        h, w = self.kernel_size
        mask[:, :, h // 2 + 1:, :] = 0
        mask[:, :, h // 2, w // 2 + (mask_type == 'B'):] = 0
        self.register_buffer('mask', mask)
    def forward(self, x):
        # Multiply in the graph: masked weights receive zero gradient, including after optimizer updates.
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

class PixelCNN(nn.Module):
    """Unconditional raster-order categorical prior on 8x8 discrete maps."""
    def __init__(self, size=256, channels=128, layers=8):
        super().__init__()
        self.size = size
        self.embedding = nn.Embedding(size, channels)
        self.first = MaskedConv2d('A', channels, channels, 7, padding=3)
        self.blocks = nn.ModuleList([MaskedConv2d('B', channels, channels, 3, padding=1) for _ in range(layers)])
        self.head = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, channels, 1), nn.ReLU(), nn.Conv2d(channels, size, 1))
    def forward(self, codes):
        h = self.first(self.embedding(codes).permute(0, 3, 1, 2))
        for block in self.blocks:
            h = h + block(F.relu(h))
        return self.head(h)
    @torch.no_grad()
    def sample(self, n, height=8, width=8, temperature=1.0):
        assert temperature > 0
        device = next(self.parameters()).device
        codes = torch.zeros(n, height, width, dtype=torch.long, device=device)
        for row in range(height):
            for col in range(width):
                probabilities = (self(codes)[:, :, row, col] / temperature).softmax(1)
                codes[:, row, col] = torch.multinomial(probabilities, 1).squeeze(1)
        return codes

class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid())
    def forward(self, z):
        return self.net(z.view(-1, self.latent_dim, 1, 1))
    def sample(self, n):
        return self(torch.randn(n, self.latent_dim, device=next(self.parameters()).device))

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        # No sigmoid and no batch normalization: each input has its own critic gradient.
        self.net = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(.2), nn.Conv2d(256, 1, 4))
    def forward(self, x):
        return self.net(x).view(-1)

def initialize_gan(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0, .02)
        if module.bias is not None: nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1, .02)
        nn.init.zeros_(module.bias)

def gradient_penalty(critic, real, fake):
    alpha = torch.rand(real.shape[0], 1, 1, 1, device=real.device)
    interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    values = critic(interpolated)
    gradients = torch.autograd.grad(values.sum(), interpolated, create_graph=True)[0]
    norms = gradients.flatten(1).norm(2, dim=1)
    return (norms - 1).square().mean(), norms.mean().detach()
