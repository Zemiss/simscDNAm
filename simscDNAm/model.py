"""直接提取 train.ipynb 第 12 单元格的模型、损失与 DDIM。"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(
        ((steps / timesteps + offset) / (1.0 + offset)) * math.pi / 2.0
    ).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    return (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, 0.999).float()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = torch.cat(
                [embedding, embedding.new_zeros((len(timesteps), 1))], dim=1
            )
        return embedding


class FiLMResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.update = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, hidden: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        scale, shift = self.film(context).chunk(2, dim=1)
        modulated = self.norm(hidden) * (1.0 + scale) + shift
        return hidden + self.update(modulated)


class LatentDenoiser(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_classes: int,
        condition_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        blocks: int,
    ):
        super().__init__()
        self.condition_token = nn.Embedding(n_classes, condition_dim)
        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.time_projection = nn.Sequential(
            SinusoidalTimeEmbedding(condition_dim),
            nn.Linear(condition_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(hidden_dim, dropout_rate) for _ in range(blocks)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        noisy_z: torch.Tensor,
        timesteps: torch.Tensor,
        class_index: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(noisy_z)
        context = self.time_projection(timesteps) + self.condition_projection(
            self.condition_token(class_index)
        )
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.output(hidden)


class LatentDiffusionVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        latent_dim: int = 32,
        condition_dim: int = 32,
        hidden_dim: int = 256,
        dropout_rate: float = 0.1,
        diffusion_timesteps: int = 1_000,
        denoiser_blocks: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.diffusion_timesteps = diffusion_timesteps
        self.register_buffer("input_center", torch.zeros(input_dim))
        self.register_buffer("latent_center", torch.zeros(latent_dim))
        self.register_buffer("latent_scale", torch.ones(latent_dim))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )
        self.denoiser = LatentDenoiser(
            latent_dim,
            n_classes,
            condition_dim,
            hidden_dim,
            dropout_rate,
            denoiser_blocks,
        )
        betas = cosine_beta_schedule(diffusion_timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer(
            "alphas_cumprod", torch.cumprod(1.0 - betas, dim=0)
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x - self.input_center)
        return self.mu(hidden), self.logvar(hidden).clamp(-12.0, 12.0)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(mu)
        return mu + noise * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def add_noise(
        self,
        z0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod[timesteps].unsqueeze(1)
        return torch.sqrt(alpha_bar) * z0 + torch.sqrt(1.0 - alpha_bar) * noise

    def predict_velocity(
        self,
        noisy_z: torch.Tensor,
        timesteps: torch.Tensor,
        class_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoiser(noisy_z, timesteps, class_index)

    def normalize_latent(self, z):
        return (z - self.latent_center) / self.latent_scale

    def denormalize_latent(self, z):
        return z * self.latent_scale + self.latent_center

    def velocity_target(self, z0, timesteps, noise):
        alpha = self.alphas_cumprod[timesteps].unsqueeze(1)
        return alpha.sqrt() * noise - (1.0 - alpha).sqrt() * z0

    def velocity_to_clean(self, noisy_z, timesteps, velocity):
        alpha = self.alphas_cumprod[timesteps].unsqueeze(1)
        clean = alpha.sqrt() * noisy_z - (1.0 - alpha).sqrt() * velocity
        noise = (1.0 - alpha).sqrt() * noisy_z + alpha.sqrt() * velocity
        return clean, noise


def latent_diffusion_vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    beta: float,
    diffusion_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observed_count = observed.sum()
    if observed_count.item() == 0:
        raise ValueError("VAE 批次没有甲基化观测")
    reconstruction_loss = (
        (reconstruction - target).square().masked_select(observed).sum()
        / observed_count
    )
    kl_loss = -0.5 * (
        1.0 + logvar - mu.square() - logvar.exp()
    ).sum(dim=1).mean()
    diffusion_loss = (predicted_noise - target_noise).square().mean()
    total = reconstruction_loss + beta * kl_loss + diffusion_weight * diffusion_loss
    return total, reconstruction_loss, kl_loss, diffusion_loss


@torch.no_grad()
def sample_latent_ddim(model, class_index, sampling_steps, seed):
    if not 1 <= sampling_steps <= model.diffusion_timesteps:
        raise ValueError("采样步数必须位于扩散日程范围内")
    device = class_index.device
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn((len(class_index), model.latent_dim), generator=generator, device=device)
    times = torch.linspace(model.diffusion_timesteps - 1, 0, sampling_steps, device=device, dtype=torch.long)
    for index, timestep in enumerate(times):
        batch_t = timestep.expand(len(class_index))
        velocity = model.predict_velocity(latent, batch_t, class_index)
        clean, epsilon = model.velocity_to_clean(latent, batch_t, velocity)
        if index == len(times) - 1:
            latent = clean
        else:
            alpha = model.alphas_cumprod[times[index + 1]]
            latent = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * epsilon
    return model.denormalize_latent(latent)


def load_cvae_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("diffusion_prediction") != "v":
        raise ValueError("此加载器需要新版本 v-prediction 模型，请重新训练；旧权重已备份")
    model = LatentDiffusionVAE(checkpoint["input_dim"], len(checkpoint["class_names"]),
                              checkpoint["latent_dim"], checkpoint["condition_dim"],
                              checkpoint["hidden_dim"], checkpoint["dropout_rate"],
                              checkpoint["diffusion_timesteps"], checkpoint["denoiser_blocks"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
