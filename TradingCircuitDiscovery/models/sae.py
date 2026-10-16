from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, top_k: int | None = None):
        super().__init__()
        if top_k is not None and (top_k <= 0 or top_k > latent_dim):
            raise ValueError("top_k must be in [1, latent_dim]")
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = F.relu(self.encoder(x))
        if self.top_k is not None and self.top_k < latent.shape[-1]:
            values, indices = torch.topk(latent, k=self.top_k, dim=-1)
            sparse = torch.zeros_like(latent)
            latent = sparse.scatter(-1, indices, values)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


def sae_loss(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    latent: torch.Tensor,
    l1_lambda: float = 0.001,
) -> torch.Tensor:
    reconstruction_loss = F.mse_loss(reconstruction, x)
    sparse_penalty = latent.abs().mean()
    return reconstruction_loss + l1_lambda * sparse_penalty
