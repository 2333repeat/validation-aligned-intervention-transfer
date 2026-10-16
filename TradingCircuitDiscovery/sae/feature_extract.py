from __future__ import annotations

import torch


def extract_top_features(model, activations: torch.Tensor, top_k: int = 10) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        _reconstruction, latent = model(activations)
    mean_activation = latent.mean(dim=0)
    values, indices = torch.topk(mean_activation, k=min(top_k, latent.shape[1]))
    return {"indices": indices.cpu(), "values": values.cpu(), "activations": latent.detach().cpu()}
