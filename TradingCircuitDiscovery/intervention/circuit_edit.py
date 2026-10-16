from __future__ import annotations

import torch


class CircuitEditor:
    @staticmethod
    def neuron_edit(activation: torch.Tensor, indices: list[int], value: float) -> torch.Tensor:
        edited = activation.clone()
        edited[..., indices] = value
        return edited

    @staticmethod
    def feature_scale(latent: torch.Tensor, feature_indices: list[int], scale: float) -> torch.Tensor:
        edited = latent.clone()
        edited[..., feature_indices] = edited[..., feature_indices] * scale
        return edited

    @staticmethod
    def sae_feature_edit(model, activation: torch.Tensor, feature_indices: list[int], scale: float) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            _reconstruction, latent = model(activation)
            edited_latent = CircuitEditor.feature_scale(latent, feature_indices, scale)
            return model.decoder(edited_latent)
