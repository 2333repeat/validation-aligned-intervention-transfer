from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from TradingCircuitDiscovery.models.sae import SparseAutoencoder, sae_loss


def train_sae_step(model, optimizer, batch: torch.Tensor, l1_lambda: float = 0.001) -> dict[str, float]:
    model.train()
    optimizer.zero_grad()
    reconstruction, latent = model(batch)
    loss = sae_loss(batch, reconstruction, latent, l1_lambda=l1_lambda)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach().cpu()), "l1": float(latent.abs().mean().detach().cpu())}


@dataclass
class SAEExperimentResult:
    model: SparseAutoencoder
    latent_dim: int
    best_epoch: int
    validation_loss: float
    validation_metrics: dict[str, float]
    candidates: list[dict]


@dataclass
class SAESparsitySweepResult:
    model: SparseAutoencoder
    selected_candidate: dict
    candidates: list[dict]


def evaluate_sae(model, activations: torch.Tensor) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    values = activations.to(device)
    with torch.no_grad():
        reconstruction, latent = model(values)
    residual = values - reconstruction
    mse = float(torch.mean(residual.square()).cpu())
    input_variance = float(torch.var(values, unbiased=False).cpu())
    residual_variance = float(torch.var(residual, unbiased=False).cpu())
    explained_variance = 1.0 - residual_variance / max(input_variance, 1e-12)
    active = latent > 1e-8
    l0 = float(active.sum(dim=1).float().mean().cpu())
    return {
        "reconstruction_mse": mse,
        "explained_variance": float(explained_variance),
        "l0": l0,
        "normalized_l0": l0 / float(latent.shape[1]),
        "l1": float(latent.abs().mean().cpu()),
        "dead_feature_rate": float((~active.any(dim=0)).float().mean().cpu()),
    }


def _fit_sae_model(
    train_activations: torch.Tensor,
    validation_activations: torch.Tensor,
    latent_dim: int,
    epochs: int,
    patience: int,
    l1_lambda: float,
    learning_rate: float,
    random_state: int,
    device: str,
    top_k: int | None = None,
) -> tuple[SparseAutoencoder, int, float, dict[str, float]]:
    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))
    model = SparseAutoencoder(
        input_dim=int(train_activations.shape[1]),
        latent_dim=int(latent_dim),
        top_k=top_k,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(int(epochs)):
        train_sae_step(model, optimizer, train_activations.to(device), l1_lambda=l1_lambda)
        model.eval()
        with torch.no_grad():
            reconstruction, latent = model(validation_activations.to(device))
            validation_loss = float(
                sae_loss(
                    validation_activations.to(device),
                    reconstruction,
                    latent,
                    l1_lambda=l1_lambda,
                ).cpu()
            )
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(patience):
                break
    if best_state is None:
        raise RuntimeError("SAE training did not produce a checkpoint")
    model.load_state_dict(best_state)
    metrics = evaluate_sae(model, validation_activations)
    return model, best_epoch, best_loss, metrics


def fit_sae_sparsity_sweep(
    train_activations: torch.Tensor,
    validation_activations: torch.Tensor,
    expansion_factors=(2, 4, 8),
    l1_lambdas=(0.001, 0.01, 0.05),
    topk_values=(4, 8, 16, 32),
    epochs: int = 40,
    patience: int = 6,
    learning_rate: float = 1e-3,
    random_state: int = 1,
    device: str = "cpu",
    explained_variance_tolerance: float = 0.02,
) -> SAESparsitySweepResult:
    if train_activations.ndim != 2 or validation_activations.ndim != 2:
        raise ValueError("SAE activations must be two-dimensional")
    if train_activations.shape[1] != validation_activations.shape[1]:
        raise ValueError("Training and validation feature dimensions must match")
    input_dim = int(train_activations.shape[1])
    candidates = []
    models = []
    for factor in expansion_factors:
        latent_dim = input_dim * int(factor)
        specifications = [
            ("l1", float(l1_lambda), None) for l1_lambda in l1_lambdas
        ] + [
            ("topk", 0.0, int(top_k)) for top_k in topk_values if int(top_k) <= latent_dim
        ]
        for mode, l1_lambda, top_k in specifications:
            model, best_epoch, validation_loss, metrics = _fit_sae_model(
                train_activations,
                validation_activations,
                latent_dim=latent_dim,
                epochs=epochs,
                patience=patience,
                l1_lambda=l1_lambda,
                learning_rate=learning_rate,
                random_state=random_state,
                device=device,
                top_k=top_k,
            )
            candidates.append(
                {
                    "expansion_factor": int(factor),
                    "latent_dim": latent_dim,
                    "sparsity_mode": mode,
                    "l1_lambda": l1_lambda if mode == "l1" else None,
                    "top_k": top_k,
                    "best_epoch": best_epoch,
                    "validation_loss": validation_loss,
                    **metrics,
                    "pareto_selected": False,
                }
            )
            models.append(model.cpu())
    if not candidates:
        raise ValueError("At least one SAE candidate is required")
    best_explained_variance = max(item["explained_variance"] for item in candidates)
    eligible = [
        index
        for index, item in enumerate(candidates)
        if item["explained_variance"] >= best_explained_variance - explained_variance_tolerance
    ]
    selected_index = min(
        eligible,
        key=lambda index: (
            candidates[index]["normalized_l0"],
            candidates[index]["reconstruction_mse"],
            candidates[index]["dead_feature_rate"],
        ),
    )
    candidates[selected_index]["pareto_selected"] = True
    return SAESparsitySweepResult(
        model=models[selected_index],
        selected_candidate=candidates[selected_index],
        candidates=candidates,
    )


def fit_sae_candidates(
    train_activations: torch.Tensor,
    validation_activations: torch.Tensor,
    expansion_factors=(2, 4, 8),
    epochs: int = 100,
    patience: int = 10,
    l1_lambda: float = 0.001,
    learning_rate: float = 1e-3,
    random_state: int = 1,
    device: str = "cpu",
) -> SAEExperimentResult:
    if train_activations.ndim != 2 or validation_activations.ndim != 2:
        raise ValueError("SAE activations must be two-dimensional")
    if train_activations.shape[1] != validation_activations.shape[1]:
        raise ValueError("Training and validation feature dimensions must match")
    input_dim = int(train_activations.shape[1])
    candidates = []
    best_result = None
    for factor in expansion_factors:
        torch.manual_seed(int(random_state))
        np.random.seed(int(random_state))
        latent_dim = input_dim * int(factor)
        model = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_loss = float("inf")
        best_epoch = -1
        best_state = None
        epochs_without_improvement = 0
        for epoch in range(int(epochs)):
            train_sae_step(model, optimizer, train_activations.to(device), l1_lambda=l1_lambda)
            model.eval()
            with torch.no_grad():
                reconstruction, latent = model(validation_activations.to(device))
                validation_loss = float(
                    sae_loss(
                        validation_activations.to(device),
                        reconstruction,
                        latent,
                        l1_lambda=l1_lambda,
                    ).cpu()
                )
            if validation_loss < best_loss - 1e-10:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(patience):
                    break
        if best_state is None:
            raise RuntimeError("SAE training did not produce a checkpoint")
        model.load_state_dict(best_state)
        metrics = evaluate_sae(model, validation_activations)
        candidate = {
            "latent_dim": latent_dim,
            "best_epoch": best_epoch,
            "validation_loss": best_loss,
            **metrics,
        }
        candidates.append(candidate)
        if best_result is None or best_loss < best_result.validation_loss:
            best_result = SAEExperimentResult(
                model=model.cpu(),
                latent_dim=latent_dim,
                best_epoch=best_epoch,
                validation_loss=best_loss,
                validation_metrics=metrics,
                candidates=candidates,
            )
    if best_result is None:
        raise ValueError("At least one SAE expansion factor is required")
    best_result.candidates = candidates
    return best_result
