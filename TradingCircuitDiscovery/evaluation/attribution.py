from __future__ import annotations

import numpy as np
import torch

from TradingCircuitDiscovery.intervention.activation_patch import resolve_module


def _top_indices(values, top_k: int) -> list[int]:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return np.argsort(-np.abs(scores), kind="stable")[: min(top_k, len(scores))].astype(int).tolist()


def rank_attribution_baselines(
    activations,
    labels,
    probe_coefficients,
    gradient_x_activation,
    integrated_gradients,
    top_k: int,
) -> dict[str, list[int]]:
    values = np.asarray(activations, dtype=np.float64)
    targets = np.asarray(labels, dtype=int)
    if values.ndim != 2 or targets.shape != (len(values),):
        raise ValueError("activations and labels must be paired")
    if set(np.unique(targets).tolist()) != {0, 1}:
        raise ValueError("attribution ranking requires binary labels")
    activation_difference = values[targets == 1].mean(axis=0) - values[targets == 0].mean(axis=0)
    activation_magnitude = np.mean(np.abs(values), axis=0)
    return {
        "activation_difference": _top_indices(activation_difference, top_k),
        "probe_weight": _top_indices(probe_coefficients, top_k),
        "activation_magnitude": _top_indices(activation_magnitude, top_k),
        "gradient_x_activation": _top_indices(gradient_x_activation, top_k),
        "integrated_gradients": _top_indices(integrated_gradients, top_k),
    }


def _forward_with_layer(adapter, states, layer: str, replacement: torch.Tensor | None = None):
    root, sub_layer = adapter._split_layer(layer)
    model = getattr(adapter.agent, root)
    model.eval()
    target_module = resolve_module(model, sub_layer)
    storage = {}

    def hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Hidden attribution requires tensor layer outputs")
        if replacement is None:
            storage["hidden"] = output
            return output
        patched = replacement.to(device=output.device, dtype=output.dtype)
        storage["hidden"] = patched
        return patched

    handle = target_module.register_forward_hook(hook)
    try:
        scores = model(adapter._state_tensor(states))
        if isinstance(scores, tuple):
            scores = scores[0]
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
    finally:
        handle.remove()
    return scores, storage["hidden"]


def compute_hidden_attributions(
    adapter,
    states,
    layer: str,
    baseline,
    ig_steps: int = 8,
) -> dict[str, np.ndarray]:
    if ig_steps <= 0:
        raise ValueError("ig_steps must be positive")
    original_scores, hidden = _forward_with_layer(adapter, states, layer)
    actions = original_scores.detach().argmax(dim=1)
    baseline_tensor = torch.as_tensor(baseline, device=hidden.device, dtype=hidden.dtype).reshape(1, -1)
    baseline_tensor = baseline_tensor.expand_as(hidden)
    objective = original_scores.gather(1, actions[:, None]).sum()
    gradients = torch.autograd.grad(objective, hidden, retain_graph=False, create_graph=False)[0]
    delta = hidden.detach() - baseline_tensor
    gradient_x_activation = torch.mean(torch.abs(gradients.detach() * delta), dim=0)

    accumulated = torch.zeros_like(hidden)
    for alpha in torch.linspace(1.0 / ig_steps, 1.0, ig_steps, device=hidden.device):
        interpolated = (baseline_tensor + alpha * delta).detach().requires_grad_(True)
        scores, patched = _forward_with_layer(adapter, states, layer, replacement=interpolated)
        objective = scores.gather(1, actions[:, None]).sum()
        step_gradient = torch.autograd.grad(objective, patched, retain_graph=False, create_graph=False)[0]
        accumulated += step_gradient.detach()
    integrated = torch.mean(torch.abs(delta * accumulated / float(ig_steps)), dim=0)
    return {
        "gradient_x_activation": gradient_x_activation.detach().cpu().numpy(),
        "integrated_gradients": integrated.detach().cpu().numpy(),
    }

