from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


def resolve_module(model: nn.Module, layer_name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise KeyError(f"Layer {layer_name!r} not found. Available: {sorted(modules)[:20]}")
    return modules[layer_name]


@contextmanager
def patch_layer_output(model: nn.Module, layer_name: str, activation: torch.Tensor) -> Iterator[None]:
    module = resolve_module(model, layer_name)

    def hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            return activation.to(device=output.device, dtype=output.dtype)
        if isinstance(output, tuple):
            patched = activation.to(device=output[0].device, dtype=output[0].dtype)
            return (patched, *output[1:])
        return activation

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def patch_layer_indices(
    model: nn.Module,
    layer_name: str,
    indices: list[int],
    values: float | torch.Tensor,
) -> Iterator[None]:
    module = resolve_module(model, layer_name)

    def hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Indexed patching requires a tensor layer output")
        edited = output.clone()
        replacement = torch.as_tensor(values, device=edited.device, dtype=edited.dtype)
        edited[..., indices] = replacement
        return edited

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
