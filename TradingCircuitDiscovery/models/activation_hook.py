from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch import nn


def _first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Hook output is not tensor-like: {type(output)!r}")


@dataclass
class ActivationRecorder:
    layers: list[str] | None = None
    detach_cpu: bool = True
    storage: dict[str, torch.Tensor] = field(default_factory=dict)

    def hook(self, name: str):
        def fn(_module, _inputs, output):
            tensor = _first_tensor(output)
            if self.detach_cpu:
                tensor = tensor.detach().clone().cpu()
            self.storage[name] = tensor

        return fn

    def clear(self) -> None:
        self.storage.clear()

    @contextmanager
    def record(self, model: nn.Module) -> Iterator["ActivationRecorder"]:
        handles = []
        self.clear()
        layer_filter = set(self.layers or [])
        for name, module in model.named_modules():
            if not name:
                continue
            if layer_filter and name not in layer_filter:
                continue
            handles.append(module.register_forward_hook(self.hook(name)))
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()
