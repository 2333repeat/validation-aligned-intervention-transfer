from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch
from torch import nn

from TradingCircuitDiscovery.intervention.activation_patch import patch_layer_output, resolve_module


class RLModelAdapter:
    supports_batch_scores = True

    def __init__(self, agent, layers: list[str] | None = None):
        self.agent = agent
        self.layers = layers or []
        self.device = getattr(agent, "device", "cpu")

    def get_action(self, state) -> int:
        distribution = self.get_action_distribution(state)
        return int(torch.argmax(distribution).item())

    def get_action_distribution(self, state) -> torch.Tensor:
        return torch.softmax(self.get_action_scores(state), dim=-1)

    def get_action_scores(self, state) -> torch.Tensor:
        raise NotImplementedError

    def get_hidden(self, state) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @contextmanager
    def patch_hidden(self, layer: str, value: torch.Tensor) -> Iterator[None]:
        root, sub_layer = self._split_layer(layer)
        module = getattr(self.agent, root)
        with patch_layer_output(module, sub_layer, value):
            yield

    @contextmanager
    def intervene_hidden(
        self,
        layer: str,
        indices: list[int],
        values: float | torch.Tensor,
    ) -> Iterator[None]:
        root, sub_layer = self._split_layer(layer)
        module = resolve_module(getattr(self.agent, root), sub_layer)

        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("Indexed hidden intervention requires a tensor layer output")
            edited = output.clone()
            replacement = torch.as_tensor(values, device=edited.device, dtype=edited.dtype)
            edited[..., indices] = replacement
            return edited

        handle = module.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def _state_tensor(self, state) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            tensor = state.detach().clone().float()
        else:
            tensor = torch.tensor(np.asarray(state), dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def _split_layer(self, layer: str) -> tuple[str, str]:
        if "." not in layer:
            raise ValueError(f"Layer must include model prefix, got {layer!r}")
        root, sub_layer = layer.split(".", 1)
        if not hasattr(self.agent, root):
            raise KeyError(f"Agent has no module {root!r}")
        return root, sub_layer

    def _record_prefixed_layers(self, state, forward_fn) -> dict[str, torch.Tensor]:
        handles = []
        storage: dict[str, torch.Tensor] = {}
        for layer in self.layers:
            root, sub_layer = self._split_layer(layer)
            module = resolve_module(getattr(self.agent, root), sub_layer)

            def make_hook(name):
                def hook(_module, _inputs, output):
                    tensor = output if isinstance(output, torch.Tensor) else output[0]
                    storage[name] = tensor.detach().clone().cpu()

                return hook

            handles.append(module.register_forward_hook(make_hook(layer)))
        try:
            _ = forward_fn(self._state_tensor(state))
        finally:
            for handle in handles:
                handle.remove()
        return storage


class PPOAdapter(RLModelAdapter):
    def get_action_scores(self, state) -> torch.Tensor:
        self.agent.actor.eval()
        with torch.no_grad():
            probs = self.agent.actor(self._state_tensor(state)).squeeze(0).detach().cpu()
        return probs

    def get_action_distribution(self, state) -> torch.Tensor:
        return self.get_action_scores(state)

    def get_hidden(self, state) -> dict[str, torch.Tensor]:
        return self._record_prefixed_layers(state, self.agent.actor)


class DQNAdapter(RLModelAdapter):
    def get_action_scores(self, state) -> torch.Tensor:
        self.agent.policy_net.eval()
        with torch.no_grad():
            q_values = self.agent.policy_net(self._state_tensor(state)).squeeze(0).detach().cpu()
        return q_values

    def get_hidden(self, state) -> dict[str, torch.Tensor]:
        return self._record_prefixed_layers(state, self.agent.policy_net)

    def get_action(self, state) -> int:
        return int(torch.argmax(self.get_action_scores(state)).item())


class A2CAdapter(PPOAdapter):
    """A2C shares the probability actor contract used by PPO."""


class SACAdapter(RLModelAdapter):
    """Adapter slot for SAC actors supplied by downstream projects."""

    def __init__(self, agent, actor_attr: str = "actor", layers: list[str] | None = None):
        super().__init__(agent, layers=layers)
        self.actor_attr = actor_attr
        if not hasattr(agent, actor_attr):
            raise ValueError("SACAdapter requires an agent with an actor module.")

    def get_action_scores(self, state) -> torch.Tensor:
        actor: nn.Module = getattr(self.agent, self.actor_attr)
        actor.eval()
        with torch.no_grad():
            output = actor(self._state_tensor(state))
            if isinstance(output, tuple):
                output = output[0]
            return output.squeeze(0).detach().cpu()

    def get_hidden(self, state) -> dict[str, torch.Tensor]:
        return self._record_prefixed_layers(state, getattr(self.agent, self.actor_attr))
