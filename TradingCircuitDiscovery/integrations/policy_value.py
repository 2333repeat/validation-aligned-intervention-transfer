from __future__ import annotations

import sys
from pathlib import Path

from TradingCircuitDiscovery.models.model_adapter import A2CAdapter, DQNAdapter, PPOAdapter, SACAdapter
from TradingCircuitDiscovery.utils.config import resolve_workspace_path


def _add_policy_value_to_path(project_root: str | Path | None = None) -> Path:
    project_dir = Path(project_root).expanduser().resolve() if project_root else resolve_workspace_path("Policy-Value")
    if not project_dir.exists():
        raise FileNotFoundError(f"Policy-Value project not found: {project_dir}")
    project_str = str(project_dir)
    if project_str not in sys.path:
        sys.path.insert(0, project_str)
    return project_dir


def _resolve_project_path(path: str | Path, project_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    parts = candidate.parts
    if parts and parts[0].lower() == "policy-value":
        candidate = Path(*parts[1:])
    project_candidate = (project_dir / candidate).resolve()
    return project_candidate if project_candidate.exists() else resolve_workspace_path(path)


def _load_agent_cfg(agent_config_path: str | Path, project_dir: Path):
    config_path = _resolve_project_path(agent_config_path, project_dir)
    try:
        from omegaconf import OmegaConf

        return OmegaConf.load(config_path)
    except ModuleNotFoundError:
        from types import SimpleNamespace

        from TradingCircuitDiscovery.utils.config import load_config

        return SimpleNamespace(**load_config(config_path))


def build_policy_value_adapter(
    algorithm: str,
    agent_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    layers: list[str] | None = None,
    project_root: str | Path | None = None,
):
    """Build an adapter for the existing Policy-Value PPO/DDQN project.

    The function only instantiates and optionally loads frozen models. It does
    not modify training code, trainers, rewards, or environment behavior.
    """

    project_dir = _add_policy_value_to_path(project_root)
    cfg = _load_agent_cfg(agent_config_path, project_dir)
    normalized = algorithm.upper().replace("_", "-")

    def resolved_checkpoint() -> str:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required")
        return str(_resolve_project_path(checkpoint_path, project_dir))

    if normalized == "PPO":
        from agents.PPOAgent import PPO

        agent = PPO(cfg)
        if checkpoint_path:
            agent.load_model(resolved_checkpoint())
        return PPOAdapter(agent, layers=layers or ["actor.fc1"])

    if normalized in {"DQN", "DDQN", "QR-DQN", "QRDQN"}:
        import torch
        from agents.DQNagent import DDQN

        agent = DDQN(cfg)
        if checkpoint_path:
            agent.load_model(resolved_checkpoint())
        return DQNAdapter(agent, layers=layers or ["policy_net.fc1"])

    if normalized == "A2C":
        from agents.A2CAgent import A2C

        agent = A2C(cfg)
        if checkpoint_path:
            agent.load_model(resolved_checkpoint())
        return A2CAdapter(agent, layers=layers or ["actor.fc1"])

    if normalized == "SAC":
        from agents.SACAgent import SAC

        agent = SAC(cfg)
        if checkpoint_path:
            agent.load_model(resolved_checkpoint())
        return SACAdapter(agent, layers=layers or ["actor.fc1"])

    raise ValueError(f"Unsupported Policy-Value algorithm: {algorithm}")
