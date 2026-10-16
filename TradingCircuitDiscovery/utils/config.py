from __future__ import annotations

from pathlib import Path
from typing import Any


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (workspace_root() / candidate).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_workspace_path(path)
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except ModuleNotFoundError:
        from omegaconf import OmegaConf

        data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data
