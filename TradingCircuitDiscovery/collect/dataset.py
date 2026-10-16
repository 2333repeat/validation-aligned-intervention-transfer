from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ActivationBank:
    records: list[dict]
    metadata: dict | None = None

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"records": self.records, "metadata": self.metadata or {}}, output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "ActivationBank":
        try:
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(Path(path), map_location="cpu")
        return cls(records=payload["records"], metadata=payload.get("metadata", {}))


def _flatten_activation(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def flatten_activation_dataset(records: list[dict], layer: str, concept: str) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for record in records:
        if layer not in record["activations"]:
            continue
        concepts = record.get("concepts", {})
        if concept not in concepts:
            continue
        x_rows.append(_flatten_activation(record["activations"][layer]))
        y_rows.append(int(concepts[concept]))
    if not x_rows:
        raise ValueError(f"No rows found for layer={layer!r}, concept={concept!r}")
    return np.vstack(x_rows), np.asarray(y_rows, dtype=np.int64)
