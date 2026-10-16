from __future__ import annotations

from pathlib import Path


def visualize_feature_activation(values, output_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(list(values))
    ax.set_xlabel("sample")
    ax.set_ylabel("activation")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
