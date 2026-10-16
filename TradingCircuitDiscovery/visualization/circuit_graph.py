from __future__ import annotations

from pathlib import Path


def draw_circuit_graph(graph: dict, output_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = [node["id"] for node in graph.get("nodes", [])]
    y_positions = list(reversed(range(len(nodes))))
    fig, ax = plt.subplots(figsize=(7, max(3, len(nodes) * 0.4)))
    for node, y in zip(nodes, y_positions):
        ax.scatter([0], [y], s=250)
        ax.text(0.08, y, node, va="center")
    node_to_y = dict(zip(nodes, y_positions))
    for edge in graph.get("edges", []):
        if edge["source"] in node_to_y and edge["target"] in node_to_y:
            ax.annotate(
                "",
                xy=(0, node_to_y[edge["target"]]),
                xytext=(0, node_to_y[edge["source"]]),
                arrowprops={"arrowstyle": "->"},
            )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
