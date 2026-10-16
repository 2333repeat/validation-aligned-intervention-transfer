from __future__ import annotations


def build_dashboard_summary(metrics: dict, circuits: dict) -> dict:
    return {
        "n_circuits": len(circuits),
        "n_metrics": len(metrics),
        "metrics": metrics,
        "circuits": circuits,
    }
