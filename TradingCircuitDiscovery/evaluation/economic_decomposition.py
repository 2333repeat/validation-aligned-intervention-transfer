from __future__ import annotations

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = (
    "run_id",
    "algorithm",
    "market",
    "seed",
    "concept",
    "policy_label",
    "family",
    "selected_layer",
)


def decompose_economic_channels(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "run_id",
        "concept",
        "transition_cost",
        "baseline_pnl",
        "intervention_pnl",
        "delta_pnl",
        "delta_turnover",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"economic decomposition missing columns: {missing}")
    group_columns = [column for column in IDENTITY_COLUMNS if column in frame.columns]
    rows = []
    for _keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        zero_cost = group.loc[np.isclose(group["transition_cost"].astype(float), 0.0)]
        if len(zero_cost) != 1:
            raise ValueError("each run/concept group requires exactly one zero-cost row")
        zero = zero_cost.iloc[0]
        gross_component = float(zero["delta_pnl"])
        for _, row in group.iterrows():
            baseline_drag = float(row["baseline_pnl"] - zero["baseline_pnl"])
            intervention_drag = float(
                row["intervention_pnl"] - zero["intervention_pnl"]
            )
            differential_cost = intervention_drag - baseline_drag
            reconstructed = gross_component + differential_cost
            output = row.to_dict()
            output.update(
                {
                    "gross_behavior_component": gross_component,
                    "baseline_cost_drag": baseline_drag,
                    "intervention_cost_drag": intervention_drag,
                    "differential_cost_component": differential_cost,
                    "reconstructed_delta_pnl": reconstructed,
                    "decomposition_residual": float(row["delta_pnl"] - reconstructed),
                }
            )
            rows.append(output)
    return pd.DataFrame(rows)
