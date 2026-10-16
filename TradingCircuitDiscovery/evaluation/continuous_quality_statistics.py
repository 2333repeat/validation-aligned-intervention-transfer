from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_continuous_quality_model(
    frame: pd.DataFrame,
    quality_column: str,
    model_name: str,
    excluded_concepts: tuple[str, ...] = (),
) -> list[dict]:
    selected = frame.copy()
    if excluded_concepts:
        selected = selected.loc[~selected["concept"].isin(excluded_concepts)].copy()
    required = {
        "effect",
        "algorithm",
        "concept",
        "market",
        "run_id",
        quality_column,
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"continuous quality model missing columns: {missing}")
    selected = selected.dropna(subset=list(required)).copy()
    if len(selected) < 24 or selected["market"].nunique() < 3:
        raise ValueError("continuous quality model requires at least 24 rows and 3 markets")
    quality_mean = float(selected[quality_column].mean())
    quality_std = float(selected[quality_column].std(ddof=0))
    if not np.isfinite(quality_std) or quality_std <= 0:
        raise ValueError(f"{quality_column} must have positive finite variance")
    selected["quality_z"] = (selected[quality_column] - quality_mean) / quality_std
    model = smf.mixedlm(
        "effect ~ C(algorithm) * quality_z + C(concept)",
        data=selected,
        groups=selected["market"],
        re_formula="1",
        vc_formula={"policy": "0 + C(run_id)"},
    )
    candidates = []
    errors = []
    for optimizer in ("lbfgs", "powell", "bfgs", "cg"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = model.fit(
                    reml=False,
                    method=optimizer,
                    maxiter=1500,
                    disp=False,
                )
            finite = (
                bool(fitted.converged)
                and np.isfinite(fitted.llf)
                and np.all(np.isfinite(fitted.fe_params.to_numpy()))
                and np.all(np.isfinite(fitted.cov_re.to_numpy()))
                and np.all(np.isfinite(np.asarray(fitted.vcomp)))
            )
            if finite:
                candidates.append(
                    (
                        float(fitted.llf),
                        optimizer,
                        fitted,
                        [str(item.message) for item in caught],
                    )
                )
            else:
                errors.append(f"{optimizer}: non-finite or unconverged fit")
        except Exception as exc:
            errors.append(f"{optimizer}: {exc}")
    if not candidates:
        raise RuntimeError("No finite continuous-quality MixedLM fit: " + "; ".join(errors))
    log_likelihood, optimizer, fitted, optimizer_warnings = max(
        candidates,
        key=lambda item: item[0],
    )
    confidence = fitted.conf_int()
    common = {
        "method": "continuous_quality_mixedlm",
        "model_name": model_name,
        "quality_metric": quality_column,
        "quality_mean": quality_mean,
        "quality_std": quality_std,
        "excluded_concepts": ",".join(excluded_concepts),
        "status": "completed",
        "observations": int(fitted.nobs),
        "policies": int(selected["run_id"].nunique()),
        "markets": int(selected["market"].nunique()),
        "optimizer": optimizer,
        "log_likelihood": log_likelihood,
        "optimizer_warnings": " | ".join(optimizer_warnings),
    }
    rows = []
    for term, estimate in fitted.fe_params.items():
        rows.append(
            {
                **common,
                "term": str(term),
                "estimate": float(estimate),
                "std_error": float(fitted.bse_fe[term]),
                "ci_low": float(confidence.loc[term, 0]),
                "ci_high": float(confidence.loc[term, 1]),
                "p_value": float(fitted.pvalues[term]),
            }
        )
    fixed_terms = list(fitted.fe_params.index)
    fixed_covariance = fitted.cov_params().loc[fixed_terms, fixed_terms]
    base_quality_term = "quality_z"
    for algorithm in sorted(selected["algorithm"].unique().tolist()):
        interaction = f"C(algorithm)[T.{algorithm}]:quality_z"
        weights = pd.Series(0.0, index=fixed_terms)
        weights[base_quality_term] = 1.0
        if interaction in weights.index:
            weights[interaction] = 1.0
        estimate = float(weights @ fitted.fe_params)
        variance = float(weights @ fixed_covariance @ weights)
        std_error = math.sqrt(max(variance, 0.0))
        z_value = estimate / std_error if std_error > 0 else math.copysign(math.inf, estimate)
        p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
        rows.append(
            {
                **common,
                "term": f"quality_slope[{algorithm}]",
                "estimate": estimate,
                "std_error": std_error,
                "ci_low": estimate - 1.959963984540054 * std_error,
                "ci_high": estimate + 1.959963984540054 * std_error,
                "p_value": p_value,
            }
        )
    rows.append(
        {
            **common,
            "term": "market_random_intercept_variance",
            "estimate": float(fitted.cov_re.iloc[0, 0]),
        }
    )
    policy_variance = float(np.asarray(fitted.vcomp).reshape(-1)[0])
    rows.append(
        {
            **common,
            "term": "policy_random_intercept_variance",
            "estimate": policy_variance,
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality-column", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--exclude-concept", action="append", default=[])
    args = parser.parse_args()
    rows = fit_continuous_quality_model(
        pd.read_csv(args.input),
        quality_column=args.quality_column,
        model_name=args.model_name,
        excluded_concepts=tuple(args.exclude_concept),
    )
    Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
