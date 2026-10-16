from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_market_random_intercept(frame: pd.DataFrame) -> list[dict]:
    model = smf.mixedlm(
        "effect ~ C(algorithm) + C(concept) + C(policy_label)",
        data=frame,
        groups=frame["market"],
    )
    candidates = []
    errors = []
    for optimizer in ("lbfgs", "powell", "bfgs", "cg"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(
                    reml=False,
                    method=optimizer,
                    maxiter=1000,
                    disp=False,
                )
            finite = (
                bool(result.converged)
                and np.isfinite(result.llf)
                and np.all(np.isfinite(result.fe_params.to_numpy()))
                and np.all(np.isfinite(result.cov_re.to_numpy()))
            )
            if finite:
                candidates.append(
                    (float(result.llf), optimizer, result, [str(item.message) for item in caught])
                )
            else:
                errors.append(f"{optimizer}: non-finite or unconverged fit")
        except Exception as exc:
            errors.append(f"{optimizer}: {exc}")
    if not candidates:
        raise RuntimeError("No finite MixedLM fit: " + "; ".join(errors))
    log_likelihood, optimizer, fitted, optimizer_warnings = max(
        candidates,
        key=lambda item: item[0],
    )
    confidence = fitted.conf_int()
    rows = []
    for term, estimate in fitted.fe_params.items():
        rows.append(
            {
                "method": "mixedlm_market_random_intercept",
                "term": str(term),
                "estimate": float(estimate),
                "std_error": float(fitted.bse_fe[term]),
                "ci_low": float(confidence.loc[term, 0]),
                "ci_high": float(confidence.loc[term, 1]),
                "p_value": float(fitted.pvalues[term]),
                "status": "completed",
                "observations": int(fitted.nobs),
                "markets": int(frame["market"].nunique()),
                "converged": bool(fitted.converged),
                "optimizer": optimizer,
                "log_likelihood": log_likelihood,
                "optimizer_warnings": " | ".join(optimizer_warnings),
            }
        )
    rows.append(
        {
            "method": "mixedlm_market_random_intercept",
            "term": "market_random_intercept_variance",
            "estimate": float(fitted.cov_re.iloc[0, 0]),
            "status": "completed",
            "observations": int(fitted.nobs),
            "markets": int(frame["market"].nunique()),
            "converged": bool(fitted.converged),
            "optimizer": optimizer,
            "log_likelihood": log_likelihood,
            "optimizer_warnings": " | ".join(optimizer_warnings),
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = fit_market_random_intercept(pd.read_csv(args.input))
    Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
