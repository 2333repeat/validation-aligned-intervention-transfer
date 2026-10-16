from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


ALGORITHMS = ("PPO", "DDQN", "A2C", "SAC")
MARKETS = (
    "AAPL",
    "AMD",
    "AMZN",
    "DJI",
    "FCHI",
    "GE",
    "GOOGL",
    "HSI",
    "INTC",
    "IXIC",
    "KS11",
    "MSFT",
    "N225",
    "NVDA",
    "SP500",
)
SEEDS = (1, 2, 3)


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    algorithm: str
    market: str
    seed: int
    checkpoint_path: Path
    account_path: Path

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["checkpoint_path"] = str(self.checkpoint_path)
        payload["account_path"] = str(self.account_path)
        return payload


def build_full_manifest(policy_value_root: str | Path, validate_paths: bool = True) -> list[ExperimentRun]:
    root = Path(policy_value_root).expanduser().resolve()
    runs: list[ExperimentRun] = []
    for algorithm in ALGORITHMS:
        for market in MARKETS:
            for seed in SEEDS:
                run_id = f"{algorithm}-{market}-cuda-s{seed}"
                agent_dir = root / "result" / run_id / "agent"
                run = ExperimentRun(
                    run_id=run_id,
                    algorithm=algorithm,
                    market=market,
                    seed=seed,
                    checkpoint_path=agent_dir / "model" / "best_model.pth",
                    account_path=agent_dir / "agent_testAccount.csv",
                )
                runs.append(run)
    if validate_paths:
        missing = audit_manifest_paths(runs)
        if missing:
            raise FileNotFoundError(f"Missing files for {len(missing)} canonical runs: {missing[:3]}")
    return runs


def audit_manifest_paths(runs: list[ExperimentRun]) -> list[dict]:
    missing = []
    for run in runs:
        checkpoint_exists = run.checkpoint_path.is_file()
        account_exists = run.account_path.is_file()
        if not checkpoint_exists or not account_exists:
            missing.append(
                {
                    "run_id": run.run_id,
                    "checkpoint_exists": checkpoint_exists,
                    "account_exists": account_exists,
                }
            )
    return missing


def write_manifest(runs: list[ExperimentRun], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([run.to_dict() for run in runs], indent=2), encoding="utf-8")
    return path
