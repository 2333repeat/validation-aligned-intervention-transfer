from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from TradingEnv import TradingEnv
from agents.A2CAgent import A2C
from agents.DQNagent import DDQN
from agents.PPOAgent import PPO
from agents.SACAgent import SAC
from trainers.A2CTrainer import A2CTrainer
from trainers.DQNTrainer import DQNTrainer
from trainers.PPOTrainer import PPOTrainer
from trainers.SACTrainer import SACTrainer
from utils.runtime import resolve_device
from utils.utils import setup_seed


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Unified Policy-Value training entry point. "
            "Supports PPO, DDQN, A2C, and discrete-action SAC with a shared "
            "environment-step budget for smoke tests and large baseline runs."
        )
    )
    ap.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "DDQN", "A2C", "SAC"])
    ap.add_argument("--dataset", type=str, required=True, help="Dataset name without the .csv suffix, e.g., HSI/KS11/IXIC.")
    ap.add_argument("--exp-name", type=str, default="", help="Output run name; generated automatically when omitted.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda", help="auto | cuda | cpu")

    ap.add_argument("--train-start", type=str, default="2012-01-01")
    ap.add_argument("--train-end", type=str, default="2021-12-31")
    ap.add_argument("--test-start", type=str, default="2022-01-01")
    ap.add_argument("--test-end", type=str, default="2024-12-31")

    ap.add_argument("--total-env-steps", type=int, default=200000)
    ap.add_argument("--eval-every-steps", type=int, default=10000)
    ap.add_argument("--learning-starts", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--buffer-size", type=int, default=100000)
    ap.add_argument("--save-best-by", type=str, default="PnL", choices=["PnL", "Sharpe", "CR"])
    ap.add_argument("--random-reset", action="store_true")
    ap.add_argument("--smoke", action="store_true")

    ap.add_argument("--transition-cost", type=float, default=0.003)
    ap.add_argument("--init-money", type=float, default=500000.0)
    return ap


def load_market_df(root: Path, dataset: str) -> pd.DataFrame:
    data_path = root / "Data" / f"{dataset}.csv"
    if not data_path.exists():
        raise FileNotFoundError(str(data_path))
    df_raw = pd.read_csv(data_path)
    df_raw = df_raw.dropna()
    if "Adj Close" in df_raw.columns:
        df_raw = df_raw.drop(["Adj Close"], axis=1)
    return df_raw


def build_cfg(root: Path, args) -> tuple:
    cfg = OmegaConf.load(str(root / "configs" / "common.yaml"))
    agent_cfg = OmegaConf.load(str(root / "configs" / f"{args.algorithm}.yaml"))
    device = resolve_device(args.device)
    exp_name = args.exp_name or f"{args.algorithm}-{args.dataset}-{device}-s{int(args.seed)}"

    cfg.dataDir = "Data"
    cfg.dataSetName = str(args.dataset)
    cfg.expName = exp_name
    cfg.time.train_startingDate = str(args.train_start)
    cfg.time.train_endingDate = str(args.train_end)
    cfg.time.test_startingDate = str(args.test_start)
    cfg.time.test_endingDate = str(args.test_end)
    cfg.agent.algorithm = str(args.algorithm)
    cfg.system.device = device
    cfg.system.seed = int(args.seed)
    cfg.train.total_env_steps = int(args.total_env_steps)
    cfg.train.eval_every_steps = int(args.eval_every_steps)
    cfg.train.learning_starts = int(min(args.learning_starts, max(32, cfg.train.total_env_steps // 4)) if args.smoke else args.learning_starts)
    cfg.train.batch_size = int(min(64, args.batch_size) if args.smoke else args.batch_size)
    cfg.train.buffer_size = int(max(2048, args.buffer_size) if args.smoke else args.buffer_size)
    cfg.train.save_best_by = str(args.save_best_by)
    cfg.train.random_reset = bool(args.random_reset or args.smoke)
    cfg.train.init_money = float(args.init_money)

    agent_cfg.device = device
    if hasattr(agent_cfg, "num_episodes"):
        agent_cfg.num_episodes = 0
    if hasattr(agent_cfg, "test_interval"):
        agent_cfg.test_interval = 0
    return cfg, agent_cfg


def build_envs(df_raw, cfg, transition_cost: float):
    train_start = datetime.strptime(cfg.time.train_startingDate, "%Y-%m-%d")
    train_end = datetime.strptime(cfg.time.train_endingDate, "%Y-%m-%d")
    test_start = datetime.strptime(cfg.time.test_startingDate, "%Y-%m-%d")
    test_end = datetime.strptime(cfg.time.test_endingDate, "%Y-%m-%d")
    env_train = TradingEnv(df_raw, train_start, train_end, init_money=float(cfg.train.init_money))
    env_test = TradingEnv(df_raw, test_start, test_end, init_money=float(cfg.train.init_money))
    env_train.transitionCost = float(transition_cost)
    env_test.transitionCost = float(transition_cost)
    env_train.init_market_data()
    env_test.init_market_data()
    return env_train, env_test


def build_agent_trainer(cfg, agent_cfg):
    algo = cfg.agent.algorithm
    if algo == "PPO":
        return PPO(agent_cfg), PPOTrainer(cfg, AgentName="agent")
    if algo == "DDQN":
        return DDQN(agent_cfg), DQNTrainer(cfg, AgentName="agent")
    if algo == "A2C":
        return A2C(agent_cfg), A2CTrainer(cfg, AgentName="agent")
    if algo == "SAC":
        return SAC(agent_cfg), SACTrainer(cfg, AgentName="agent")
    raise ValueError(f"Unsupported algorithm: {algo}")


def main() -> None:
    args = build_parser().parse_args()
    setup_seed(int(args.seed))
    root = Path(__file__).resolve().parent
    df_raw = load_market_df(root, args.dataset)
    cfg, agent_cfg = build_cfg(root, args)
    env_train, env_test = build_envs(df_raw, cfg, args.transition_cost)
    agent, trainer = build_agent_trainer(cfg, agent_cfg)

    print(
        f"[train_cli] algo={cfg.agent.algorithm} dataset={args.dataset} seed={int(args.seed)} "
        f"device={cfg.system.device} exp={cfg.expName}",
        flush=True,
    )
    print(
        f"[train_cli] train={cfg.time.train_startingDate}..{cfg.time.train_endingDate} "
        f"test={cfg.time.test_startingDate}..{cfg.time.test_endingDate} "
        f"steps={int(cfg.train.total_env_steps)} eval_every={int(cfg.train.eval_every_steps)} "
        f"batch={int(cfg.train.batch_size)} buffer={int(cfg.train.buffer_size)} "
        f"learning_starts={int(cfg.train.learning_starts)} random_reset={bool(cfg.train.random_reset)}",
        flush=True,
    )

    trainer.train_with_validation(
        env_train,
        env_test,
        agent,
        total_env_steps=int(cfg.train.total_env_steps),
        eval_every_steps=int(cfg.train.eval_every_steps),
        random_reset=bool(cfg.train.random_reset),
        save_best_by=str(cfg.train.save_best_by),
    )

    ckpt = root / "result" / str(cfg.expName) / "agent" / "model" / "best_model.pth"
    print(f"[train_cli] best_ckpt={ckpt.as_posix()} exists={ckpt.exists()}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
