import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.estimator import PerformanceEstimator
from utils.RewardRecord import RewardRecord
from utils.utils import confirm_makedirs
from utils.ReplayBuffer import ReplayBuffer
import os
import numpy as np
from utils.Visualizer import Visualizer
from utils.runtime import choose_metric, choose_start_index, EvalResult


class DQNTrainer:
    def __init__(self, cfg, AgentName):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.AgentName = AgentName
        self.saveDir = os.path.join('result', cfg.expName, AgentName)
        self.saveModelDir = os.path.join(self.saveDir, 'model')
        self.lossRecord = RewardRecord(self.saveDir, self.cfg)
        self.memory = ReplayBuffer(self.cfg.train.buffer_size)
        self.batch_size = self.cfg.train.batch_size
        confirm_makedirs(self.saveModelDir)

    def _reset_env(self, env, random_reset: bool):
        start = choose_start_index(env, random_reset)
        if start is None:
            return env.reset()
        return env.reset(startingPoint=start)

    def _evaluate(self, test_env, agent, metric_name: str) -> EvalResult:
        test_env.init_market_data()
        state = test_env.reset()
        done = False
        while not done:
            action = agent.act(state, train=False)
            state, _, done = test_env.step(action)
        est = PerformanceEstimator(self.cfg, test_env.account, self.saveDir)
        est.computePerformance(len(test_env.account))
        metric = choose_metric(est, metric_name)
        return EvalResult(metric=metric, pnl=float(est.PnL), sharpe=float(est.sharpeRatio), cr=float(est.CR))

    def train_with_validation(self, train_env, test_env, agent, total_env_steps=200000, eval_every_steps=10000, *, random_reset: bool = False, save_best_by: str = "PnL"):
        best_metric = -np.inf
        global_step = 0
        state = self._reset_env(train_env, random_reset)
        total_reward = 0.0
        learning_starts = int(getattr(self.cfg.train, "learning_starts", self.batch_size))
        train_freq = int(getattr(self.cfg.train, "train_freq", 1))
        gradient_steps = int(getattr(self.cfg.train, "gradient_steps", 1))
        pbar = tqdm(total=int(total_env_steps), desc='DQN-Train')
        while global_step < int(total_env_steps):
            action = agent.act(state, train=True)
            next_state, reward, done = train_env.step(action)
            self.memory.add(
                state=torch.tensor([state], dtype=torch.float32, device=self.device),
                action=torch.tensor(action, dtype=torch.long).view(-1, 1).to(self.device),
                reward=torch.tensor(reward, dtype=torch.float32).view(-1, 1).to(self.device),
                next_state=torch.tensor([next_state], dtype=torch.float32, device=self.device),
                done=torch.tensor(done, dtype=torch.float32).view(-1, 1).to(self.device),
            )
            state = next_state
            total_reward += reward
            global_step += 1
            pbar.update(1)
            if len(self.memory) >= max(self.batch_size, learning_starts) and global_step % train_freq == 0:
                for _ in range(max(1, gradient_steps)):
                    agent.learn(self.memory.sample(self.batch_size, 'dqn'))
            if done:
                self.lossRecord.appendRewardWithVal(total_reward, self.AgentName)
                total_reward = 0.0
                state = self._reset_env(train_env, random_reset)
            if global_step % int(eval_every_steps) == 0 or global_step == int(total_env_steps):
                result = self._evaluate(test_env, agent, save_best_by)
                if result.metric > best_metric:
                    best_metric = result.metric
                    est = PerformanceEstimator(self.cfg, test_env.account, self.saveDir)
                    est.computePerformance(len(test_env.account))
                    est.displayPerformance(self.AgentName, len(test_env.account))
                    self.lossRecord.CapitalPlot(test_env.account, self.AgentName)
                    self.lossRecord.draw(test_env.account, self.AgentName)
                    agent.save_model(self.saveModelDir)
                pbar.set_postfix(metric=result.metric, pnl=result.pnl, sharpe=result.sharpe)
        pbar.close()
