import numpy as np
from tqdm import tqdm
from utils.estimator import PerformanceEstimator
from utils.Visualizer import Visualizer
import os
from utils.utils import confirm_makedirs
from utils.RewardRecord import RewardRecord
import torch
from utils.runtime import choose_metric, choose_start_index, EvalResult


class PPOTrainer:
    def __init__(self, cfg, AgentName):
        self.cfg = cfg
        self.AgentName = AgentName
        self.saveDir = os.path.join('result', cfg.expName, AgentName)
        self.saveModelDir = os.path.join(self.saveDir, 'model')
        self.lossRecord = RewardRecord(self.saveDir, self.cfg)
        confirm_makedirs(self.saveModelDir)

    def _reset_env(self, env, random_reset: bool):
        start = choose_start_index(env, random_reset)
        if start is None:
            return env.reset()
        return env.reset(startingPoint=start)

    def _evaluate(self, test_env, agent, metric_name: str) -> EvalResult:
        agent.actor.eval()
        agent.critic.eval()
        test_env.init_market_data()
        state = test_env.reset()
        done = False
        while not done:
            action = agent.take_action(state, greedy=True)
            state, _, done = test_env.step(action)
        est = PerformanceEstimator(self.cfg, test_env.account, self.saveDir)
        est.computePerformance(len(test_env.account))
        metric = choose_metric(est, metric_name)
        return EvalResult(metric=metric, pnl=float(est.PnL), sharpe=float(est.sharpeRatio), cr=float(est.CR))

    def train_with_validation(self, train_env, test_env, agent, total_env_steps=200000, eval_every_steps=10000, *, random_reset: bool = False, save_best_by: str = "PnL"):
        """按环境步数训练，适合跨算法统一比较。"""
        best_metric = -np.inf
        global_step = 0
        state = self._reset_env(train_env, random_reset)
        episode_return = 0.0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        pbar = tqdm(total=int(total_env_steps), desc='PPO-Train')
        while global_step < int(total_env_steps):
            agent.actor.train()
            agent.critic.train()
            action = agent.take_action(state)
            next_state, reward, done = train_env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(float(done))
            state = next_state
            episode_return += reward
            global_step += 1
            pbar.update(1)
            if len(transition_dict['states']) >= int(agent.rollout_steps) or done or global_step == int(total_env_steps):
                agent.update(transition_dict)
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            if done:
                self.lossRecord.appendRewardWithVal(episode_return, self.AgentName)
                episode_return = 0.0
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
        print(f'训练完成，最佳 {save_best_by}: {best_metric:.4f}')

    def test(self, test_env, agent,):
        test_env.init_market_data()
        agent.load_model('result/TEST/agent/model/best_model.pth')
        agent.actor.eval()
        state = test_env.reset()
        done = False
        while not done:
            action = agent.take_action(state, greedy=True)
            state, _, done = test_env.step(action)
        est = PerformanceEstimator(self.cfg, test_env.account, self.saveDir)
        est.computePerformance(len(test_env.account))
        est.displayPerformance(self.AgentName, len(test_env.account))
        self.lossRecord.CapitalPlot(test_env.account, self.AgentName)
        self.lossRecord.draw(test_env.account, self.AgentName)
