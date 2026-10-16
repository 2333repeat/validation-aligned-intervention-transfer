import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import math
from random import random
from model.DNN import QNet
from utils.runtime import resolve_device

class DDQN:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = resolve_device(getattr(cfg, "device", "auto"))
        self.policy_net = QNet(cfg.input_size*cfg.window_size, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.target_net = QNet(cfg.input_size*cfg.window_size, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma
        # self.batch_size = cfg.train.batch_size
        self.target_update = int(getattr(cfg, "target_update", 1000))
        self.loss_fn = nn.MSELoss()
        self.grad_clip = float(getattr(cfg, "max_grad_norm", 1.0))
        self.num_updates = 0
        self.EPS_START = float(getattr(cfg, "EPS_START", 1.0))
        self.EPS_END = float(getattr(cfg, "EPS_END", 0.05))
        self.EPS_DECAY = float(getattr(cfg, "EPS_DECAY", 2000))
        self.steps_done = 0
        self.memory_counter = 0
        self.tau = float(getattr(cfg, "tau", 0.005))
        self.epsilon = 0

    def epsilonGreedAction(self, q_values):

        epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if random() > epsilon:
            return torch.argmax(q_values).item()
        else:
            return np.random.randint(0, 3)

    def act(self, state, train=True):

        # state 本身是 1D numpy array，避免 torch.tensor([state]) 的慢路径
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze()
            if train:
                action = self.epsilonGreedAction(q_values)
            else:
                action = int(torch.argmax(q_values).item())
        return int(action)

    def learn(self, batch):
        s, a, r, s_, done = batch  # 需在回放缓冲区存储done标志
        # Q值计算
        q_eval = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze()

        with torch.no_grad():
            # 双重Q学习目标
            policy_net_actions = self.policy_net(s_).argmax(dim=1)
            q_target = r + (1 - done.squeeze()) * self.gamma * self.target_net(s_).gather(1, policy_net_actions.unsqueeze(
                1)).squeeze()

        # 损失计算与反向传播
        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)  # 梯度裁剪
        self.optimizer.step()
        # 目标网络软更新
        self.soft_update_target()

    def save_model(self, path):
        path = os.path.join(path, 'best_model.pth')
        torch.save({'policy_net': self.policy_net.state_dict()}, path)

    def load_model(self, path, map_location=None):
        if map_location is None:
            map_location = self.device
        ckpt = torch.load(path, map_location=map_location)
        if "policy_net" in ckpt:
            self.policy_net.load_state_dict(ckpt["policy_net"])
        else:
            self.policy_net.load_state_dict(ckpt)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update_target(self):

        for t, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
            t.data.copy_(self.tau * p.data + (1 - self.tau) * t.data)
