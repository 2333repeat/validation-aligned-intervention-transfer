from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from model.DNN import LogitPolicyNet, QNet
from utils.runtime import resolve_device


class SAC:
    """离散动作 SAC。对外命名保持 SAC，内部实现为 discrete-action SAC。"""

    def __init__(self, cfg):
        self.device = resolve_device(getattr(cfg, "device", "auto"))
        state_dim = cfg.input_size * cfg.window_size
        hidden_dim = cfg.hidden_dim
        action_dim = cfg.action_dim
        self.action_dim = int(action_dim)
        self.actor = LogitPolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.q1 = QNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.q2 = QNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_q1 = QNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_q2 = QNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.actor_lr))
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=float(cfg.critic_lr))
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=float(cfg.critic_lr))

        self.gamma = float(cfg.gamma)
        self.tau = float(getattr(cfg, "tau", 0.005))
        self.max_grad_norm = float(getattr(cfg, "max_grad_norm", 1.0))
        self.fixed_alpha = float(getattr(cfg, "alpha", 0.2))
        self.alpha_mode = str(getattr(cfg, "alpha_mode", "auto"))
        if self.alpha_mode == "auto":
            scale = float(getattr(cfg, "target_entropy_scale", 0.7))
            self.target_entropy = math.log(self.action_dim) * scale
            self.log_alpha = torch.tensor(np.log(self.fixed_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(getattr(cfg, "alpha_lr", 3e-4)))
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optimizer = None

    @property
    def alpha(self):
        if self.log_alpha is None:
            return torch.tensor(self.fixed_alpha, dtype=torch.float32, device=self.device)
        return self.log_alpha.exp()

    def _policy(self, states):
        logits = self.actor(states)
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        return probs, log_probs

    def take_action(self, state, greedy: bool = False):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self._policy(state_t)
            if greedy:
                return int(torch.argmax(probs, dim=1).item())
            return int(torch.distributions.Categorical(probs).sample().item())

    def _soft_value(self, next_states):
        next_probs, next_log_probs = self._policy(next_states)
        q1_next = self.target_q1(next_states)
        q2_next = self.target_q2(next_states)
        q_next = torch.min(q1_next, q2_next)
        return (next_probs * (q_next - self.alpha.detach() * next_log_probs)).sum(dim=1, keepdim=True)

    def compute_alpha_loss(self, entropy):
        if self.alpha_optimizer is None:
            raise RuntimeError("alpha_mode is not auto")
        entropy_gap = entropy.detach() - self.target_entropy
        return (self.log_alpha * entropy_gap).mean()

    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch
        actions = actions.view(-1, 1)
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)

        with torch.no_grad():
            target_v = self._soft_value(next_states)
            target_q = rewards + self.gamma * (1.0 - dones) * target_v

        q1_pred = self.q1(states).gather(1, actions)
        q2_pred = self.q2(states).gather(1, actions)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.max_grad_norm)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.max_grad_norm)
        self.q2_optimizer.step()

        probs, log_probs = self._policy(states)
        q_min = torch.min(self.q1(states), self.q2(states))
        actor_loss = (probs * (self.alpha.detach() * log_probs - q_min)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        if self.alpha_optimizer is not None:
            entropy = -(probs.detach() * log_probs.detach()).sum(dim=1, keepdim=True)
            alpha_loss = self.compute_alpha_loss(entropy)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._soft_update()

    def _soft_update(self):
        for target, online in ((self.target_q1, self.q1), (self.target_q2, self.q2)):
            for target_param, param in zip(target.parameters(), online.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, path):
        path = os.path.join(path, 'best_model.pth')
        payload = {
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'target_q1': self.target_q1.state_dict(),
            'target_q2': self.target_q2.state_dict(),
        }
        if self.log_alpha is not None:
            payload['log_alpha'] = self.log_alpha.detach().cpu()
        torch.save(payload, path)

    def load_model(self, path, map_location=None):
        if map_location is None:
            map_location = self.device
        ckpt = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(ckpt['actor'])
        self.q1.load_state_dict(ckpt['q1'])
        self.q2.load_state_dict(ckpt['q2'])
        self.target_q1.load_state_dict(ckpt.get('target_q1', ckpt['q1']))
        self.target_q2.load_state_dict(ckpt.get('target_q2', ckpt['q2']))
        if self.log_alpha is not None and 'log_alpha' in ckpt:
            self.log_alpha.data.copy_(ckpt['log_alpha'].to(self.device))
