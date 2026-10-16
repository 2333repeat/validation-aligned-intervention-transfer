from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from model.DNN import PolicyNet, ValueNet
from utils.runtime import resolve_device


class A2C:
    def __init__(self, cfg):
        self.device = resolve_device(getattr(cfg, "device", "auto"))
        self.actor = PolicyNet(cfg.input_size * cfg.window_size, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.critic = ValueNet(cfg.input_size * cfg.window_size, cfg.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.actor_lr))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(cfg.critic_lr))
        self.gamma = float(cfg.gamma)
        self.entropy_coef = float(getattr(cfg, "entropy_coef", 0.01))
        self.value_coef = float(getattr(cfg, "value_coef", 0.5))
        self.max_grad_norm = float(getattr(cfg, "max_grad_norm", 0.5))
        self.n_steps = int(getattr(cfg, "n_steps", 20))

    def take_action(self, state, greedy: bool = False):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state_t)
            if greedy:
                return int(torch.argmax(probs, dim=1).item())
            return int(torch.distributions.Categorical(probs).sample().item())

    def compute_nstep_returns(self, rewards, dones, bootstrap_value):
        returns = torch.zeros_like(rewards, device=self.device)
        running = bootstrap_value.to(self.device)
        for step in range(rewards.shape[0] - 1, -1, -1):
            running = rewards[step:step + 1] + self.gamma * running * (1.0 - dones[step:step + 1])
            returns[step:step + 1] = running
        return returns

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float32, device=self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float32, device=self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            bootstrap_value = self.critic(next_states[-1:]) * (1.0 - dones[-1:])
            targets = self.compute_nstep_returns(rewards, dones, bootstrap_value)
        values = self.critic(states)
        advantages = targets - values

        probs = self.actor(states)
        log_probs = torch.log(torch.clamp(probs.gather(1, actions), min=1e-12))
        entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=1, keepdim=True).mean()

        actor_loss = -(log_probs * advantages.detach()).mean() - self.entropy_coef * entropy
        critic_loss = F.mse_loss(values, targets.detach())
        loss = actor_loss + self.value_coef * critic_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def save_model(self, path):
        path = os.path.join(path, 'best_model.pth')
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)

    def load_model(self, path, map_location=None):
        if map_location is None:
            map_location = self.device
        ckpt = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
