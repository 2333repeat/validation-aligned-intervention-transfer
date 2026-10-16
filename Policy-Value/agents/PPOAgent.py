import torch
import torch.nn.functional as F
import numpy as np
from model.DNN import PolicyNet
from model.DNN import ValueNet
import os
from utils.runtime import resolve_device


def compute_advantage(gamma, lmbda, td_delta, device):
    """GAE浼樺娍鍑芥暟璁＄畻"""
    td_delta = td_delta.detach().cpu().numpy().flatten()
    advantage = 0.0
    advantages = []
    for delta in reversed(td_delta):
        advantage = delta + gamma * lmbda * advantage
        advantages.append(advantage)
    advantages = torch.tensor(advantages[::-1], dtype=torch.float, device=device)
    return advantages.view(-1, 1)


class PPO:
    """PPO绠楁硶瀹炵幇"""

    def __init__(self, cfg):
        self.device = resolve_device(getattr(cfg, "device", "auto"))
        self.actor = PolicyNet(cfg.input_size*cfg.window_size, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.critic = ValueNet(cfg.input_size*cfg.window_size, cfg.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.gamma = cfg.gamma  # 鎶樻墸鍥犲瓙
        self.lmbda = cfg.lmbda  # GAE鍙傛暟
        self.epochs = cfg.epochs  # 绛栫暐鏇存柊杞暟
        self.eps = cfg.eps  # PPO鎴柇鑼冨洿
        self.rollout_steps = int(getattr(cfg, "rollout_steps", 512))
        # 熵正则（可选）：缓解“长期不交易/一直 hold”的退化策略。
        # 兼容旧配置：未提供时默认 0，行为不变。
        self.entropy_coef = float(getattr(cfg, "entropy_coef", 0.0))

    def take_action(self, state, greedy: bool = False):
        """鏍规嵁鐘舵€侀€夋嫨鍔ㄤ綔銆?

        璇存槑锛?
        - 璁粌闃舵榛樿閲囨牱锛坓reedy=False锛?
        - 楠岃瘉/鍥炴祴闃舵寤鸿璐績锛坓reedy=True锛変互闄嶄綆闅忔満鎬?
        """
        # state 鏈韩鏄?1D numpy array锛岄伩鍏?torch.tensor([state]) 鐨勬參璺緞涓庡憡璀?
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state_t)
            if greedy:
                return int(torch.argmax(probs, dim=1).item())
            action_dist = torch.distributions.Categorical(probs)
            return int(action_dist.sample().item())

    def save_model(self, path):
        path = os.path.join(path,'best_model.pth')
        """淇濆瓨妯″瀷"""
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)

    def load_model(self, path, map_location=None):
        """鍔犺浇妯″瀷"""
        if map_location is None:
            # 榛樿鎸夊綋鍓?agent.device 鍔犺浇锛涘湪闇€瑕佸己鍒?CPU 鎺ㄧ悊鏃跺彲鏄惧紡浼犲叆 map_location="cpu"
            map_location = self.device
        ckpt = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

    def update(self, transition_dict):
        """PPO绠楁硶鏇存柊"""
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float, device=self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float, device=self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 璁＄畻TD鐩爣鍜屼紭鍔垮嚱鏁?
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta, self.device)

        # 淇濆瓨鏃х瓥鐣ョ殑姒傜巼鐢ㄤ簬閲嶈鎬ч噰鏍?
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # 澶氳疆娆℃洿鏂?
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)

            # PPO-Clip鐩爣鍑芥暟
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # 姊害涓嬮檷
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            if self.entropy_coef > 0:
                # H(p) = -sum p log p
                probs = self.actor(states)
                ent = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=1).mean()
                actor_loss = actor_loss - self.entropy_coef * ent
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

