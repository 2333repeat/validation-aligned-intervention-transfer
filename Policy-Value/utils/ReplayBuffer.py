from collections import deque
import random
import torch
from dataclasses import dataclass

@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    prob: torch.Tensor = None        # PPO
    done: torch.Tensor = None
    target_value: torch.Tensor = None # A2C
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def clear(self):
        self.buffer.clear()

    def add(self, **kwargs):
        transition = Transition(**{
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v  # keep batch dimensions consistent
            for k, v in kwargs.items()
        })
        self.buffer.append(transition)

    def sample(self, batch_size, algo_type='dqn'):
        """Sample transitions and return the fields required by the selected algorithm."""
        samples = random.sample(self.buffer, batch_size)

        # Collect fields that are present for all sampled transitions.
        fields = {}
        for field in Transition.__annotations__:
            if all(getattr(s, field) is not None for s in samples):
                fields[field] = torch.cat([getattr(s, field) for s in samples])

        # Return the field layout expected by each algorithm family.
        if algo_type.lower() == 'ppo':
            return (
                fields['state'].squeeze(),
                fields['action'].squeeze(),
                fields['reward'].squeeze(),
                fields['next_state'].squeeze(),
                fields['prob'].squeeze(),
                fields['done'].squeeze(),
            )

        else:  # DQN/DDQN
            return (
                fields['state'].squeeze(),
                fields['action'].squeeze(),
                fields['reward'].squeeze(),
                fields['next_state'].squeeze(),
                fields['done'].squeeze(1),
            )



    def __len__(self):
        return len(self.buffer)
