from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class TrajectoryStep:
    t: int
    state: np.ndarray
    action: int
    reward: float
    done: bool
    concepts: dict


def rollout_environment(adapter, env, max_steps: int | None = None, concept_fn: Callable | None = None) -> list[TrajectoryStep]:
    state = env.reset()
    steps: list[TrajectoryStep] = []
    done = False
    count = 0
    while not done:
        action = adapter.get_action(state)
        next_state, reward, done = env.step(action)
        concepts = concept_fn(env, env.t - 1) if concept_fn else {}
        steps.append(
            TrajectoryStep(
                t=env.t - 1,
                state=np.asarray(state, dtype=np.float32),
                action=action,
                reward=float(reward),
                done=bool(done),
                concepts=concepts,
            )
        )
        state = next_state
        count += 1
        if max_steps is not None and count >= max_steps:
            break
    return steps
