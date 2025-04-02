import random
from typing import Any, Dict, List, Tuple

import gym
import metaworld
import numpy as np
from gym.spaces import Box


class SuccessCounter(gym.Wrapper):
    """Helper class to keep count of successes in MetaWorld environments."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.successes = []
        self.current_success = False

    def step(self, action: Any) -> Tuple:
        # 处理新版gym (gymnasium)
        result = self.env.step(action)
        if len(result) == 5:
            # 新版本的gymnasium返回5个值
            obs, reward, terminated, truncated, info = result
        else:
            # 旧版本的gym返回4个值
            obs, reward, done, info = result
            terminated = done
            truncated = False
            
        if info.get("success", False):
            self.current_success = True
        if terminated or truncated:
            self.successes.append(self.current_success)
        
        return obs, reward, terminated, truncated, info

    def pop_successes(self) -> List[bool]:
        res = self.successes
        self.successes = []
        return res

    def reset(self, **kwargs) -> tuple:
        self.current_success = False
        result = self.env.reset(**kwargs)
        # 处理新旧版本的reset接口
        if isinstance(result, tuple) and len(result) == 2:
            # 新版本返回(obs, info)
            return result
        else:
            # 旧版本直接返回obs
            return result, {}


class OneHotAdder(gym.Wrapper):
    """Appends one-hot encoding to the observation. Can be used e.g. to encode the task."""

    def __init__(
        self, env: gym.Env, one_hot_idx: int, one_hot_len: int, orig_one_hot_dim: int = 0
    ) -> None:
        super().__init__(env)
        assert 0 <= one_hot_idx < one_hot_len
        self.to_append = np.zeros(one_hot_len)
        self.to_append[one_hot_idx] = 1.0

        orig_obs_low = self.env.observation_space.low
        orig_obs_high = self.env.observation_space.high
        if orig_one_hot_dim > 0:
            orig_obs_low = orig_obs_low[:-orig_one_hot_dim]
            orig_obs_high = orig_obs_high[:-orig_one_hot_dim]
        self.observation_space = Box(
            np.concatenate([orig_obs_low, np.zeros(one_hot_len)]),
            np.concatenate([orig_obs_high, np.ones(one_hot_len)]),
        )
        self.orig_one_hot_dim = orig_one_hot_dim

    def _append_one_hot(self, obs: np.ndarray) -> np.ndarray:
        if self.orig_one_hot_dim > 0:
            obs = obs[: -self.orig_one_hot_dim]
        return np.concatenate([obs, self.to_append])

    def step(self, action: Any) -> Tuple:
        # 处理新版gym (gymnasium)
        result = self.env.step(action)
        if len(result) == 5:
            # 新版本的gymnasium返回5个值
            obs, reward, terminated, truncated, info = result
        else:
            # 旧版本的gym返回4个值
            obs, reward, done, info = result
            terminated = done
            truncated = False
            
        return self._append_one_hot(obs), reward, terminated, truncated, info

    def reset(self, **kwargs) -> tuple:
        result = self.env.reset(**kwargs)
        # 处理新旧版本的reset接口
        if isinstance(result, tuple) and len(result) == 2:
            # 新版本返回(obs, info)
            return self._append_one_hot(result[0]), result[1]
        else:
            # 旧版本直接返回obs
            return self._append_one_hot(result), {}


class RandomizationWrapper(gym.Wrapper):
    """Manages randomization settings in MetaWorld environments."""

    ALLOWED_KINDS = [
        "deterministic",
        "random_init_all",
        "random_init_fixed20",
        "random_init_small_box",
    ]

    def __init__(self, env: gym.Env, subtasks: List[metaworld.Task], kind: str) -> None:
        assert kind in RandomizationWrapper.ALLOWED_KINDS
        super().__init__(env)
        self.subtasks = subtasks
        self.kind = kind

        env.set_task(subtasks[0])
        if kind == "random_init_all":
            env._freeze_rand_vec = False

        if kind == "random_init_fixed20":
            assert len(subtasks) >= 20

        if kind == "random_init_small_box":
            diff = env._random_reset_space.high - env._random_reset_space.low
            self.reset_space_low = env._random_reset_space.low + 0.45 * diff
            self.reset_space_high = env._random_reset_space.low + 0.55 * diff

    def reset(self, **kwargs) -> tuple:
        if self.kind == "random_init_fixed20":
            self.env.set_task(self.subtasks[random.randint(0, 19)])
        elif self.kind == "random_init_small_box":
            rand_vec = np.random.uniform(
                self.reset_space_low, self.reset_space_high, size=self.reset_space_low.size
            )
            self.env._last_rand_vec = rand_vec

        result = self.env.reset(**kwargs)
        # 处理新旧版本的reset接口
        if isinstance(result, tuple) and len(result) == 2:
            # 新版本返回(obs, info)
            return result
        else:
            # 旧版本直接返回obs
            return result, {}
