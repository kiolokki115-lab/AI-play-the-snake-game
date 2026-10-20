"""通用动作掩码 Wrapper（兼容 SB3 的 PPO / DQN / SAC 等所有算法）。

约定：被包裹的环境需要在 info 字典中返回 "action_mask"
（bool 数组，True 表示该动作合法）。

Wrapper 在 step 时检查模型选中的动作：若为非法动作（会立即撞墙/撞自己），
则从合法动作中随机替换一个，从而以硬约束方式杜绝主动送死。
"""

import gymnasium as gym
import numpy as np


class ActionMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._n_actions = int(env.action_space.n)
        self._last_mask = None

    def _all_true_mask(self):
        """环境未提供掩码时的兜底：全部动作视为合法，行为等同原环境。"""
        return np.ones(self._n_actions, dtype=bool)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_mask = info.get("action_mask", self._all_true_mask())
        return obs, info

    def step(self, action):
        action = int(action)

        # 若选中非法动作，从合法动作中随机替换
        if self._last_mask is not None and not self._last_mask[action]:
            valid_actions = np.where(self._last_mask)[0]
            if len(valid_actions) > 0:
                action = int(np.random.choice(valid_actions))

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_mask = info.get("action_mask", self._all_true_mask())
        return obs, reward, terminated, truncated, info
