"""消融实验：v2 观测 + 旧超参（eps 0.10 / buffer 1万 / batch 32），500k 步。

目的：v2 全量改动（新观测 + 新超参）比 v1-b 差 14.6 个食物，
      本实验只改回超参、观测保持 v2，用来判断是哪一边拖的后腿。
"""

import os
import sys

sys.path.insert(0, r"C:\Users\35630\Desktop\AI玩游戏")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from mask_wrapper import ActionMaskWrapper
from snake import SnakeEnv

ROOT = r"C:\Users\35630\Desktop\AI玩游戏"
OUT = os.path.join(ROOT, "snake_logs", "ablation_v2obs_oldhp")

env = DummyVecEnv(
    [lambda: ActionMaskWrapper(SnakeEnv(render_mode=None, grid_size=15, obs_version=2))]
)
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=10_000,
    exploration_fraction=0.1,
    exploration_final_eps=0.10,
    train_freq=4,
    target_update_interval=1000,
    gamma=0.99,
)
print("=== 训练：v2 观测 + 旧超参，500k 步 ===", flush=True)
model.learn(total_timesteps=500_000)
model.save(OUT)
env.close()
print(f"=== 已保存 {OUT}.zip ===", flush=True)

# 评估（与之前完全相同的 20 个种子）
m = DQN.load(OUT, device="cpu")
test_env = ActionMaskWrapper(
    SnakeEnv(render_mode=None, grid_size=15, obs_version=2, max_steps=3000)
)
foods, steps, caps = [], [], 0
for seed in range(700, 720):
    obs, info = test_env.reset(seed=seed)
    n = 0
    while True:
        action, _ = m.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        n += 1
        if terminated or truncated:
            break
    foods.append(info["foods"])
    steps.append(n)
    caps += int(truncated and not terminated)
test_env.close()

foods = np.array(foods)
print(
    f"RESULT v2obs+oldhp: mean={foods.mean():.1f} min={foods.min()} max={foods.max()} "
    f"steps={np.mean(steps):.0f} caps={caps}",
    flush=True,
)
print("foods:", foods.tolist(), flush=True)
