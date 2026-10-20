"""在消融模型（v2 观测 + 旧超参，500k）基础上继续训练 650k 步，
总步数对齐 v1-b 的 1.12M，用来回答“room 观测 + 旧超参 + 足量训练能不能超过 57.0”。
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
SRC = os.path.join(ROOT, "snake_logs", "ablation_v2obs_oldhp")
OUT = os.path.join(ROOT, "snake_logs", "v2_oldhp_1150k")

venv = DummyVecEnv(
    [lambda: ActionMaskWrapper(SnakeEnv(render_mode=None, grid_size=15, obs_version=2))]
)
model = DQN.load(SRC, env=venv)
print(f"起点 {model.num_timesteps:,} 步，再训练 650k 步", flush=True)
model.learn(total_timesteps=650_000, reset_num_timesteps=False)
model.save(OUT)
venv.close()
print(f"=== 已保存 {OUT}.zip（总 {model.num_timesteps:,} 步）===", flush=True)

m = DQN.load(OUT, device="cpu")
env = ActionMaskWrapper(
    SnakeEnv(render_mode=None, grid_size=15, obs_version=2, max_steps=3000)
)
foods, steps, caps = [], [], 0
for seed in range(700, 720):
    obs, info = env.reset(seed=seed)
    n = 0
    while True:
        action, _ = m.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        n += 1
        if terminated or truncated:
            break
    foods.append(info["foods"])
    steps.append(n)
    caps += int(truncated and not terminated)
env.close()

foods = np.array(foods)
print(
    f"RESULT v2obs+oldhp+1.15M: mean={foods.mean():.1f} min={foods.min()} max={foods.max()} "
    f"steps={np.mean(steps):.0f} caps={caps}",
    flush=True,
)
print("foods:", foods.tolist(), flush=True)
