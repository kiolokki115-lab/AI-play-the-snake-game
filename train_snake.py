import os
import sys
import time
import pygame
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from snake import SnakeEnv
from mask_wrapper import ActionMaskWrapper

# 自定义保存路径
# 正式模型：v1 观测（绝对 escape_ratio）。修正评估器后的实测结果见下，
# room 观测（v2）在 1.15M 步只有 46.4，明显不如 v1 的 58.2，故不作为默认。
MODEL_PATH = r"C:\Users\35630\Desktop\AI玩游戏\snake_dqn_model"
# 实验模型：room 观测（第 7 维改为相对空间），保留用于对比/演示：--watch-v2
V2_MODEL_PATH = r"C:\Users\35630\Desktop\AI玩游戏\snake_dqn_model_v2"
LOG_DIR = r"C:\Users\35630\Desktop\AI玩游戏\snake_logs"


class RenderCallback(BaseCallback):
    def __init__(self, env_instance, fps=10, delay_ms=None, **kwargs):
        super().__init__()
        self.env_instance = env_instance
        if delay_ms is not None:
            self.delay_ms = delay_ms
        else:
            self.delay_ms = int(1000.0 / max(fps, 1))

    def _on_step(self) -> bool:
        self.env_instance.render()
        pygame.event.pump()
        pygame.time.delay(self.delay_ms)
        return True


def train(silent=True):
    os.makedirs(LOG_DIR, exist_ok=True)

    # 明确打印“训练的是哪个”，避免续训/覆盖搞混
    print(f"📁 训练目标: {MODEL_PATH}.zip")
    print(f"🧠 观测版本: v1 (绝对 escape_ratio)")

    render_mode = None if silent else "human"
    # 通关（清屏）终局奖励默认 2000，可通过 SnakeEnv(win_bonus=...) 调整；
    # 别调成 1e7 这种量级，值函数学不动。
    raw_env = SnakeEnv(render_mode=render_mode, grid_size=15, obs_version=1)
    # 👈 关键：动作掩码硬约束，禁止 AI 选择立即致死动作
    masked_env = ActionMaskWrapper(raw_env)

    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([lambda: masked_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=LOG_DIR,
        name_prefix="snake_rl_v2",
        save_replay_buffer=False,
        verbose=1,
    )

    callbacks = [checkpoint_callback]
    if not silent:
        callbacks.append(RenderCallback(raw_env, fps=30))

    if os.path.exists(f"{MODEL_PATH}.zip"):
        model = DQN.load(MODEL_PATH, env=env)
        print(f"🔄 已存在同名模型 → 继续训练，起点 {model.num_timesteps:,} 步")
        print(f"   ⚠️ 训练结束会覆盖 {MODEL_PATH}.zip")
    else:
        print("🆕 不存在同名模型 → 从零开始训练")
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0001,
            # 【实测回退】曾试过 buffer 10万 / batch 64 / eps 0.02：
            # 同样 500k 步、同样 room 观测，平均食物 47.0 -> 40.6，明显更差，故全部回退。
            buffer_size=10_000,
            exploration_fraction=0.1,
            exploration_final_eps=0.10,
            train_freq=4,
            target_update_interval=1000,
            gamma=0.99,
        )

    mode_text = "🔇 静默模式" if silent else "👁️ 观摩模式"
    print(f"{mode_text} | Ctrl+C 安全中断")

    try:
        model.learn(
            total_timesteps=1_000_000,
            callback=callbacks,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n⏹️ 中断训练，保存中...")
    finally:
        model.save(MODEL_PATH)
        print(f"✅ 已保存至 {MODEL_PATH}.zip")
        env.close()


def watch(model_path=MODEL_PATH, obs_version=1):
    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ 未找到模型文件 {model_path}.zip")
        print("   → 先跑训练：python train_snake.py")
        print("   → 只想看旧模型（v1 观测）：python train_snake.py --watch-old")
        return

    # 环境观测必须和模型训练时一致，否则第 7 维含义错位，行为会莫名其妙
    env = SnakeEnv(render_mode="human", grid_size=15, obs_version=obs_version)
    env = ActionMaskWrapper(env)  # 👈 推理时同样生效
    model = DQN.load(model_path, env=env)
    print(f"🧠 模型: {model_path}.zip | 观测 v{obs_version} | 训练步数 {model.num_timesteps:,}")

    print("🎮 AI 演示中... | 关闭窗口或按 Ctrl+C 退出")
    obs, _ = env.reset()
    total_reward = 0
    clock = pygame.time.Clock()

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()

            if terminated or truncated:
                foods = info.get("foods", 0)
                score = info.get("score", 0)
                if info.get("win"):
                    print(
                        f"🏆 通关清屏！食物 {foods} | 展示分: {score:,} "
                        f"| 训练奖励: {total_reward:.1f}"
                    )
                else:
                    print(
                        f"💀 本局结束: 食物 {foods} | 展示分: {score:,} "
                        f"| 训练奖励: {total_reward:.1f}"
                    )
                obs, _ = env.reset()
                total_reward = 0

            clock.tick(15)
    except KeyboardInterrupt:
        print("\n⏹️ 退出演示")
    finally:
        env.close()


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--watch" in args:
        watch()
    elif "--watch-v2" in args or "--watch-old" in args:
        watch(V2_MODEL_PATH, obs_version=2)
    elif "--observe" in args:
        train(silent=False)
    else:
        train(silent=True)