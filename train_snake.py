import os
import sys
import time
import pygame
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from snake import SnakeEnv
from mask_wrapper import ActionMaskWrapper

# 自定义保存路径
MODEL_PATH = r"C:\Users\35630\Desktop\AI玩游戏\snake_dqn_model"
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
    
    # 调试：打印实际保存路径，防止路径不对
    print(f"📁 模型将保存至: {MODEL_PATH}.zip")

    render_mode = None if silent else "human"
    # 通关（清屏）终局奖励默认 2000，可通过 SnakeEnv(win_bonus=...) 调整；
    # 别调成 1e7 这种量级，值函数学不动。
    raw_env = SnakeEnv(render_mode=render_mode, grid_size=15)
    # 👈 关键：动作掩码硬约束，禁止 AI 选择立即致死动作
    masked_env = ActionMaskWrapper(raw_env)

    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([lambda: masked_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=LOG_DIR,
        name_prefix="snake_rl",
        save_replay_buffer=False,
        verbose=1,
    )

    callbacks = [checkpoint_callback]
    if not silent:
        callbacks.append(RenderCallback(raw_env, fps=30))

    if os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"🔄 加载已有模型继续训练...")
        model = DQN.load(MODEL_PATH, env=env)
    else:
        print("🆕 从零开始训练...")
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0001,
            buffer_size=10000,
            # 调高最终探索率：从 0.05 提升到 0.10，保留更多探索空间
            exploration_fraction=0.1,
            exploration_final_eps=0.10,
            train_freq=4,
            target_update_interval=1000,
            # 调高折扣因子：让 AI 更愿意为吃到食物而冒险
            gamma=0.99
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


def watch(model_path=MODEL_PATH):
    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ 未找到模型文件 {model_path}.zip，请先运行训练！")
        return

    env = SnakeEnv(render_mode="human", grid_size=15)
    env = ActionMaskWrapper(env)  # 👈 推理时同样生效
    model = DQN.load(model_path, env=env)

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
    elif "--observe" in args:
        train(silent=False)
    else:
        train(silent=True)