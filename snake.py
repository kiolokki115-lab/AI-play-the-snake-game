import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import os
from collections import deque

# ==========================================
# 强制初始化 pygame-ce (防止被外部调用时崩溃)
# ==========================================
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
pygame.mixer.quit()
pygame.display.init()
pygame.init()
# ==========================================


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        render_mode=None,
        grid_size=10,
        win_bonus=2000.0,
        food_reward_base=70.0,
        milestone_every=25,
        milestone_bonus=100.0,
        obs_version=2,
        max_steps=3000,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.window_size = 512
        # 观测版本：2 = 第 7 维用相对空间 room（新）；1 = 用绝对 escape_ratio（旧模型专用）
        self.obs_version = obs_version
        # 单局步数上限：蛇可能绕圈永远不死，没有上限会把训练/评估直接挂住
        self.max_steps = max_steps
        # 通关（清屏）终局奖励：故意与单个食物的 +70 保持同一量级，
        # 这样值函数不会被顶到 1e7 那种无法学习的数值上。
        self.win_bonus = win_bonus
        # 展示用分数：通关额外加 1000 万分，只进 info，与训练奖励完全解耦。
        self.win_score = 10_000_000
        # 单个食物的奖励上限；实际按“吃完后的相对空间”在 0.25~1 倍之间取值。
        self.food_reward_base = food_reward_base
        # 长度里程碑：每吃 milestone_every 个食物额外奖励一次，抵消终局奖励被折扣吃掉。
        self.milestone_every = milestone_every
        self.milestone_bonus = milestone_bonus
        self.win = False
        self.action_space = spaces.Discrete(4)  # 0:上 1:下 2:左 3:右
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.render_mode = render_mode
        self._last_direction = (0, 1)  # 记录最后移动方向，默认向下

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [[self.grid_size // 2, self.grid_size // 2]]
        self.food = self._place_food()
        self.steps = 0
        self.win = False
        self.foods_eaten = 0
        self._last_direction = (0, 1)
        if self.obs_version == 1:
            # v1 观测开局就是 0.0，为公平对比/演示旧模型保持原样
            self._cached_escape_ratio = 0.0
        else:
            # v2 的 room 开局必须是真实值（否则恒为 0，等于给策略喂了个坏特征）
            self._cached_escape_ratio = self._compute_escape_ratio(*self.snake[0])
        return self._get_obs(), self._get_info()

    def _simulate_move(self, hx, hy, action):
        if action == 0:
            return hx, hy - 1
        elif action == 1:
            return hx, hy + 1
        elif action == 2:
            return hx - 1, hy
        elif action == 3:
            return hx + 1, hy

    def _get_action_mask(self):
        """返回 4 个动作的合法性掩码（True=安全，False=会立即死亡）。

        死亡判定与 step() 中完全一致：越界，或撞进 self.snake[:-1]
        （蛇尾格在下一步会 pop，因此允许进入，不算撞自己）。

        兜底：四向全为非法（必死局）时全部置 True，避免下游采样崩溃。
        """
        head_x, head_y = self.snake[0]
        body_set = set(map(tuple, self.snake[:-1]))

        mask = np.zeros(4, dtype=bool)
        for action in range(4):
            nx, ny = self._simulate_move(head_x, head_y, action)
            in_bounds = 0 <= nx < self.grid_size and 0 <= ny < self.grid_size
            mask[action] = in_bounds and (nx, ny) not in body_set

        if not mask.any():
            mask[:] = True

        return mask

    def _count_free_space(self, start, snake_body):
        """BFS计算从start出发的可达自由格数"""
        body_set = set(map(tuple, snake_body[:-1]))
        visited = set()
        queue = deque([tuple(start)])
        visited.add(tuple(start))
        count = 0
        max_search = self.grid_size ** 2
        while queue and count < max_search:
            x, y = queue.popleft()
            count += 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if (nx, ny) not in visited and (nx, ny) not in body_set:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return count

    def _compute_escape_ratio(self, head_x, head_y):
        """计算蛇头四个方向中最大的可达空间比率"""
        body = self.snake
        free_spaces = []
        for a in range(4):
            nx, ny = self._simulate_move(head_x, head_y, a)
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                free = self._count_free_space([nx, ny], body)
                free_spaces.append(free)
            else:
                free_spaces.append(0)
        max_free = max(free_spaces) if free_spaces else 0
        max_possible = self.grid_size ** 2
        return max_free / max_possible if max_possible > 0 else 0.0

    def _relative_room(self, escape_ratio):
        """把逃生空间比率换算成「相对当前蛇长的可达比例」（0~1）。

        escape_ratio 是绝对比率（可达格数 / 全盘格数），它的上限随蛇长单调下降：
        蛇长 200 时全盘最多也只剩 ~11% 的可达空间，用绝对阈值（0.5 / 0.1）
        去衡量，长蛇永远被判成“不安全”，后期每一步都在扣分。
        这里改成以「当前理论上还能到达的格数」为分母，让尺子随蛇长一起缩放。
        """
        total = self.grid_size ** 2
        # _count_free_space 把蛇尾当空格，因此可达上限是 全盘 -（蛇长 - 1）
        reachable_cap = total - len(self.snake) + 1
        if reachable_cap <= 0:
            return 0.0
        room = escape_ratio * total / reachable_cap
        return float(min(1.0, max(0.0, room)))

    def _compute_directional_freedom(self, head_x, head_y):
        """【修复】计算四个方向各自的BFS可达自由格数，替代原直线隧道宽度"""
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 上、下、左、右
        body_set = set(map(tuple, self.snake[:-1]))
        freedoms = []
        max_possible = self.grid_size ** 2

        for dx, dy in directions:
            nx, ny = head_x + dx, head_y + dy
            # 如果相邻格就是墙或身体，该方向自由度为0
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size or (nx, ny) in body_set:
                freedoms.append(0.0)
            else:
                # 从该方向的相邻格开始做BFS，计算真实可达空间
                free = self._count_free_space([nx, ny], self.snake)
                freedoms.append(free / max_possible if max_possible > 0 else 0.0)

        return freedoms

    def step(self, action):
        self.steps += 1
        head_x, head_y = self.snake[0]
        fx, fy = self.food
        old_dist = abs(head_x - fx) + abs(head_y - fy)

        # 执行移动
        if action == 0:
            head_y -= 1
            self._last_direction = (0, -1)
        elif action == 1:
            head_y += 1
            self._last_direction = (0, 1)
        elif action == 2:
            head_x -= 1
            self._last_direction = (-1, 0)
        elif action == 3:
            head_x += 1
            self._last_direction = (1, 0)

        new_dist = abs(head_x - fx) + abs(head_y - fy)

        # 死亡检测
        terminated = (
            head_x < 0
            or head_x >= self.grid_size
            or head_y < 0
            or head_y >= self.grid_size
            or [head_x, head_y] in self.snake[:-1]
        )

        if terminated:
            self.snake.insert(0, [head_x, head_y])
            self._cached_escape_ratio = 0.0
            return self._get_obs(), -20.0, True, False, self._get_info()

        # 合法移动
        self.snake.insert(0, [head_x, head_y])

        if [head_x, head_y] == self.food:
            # 吃到食物：检查逃生空间
            self.foods_eaten += 1
            self._cached_escape_ratio = self._compute_escape_ratio(head_x, head_y)

            # 通关判定：这一口把蛇身填满整个棋盘（清屏），先判胜再放食物，
            # 否则满屏时 _place_food 找不到空格。
            if len(self.snake) >= self.grid_size ** 2:
                self.win = True
                return (
                    self._get_obs(),
                    float(self.win_bonus),
                    True,
                    False,
                    self._get_info(),
                )

            # 【修复】食物奖励不再有 −2 断崖：按“吃完后还剩多少相对空间”给分，
            # 最差也有 0.25×70≈17.5，长蛇不会掉进“吃一口就亏”的负收益区。
            room = self._relative_room(self._cached_escape_ratio)
            reward = self.food_reward_base * (0.25 + 0.75 * room)

            # 长度里程碑：每 milestone_every 个食物额外给一次奖励
            if self.milestone_every > 0 and self.foods_eaten % self.milestone_every == 0:
                reward += self.milestone_bonus

            self.food = self._place_food()
        else:
            self.snake.pop()
            self._cached_escape_ratio = self._compute_escape_ratio(head_x, head_y)

            # 【修复】混合奖励：只奖励接近 + 方向对齐 + 加重步罚
            max_dist = 2 * (self.grid_size - 1) or 1

            # 只在靠近食物时给正奖励，远离不惩罚（彻底杜绝追尾动机）
            approach_reward = max(0.0, old_dist - new_dist) * 4.0 / max_dist

            # 离散方向对齐奖励
            food_dir_reward = 0.0
            if fx > head_x and action == 3:
                food_dir_reward = 0.3
            elif fx < head_x and action == 2:
                food_dir_reward = 0.3
            elif fy > head_y and action == 1:
                food_dir_reward = 0.3
            elif fy < head_y and action == 0:
                food_dir_reward = 0.3

            # 基础步罚加重，逼迫AI主动觅食而非绕圈
            base_penalty = -0.15

            reward = base_penalty + approach_reward + food_dir_reward

            # 逃生空间惩罚：同样改成相对阈值，否则长蛇会被一路扣分
            room = self._relative_room(self._cached_escape_ratio)
            if room < 0.25:
                reward -= 1.0
            if len(self.snake) > self.grid_size * 2 and room < 0.6:
                reward -= 0.5

        # 步数上限：绕圈不死的局在这里被截断，避免训练/评估无限挂起
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_info(self):
        """统一的 info：动作掩码 + 通关标记 + 展示用分数（与训练奖励无关）。"""
        score = self.foods_eaten * 100
        if self.win:
            score += self.win_score
        return {
            "action_mask": self._get_action_mask(),
            "win": self.win,
            "foods": self.foods_eaten,
            "score": score,
        }

    def _place_food(self):
        """在空格中均匀随机放置食物；棋盘被填满时返回 None（调用方需先判胜）。"""
        body = set(map(tuple, self.snake))
        free_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in body
        ]
        if not free_cells:
            return None
        return list(random.choice(free_cells))

    def _get_obs(self):
        head = self.snake[0]
        hx, hy = head
        if self.food is None:  # 棋盘已填满（终局）时的兜底
            fx, fy = hx, hy
        else:
            fx, fy = self.food

        food_dx = 0
        if fx > hx:
            food_dx = 1
        elif fx < hx:
            food_dx = -1

        food_dy = 0
        if fy > hy:
            food_dy = 1
        elif fy < hy:
            food_dy = -1

        def is_danger(x, y):
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                return 1
            if [x, y] in self.snake[:-1]:
                return 1
            return 0

        danger_up = is_danger(hx, hy - 1)
        danger_down = is_danger(hx, hy + 1)
        danger_left = is_danger(hx - 1, hy)
        danger_right = is_danger(hx + 1, hy)

        length_ratio = len(self.snake) / (self.grid_size ** 2)
        if self.obs_version == 1:
            # v1：绝对逃生空间比率（旧模型就是吃这个特征训练的）
            escape_feature = float(self._cached_escape_ratio)
        else:
            # v2：相对逃生空间 room，0=被自己封死，1=还能走到所有剩余空格。
            # 旧的绝对比率在长蛇时恒接近 0，策略无法区分“用满剩余空间”和“真被堵死”。
            escape_feature = self._relative_room(self._cached_escape_ratio)

        # 【修复】用新的方向自由度替换旧的直线隧道宽度
        dir_freedoms = self._compute_directional_freedom(hx, hy)

        return np.array(
            [
                (food_dx + 1) / 2.0,       # 0: 食物x方向
                (food_dy + 1) / 2.0,       # 1: 食物y方向
                float(danger_up),          # 2: 上方危险
                float(danger_down),        # 3: 下方危险
                float(danger_left),        # 4: 左方危险
                float(danger_right),       # 5: 右方危险
                length_ratio,                 # 6: 蛇身长度占比
                escape_feature,               # 7: 逃生空间（v1=绝对比率, v2=相对 room）
                dir_freedoms[0],           # 8: 上方可达自由度
                dir_freedoms[1],           # 9: 下方可达自由度
                dir_freedoms[2],           # 10: 左方可达自由度
                dir_freedoms[3],           # 11: 右方可达自由度
            ],
            dtype=np.float32,
        )

    def render(self):
        if self.render_mode == "human":
            self._draw_frame()

    def _draw_frame(self):
        cell_size = self.window_size // self.grid_size
        window = pygame.display.set_mode(
            (self.window_size, self.window_size), pygame.DOUBLEBUF
        )
        pygame.display.set_caption("AI 贪吃蛇 - 环境测试")

        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)

        window.fill(BLACK)
        for x, y in self.snake:
            pygame.draw.rect(
                window,
                GREEN,
                (x * cell_size, y * cell_size, cell_size - 1, cell_size - 1),
            )
        x, y = self.food
        pygame.draw.rect(
            window, RED, (x * cell_size, y * cell_size, cell_size - 1, cell_size - 1)
        )

        pygame.display.flip()
        pygame.time.delay(120)


if __name__ == "__main__":
    # ⚠️ 注意：这里测试用的grid_size应与训练时保持一致
    env = SnakeEnv(render_mode="human", grid_size=10)
    env.metadata["render_fps"] = 8

    state, _ = env.reset()

    key_map = {
        pygame.K_UP: 0,
        pygame.K_w: 0,
        pygame.K_DOWN: 1,
        pygame.K_s: 1,
        pygame.K_LEFT: 2,
        pygame.K_a: 2,
        pygame.K_RIGHT: 3,
        pygame.K_d: 3,
    }
    opposite = {0: 1, 1: 0, 2: 3, 3: 2}

    current_action = 3
    clock = pygame.time.Clock()

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit

            keys = pygame.key.get_pressed()
            desired_action = None
            for key, action in key_map.items():
                if keys[key]:
                    desired_action = action
                    break

            if (
                desired_action is not None
                and desired_action != opposite.get(current_action)
            ):
                current_action = desired_action

            state, reward, terminated, truncated, info = env.step(current_action)
            env.render()

            if terminated or truncated:
                if info.get("win"):
                    print(f"🏆 通关清屏！食物 {info['foods']} | 展示分: {info['score']:,}")
                state, _ = env.reset()
                current_action = 3

            clock.tick(env.metadata["render_fps"])
    finally:
        env.close()