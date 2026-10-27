"""训练曲线：记录每局得分，并画成「平均得分 / 最高得分 随训练轮次变化」的图。

两种用法
--------
1) 训练时自动记录（train_snake.py 已接好 CurveCallback）
       python train_snake.py            # 训练过程中写入 snake_logs/curve.csv
       python training_curve.py         # 画 snake_logs/curve.csv -> snake_logs/curve.png

2) 用已有检查点补画历史曲线（不需要重新训练）
       python training_curve.py --eval
       python training_curve.py --eval --pattern "snake_logs/snake_rl_v2_*.zip" --seeds 8 --points 12

图里三条线：灰=每局原始得分，蓝=滑动平均，橙=滑动最高。
下半部分是「食物数」的同样统计（食物数比奖励更直观）。
"""

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "snake_logs")
CURVE_CSV = os.path.join(LOG_DIR, "curve.csv")
CURVE_PNG = os.path.join(LOG_DIR, "curve.png")
EVAL_CSV = os.path.join(LOG_DIR, "curve_eval.csv")
EVAL_PNG = os.path.join(LOG_DIR, "curve_eval.png")

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


# --------------------------------------------------------------------------
# 1) 训练时记录
# --------------------------------------------------------------------------
def make_callback(csv_path=CURVE_CSV):
    """返回一个 SB3 callback；每局结束向 csv 追加一行。"""
    from stable_baselines3.common.callbacks import BaseCallback

    class CurveCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.csv_path = csv_path
            self._acc = None
            self._episode = 0
            self._fh = None
            self._writer = None

        def _on_training_start(self):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            is_new = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
            self._fh = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._fh)
            if is_new:
                self._writer.writerow(["timestep", "episode", "reward", "foods", "win", "epsilon"])
            self._acc = np.zeros(self.training_env.num_envs, dtype=np.float64)

        def _on_step(self):
            self._acc += np.asarray(self.locals["rewards"], dtype=np.float64)
            for i, done in enumerate(self.locals["dones"]):
                if not done:
                    continue
                info = self.locals["infos"][i]
                self._episode += 1
                eps = getattr(self.model, "exploration_rate", "")
                self._writer.writerow(
                    [
                        self.num_timesteps,
                        self._episode,
                        round(float(self._acc[i]), 2),
                        info.get("foods", ""),
                        int(bool(info.get("win", False))),
                        round(float(eps), 4) if eps != "" else "",
                    ]
                )
                self._acc[i] = 0.0
            if self._episode and self.num_timesteps % 5000 == 0:
                self._fh.flush()
            return True

        def _on_training_end(self):
            if self._fh:
                self._fh.flush()
                self._fh.close()
                self._fh = None
            return True

    return CurveCallback()


# --------------------------------------------------------------------------
# 2) 用检查点补画历史
# --------------------------------------------------------------------------
def _steps_of(path):
    m = re.search(r"_(\d+)_steps", os.path.basename(path))
    return int(m.group(1)) if m else 0


def eval_checkpoints(pattern, seeds, points, max_steps=3000, obs_version=2):
    """对一批检查点各跑 seeds 局，记录平均分/最高分，写入 csv 并返回数据。"""
    from stable_baselines3 import DQN

    from mask_wrapper import ActionMaskWrapper
    from snake import SnakeEnv

    files = [p for p in glob.glob(pattern) if p.endswith(".zip")]
    if not files:
        raise SystemExit(f"没找到检查点：{pattern}")
    # 同一批里可能混着多次训练的断点，按步数去重后均匀取样
    by_steps = {}
    for p in files:
        by_steps.setdefault(_steps_of(p), p)
    ordered = [by_steps[s] for s in sorted(by_steps)]
    if len(ordered) > points:
        idx = np.linspace(0, len(ordered) - 1, points).round().astype(int)
        ordered = [ordered[i] for i in sorted(set(idx))]

    rows = []
    for p in ordered:
        steps = _steps_of(p)
        env = ActionMaskWrapper(
            SnakeEnv(render_mode=None, grid_size=15, max_steps=max_steps, obs_version=obs_version)
        )
        model = DQN.load(p, device="cpu")
        rewards, foods = [], []
        for seed in range(seeds):
            obs, info = env.reset(seed=700 + seed)
            total = 0.0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            rewards.append(total)
            foods.append(info["foods"])
        env.close()
        row = (
            steps,
            float(np.mean(rewards)),
            float(np.max(rewards)),
            float(np.mean(foods)),
            float(np.max(foods)),
        )
        rows.append(row)
        print(
            f"  {steps:>8,} 步 | 平均分 {row[1]:8.1f} | 最高分 {row[2]:8.1f} "
            f"| 平均食物 {row[3]:5.1f} | 最高 {row[4]:5.1f}",
            flush=True,
        )

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(EVAL_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["timestep", "mean_reward", "max_reward", "mean_foods", "max_foods"])
        w.writerows(rows)
    print(f"\n已写入 {EVAL_CSV}")
    return rows


# --------------------------------------------------------------------------
# 3) 画图（Pillow，无第三方绘图库）
# --------------------------------------------------------------------------
def _font(size, bold=False):
    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.truetype(FONT_CANDIDATES[0], size) if bold else ImageFont.load_default()
    except OSError:
        return ImageFont.load_default()


def _rolling(values, window, fn):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(fn(values[lo : i + 1]))
    return out


def _draw_panel(draw, box, xs, series, title, font, font_small, s=1, show_x=True):
    """series: [(label, ys, color, width)]，xs 与 ys 等长。"""
    x0, y0, x1, y1 = box
    all_ys = np.concatenate([np.asarray(s[1], dtype=float) for s in series])
    ymin, ymax = float(np.min(all_ys)), float(np.max(all_ys))
    if ymax - ymin < 1e-9:
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.10
    ymin -= pad
    ymax += pad

    def px(i):
        return x0 + (x1 - x0) * (i / max(1, len(xs) - 1))

    def py(v):
        return y1 - (y1 - y0) * ((v - ymin) / (ymax - ymin))

    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(205, 205, 210))
    for k in range(5):
        v = ymin + (ymax - ymin) * k / 4
        y = py(v)
        draw.line([x0, y, x1, y], fill=(236, 236, 240))
        draw.text((x0 - 12, y), f"{v:,.0f}", font=font_small, fill=(95, 95, 100), anchor="rm")

    for label, ys, color, width in series:
        pts = [(px(i), py(float(v))) for i, v in enumerate(ys)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=width, joint="curve")

    # 面板标题（画在框外上方，留足空间不压刻度）
    draw.text((x0, y0 - 44 * s), title, font=font, fill=(35, 35, 40))

    # x 轴刻度
    if show_x:
        n = len(xs)
        for k in range(5):
            i = round(k * (n - 1) / 4)
            draw.line([px(i), y1, px(i), y1 + 6 * s], fill=(150, 150, 155))
            draw.text(
                (px(i), y1 + 12 * s), f"{xs[i]:,}", font=font_small, fill=(95, 95, 100), anchor="mt"
            )

    # 图例：画在面板内右上角，带白底避免压住曲线
    labels = [s0[0] for s0 in series]
    lw = max(draw.textlength(t, font=font_small) for t in labels) + 62 * s
    lh = len(series) * 26 * s + 12 * s
    lx, ly = x1 - lw - 16 * s, y0 + 12 * s
    draw.rectangle([lx, ly, lx + lw, ly + lh], fill=(255, 255, 255), outline=(226, 226, 231))
    for i, (label, _, color, width) in enumerate(series):
        yy = ly + 10 * s + i * 26 * s + 6 * s
        draw.line([lx + 12 * s, yy, lx + 40 * s, yy], fill=color, width=width)
        draw.text((lx + 48 * s, yy), label, font=font_small, fill=(40, 40, 45), anchor="lm")


def _save(im, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)


def plot_live(csv_path=CURVE_CSV, out_png=CURVE_PNG, window=0):
    if not os.path.exists(csv_path):
        raise SystemExit(f"没有找到 {csv_path}，先训练一会儿（train_snake.py 会自动记录）。")

    episodes, rewards, foods = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            foods.append(float(row["foods"]) if row["foods"] not in ("", None) else np.nan)
    if len(episodes) < 3:
        raise SystemExit(f"{csv_path} 里只有 {len(episodes)} 局，先多训练一会儿再画。")

    if window <= 0:
        window = max(5, min(100, len(episodes) // 10))

    avg = _rolling(rewards, window, np.mean)
    best = _rolling(rewards, window, np.max)
    if np.all(np.isnan(foods)):
        foods = None
    else:
        f_avg = _rolling(foods, window, np.nanmean)
        f_best = _rolling(foods, window, np.nanmax)

    from PIL import Image, ImageDraw

    W, H = 1400, 1000
    S = 2
    im = Image.new("RGB", (W * S, H * S), (250, 250, 252))
    d = ImageDraw.Draw(im)
    f_title, f_sub = _font(30 * S, True), _font(18 * S)
    f_panel, f_axis = _font(20 * S, True), _font(16 * S)

    d.text((60 * S, 36 * S), "训练曲线：平均 / 最高得分", font=f_title, fill=(20, 20, 25))
    d.text(
        (60 * S, 84 * S),
        f"共 {len(episodes)} 局 · 滑动窗口 {window} 局 · 数据 {os.path.basename(csv_path)}",
        font=f_sub,
        fill=(110, 110, 115),
    )

    xs = list(range(1, len(episodes) + 1))
    gray = (205, 205, 210)
    blue = (31, 119, 180)
    orange = (255, 127, 14)
    series_score = [
        ("每局得分", rewards, gray, 2 * S),
        (f"平均（{window} 局）", avg, blue, 4 * S),
        (f"最高（{window} 局）", best, orange, 4 * S),
    ]
    _draw_panel(
        d, (150 * S, 190 * S, 1340 * S, 560 * S), xs, series_score, "得分（奖励）", f_panel, f_axis,
        s=S, show_x=False,
    )
    if foods is not None:
        series_food = [
            ("每局食物", foods, gray, 2 * S),
            (f"平均（{window} 局）", f_avg, blue, 4 * S),
            (f"最高（{window} 局）", f_best, orange, 4 * S),
        ]
        _draw_panel(
            d, (150 * S, 710 * S, 1340 * S, 950 * S), xs, series_food, "食物数", f_panel, f_axis, s=S
        )
        d.text((150 * S, 976 * S), "训练局数（episode）", font=f_axis, fill=(90, 90, 95))

    im = im.resize((W, H), Image.LANCZOS)
    _save(im, out_png)
    print(f"已生成 {out_png}")
    n = len(rewards)
    print(
        f"最近 {min(window, n)} 局：平均得分 {np.mean(rewards[-window:]):.1f}，"
        f"最高 {np.max(rewards[-window:]):.1f}"
    )
    return out_png


def plot_eval(csv_path=EVAL_CSV, out_png=EVAL_PNG):
    if not os.path.exists(csv_path):
        raise SystemExit(f"没有找到 {csv_path}，先跑 python training_curve.py --eval")

    steps, mean_r, max_r, mean_f, max_f = [], [], [], [], []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            steps.append(int(row["timestep"]))
            mean_r.append(float(row["mean_reward"]))
            max_r.append(float(row["max_reward"]))
            mean_f.append(float(row["mean_foods"]))
            max_f.append(float(row["max_foods"]))

    from PIL import Image, ImageDraw

    W, H = 1400, 1000
    S = 2
    im = Image.new("RGB", (W * S, H * S), (250, 250, 252))
    d = ImageDraw.Draw(im)
    f_title, f_sub = _font(30 * S, True), _font(18 * S)
    f_panel, f_axis = _font(20 * S, True), _font(16 * S)

    d.text((60 * S, 36 * S), "训练曲线：检查点评测（平均 / 最高）", font=f_title, fill=(20, 20, 25))
    d.text(
        (60 * S, 84 * S),
        f"x 轴为累计训练步数 · 每个点在该检查点上评测多局 · 数据 {os.path.basename(csv_path)}",
        font=f_sub,
        fill=(110, 110, 115),
    )

    blue, orange = (31, 119, 180), (255, 127, 14)
    _draw_panel(
        d, (150 * S, 190 * S, 1340 * S, 560 * S), steps,
        [("平均得分", mean_r, blue, 4 * S), ("最高得分", max_r, orange, 4 * S)],
        "得分（奖励）", f_panel, f_axis, s=S, show_x=False,
    )
    _draw_panel(
        d, (150 * S, 710 * S, 1340 * S, 950 * S), steps,
        [("平均食物", mean_f, blue, 4 * S), ("最高食物", max_f, orange, 4 * S)],
        "食物数", f_panel, f_axis, s=S,
    )
    d.text((150 * S, 976 * S), "累计训练步数（timesteps）", font=f_axis, fill=(90, 90, 95))

    im = im.resize((W, H), Image.LANCZOS)
    _save(im, out_png)
    print(f"已生成 {out_png}")
    return out_png


def main(argv=None):
    ap = argparse.ArgumentParser(description="训练曲线：记录 / 绘图")
    ap.add_argument("--eval", action="store_true", help="用已有检查点补画历史曲线")
    ap.add_argument("--pattern", default=os.path.join(LOG_DIR, "snake_rl_v2_*.zip"))
    ap.add_argument("--seeds", type=int, default=8, help="每个检查点评测多少局")
    ap.add_argument("--points", type=int, default=12, help="最多评测多少个检查点")
    ap.add_argument("--window", type=int, default=0, help="滑动窗口局数，0=自动")
    ap.add_argument("--obs", type=int, default=2, choices=[1, 2], help="检查点用的观测版本")
    ap.add_argument("--csv", default=CURVE_CSV)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    if args.eval:
        print(f"评测检查点：{args.pattern}（每个 {args.seeds} 局，最多 {args.points} 个）")
        eval_checkpoints(args.pattern, args.seeds, args.points, obs_version=args.obs)
        plot_eval()
    else:
        plot_live(args.csv, args.out or CURVE_PNG, args.window)


if __name__ == "__main__":
    sys.exit(main())
