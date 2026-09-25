#!/usr/bin/env python3
"""两电机竞速可视化 - MuJoCo 原生窗口 + 终端中文"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from racer_env import TwoWheelRacerEnv
from stable_baselines3 import PPO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "training_output", "best_model", "best_model.zip")

print("=" * 40, flush=True)
print("  两电机竞速机器人 · 策略演示", flush=True)
print("=" * 40, flush=True)
print("加载模型中...", flush=True)
model = PPO.load(MODEL_PATH, device='cpu')
print(f"模型就绪 (训练步数: {model.num_timesteps:,})", flush=True)
print("")
print("窗口说明:", flush=True)
print("  鼠标拖拽 = 旋转视角", flush=True)
print("  鼠标滚轮 = 缩放", flush=True)
print("  右键拖拽 = 平移", flush=True)
print("  Ctrl+C   = 退出", flush=True)
print("", flush=True)

env = TwoWheelRacerEnv(render_mode="human", max_steps=3000)

for ep in range(1, 9999):
    obs, _ = env.reset()
    total_r, max_spd, steps = 0.0, 0.0, 0
    t0 = time.time()
    for i in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        spd = info.get('forward_speed', 0)
        max_spd = max(max_spd, spd)
        steps = i + 1
        env.render()
        if term:
            break

    elapsed = time.time() - t0
    print(f"第{ep:>2}局 | 最高速度 {max_spd*3.6:.1f} km/h | 奖励 {total_r:.0f} | "
          f"存活 {steps}/3000 | 耗时 {elapsed:.0f}秒", flush=True)

env.close()
print("已退出", flush=True)
