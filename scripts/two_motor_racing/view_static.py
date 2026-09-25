#!/usr/bin/env python3
"""静态模型查看 - 机器人不动，纯浏览"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from racer_env import TwoWheelRacerEnv

print("打开静态模型...", flush=True)
env = TwoWheelRacerEnv(render_mode="human", max_steps=3000)
obs, _ = env.reset()

print("")
print("机器人已加载，处于静止状态", flush=True)
print("  鼠标左键拖拽 = 旋转", flush=True)
print("  鼠标滚轮     = 缩放", flush=True)
print("  鼠标右键拖拽 = 平移", flush=True)
print("  Ctrl+C       = 退出", flush=True)
print("")

try:
    while True:
        env.render()
        time.sleep(1/60)
except KeyboardInterrupt:
    env.close()
    print("已退出", flush=True)
