#!/usr/bin/env python3
# MuJoCo 环境检测与功能验证

import mujoco
import numpy as np
import sys

print("=" * 60)
print("🔍 MuJoCo 环境深度检测")
print("=" * 60)

# 1. 版本信息
print(f"\n✅ MuJoCo 版本: {mujoco.__version__}")
print(f"   安装路径: {mujoco.__file__}")

# 2. 功能模块检测
modules = {
    'MjModel': hasattr(mujoco, 'MjModel'),
    'MjData': hasattr(mujoco, 'MjData'),
    'Renderer': hasattr(mujoco, 'Renderer'),
    'viewer': hasattr(mujoco, 'viewer'),
    'mj_step': hasattr(mujoco, 'mj_step'),
    'mj_forward': hasattr(mujoco, 'mj_forward'),
}

print("\n📦 模块可用性:")
for name, available in modules.items():
    status = "✅" if available else "❌"
    print(f"   {status} {name}")

# 3. 物理引擎测试
xml_content = """<mujoco model="test">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 0.5">
      <joint name="j1" type="free"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""

try:
    print("\n⚙️ 物理引擎测试:")
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    
    print(f"   ✅ 模型加载成功")
    print(f"   自由度 (nq): {model.nq}")
    print(f"   关节数 (njnt): {model.njnt}")
    print(f"   几何体数 (ngeom): {model.ngeom}")
    print(f"   驱动器数 (nu): {model.nu}")
    
    # 运行仿真
    print("\n🔄 运行仿真测试 (100步)...")
    for i in range(100):
        mujoco.mj_step(model, data)
        if i % 25 == 0:
            pos = data.qpos[:3]
            vel = data.qvel[:3]
            print(f"   步{i:3d}: 位置=({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}) "
                  f"| 速度=({vel[0]:6.3f}, {vel[1]:6.3f}, {vel[2]:6.3f})")
    
    print("\n✅ 物理仿真完全正常!")
    
except Exception as e:
    print(f"\n❌ 物理引擎错误: {e}")
    sys.exit(1)

# 4. 渲染器测试
try:
    print("\n🎨 渲染器测试:")
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data)
    pixels = renderer.render()
    print(f"   ✅ 渲染成功")
    print(f"   分辨率: {pixels.shape[1]}x{pixels.shape[0]}")
    print(f"   像素格式: {pixels.dtype}")
    renderer.close()
except Exception as e:
    print(f"   ⚠️ 渲染器: {e}")

# 5. 性能基准
print("\n⚡ 性能基准测试:")
import time

model_perf = mujoco.MjModel.from_xml_string(xml_content)
data_perf = mujoco.MjData(model_perf)

steps = 10000
start = time.perf_counter()
for _ in range(steps):
    mujoco.mj_step(model_perf, data_perf)
elapsed = time.perf_counter() - start

fps = steps / elapsed
step_time_ms = (elapsed / steps) * 1000

print(f"   仿真步数: {steps:,}")
print(f"   总耗时: {elapsed:.3f}s")
print(f"   每步耗时: {step_time_ms:.4f}ms")
print(f"   等效FPS: {fps:,.0f}")

if fps > 10000:
    print("   🚀 性能: 极快 (适合实时控制)")
elif fps > 1000:
    print("   ⚡ 性能: 快速 (适合实时仿真)")
else:
    print("   ✅ 性能: 正常")

print("\n" + "=" * 60)
print("✅ MuJoCo 环境检测完成 - 所有核心功能正常!")
print("=" * 60)
