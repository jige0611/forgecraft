"""测试模板机器人的实际运动能力"""
import sys
import numpy as np
import mujoco
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))

from locomotion_templates import LocomotionTemplateGenerator
from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
from forgecraft.simulation.builder import build_mjcf_model

logging.basicConfig(level=logging.INFO, format='%(message)s')
catalog = load_catalog()
gen = LocomotionTemplateGenerator(seed=123)

print("=" * 65)
print("  TEMPLATE ROBOT MOTION TEST")
print("=" * 65)

# 测试每个模板类型
template_creators = {
    "Diff-Wheel": gen._create_differential_wheeled,
    "Quad-Wheel": gen._create_quad_wheeled,
    "Bipedal": gen._create_bipedal,
    "Quadruped": gen._create_quadruped,
    "Crawler": gen._create_crawler,
}

results = {}

for name, creator in template_creators.items():
    print(f"\n{'─'*50}")
    print(f"Testing: {name}")
    
    # 尝试多次直到获得有效机器人
    body = None
    for attempt in range(20):
        try:
            body = creator()
            if body is None:
                continue
            
            xml_string, jm, mm, si = build_mjcf_model(body, catalog)
            model = mujoco.MjModel.from_xml_string(xml_string)
            
            if model.nu > 0:
                break  # 有效！
            body = None
        except:
            body = None
    
    if body is None or model.nu == 0:
        print(f"  ✗ Failed to create valid robot")
        results[name] = None
        continue
    
    data = mujoco.MjData(model)
    n_motors = len(body.actuated_joints())
    
    print(f"  Parts={body.num_parts()}, Motors={n_motors}, Actuators={model.nu}")
    
    # 运行直接驱动测试
    mass = np.array(model.body_mass)
    def get_com():
        xpos = np.array(data.xpos[:, :3])
        return (mass[:, None] * xpos).sum(axis=0) / mass.sum()
    
    mujoco.mj_resetData(model, data)
    
    # 落地期（500步）
    for _ in range(500):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data, nstep=4)
    
    init_com = get_com().copy()
    
    # 驱动期（2000步，混合信号）
    settling = 200
    total_disp = 0
    prev_pos = init_com[:2].copy()
    speed_sum = 0
    drive_steps = 0
    fell = False
    max_height = 0
    
    for step in range(2000):
        if step < settling:
            data.ctrl[:] = 0.0
        else:
            t = (step - settling) * 0.008
            freq = 3.0
            phases = np.linspace(0, 2*np.pi, model.nu+1)[:-1]
            ctrl = np.sin(2*np.pi*freq*t + phases) * np.cos(2*np.pi*freq*0.7*t + phases*0.5)
            data.ctrl[:] = ctrl
        
        mujoco.mj_step(model, data, nstep=4)
        
        com_now = get_com()
        max_height = max(max_height, com_now[2])
        
        if step >= settling:
            step_disp = np.linalg.norm(com_now[:2] - prev_pos)
            total_disp += step_disp
            prev_pos = com_now[:2].copy()
            vel = np.array(data.qvel[:model.nv]) if model.nv > 0 else np.zeros(3)
            speed_sum += np.linalg.norm(com_now[:2] - prev_pos) / 0.008 if step > settling else 0
            drive_steps += 1
            
            if com_now[2] < 0.03:
                fell = True
                break
    
    final_com = get_com()
    net_disp = np.linalg.norm(final_com[:2] - init_com[:2])
    survival = (2000 - step) / 2000 if fell else 1.0
    
    results[name] = {
        "disp": net_disp,
        "survival": survival,
        "max_h": max_height,
        "motors": n_motors,
        "fell": fell,
    }
    
    status = "MOVED!" if net_disp > 0.001 else ("FELL" if fell else "STABLE")
    print(f"  Displacement: {net_disp:.4f}m | Survival: {survival:.1%} | MaxH: {max_height:.3f}m")
    print(f"  Status: {status}")

# 总结
print(f"\n{'='*65}")
print("  SUMMARY")
print(f"{'='*65}")

moved_count = sum(1 for r in results.values() if r and r["disp"] > 0.001)
total_valid = sum(1 for r in results.values() if r is not None)

for name, res in results.items():
    if res:
        icon = "✓" if res["disp"] > 0.001 else ("✗" if res["fell"] else "~")
        print(f"  {icon} {name:>10}: disp={res['disp']:.4f}m  surv={res['survival']:.0%}  motors={res['motors']}")
    else:
        print(f"  ✗ {name:>10}: FAILED")

print(f"\n  Motion capable: {moved_count}/{total_valid}")
