"""优化的运动测试：针对不同类型使用专用驱动信号"""
import sys
import numpy as np
import mujoco
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from locomotion_templates import LocomotionTemplateGenerator
from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
from forgecraft.simulation.builder import build_mjcf_model

catalog = load_catalog()
gen = LocomotionTemplateGenerator(seed=42)

print("=" * 65)
print("  OPTIMIZED MOTION TEST (Type-Specific Drive)")
print("=" * 65)

def test_robot(body, name, drive_type="mixed"):
    """测试单个机器人，使用专用驱动信号"""
    try:
        xml_string, jm, mm, si = build_mjcf_model(body, catalog)
        model = mujoco.MjModel.from_xml_string(xml_string)
        
        if model.nu == 0:
            return None
        
        data = mujoco.MjData(model)
        mass = np.array(model.body_mass)
        
        def get_com():
            xpos = np.array(data.xpos[:, :3])
            return (mass[:, None] * xpos).sum(axis=0) / mass.sum()
        
        mujoco.mj_resetData(model, data)
        
        # 落地期
        for _ in range(500):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data, nstep=4)
        
        init_com = get_com().copy()
        
        # 驱动期（3000步）
        settling = 200
        total_disp = 0
        prev_pos = init_com[:2].copy()
        fell = False
        
        for step in range(3000):
            t = (step - settling) * 0.008
            
            if step < settling:
                data.ctrl[:] = 0.0
            elif drive_type == "differential":
                # 差速驱动：左轮+右轮- → 前进
                if model.nu >= 2:
                    base_speed = 0.8
                    data.ctrl[0] = base_speed   # 左轮前进
                    data.ctrl[1] = base_speed   # 右轮前进
                    if model.nu > 2:
                        data.ctrl[2:] = 0.3    # 其他轮子低速
                else:
                    data.ctrl[:] = 0.5
                    
            elif drive_type == "gait":
                # 步态驱动：交替摆动
                n = model.nu
                phases = np.linspace(0, 2*np.pi, n+1)[:-1]
                freq = 2.0  # 步频
                
                if name.startswith("bi"):  # 双足：左右交替
                    for i in range(n):
                        phase = phases[i] + 2*np.pi*freq*t
                        if i % 2 == 0:  # 左腿
                            data.ctrl[i] = np.sin(phase) * 0.7
                        else:  # 右腿
                            data.ctrl[i] = np.sin(phase + np.pi) * 0.7
                            
                elif name.startswith("quad"):  # 四足：对角步态
                    for i in range(n):
                        phase = phases[i] + 2*np.pi*freq*t
                        if i % 4 in [0, 3]:  # 对角1
                            data.ctrl[i] = np.sin(phase) * 0.6
                        elif i % 4 in [1, 2]:  # 对角2
                            data.ctrl[i] = np.sin(phase + np.pi) * 0.6
                        else:
                            data.ctrl[i] = np.sin(phase) * 0.5
                else:
                    data.ctrl[:] = np.sin(2*np.pi*freq*t + phases) * 0.6
                    
            elif drive_type == "wave":
                # 波浪驱动（爬行器）
                n = model.nu
                for i in range(n):
                    phase = i * 0.7  # 相位延迟
                    data.ctrl[i] = np.sin(2*np.pi*3.0*t - phase) * 0.8
                    
            else:  # mixed
                freq = 3.0
                phases = np.linspace(0, 2*np.pi, model.nu+1)[:-1]
                ctrl = np.sin(2*np.pi*freq*t + phases) * np.cos(2*np.pi*freq*0.7*t + phases*0.5)
                data.ctrl[:] = ctrl
            
            mujoco.mj_step(model, data, nstep=4)
            
            com_now = get_com()
            
            if step >= settling:
                step_disp = np.linalg.norm(com_now[:2] - prev_pos)
                total_disp += step_disp
                prev_pos = com_now[:2].copy()
                
                if com_now[2] < 0.03:
                    fell = True
                    break
        
        final_com = get_com()
        net_disp = np.linalg.norm(final_com[:2] - init_com[:2])
        survival = 1.0 if not fell else max(0, (3000-step)/3000)
        
        return {
            "disp": net_disp,
            "survival": survival,
            "motors": model.nu,
            "fell": fell,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


# 测试每种类型（多次取最佳）
template_configs = [
    ("Diff-Wheel", gen._create_differential_wheeled, "differential"),
    ("Quad-Wheel", gen._create_quad_wheeled, "differential"),
    ("Bipedal", gen._create_bipedal, "gait"),
    ("Quadruped", gen._create_quadruped, "gait"),
    ("Crawler", gen._create_crawler, "wave"),
]

results = {}

for name, creator, dtype in template_configs:
    best_res = {"disp": 0, "survival": 0}
    
    for attempt in range(10):
        try:
            body = creator()
            if body is None:
                continue
            
            res = test_robot(body, name, dtype)
            if res and res["disp"] > best_res["disp"]:
                best_res = res
                
        except Exception as e:
            continue
    
    results[name] = best_res
    icon = "✓" if best_res["disp"] > 0.005 else ("~" if best_res["survival"] > 0.5 else "✗")
    print(f"  {icon} {name:>10}: disp={best_res['disp']:.4f}m  surv={best_res['survival']:.0%}  motors={best_res.get('motors',0)}")

# 总结
moved = sum(1 for r in results.values() if r["disp"] > 0.005)
valid = len(results)
print(f"\n  Motion capable: {moved}/{valid}")

if moved > 0:
    best_name = max(results, key=lambda k: results[k]["disp"])
    print(f"  Best: {best_name} ({results[best_name]['disp']:.4f}m)")
