"""完整诊断：落地+驱动全过程"""
import sys
import numpy as np
import mujoco
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
from forgecraft.core.generator import BodyGenerator
from forgecraft.simulation.builder import build_mjcf_model

catalog = load_catalog()
generator = BodyGenerator(catalog)

print("Searching for robot with motors...")
for attempt in range(50):
    pop = generator.generate_initial_population(size=1, min_parts=4, max_parts=10)
    body = pop[0]
    
    xml_string, joint_map, motor_map, sensor_info = build_mjcf_model(body, catalog)
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    if model.nu > 0:
        n_motors = len(body.actuated_joints())
        print(f"\n=== Robot (motors={n_motors}, parts={body.num_parts()}, actuators={model.nu}) ===")
        
        mass = np.array(model.body_mass)
        def get_com():
            xpos = np.array(data.xpos[:, :3])
            return (mass[:, None] * xpos).sum(axis=0) / mass.sum() if mass.sum() > 0 else xpos[0]
        
        mujoco.mj_resetData(model, data)
        com_init = get_com()
        print(f"Initial: COM_z={com_init[2]:.4f}")
        
        # 阶段1：落地期（更长）
        print("\n--- Grounding (500 steps) ---")
        for gstep in range(500):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data, nstep=4)
            com_g = get_com()
            if gstep % 100 == 0 or gstep < 5:
                print(f"  g{gstep:>3}: z={com_g[2]:.4f}")
            if com_g[2] < 0.1:
                print(f"  *** LANDED at {gstep}! ***")
                break
        
        com_post_ground = get_com()
        print(f"Post-ground: z={com_post_ground[2]:.4f}")
        
        # 阶段2：稳定+驱动
        print("\n--- Drive Phase (1500 steps, mixed signal) ---")
        settling = 200
        drive_start_com = None
        
        for step in range(1500):
            if step < settling:
                data.ctrl[:] = 0.0
            else:
                if drive_start_com is None:
                    drive_start_com = get_com().copy()
                    print(f"  Drive START: COM=({drive_start_com[0]:.3f},{drive_start_com[1]:.3f},{drive_start_com[2]:.3f})")
                
                # 混合驱动信号
                t = (step - settling) * 0.008  # dt*substeps = 0.008
                freq = 3.0
                phases = np.linspace(0, 2*np.pi, model.nu+1)[:-1]
                ctrl = np.sin(2*np.pi*freq*t + phases) * np.cos(2*np.pi*freq*0.7*t + phases*0.5)
                data.ctrl[:] = ctrl
            
            mujoco.mj_step(model, data, nstep=4)
            
            com_now = get_com()
            if drive_start_com is not None and (step-settling) % 200 == 0:
                disp = np.linalg.norm(com_now[:2] - drive_start_com[:2])
                print(f"  d{step-settling:>4}: z={com_now[2]:.4f} disp={disp:.4f} contacts={data.ncon}")
            
            if com_now[2] < 0.03:
                print(f"  *** FELL at d{max(0,step-settling)}! ***")
                break
        
        final_com = get_com()
        total_disp = np.linalg.norm(final_com[:2] - drive_start_com[:2]) if drive_start_com is not None else 0
        print(f"\nFinal: z={final_com[2]:.4f}, disp={total_disp:.4f}m")
        break
else:
    print("No actuated robot found!")
