"""诊断差速轮式机器人"""
import sys
import numpy as np
import mujoco
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from locomotion_templates import LocomotionTemplateGenerator

gen = LocomotionTemplateGenerator(seed=42)

for attempt in range(10):
    try:
        robot = gen._create_differential_wheeled()
        if robot is None:
            print(f"[{attempt}] create_part returned None")
            continue
        
        # 检查零件质量
        print(f"\n[{attempt}] Robot parts={robot.num_parts()}")
        for node_id in robot.graph.nodes:
            part = robot.graph.nodes[node_id]["part"]
            mass = part.params.get("mass", 0)
            actuated = part.params.get("actuated", 0)
            print(f"  {node_id}: type={part.part_type}, mass={mass:.4f}, actuated={actuated}")
        
        # 尝试构建MuJoCo模型
        from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
        from forgecraft.simulation.builder import build_mjcf_model
        
        catalog = load_catalog()
        xml_string, jm, mm, si = build_mjcf_model(robot, catalog)
        model = mujoco.MjModel.from_xml_string(xml_string)
        print(f"SUCCESS! Actuators={model.nu}")
        break
        
    except Exception as e:
        print(f"[{attempt}] ERROR: {e}")
