"""
YAML 配置加载器

从 YAML 文件加载零件箱 (catalog) 和任务 (task) 配置:
  catalog YAML → CatalogConstraints + CatalogSpec[] → Catalog → PartSpec{}
  task YAML    → TaskSpec → SimConfig + RLConfig + TaskConfig

用户入口:
  load_catalog(name) → Catalog
  load_task(name)    → TaskSpec
  list_catalogs()    → [name, ...]
  list_tasks()       → [name, ...]
"""

# Standard library
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

# Third-party
import yaml

# ── 配置加载器 ──
# 从 YAML 文件加载 catalog (零件箱) 和 task (任务) 配置。
# YAML 结构: parts[].physics/geometry/joint/sensor → PartSpec
#            task.simulation/termination/reward/pareto → SimConfig + TaskConfig

from forgecraft.config import (
    EvolutionConfig,
    PartSpec,
    RewardComponent,
    RLConfig,
    SimConfig,
    TaskConfig,
)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_CATALOG_DIR = _PACKAGE_DIR / "configs" / "catalogs"
_DEFAULT_TASK_DIR = _PACKAGE_DIR / "configs" / "tasks"

__all__ = [
    "CatalogConstraints",
    "CatalogSpec",
    "TaskSpec",
    "Catalog",
    "load_catalog",
    "load_task",
    "list_catalogs",
    "list_tasks",
]


class CatalogConstraints:
    def __init__(self, data: dict):
        self.max_depth = data.get("max_depth", 6)
        self.max_children_per_node = data.get("max_children_per_node", 4)
        self.require_root = data.get("require_root", True)
        self.symmetry = data.get("symmetry", "none")
        self.max_parts = data.get("max_parts", 30)

    def to_dict(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "max_children_per_node": self.max_children_per_node,
            "require_root": self.require_root,
            "symmetry": self.symmetry,
            "max_parts": self.max_parts,
        }


class CatalogSpec:
    """单个零件类型规格 — YAML parts[].* → PartSpec 中间表示

    解析 YAML 中的:
      geometry.{type,color} + physics.{mass,friction} + size.{length,radius,...}
      + joint.{type,axis,range,torque,velocity,damping}
      + sensor.{touch,imu}
    统一转为 PartSpec 供进化系统使用。
    """
    def __init__(self, name: str, part_type: str, data: dict):
        self.name = name
        self.part_type = part_type
        self._data = data
        self.geometry_type = data.get("geometry", {}).get("type", "box")
        self.color = data.get("geometry", {}).get("color", [0.6, 0.6, 0.6, 1.0])
        self.mass = data.get("physics", {}).get("mass", 0.1)
        self.friction = data.get("physics", {}).get("friction", [0.6, 0.01, 0.01])
        self.size = data.get("size", {})
        self.actuated = data.get("actuated", False)
        self.faces = data.get("faces", ["front"])

        joint_data = data.get("joint", {})
        self.joint_type = joint_data.get("type", "hinge")
        self.joint_axis = joint_data.get("axis", [0, 1, 0])
        self.joint_range = joint_data.get("range", None)
        self.joint_torque_range = joint_data.get("torque", [0.5, 5.0])
        self.joint_velocity_range = joint_data.get("velocity", [5.0, 15.0])
        self.joint_damping = joint_data.get("damping", 0.5)

        sensor_data = data.get("sensor", {})
        self.has_touch = sensor_data.get("touch", False)
        self.has_imu = sensor_data.get("imu", False)
        
        # ── 工程特征 (YAML 增强) ──
        mounting_data = data.get("mounting", {})
        self.mounting = {
            "pattern": mounting_data.get("pattern", "4xM3"),       # e.g. "4xM3", "6xM4"
            "spacing": mounting_data.get("spacing", [25.0, 25.0]), # [x_spacing, y_spacing] mm
            "hole_diameter": mounting_data.get("hole_diameter", 3.2),
            "face": mounting_data.get("face", "+z"),               # 安装面方向
        }
        shaft_data = data.get("shaft", {})
        self.shaft = {
            "diameter": shaft_data.get("diameter", None),          # None → 自动推算
            "tolerance": shaft_data.get("tolerance", "H7"),        # 配合公差
            "length": shaft_data.get("length", None),
        }
        keyway_data = data.get("keyway", {})
        self.keyway = {
            "width": keyway_data.get("width", None),
            "depth": keyway_data.get("depth", None),
            "standard": keyway_data.get("standard", "DIN6885"),    # 键槽标准
        }
        thread_data = data.get("thread", {})
        self.thread = {
            "size": thread_data.get("size", None),                 # e.g. "M3", "M4"
            "pitch": thread_data.get("pitch", None),
            "length": thread_data.get("length", None),
            "type": thread_data.get("type", "metric"),             # metric/unified
        }

    def to_part_spec(self) -> PartSpec:
        size_min = 0.01
        size_max = 0.1
        for key in ["length", "radius", "width"]:
            if key in self.size:
                if isinstance(self.size[key], list) and len(self.size[key]) == 2:
                    size_min = min(self.size[key])
                    size_max = max(self.size[key])
                break

        param_ranges = {}
        for key, val in self.size.items():
            if isinstance(val, list) and len(val) == 2:
                param_ranges[key] = [float(val[0]), float(val[1])]

        if self.actuated:
            if isinstance(self.joint_torque_range, list) and len(self.joint_torque_range) == 2:
                param_ranges["max_torque"] = [float(self.joint_torque_range[0]), float(self.joint_torque_range[1])]
            if isinstance(self.joint_velocity_range, list) and len(self.joint_velocity_range) == 2:
                param_ranges["max_velocity"] = [float(self.joint_velocity_range[0]), float(self.joint_velocity_range[1])]

        mass_range = self._data.get("mass_range", self._data.get("physics", {}).get("mass_range", []))
        if not isinstance(mass_range, list) or len(mass_range) != 2:
            mass_range = []

        density = self._data.get("density", self._data.get("physics", {}).get("density", 1000.0))

        return PartSpec(
            part_type=self.name,
            mass=self.mass,
            shape=self.geometry_type,
            size_range=[size_min, size_max],
            can_actuate=self.actuated,
            max_torque=max(self.joint_torque_range) if self.actuated else 0.0,
            color=self.color,
            param_ranges=param_ranges,
            mass_range=mass_range,
            density=float(density),
            has_touch=self.has_touch,
            has_imu=self.has_imu,
            friction=list(self.friction) if isinstance(self.friction, (list, tuple)) else [0.6, 0.01, 0.01],
            joint_type=self.joint_type if self.actuated else "fixed",
            joint_axis=list(self.joint_axis) if isinstance(self.joint_axis, (list, tuple)) else [0.0, 1.0, 0.0],
            joint_range_min=self.joint_range[0] if self.joint_range and len(self.joint_range) == 2 else -1.5,
            joint_range_max=self.joint_range[1] if self.joint_range and len(self.joint_range) == 2 else 1.5,
            joint_damping=float(self.joint_damping),
        )

    def to_dict(self) -> dict:
        return {
            "part_type": self.name,
            "geometry": self.geometry_type,
            "color": self.color,
            "mass": self.mass,
            "friction": self.friction,
            "size": self.size,
            "actuated": self.actuated,
            "faces": self.faces,
            "joint": {
                "type": self.joint_type,
                "axis": self.joint_axis,
                "range": self.joint_range,
                "torque": self.joint_torque_range,
                "velocity": self.joint_velocity_range,
                "damping": self.joint_damping,
            } if self.actuated else None,
        }


class TaskSpec:
    def __init__(self, data: dict):
        self.name = data.get("name", "unnamed")
        self.description = data.get("description", "")

        sim = data.get("simulation", {})
        self.timestep = sim.get("timestep", 0.005)
        self.substeps = sim.get("substeps", 10)
        self.gravity = sim.get("gravity", [0, 0, -9.81])
        self.friction = sim.get("friction", 0.6)

        terrain = data.get("terrain", {})
        self.terrain_type = terrain.get("type", "flat")
        self.terrain_size = terrain.get("size", [10, 10])

        term = data.get("termination", {})
        self.max_steps = term.get("max_steps", 500)
        self.fall_height = term.get("fall_height", 0.05)

        reward = data.get("reward", {})
        self.reward_components: List[Dict[str, Any]] = reward.get("components", [])
        self.fitness_formula: str = reward.get("fitness_formula", "")
        self.fitness_components: List[str] = reward.get("fitness_components", data.get("metrics", []))
        self.domain: str = reward.get("domain", "robot")

        pareto = data.get("pareto", {})
        self.pareto_enabled = pareto.get("enabled", False)
        self.pareto_objectives = pareto.get("objectives", [])

        self.metrics = data.get("metrics", [])

    def to_sim_config(self) -> SimConfig:
        gz = self.gravity[2] if len(self.gravity) > 2 else -9.81
        return SimConfig(
            gravity=gz,
            timestep=self.timestep,
            friction=self.friction,
            max_steps=self.max_steps,
            substeps=self.substeps,
        )

    def to_task_config(self) -> TaskConfig:
        components = []
        for comp in self.reward_components:
            components.append(RewardComponent(
                name=comp.get("name", "unknown"),
                weight=comp.get("weight", 1.0),
                expression=comp.get("expression", ""),
                description=comp.get("description", ""),
            ))

        fitness_formula = self.fitness_formula
        fitness_components = self.fitness_components

        return TaskConfig(
            reward_components=components,
            fitness_formula=fitness_formula,
            fitness_components=fitness_components,
            max_episode_steps=self.max_steps,
            fall_height=self.fall_height,
            domain=self.domain,
        )


class Catalog:
    """零件箱 — 聚合约束 + 零件类型列表

    提供:
      - to_part_specs()  → {type_name: PartSpec} 供进化系统使用
      - get_root_options() → 可作为根节点的非驱动类型
      - get_actuated() / get_passive() → 按驱动能力分类
    """
    def __init__(
        self,
        name: str,
        constraints: CatalogConstraints,
        parts: Dict[str, CatalogSpec],
    ):
        self.name = name
        self.constraints = constraints
        self.parts = parts

    def to_part_specs(self) -> Dict[str, PartSpec]:
        return {name: spec.to_part_spec() for name, spec in self.parts.items()}

    def get_root_options(self) -> List[str]:
        return [name for name, spec in self.parts.items() if not spec.actuated]

    def get_actuated(self) -> List[str]:
        return [name for name, spec in self.parts.items() if spec.actuated]

    def get_passive(self) -> List[str]:
        return [name for name, spec in self.parts.items() if not spec.actuated]

    def describe(self) -> str:
        lines = [f"Catalog: {self.name}"]
        lines.append(f"  Constraints: {self.constraints.to_dict()}")
        for name, spec in self.parts.items():
            act = "[驱动]" if spec.actuated else "[固定]"
            lines.append(f"  {name:16s} {act} {spec.geometry_type:10s} mass={spec.mass:.3f}")
        return "\n".join(lines)


def load_catalog(path: Optional[str] = None, name: str = "primitives") -> Catalog:
    if path is not None:
        full_path = Path(path)
    else:
        full_path = _DEFAULT_CATALOG_DIR / f"{name}.yaml"

    if not full_path.exists():
        available = list(_DEFAULT_CATALOG_DIR.glob("*.yaml"))
        available_names = [p.stem for p in available]
        raise FileNotFoundError(
            f"零件箱 '{full_path}' 不存在。\n"
            f"可用预设: {available_names}\n"
            f"可用文件: {[str(p) for p in available]}"
        )

    with open(full_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    constraints = CatalogConstraints(data.get("constraints", {}))
    parts = {}
    for part_name, part_data in data.get("parts", {}).items():
        parts[part_name] = CatalogSpec(part_name, part_name, part_data)

    return Catalog(name=full_path.stem, constraints=constraints, parts=parts)


def load_task(path: Optional[str] = None, name: str = "speed") -> TaskSpec:
    if path is not None:
        full_path = Path(path)
    else:
        full_path = _DEFAULT_TASK_DIR / f"{name}.yaml"

    if not full_path.exists():
        available = list(_DEFAULT_TASK_DIR.glob("*.yaml"))
        available_names = [p.stem for p in available]
        raise FileNotFoundError(
            f"任务 '{full_path}' 不存在。\n"
            f"可用预设: {available_names}"
        )

    with open(full_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return TaskSpec(data)


def list_catalogs() -> List[str]:
    if not _DEFAULT_CATALOG_DIR.exists():
        return []
    return sorted([p.stem for p in _DEFAULT_CATALOG_DIR.glob("*.yaml")])


def list_tasks() -> List[str]:
    if not _DEFAULT_TASK_DIR.exists():
        return []
    return sorted([p.stem for p in _DEFAULT_TASK_DIR.glob("*.yaml")])
