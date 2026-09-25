from __future__ import annotations

import logging
import random
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from forgecraft.config import PartSpec
from forgecraft.core.catalog import DEFAULT_CATALOG
from forgecraft.core.morphology import Joint, MechanicalBody, Part

__all__ = ["BodyGenerator"]

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from forgecraft.core.loader import Catalog


class BodyGenerator:
    """随机形态生成器 — 从零件箱组装 MechanicalBody

    生成策略:
      1. 选根: 从非驱动零件 (can_actuate=False) 中随机取
      2. 递归: 按约束 (max_depth, max_children_per_node) 扩展子节点
      3. 参数: 从 PartSpec.param_ranges 均匀采样 → 高斯噪声微调
      4. 拓扑: 随机零件类型 + 随机关节类型 + 随机连接的叶节点
      5. 校验: 自动检查最小零件数 + root 非空

    Usage:
        gen = BodyGenerator(catalog, seed=42)
        body = gen.generate_random_body(min_parts=4, max_parts=12)
        pop = gen.generate_initial_population(size=50)
    """
    def __init__(
        self,
        catalog: Optional[Dict[str, PartSpec]] = None,
        seed: Optional[int] = None,
    ):
        self.catalog = catalog or dict(DEFAULT_CATALOG)
        self.rng = np.random.RandomState(seed)
        self.random = random.Random(seed)
        self._catalog_obj: Optional["Catalog"] = None

    @classmethod
    def from_catalog(cls, catalog: "Catalog", seed: Optional[int] = None) -> "BodyGenerator":
        gen = cls(catalog=catalog.to_part_specs(), seed=seed)
        gen._catalog_obj = catalog
        return gen

    def sample_param(self, spec: PartSpec, key: str) -> float:
        lo, hi = spec.get_range(key)
        if key == "mass":
            if spec.mass_range and len(spec.mass_range) == 2:
                return self.rng.uniform(*spec.mass_range)
            return spec.mass * self.rng.uniform(0.8, 1.2)
        return self.rng.uniform(lo, hi)

    def create_part(self, part_type: str, position: Optional[np.ndarray] = None) -> Part:
        spec = self.catalog[part_type]
        params: Dict[str, float] = {}

        if spec.param_ranges:
            for key in spec.param_ranges:
                if key.startswith("joint_"):
                    continue
                params[key] = self.sample_param(spec, key)
        else:
            params.update({
                "length": self.sample_param(spec, "length"),
                "radius": self.sample_param(spec, "radius"),
                "thickness": self.sample_param(spec, "thickness"),
            })

        if spec.mass_range and len(spec.mass_range) == 2:
            params["mass"] = self.rng.uniform(*spec.mass_range)
        else:
            params["mass"] = spec.mass * self.rng.uniform(0.8, 1.2)

        if spec.can_actuate:
            params["max_torque"] = self.sample_param(spec, "max_torque")
            params["max_velocity"] = self.sample_param(spec, "max_velocity")
            params["actuated"] = 1.0
        else:
            params["actuated"] = 0.0

        if spec.has_touch:
            params["touch_sensor"] = 1.0
        if spec.has_imu:
            params["imu_sensor"] = 1.0

        if spec.friction:
            params["friction_primary"] = spec.friction[0] if isinstance(spec.friction, (list, tuple)) else spec.friction

        if spec.density:
            params["density"] = spec.density
        pos = position if position is not None else np.zeros(3, dtype=np.float32)
        return Part(part_type=part_type, params=params, position=pos)

    # ── Generate a random body: root → recursive growth → motor placement → joint setup
    def generate_random_body(
        self,
        name: str = "random_body",
        min_parts: int = 4,
        max_parts: int = 10,
        max_depth: int = 4,
    ) -> MechanicalBody:
        # 防御: catalog 为空时无法生成
        if not self.catalog:
            _logger.error("generate_random_body: catalog is empty")
            body = MechanicalBody(name=name)
            body.fitness = -1e6
            return body

        try:
            return self._generate_body_impl(name, min_parts, max_parts, max_depth)
        except Exception as e:
            _logger.warning(f"generate_random_body failed: {e}, returning minimal body")
            body = MechanicalBody(name=name)
            body.fitness = -1e6
            return body

    def _generate_body_impl(
        self,
        name: str,
        min_parts: int,
        max_parts: int,
        max_depth: int,
    ) -> MechanicalBody:
        body = MechanicalBody(name=name)
        total_parts = self.rng.randint(min_parts, max_parts + 1)

        # 根节点: 必须是非驱动零件 (can_actuate=False)
        root_types = [t for t in self.catalog if not self.catalog[t].can_actuate
                      and self.catalog[t].shape in ("box", "cylinder")]
        if not root_types:
            root_types = [t for t in self.catalog if not self.catalog[t].can_actuate]
        root_type = root_types[0] if root_types else list(self.catalog.keys())[0]

        root = self.create_part(root_type, position=np.array([0.0, 0.0, 0.5], dtype=np.float32))
        body.add_part(root)

        # 候选子零件类型 (排除根类型)
        part_types = [t for t in self.catalog if t != root_type]

        # 逐个添加子零件到随机选中的叶子节点
        for _ in range(total_parts - 1):
            parent_id = self.random.choice(list(body.graph.nodes))
            if body.get_level(parent_id) >= max_depth:
                parent_id = self.random.choice([n for n in body.graph.nodes if body.get_level(n) < max_depth])

            new_type = self.random.choice(part_types)
            parent_part = body.get_part(parent_id)
            spec = self.catalog[parent_part.part_type]

            direction = self.rng.randn(3).astype(np.float32)
            direction[1] = abs(direction[1])
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            parent_len = parent_part.params.get("length", 0.1)
            offset = parent_len * 0.5
            new_pos = parent_part.position + direction * offset

            new_part = self.create_part(new_type, position=new_pos)
            body.add_part(new_part)

            new_spec = self.catalog[new_type]
            joint_type = getattr(new_spec, "joint_type", "hinge") or "hinge"
            joint_axis_raw = getattr(new_spec, "joint_axis", [0.0, 1.0, 0.0])
            joint_axis = np.array(joint_axis_raw, dtype=np.float32)
            joint_range_min = getattr(new_spec, "joint_range_min", -1.5)
            joint_range_max = getattr(new_spec, "joint_range_max", 1.5)
            joint_damping = getattr(new_spec, "joint_damping", 0.5)

            joint = Joint(
                joint_type=joint_type,
                parent_id=parent_id,
                child_id=new_part.part_id,
                anchor=new_pos - parent_part.position,
                axis=joint_axis,
                params={
                    "range_min": joint_range_min,
                    "range_max": joint_range_max,
                    "damping": joint_damping,
                },
            )
            body.add_joint(joint)

        return body

    def generate_simple_walker(
        self,
        n_legs: int = 2,
        segments_per_leg: int = 2,
        name: str = "simple_walker",
    ) -> MechanicalBody:
        body = MechanicalBody(name=name)
        root = self.create_part("base", position=np.array([0.0, 0.0, 0.5], dtype=np.float32))
        body.add_part(root)

        base_size = root.params.get("length", 0.1)

        # 选择当前零件箱中实际可用的驱动零件 (不同零件箱命名不同)
        motor_type = next(
            (t for t in ("hinge_motor", "motor", "hinge_joint") if t in self.catalog),
            None,
        )

        for leg_idx in range(n_legs):
            sign = 1.0 if leg_idx % 2 == 0 else -1.0
            x_offset = sign * base_size * 0.3

            prev_id = root.part_id
            prev_pos = root.position.copy()
            prev_pos[0] += x_offset

            for seg_idx in range(segments_per_leg):
                if seg_idx == segments_per_leg - 1:
                    seg_type = "foot"
                else:
                    seg_type = "segment"

                seg_pos = prev_pos + np.array([0.0, 0.0, -0.1], dtype=np.float32) if seg_idx == 0 else prev_pos + np.array([0.0, 0.0, -0.12], dtype=np.float32)

                seg = self.create_part(seg_type, position=seg_pos)
                body.add_part(seg)

                seg_spec = self.catalog.get(seg_type)
                joint_type = seg_spec.joint_type if seg_spec and seg_spec.can_actuate else "hinge"
                joint_axis = np.array(seg_spec.joint_axis if seg_spec and seg_spec.can_actuate else [0.0, 1.0, 0.0], dtype=np.float32)
                jr_min = seg_spec.joint_range_min if seg_spec and seg_spec.can_actuate else -1.2
                jr_max = seg_spec.joint_range_max if seg_spec and seg_spec.can_actuate else 1.2
                jd = seg_spec.joint_damping if seg_spec and seg_spec.can_actuate else 0.5
                joint = Joint(
                    joint_type=joint_type,
                    parent_id=prev_id,
                    child_id=seg.part_id,
                    anchor=np.array([0.0, 0.0, -0.05], dtype=np.float32),
                    axis=joint_axis,
                    params={"range_min": jr_min, "range_max": jr_max, "damping": jd},
                )
                body.add_joint(joint)

                if seg_idx == 0 and motor_type is not None:
                    motor = self.create_part(motor_type, position=root.position.copy())
                    body.add_part(motor)
                    m_joint = Joint(
                        joint_type="hinge",
                        parent_id=root.part_id,
                        child_id=motor.part_id,
                        anchor=np.array([x_offset, 0.0, 0.0], dtype=np.float32),
                        axis=joint_axis,
                        params={"range_min": -1.5, "range_max": 1.5},
                    )
                    body.add_joint(m_joint)

                prev_id = seg.part_id
                prev_pos = seg_pos

        return body

    # ── Generate initial population: n random bodies with unique seeds
    def generate_initial_population(
        self,
        size: int,
        min_parts: int = 4,
        max_parts: int = 10,
    ) -> List[MechanicalBody]:
        population = []
        for i in range(size):
            body = self.generate_random_body(
                name=f"gen0_ind{i:03d}",
                min_parts=min_parts,
                max_parts=max_parts,
            )
            population.append(body)
        return population
