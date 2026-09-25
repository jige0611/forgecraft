import random
from typing import Dict, Optional
import logging

import numpy as np

from forgecraft.config import EvolutionConfig, PartSpec
from forgecraft.core.catalog import DEFAULT_CATALOG
from forgecraft.core.morphology import Joint, MechanicalBody, Part


# ══════════════════════════════════════════════════════════
# 遗传算子 (Genetic Operators)
#
# 变异算子 (Mutation):
#   - mutate_params:      高斯噪声缩放零件/joint参数
#   - add_random_part:    随机零件 + 随机父节点的叶插入
#   - delete_random_part: 随机删除非根叶节点
#   - swap_part:          同类型零件互换
#   - apply_mutations:    批量变异 (多轮, 概率采样)
#
# 交叉算子 (Crossover):
#   - crossover:          子树交换: 从两个父体中各取一个子树,
#                          在 root 处组装成新子代
#
# 变异策略:
#   高斯噪声: param' = param + ε·σ·max(|param|, 0.001)
#   裁剪:      param' = clamp(param', [min, max])
# ══════════════════════════════════════════════════════════

__all__ = [
    "mutate_params",
    "add_part_mutation",
    "delete_part_mutation",
    "topological_mutation",
    "crossover",
    "apply_mutations",
]

_logger = logging.getLogger(__name__)



def mutate_params(
    body: MechanicalBody,
    scale: float = 0.1,
    catalog: Optional[Dict[str, PartSpec]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> MechanicalBody:
    if rng is None:
        rng = np.random.RandomState()
    catalog = catalog or dict(DEFAULT_CATALOG)
    mutated = body.clone()

    for part in mutated.parts():
        spec = catalog.get(part.part_type)
        if spec is None:
            continue

        param_keys = []
        if spec.param_ranges:
            param_keys = list(spec.param_ranges.keys())
        else:
            param_keys = ["length", "radius", "thickness", "mass"]

        for key in param_keys:
            if key in part.params:
                current = part.params[key]
                noise = rng.randn() * scale * max(abs(current), 0.001)
                new_val = spec.clamp_param(key, current + noise) if spec.param_ranges else max(0.0001, current + noise)
                part.params[key] = new_val

        if spec.can_actuate:
            for key in ["max_torque", "max_velocity"]:
                if key in part.params:
                    current = part.params[key]
                    noise = rng.randn() * scale * max(abs(current), 0.001)
                    if key in spec.param_ranges:
                        part.params[key] = spec.clamp_param(key, current + noise)
                    else:
                        part.params[key] = max(0.001, current + noise)

    for joint in mutated.joints():
        if joint.joint_type == "hinge":
            for key in ["range_min", "range_max"]:
                if key in joint.params:
                    current = joint.params[key]
                    noise = rng.randn() * scale * max(abs(current), 0.01)
                    joint.params[key] = current + noise
            if "damping" in joint.params:
                current = joint.params["damping"]
                noise = rng.randn() * scale * max(abs(current), 0.01)
                joint.params["damping"] = max(0.005, current + noise)

    mutated.invalidate_feature_cache()
    return mutated


def add_part_mutation(
    body: MechanicalBody,
    catalog: Optional[Dict[str, PartSpec]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> Optional[MechanicalBody]:
    if rng is None:
        rng = np.random.RandomState()
    catalog = catalog or dict(DEFAULT_CATALOG)
    mutated = body.clone()

    parent_id = random.choice(list(mutated.graph.nodes))
    parent_part = mutated.get_part(parent_id)

    available_types = list(catalog.keys())

    new_type = random.choice(available_types)
    spec = catalog[new_type]

    max_children = 4
    if len(mutated.children_of(parent_id)) >= max_children:
        candidates = [n for n in mutated.graph.nodes if len(mutated.children_of(n)) < max_children]
        if candidates:
            parent_id = random.choice(candidates)
            parent_part = mutated.get_part(parent_id)

    direction = np.array([rng.randn(), abs(rng.randn()), rng.randn()], dtype=np.float32)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    parent_len = parent_part.params.get("length", 0.1)
    offset = parent_len * 0.6 + rng.uniform(0.01, 0.04)
    new_pos = parent_part.position + direction * offset

    if spec.param_ranges:
        params = {}
        for key, (lo, hi) in spec.param_ranges.items():
            if not key.startswith("joint_"):
                params[key] = rng.uniform(lo, hi)
    else:
        params = {
            "length": rng.uniform(*spec.size_range),
            "radius": rng.uniform(spec.size_range[0] * 0.3, spec.size_range[1] * 0.5),
            "thickness": rng.uniform(spec.size_range[0] * 0.1, spec.size_range[1] * 0.3),
        }

    if spec.mass_range and len(spec.mass_range) == 2:
        params["mass"] = rng.uniform(*spec.mass_range)
    elif "mass" not in params:
        params["mass"] = spec.mass * rng.uniform(0.8, 1.2)

    if spec.has_touch:
        params["touch_sensor"] = 1.0
    if spec.has_imu:
        params["imu_sensor"] = 1.0

    if spec.can_actuate:
        if "max_torque" not in params:
            params["max_torque"] = spec.max_torque * rng.uniform(0.5, 1.5)
        if "max_velocity" not in params:
            params["max_velocity"] = rng.uniform(5.0, 15.0)

    new_part = Part(part_type=new_type, params=params, position=new_pos)
    mutated.add_part(new_part)

    joint_type = spec.joint_type if spec.can_actuate else "fixed"
    joint_axis = np.array(spec.joint_axis, dtype=np.float32) if spec.can_actuate else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    joint_range_min = spec.joint_range_min if spec.can_actuate else 0.0
    joint_range_max = spec.joint_range_max if spec.can_actuate else 0.0
    joint_damping = spec.joint_damping if spec.can_actuate else 0.0

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
    mutated.add_joint(joint)

    mutated.invalidate_feature_cache()
    return mutated


def delete_part_mutation(
    body: MechanicalBody,
    rng: Optional[np.random.RandomState] = None,
) -> Optional[MechanicalBody]:
    if rng is None:
        rng = np.random.RandomState()

    leaves = body.leaf_parts()
    non_root_leaves = [n for n in leaves if n != body.root_id]
    if not non_root_leaves:
        return None

    to_delete = random.choice(non_root_leaves)
    mutated = body.clone()
    mutated.graph.remove_node(to_delete)
    mutated.sync_from_graph(mutated.graph)
    mutated.invalidate_feature_cache()
    return mutated


def topological_mutation(
    body: MechanicalBody,
    catalog: Optional[Dict[str, PartSpec]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> MechanicalBody:
    """随机选择一种拓扑变异: add_part (50%) 或 delete_part (50%)"""
    if rng is None:
        rng = np.random.RandomState()
    catalog = catalog or dict(DEFAULT_CATALOG)

    # 50/50 概率: 添加或删除零件
    if rng.random() < 0.5:
        return add_part_mutation(body, catalog, rng)
    else:
        result = delete_part_mutation(body, rng)
        # 删除后至少保留 4 个零件 (root + 至少 3 个子)
        if result is not None and result.num_parts() >= 4:
            return result
        return body.clone()


def crossover(
    parent_a: MechanicalBody,
    parent_b: MechanicalBody,
    rng: Optional[np.random.RandomState] = None,
) -> MechanicalBody:
    """子树交换交叉: 从 parent_b 随机取一个子树, 嫁接/堆叠到 parent_a 的随机节点上"""
    if rng is None:
        rng = np.random.RandomState()

    child = parent_a.clone()

    # 1. 从 parent_b 中随机选一个子树根 (非 root)
    non_root_b = [n for n in parent_b.graph.nodes if n != parent_b.root_id]
    if len(non_root_b) < 2:
        return child

    subroot = random.choice(non_root_b)
    # 2. BFS 收集子树所有后代
    descendants = set()
    stack = [subroot]
    while stack:
        node = stack.pop()
        if node in descendants:
            continue
        descendants.add(node)
        for child_node in parent_b.children_of(node):
            if child_node not in descendants:
                stack.append(child_node)

    if len(descendants) < 1:
        return child

    # 3. 在 parent_a 中随机选一个挂载点
    target_parent = random.choice(list(child.graph.nodes))
    target_part = child.get_part(target_parent)

    # 4. 计算子树根到目标点的偏移量
    subroot_part = parent_b.get_part(subroot)
    root_offset = subroot_part.position - target_part.position

    # 5. 拷贝子树节点到 child (按深度排序, 父节点先于子节点)
    id_mapping = {}
    for old_id in sorted(descendants, key=lambda n: parent_b.get_level(n)):
        old_part = parent_b.get_part(old_id)
        new_part = old_part.clone()
        new_part.part_id = old_part.part_id + "_x"
        # 加小随机偏移避免精确重叠
        new_part.position = old_part.position - root_offset + np.array([rng.uniform(-0.02, 0.02), 0.0, rng.uniform(-0.02, 0.02)], dtype=np.float32)
        id_mapping[old_id] = new_part.part_id
        child.add_part(new_part)

    for u, v in parent_b.graph.edges:
        if u in descendants and v in descendants:
            joint = parent_b.get_joint(u, v)
            new_joint = joint.clone()
            new_joint.parent_id = id_mapping[u]
            new_joint.child_id = id_mapping[v]
            child.add_joint(new_joint)

    subroot_new_id = id_mapping[subroot]
    joint_type = "hinge"
    joint = Joint(
        joint_type=joint_type,
        parent_id=target_parent,
        child_id=subroot_new_id,
        anchor=np.array([rng.uniform(-0.03, 0.03), 0.03, rng.uniform(-0.03, 0.03)], dtype=np.float32),
        axis=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        params={"range_min": -1.5, "range_max": 1.5, "damping": 0.5},
    )
    child.add_joint(joint)

    return child


def apply_mutations(
    body: MechanicalBody,
    config: EvolutionConfig,
    catalog: Optional[Dict[str, PartSpec]] = None,
    rng: Optional[np.random.RandomState] = None,
    param_scale_override: Optional[float] = None,
    topo_prob_override: Optional[float] = None,
) -> MechanicalBody:
    if rng is None:
        rng = np.random.RandomState()

    result = body.clone()

    try:
        param_prob = config.param_mutation_prob
        param_scale = config.param_mutation_scale
        topo_prob = config.topo_mutation_prob

        if param_scale_override is not None:
            param_scale = param_scale_override
            param_prob = min(1.0, param_prob * 1.5)
        if topo_prob_override is not None:
            topo_prob = topo_prob_override

        if rng.random() < param_prob:
            result = mutate_params(result, param_scale, catalog, rng)

        if rng.random() < topo_prob:
            topo_result = topological_mutation(result, catalog, rng)
            if topo_result is not None:
                result = topo_result
    except Exception:
        # 变异失败时回退到原始克隆体 (防御: 非法参数 / 空 catalog / 0 零件)
        _logger.warning(f"apply_mutations failed, returning clone")
        result = body.clone()

    return result
