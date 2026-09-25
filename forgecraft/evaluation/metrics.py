"""
可制造性评估指标

基于形态图 + part mesh 计算:
  - 打印床尺寸检查 (_check_bed_size)
  - 公差分析 (joint clearance, overhang)
  - 整体可制造性评分 (0-1)

所有指标为字典形式, 供 fitness 函数加权使用。
"""

import math
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np

from forgecraft.core.morphology import MechanicalBody

__all__ = [
    "compute_body_metrics",
    "compute_manufacturability",
    "compute_manufacturability_detail",
    "aggregate_fitness",
    "aggregate_fitness_generic",
    "_FITNESS_SENSORS",
]

_logger = logging.getLogger(__name__)


_FITNESS_SENSORS = {
    "speed": lambda info: info.get("speed", 0.0),
    "energy": lambda info: info.get("energy", 0.0),
    "displacement": lambda info: info.get("displacement", 0.0),
    "upright": lambda info: info.get("upright", 0.0),
    "height": lambda info: info.get("height", 0.0),
    "velocity": lambda info: info.get("velocity", 0.0),
    "joint_effort": lambda info: info.get("joint_effort", 0.0),
    "manufacturability": lambda info: info.get("manufacturability", 0.0),
    "survival": lambda info: info.get("survival_ratio", 0.0),
    "fall_penalty": lambda info: -1.0 if info.get("fell", False) else 0.0,
    # 反常识运动指标 (与 env.py _REWARD_SENSORS 对应的适应度键)
    "novelty_score": lambda info: info.get("novelty_score", 0.0),
    "asymmetry_index": lambda info: info.get("asymmetry_index", 0.0),
    "irregularity_metric": lambda info: info.get("irregularity_metric", 0.0),
    "bilateral_symmetry": lambda info: info.get("bilateral_symmetry", 0.0),
    "shape_commonality_score": lambda info: info.get("shape_commonality_score", 0.0),
    "efficiency_ratio": lambda info: info.get("efficiency_ratio", 0.0),
}


_FDM_BED_X = 0.22   # FDM 打印床 X 尺寸 (22cm)
_FDM_BED_Y = 0.22   # FDM 打印床 Y 尺寸 (22cm)
_FDM_BED_Z = 0.25   # FDM 打印床 Z 高度 (25cm)

_MIN_WALL_THICKNESS = 0.0008        # 最小壁厚 0.8mm (FDM 喷嘴直径 0.4mm → 2 层)
_MIN_MOTOR_TORQUE_RATIO = 0.3       # 电机扭矩安全裕度
_MAX_CANTILEVER_LENGTH = 0.6        # 最大悬臂长度 (m)
_MAX_ASSEMBLY_PARTS_BEFORE_PENALTY = 15  # 超此零件数开始惩罚装配复杂度


def _estimate_part_bounds(position: np.ndarray, params: dict, part_type: str) -> Tuple[float, float, float]:
    half_x = 0.025
    half_y = 0.015
    half_z = 0.025
    length = params.get("length", 0.1)
    radius = params.get("radius", 0.015)
    width = params.get("width", 0.02)
    height = params.get("height", 0.02)
    thickness = params.get("thickness", 0.02)

    if part_type in ("rod", "segment", "limb_segment", "upper_leg", "lower_leg"):
        half_x = radius
        half_y = length / 2.0
        half_z = radius
    elif part_type in ("box_body", "base", "body", "core", "main_body", "chassis", "box"):
        half_x = length / 2.0
        half_y = width / 2.0
        half_z = height / 2.0
    elif part_type in ("hinge_motor", "motor", "hinge_joint"):
        half_x = radius
        half_y = length / 2.0
        half_z = radius
    elif part_type in ("foot_contact", "foot", "foot_pad", "foot_spike"):
        half_x = radius
        half_y = radius
        half_z = radius

    return float(half_x), float(half_y), float(half_z)


def _check_bed_size(body: MechanicalBody) -> Tuple[float, Dict]:
    """打印床尺寸检查 — 每个零件 extent 是否在 FDM 床范围内"""
    violations = 0
    part_details = []

    for part in body.parts():
        pos = part.position
        hx, hy, hz = _estimate_part_bounds(pos, part.params, part.part_type)
        extents = np.array([hx * 2, hy * 2, hz * 2])
        ok_x = extents[0] <= _FDM_BED_X
        ok_y = extents[1] <= _FDM_BED_Y
        ok_z = extents[2] <= _FDM_BED_Z

        if not (ok_x and ok_y and ok_z):
            violations += 1
            part_details.append({
                "part_id": part.part_id,
                "part_type": part.part_type,
                "size": [float(v) for v in extents],
            })

    detail = {
        "violations": violations,
        "bed_limit": [_FDM_BED_X, _FDM_BED_Y, _FDM_BED_Z],
        "largest_violations": part_details[:3],
    }
    if violations == 0:
        return 1.0, detail
    penalty = min(0.25, violations * 0.05)
    return 1.0 - penalty, detail


def _check_wall_thickness(body: MechanicalBody) -> Tuple[float, Dict]:
    violations = []
    for part in body.parts():
        params = part.params
        checks = []
        if "thickness" in params:
            checks.append(("thickness", params["thickness"]))
        if "radius" in params and part.part_type in ("rod", "hinge_motor", "hinge_joint", "motor"):
            checks.append(("radius", params["radius"]))
        if "height" in params:
            checks.append(("height", params["height"]))
        if "width" in params:
            checks.append(("width", params["width"]))

        for key, val in checks:
            if val < _MIN_WALL_THICKNESS:
                violations.append((part.part_id, part.part_type, key, val))

    if not violations:
        return 1.0, {"violations": []}
    penalty = min(0.2, len(violations) * 0.05)
    return 1.0 - penalty, {"violations": violations, "min_wall": _MIN_WALL_THICKNESS}


def _check_assembly_complexity(body: MechanicalBody) -> Tuple[float, Dict]:
    n_parts = body.num_parts()
    n_joints = body.num_joints()
    n_motors = len(body.actuated_joints())
    max_depth = body.max_depth()
    unique_types = len(set(p.part_type for p in body.parts()))

    if n_parts <= _MAX_ASSEMBLY_PARTS_BEFORE_PENALTY:
        return 1.0, {"n_parts": n_parts, "n_joints": n_joints, "n_motors": n_motors,
                       "max_depth": max_depth, "unique_types": unique_types}

    excess = n_parts - _MAX_ASSEMBLY_PARTS_BEFORE_PENALTY
    penalty_parts = min(0.10, excess * 0.005)
    penalty_depth = max(0, max_depth - 6) * 0.01
    total_penalty = min(0.15, penalty_parts + penalty_depth)
    return 1.0 - total_penalty, {"n_parts": n_parts, "n_joints": n_joints,
                                   "n_motors": n_motors, "max_depth": max_depth,
                                   "unique_types": unique_types, "excess": excess}


def _check_torque_budget(body: MechanicalBody) -> Tuple[float, Dict]:
    total_mass = 0.0
    total_torque = 0.0

    for part in body.parts():
        total_mass += part.params.get("mass", 0.1)

    for joint in body.joints():
        if joint.joint_type == "hinge":
            child = body.get_part(joint.child_id)
            torque = child.params.get("max_torque", 0.0)
            total_torque += torque

    gravity = 9.81
    required_torque = total_mass * gravity * 0.05

    if total_torque == 0.0:
        return 1.0, {"total_mass": total_mass, "total_torque": 0.0, "required": required_torque,
                       "warning": "no actuators"}

    ratio = total_torque / max(required_torque, 1e-6)
    if ratio >= _MIN_MOTOR_TORQUE_RATIO:
        return 1.0, {"total_mass": total_mass, "total_torque": total_torque,
                       "required": required_torque, "ratio": ratio}

    shortage = (_MIN_MOTOR_TORQUE_RATIO - ratio) / _MIN_MOTOR_TORQUE_RATIO
    penalty = min(0.2, shortage * 0.2)
    return 1.0 - penalty, {"total_mass": total_mass, "total_torque": total_torque,
                             "required": required_torque, "ratio": ratio,
                             "min_ratio": _MIN_MOTOR_TORQUE_RATIO}


def _check_cantilever(body: MechanicalBody) -> Tuple[float, Dict]:
    max_chain = 0
    worst_chain_nodes = []
    visited = set()  # 防止环状图导致无限递归

    def _dfs(node_id, depth, length_acc):
        nonlocal max_chain, worst_chain_nodes
        if node_id in visited:
            return
        visited.add(node_id)
        children = body.children_of(node_id)
        if not children:
            if length_acc > max_chain:
                max_chain = length_acc
                worst_chain_nodes = [node_id]
            return
        for child_id in children:
            child = body.get_part(child_id)
            child_len = child.params.get("length", 0.1)
            is_actuated = child.params.get("actuated", 0.0) > 0.5
            new_depth = depth + 1 if is_actuated else depth
            new_len = 0.0 if is_actuated else length_acc + child_len
            _dfs(child_id, new_depth, new_len)

    if body.root_id:
        _dfs(body.root_id, 1, 0.0)

    if max_chain <= _MAX_CANTILEVER_LENGTH:
        return 1.0, {"max_cantilever": max_chain, "limit": _MAX_CANTILEVER_LENGTH}

    excess = max_chain - _MAX_CANTILEVER_LENGTH
    penalty = min(0.15, excess * 0.15)
    return 1.0 - penalty, {"max_cantilever": max_chain, "limit": _MAX_CANTILEVER_LENGTH, "excess": excess}


def compute_body_metrics(body: MechanicalBody, env_info: Dict) -> Dict[str, float]:
    metrics = {
        "fitness": body.fitness,
        "num_parts": float(body.num_parts()),
        "num_joints": float(body.num_joints()),
        "max_depth": float(body.max_depth()),
        "num_motors": float(len(body.actuated_joints())),
    }
    for key, value in env_info.items():
        if isinstance(value, (int, float, np.floating)):
            metrics[key] = float(value)
    return metrics


# ── 可制造性综合评分 ── bed_size/wall_thickness/assembly/torque/cantilever 加权
def compute_manufacturability(body: MechanicalBody) -> float:
    checks = {
        "bed_size": _check_bed_size(body),
        "wall_thickness": _check_wall_thickness(body),
        "assembly": _check_assembly_complexity(body),
        "torque_budget": _check_torque_budget(body),
        "cantilever": _check_cantilever(body),
    }

    weighted = 0.0
    weights = {"bed_size": 0.25, "wall_thickness": 0.25, "assembly": 0.15,
               "torque_budget": 0.20, "cantilever": 0.15}
    for name, (score, _detail) in checks.items():
        weighted += weights.get(name, 0.2) * score

    return round(max(0.0, min(1.0, weighted)), 4)


def compute_manufacturability_detail(body: MechanicalBody) -> Dict:
    checks = {
        "bed_size": _check_bed_size(body),
        "wall_thickness": _check_wall_thickness(body),
        "assembly": _check_assembly_complexity(body),
        "torque_budget": _check_torque_budget(body),
        "cantilever": _check_cantilever(body),
    }
    result = {"overall": compute_manufacturability(body)}
    for name, (score, detail) in checks.items():
        result[name] = {"score": round(score, 4), "detail": detail}
    return result


# ── 经典加权适应度 ── speed*1.0 + upright*0.1 - energy*0.1 + mfg*0.05
def aggregate_fitness(
    body: MechanicalBody,
    speed: float,
    energy: float,
    upright: float,
    manufacturability: float,
    speed_weight: float = 1.0,
    energy_weight: float = 0.01,
    upright_weight: float = 0.1,
    manufacturability_weight: float = 0.05,
) -> float:
    fitness = (
        speed_weight * speed
        + upright_weight * upright
        - energy_weight * energy * 10.0
        + manufacturability_weight * manufacturability
    )
    body.fitness = fitness
    body.fitness_components = {
        "speed": speed,
        "energy": energy,
        "upright": upright,
        "manufacturability": manufacturability,
    }
    return fitness


# ── 通用适应度 ── 按 TaskConfig.fitness_components 动态选择加权公式
def aggregate_fitness_generic(
    body: MechanicalBody,
    episode_info: Dict[str, float],
    task_config=None,
) -> float:
    if task_config is not None and task_config.fitness_components:
        mfg = compute_manufacturability(body)
        episode_info["manufacturability"] = mfg
        survival_ratio = episode_info.get("survival_ratio", 0.5)

        total = 0.0
        for comp_name in task_config.fitness_components:
            w = task_config.get_weight(comp_name)
            if w == 0.0:
                if comp_name == "speed":
                    w = 1.0
                elif comp_name == "displacement":
                    w = 1.0
                elif comp_name == "upright":
                    w = 0.1
                elif comp_name == "energy":
                    w = -0.1
                elif comp_name == "manufacturability":
                    w = 0.05
                elif comp_name == "height":
                    w = 1.0
                elif comp_name == "survival":
                    w = 0.2
                else:
                    w = 1.0
            if comp_name in _FITNESS_SENSORS:
                val = _FITNESS_SENSORS[comp_name](episode_info)
            elif comp_name == "manufacturability":
                val = mfg
            else:
                val = episode_info.get(comp_name, 0.0)
            total += w * val
        if "survival" not in task_config.fitness_components:
            total += 0.2 * survival_ratio
        fitness = total
    else:
        mfg = compute_manufacturability(body)
        speed = episode_info.get("speed", 0.0)
        energy = episode_info.get("energy", 0.0)
        upright = episode_info.get("upright", 0.0)
        survival_ratio = episode_info.get("survival_ratio", 0.5)
        fitness = speed * 1.0 + upright * 0.1 - energy * 0.01 * 10.0 + mfg * 0.05 + survival_ratio * 0.2

    body.fitness = fitness
    body.fitness_components = {
        k: float(v) for k, v in episode_info.items()
        if isinstance(v, (int, float, np.floating))
    }
    body.fitness_components["manufacturability"] = mfg

    return fitness
