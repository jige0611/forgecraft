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


def _ramp(x: float, lo: float, hi: float) -> float:
    """把 x 线性映射到 [0, 1]，区间外截断。

    可制造性评分必须对形态参数连续可导，否则在多目标搜索里只能充当常数偏移，
    无法产生选择压力。所有分段评分都以此为基础构造。
    """
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _check_bed_size(body: MechanicalBody) -> Tuple[float, Dict]:
    """打印床尺寸检查 — 连续评分。

    设计阈值（零件恰好铺满打印床）落在 0.5 分：留出余量向 1.0 上升，
    超出则向 0.0 下降。原实现采用二值判定加封顶扣分，绝大多数机体恒得 1.0，
    评分因而没有梯度。
    """
    violations = 0
    usage = 0.0
    worst = None
    worst_violations: List[Dict] = []

    bed = np.array([_FDM_BED_X, _FDM_BED_Y, _FDM_BED_Z])
    for part in body.parts():
        pos = part.position
        hx, hy, hz = _estimate_part_bounds(pos, part.params, part.part_type)
        extents = np.array([hx * 2, hy * 2, hz * 2])
        ratio = float(np.max(extents / bed))
        if ratio > usage:
            usage = ratio
            worst = {"part_id": part.part_id, "part_type": part.part_type,
                     "size": [float(v) for v in extents], "usage": round(ratio, 4)}
        if ratio > 1.0:
            violations += 1
            if len(worst_violations) < 3:
                worst_violations.append({"part_id": part.part_id,
                                         "part_type": part.part_type,
                                         "size": [float(v) for v in extents]})

    if usage <= 1.0:
        score = 1.0 - 0.5 * _ramp(usage, 0.5, 1.0)
    else:
        score = 0.5 * (1.0 - _ramp(usage, 1.0, 1.5))

    detail = {
        "score": round(score, 4),
        "bed_usage": round(usage, 4),
        "violations": violations,
        "bed_limit": [_FDM_BED_X, _FDM_BED_Y, _FDM_BED_Z],
        "worst_part": worst,
        "largest_violations": worst_violations,
    }
    return score, detail


def _check_wall_thickness(body: MechanicalBody) -> Tuple[float, Dict]:
    """壁厚检查 — 连续评分，按最薄特征相对可打印下限的比值计分。

    比值为 1（恰好等于最小壁厚）落在 0.5 分，两倍以上视为完全安全。
    """
    violations = []
    min_ratio = float("inf")
    worst = None

    for part in body.parts():
        params = part.params
        features = []
        if "thickness" in params:
            features.append(("thickness", params["thickness"]))
        if "radius" in params and part.part_type in ("rod", "hinge_motor", "hinge_joint", "motor"):
            features.append(("radius", params["radius"]))
        if "height" in params:
            features.append(("height", params["height"]))
        if "width" in params:
            features.append(("width", params["width"]))

        for key, val in features:
            ratio = val / _MIN_WALL_THICKNESS
            if ratio < 1.0:
                violations.append((part.part_id, part.part_type, key, val))
            if ratio < min_ratio:
                min_ratio = ratio
                worst = {"part_id": part.part_id, "part_type": part.part_type,
                         "feature": key, "value": val, "ratio": round(ratio, 4)}

    if math.isinf(min_ratio):
        score = 1.0
    elif min_ratio >= 1.0:
        score = 0.5 + 0.5 * _ramp(min_ratio, 1.0, 2.0)
    else:
        score = 0.5 * _ramp(min_ratio, 0.5, 1.0)

    detail = {
        "score": round(score, 4),
        "min_ratio": None if math.isinf(min_ratio) else round(min_ratio, 4),
        "violations": violations,
        "min_wall": _MIN_WALL_THICKNESS,
        "worst": worst,
    }
    return score, detail


def _check_assembly_complexity(body: MechanicalBody) -> Tuple[float, Dict]:
    """装配复杂度 — 连续评分，零件数与装配深度共同决定。

    原实现在零件数未超阈值时直接返回 1.0，且惩罚封顶 0.15，
    因此在 4-25 个零件的实际分布上几乎恒为 1.0。
    """
    n_parts = body.num_parts()
    n_joints = body.num_joints()
    n_motors = len(body.actuated_joints())
    max_depth = body.max_depth()
    unique_types = len(set(p.part_type for p in body.parts()))
    threshold = _MAX_ASSEMBLY_PARTS_BEFORE_PENALTY

    if n_parts <= threshold:
        parts_score = 1.0 - 0.5 * _ramp(n_parts, max(1.0, threshold * 0.5), threshold)
    else:
        parts_score = 0.5 * (1.0 - _ramp(n_parts, threshold, threshold + 25))

    depth_score = 1.0 - 0.3 * _ramp(max_depth, 6, 12)
    score = parts_score * depth_score

    detail = {
        "score": round(score, 4),
        "n_parts": n_parts,
        "n_joints": n_joints,
        "n_motors": n_motors,
        "max_depth": max_depth,
        "unique_types": unique_types,
        "parts_score": round(parts_score, 4),
        "depth_score": round(depth_score, 4),
    }
    return score, detail


def _check_torque_budget(body: MechanicalBody) -> Tuple[float, Dict]:
    """扭矩预算 — 连续评分。

    原实现在总扭矩为 0 时直接返回 1.0，等于给"没有执行器"的机体满分；
    同时铰链存在却无扭矩应当判为不可驱动。现在把两种情况分开处理，
    并对没有铰链的机体标记为不适用，由上层重新归一化权重。
    """
    total_mass = 0.0
    total_torque = 0.0

    for part in body.parts():
        total_mass += part.params.get("mass", 0.1)

    hinge_joints = [j for j in body.joints() if j.joint_type == "hinge"]
    for joint in hinge_joints:
        child = body.get_part(joint.child_id)
        total_torque += child.params.get("max_torque", 0.0)

    gravity = 9.81
    required_torque = total_mass * gravity * 0.05

    if not hinge_joints:
        return 1.0, {"score": 1.0, "applicable": False, "n_hinges": 0,
                     "total_mass": total_mass, "required": required_torque,
                     "note": "no hinge joints, torque budget not applicable"}

    if total_torque <= 0.0:
        return 0.0, {"score": 0.0, "applicable": True, "n_hinges": len(hinge_joints),
                     "total_mass": total_mass, "total_torque": 0.0,
                     "required": required_torque, "ratio": 0.0,
                     "note": "hinge joints present but no actuation torque"}

    ratio = total_torque / max(required_torque, 1e-6)
    if ratio <= _MIN_MOTOR_TORQUE_RATIO:
        score = 0.5 * _ramp(ratio, 0.05, _MIN_MOTOR_TORQUE_RATIO)
    else:
        score = 0.5 + 0.5 * _ramp(ratio, _MIN_MOTOR_TORQUE_RATIO, _MIN_MOTOR_TORQUE_RATIO * 2)

    detail = {
        "score": round(score, 4),
        "applicable": True,
        "n_hinges": len(hinge_joints),
        "total_mass": total_mass,
        "total_torque": total_torque,
        "required": required_torque,
        "ratio": round(ratio, 4),
        "min_ratio": _MIN_MOTOR_TORQUE_RATIO,
    }
    return score, detail


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
        score = 1.0 - 0.5 * _ramp(max_chain, _MAX_CANTILEVER_LENGTH / 3.0, _MAX_CANTILEVER_LENGTH)
    else:
        score = 0.5 * (1.0 - _ramp(max_chain, _MAX_CANTILEVER_LENGTH, _MAX_CANTILEVER_LENGTH * 2))

    detail = {
        "score": round(score, 4),
        "max_cantilever": max_chain,
        "limit": _MAX_CANTILEVER_LENGTH,
        "worst_chain_nodes": worst_chain_nodes,
    }
    return score, detail


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
# 权重按各检查在实际设计空间中的分辨力分配：装配复杂度与扭矩预算波动最大，
# 打印床与壁厚在多数形态上不触发，因此权重较低。
_MFG_WEIGHTS = {
    "bed_size": 0.10,
    "wall_thickness": 0.15,
    "assembly": 0.35,
    "torque_budget": 0.25,
    "cantilever": 0.15,
}


def _run_mfg_checks(body: MechanicalBody) -> Dict[str, Tuple[float, Dict]]:
    return {
        "bed_size": _check_bed_size(body),
        "wall_thickness": _check_wall_thickness(body),
        "assembly": _check_assembly_complexity(body),
        "torque_budget": _check_torque_budget(body),
        "cantilever": _check_cantilever(body),
    }


def _weighted_manufacturability(checks: Dict[str, Tuple[float, Dict]]) -> float:
    """按权重合成。不适用的检查（如无铰链机体的扭矩预算）剔除并重新归一化，
    避免"不适用即满分"把分数整体抬高。"""
    weighted = 0.0
    total_weight = 0.0
    for name, (score, detail) in checks.items():
        if detail.get("applicable", True) is False:
            continue
        weight = _MFG_WEIGHTS.get(name, 0.2)
        weighted += weight * score
        total_weight += weight
    if total_weight <= 0.0:
        return 1.0
    return round(max(0.0, min(1.0, weighted / total_weight)), 4)


def compute_manufacturability(body: MechanicalBody) -> float:
    """可制造性综合评分 (0-1)，各项均为连续评分而非阈值判定。"""
    return _weighted_manufacturability(_run_mfg_checks(body))


def compute_manufacturability_detail(body: MechanicalBody) -> Dict:
    checks = _run_mfg_checks(body)
    result = {"overall": _weighted_manufacturability(checks), "weights": dict(_MFG_WEIGHTS)}
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
_FALLBACK_WEIGHT_WARNED: set = set()


def _fallback_fitness_weight(comp_name: str) -> float:
    """legacy 兜底：该分量在 fitness_weights 与 reward_components 中均未声明。

    原实现静默套用硬编码值，使任务 yaml 里 fitness_formula 写下的权重形同虚设；
    这里保留兜底行为以免破坏既有实验，但至少留下告警。
    """
    if comp_name not in _FALLBACK_WEIGHT_WARNED:
        _FALLBACK_WEIGHT_WARNED.add(comp_name)
        _logger.warning(
            "fitness component %r has no declared weight in reward.fitness_weights; "
            "using a hardcoded default. Declare it explicitly to avoid silent drift.",
            comp_name,
        )
    return {
        "speed": 1.0,
        "displacement": 1.0,
        "upright": 0.1,
        "energy": -0.1,
        "manufacturability": 0.05,
        "height": 1.0,
        "survival": 0.2,
    }.get(comp_name, 1.0)


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
            declared = task_config.get_fitness_weight(comp_name)
            if declared is not None:
                w = declared
            else:
                w = task_config.get_weight(comp_name)
                if w == 0.0:
                    w = _fallback_fitness_weight(comp_name)
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
