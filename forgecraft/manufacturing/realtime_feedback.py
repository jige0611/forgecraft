"""实时制造性反馈模块

进化过程中同步计算制造性指标，指导选择/变异方向。
所有检查 ≤ 1ms (30 零件形态)。

检查维度:
  1. 壁厚检查 — 防止薄壁打印失败
  2. 悬垂检查 — 零件底面 ≤ 45° 可自支撑
  3. 干涉检查 — AABB + 近似体积重叠检测
  4. 连接强度 — joint 区域体积比评估
  5. 材料效率 — 体积利用率

集成方式:
  from forgecraft.manufacturing.realtime_feedback import RealtimeManufacturability
  mfg = RealtimeManufacturability(catalog)
  score, details = mfg.evaluate(body)
  body.fitness_components['manufacturability'] = score
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from forgecraft.core.morphology import MechanicalBody, Part
from forgecraft.config import PartSpec

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "ManufacturingCheck",
    "WallThicknessCheck",
    "OverhangCheck",
    "InterferenceCheck",
    "JointStressCheck",
    "MaterialEfficiencyCheck",
    "RealtimeManufacturability",
]


# ── 基础检查接口 ──

class ManufacturingCheck:
    """制造性检查基类"""
    name: str = "base"
    weight: float = 1.0  # 在总分中的权重

    def evaluate(
        self, body: MechanicalBody, catalog: Dict[str, PartSpec]
    ) -> Tuple[float, str]:
        """评估制造性

        Returns
        -------
        score : float
            0.0 (不可制造) ~ 1.0 (完美)
        detail : str
            人类可读的评估详情。
        """
        raise NotImplementedError


# ── 具体检查 ──

class WallThicknessCheck(ManufacturingCheck):
    """壁厚检查

    规则: 所有零件壁厚 ≥ spec.min_wall_thickness (默认 0.8mm)
    """

    name = "wall_thickness"
    weight = 1.0

    def evaluate(
        self, body: MechanicalBody, catalog: Dict[str, PartSpec]
    ) -> Tuple[float, str]:
        parts = list(body.parts())
        if not parts:
            return 1.0, "no parts"

        n_thin = 0
        for part in parts:
            thickness = part.params.get("thickness", 0.02)
            spec = catalog.get(part.part_type)
            min_thickness = getattr(spec, "min_wall_thickness", 0.0008) if spec else 0.0008

            if thickness < min_thickness:
                n_thin += 1

        total = len(parts)
        score = 1.0 - (n_thin / total) * 0.5
        return max(0.0, min(1.0, score)), f"{n_thin}/{total} thin walls"


class OverhangCheck(ManufacturingCheck):
    """悬垂检查

    规则: 零件底面悬垂角度 ≤ 45° 才能无支撑打印。
    简化计算: 检测没有下方支撑的零件比例。
    """

    name = "overhang"
    weight = 0.9

    def evaluate(
        self, body: MechanicalBody, catalog: Dict[str, PartSpec]
    ) -> Tuple[float, str]:
        parts = list(body.parts())
        if not parts:
            return 1.0, "no parts"

        # 获取每个零件的质心高度
        centers = np.array([
            [p.position[0], p.position[1], p.position[2]]
            if hasattr(p, 'position') and p.position is not None
            else [0.0, 0.0, 0.0]
            for p in parts
        ])

        n_overhang = 0
        for i, part in enumerate(parts):
            z = centers[i, 2]
            # 检查是否有下方支撑 (高度差 < 0.05 且水平距离 < 0.1)
            supported = False
            for j, other in enumerate(parts):
                if i == j:
                    continue
                dz = z - centers[j, 2]
                dx = np.linalg.norm(centers[i, :2] - centers[j, :2])
                if 0 < dz < 0.05 and dx < 0.1:
                    supported = True
                    break

            # 底盘零件 (z < 0.05) 自动有支撑
            if z < 0.05:
                supported = True

            if not supported:
                n_overhang += 1

        total = len(parts)
        score = 1.0 - (n_overhang / total) * 0.3
        return max(0.0, min(1.0, score)), f"{n_overhang}/{total} overhangs"


class InterferenceCheck(ManufacturingCheck):
    """干涉检查

    规则: 相邻零件不能有体积重叠。
    简化: AABB (轴对齐包围盒) 重叠检测。
    """

    name = "interference"
    weight = 1.0

    def evaluate(
        self, body: MechanicalBody, catalog: Dict[str, PartSpec]
    ) -> Tuple[float, str]:
        parts = list(body.parts())
        n = len(parts)
        if n < 2:
            return 1.0, "single part"

        # 为每个零件估算 AABB
        aabbs = []
        for part in parts:
            pos = np.array(part.position) if hasattr(part, 'position') and part.position is not None else np.zeros(3)
            # 简化的半尺寸
            half = np.array([
                abs(part.params.get("length", 0.1)) / 2,
                abs(part.params.get("width", part.params.get("thickness", 0.02))) / 2,
                abs(part.params.get("height", part.params.get("thickness", 0.02))) / 2,
            ])
            aabbs.append((pos - half, pos + half))

        overlaps = 0
        for i in range(n):
            for j in range(i + 1, n):
                min_i, max_i = aabbs[i]
                min_j, max_j = aabbs[j]
                if np.all(max_i > min_j) and np.all(max_j > min_i):
                    overlaps += 1

        total_pairs = n * (n - 1) / 2
        if total_pairs == 0:
            return 1.0, "single part"

        score = 1.0 - (overlaps / total_pairs)
        return max(0.0, min(1.0, score)), f"{overlaps}/{int(total_pairs)} overlapping"


class JointStressCheck(ManufacturingCheck):
    """连接强度检查

    规则: joint 区域零件截面面积 ≥ 最小连接面积。
    简化: 检查连接零件的厚度/半径乘积。
    """

    name = "joint_stress"
    weight = 0.8

    def evaluate(
        self, body: MechanicalBody, catalog: Dict[str, PartSpec]
    ) -> Tuple[float, str]:
        joints = list(body.joints()) if hasattr(body, 'joints') else []
        if not joints:
            return 1.0, "no joints"

        n_weak = 0
        min_area = 1e-6  # 1 mm² = 1e-6 m²

        for joint in joints:
            parent_idx = joint.get("parent", -1)
            child_idx = joint.get("child", -1)
            parts = list(body.parts())

            if 0 <= parent_idx < len(parts):
                p = parts[parent_idx]
                area = p.params.get("thickness", 0.02) * p.params.get("radius", 0.03)
                if area < min_area:
                    n_weak += 1

        total = len(joints)
        if total == 0:
            return 1.0, "no joints"

        score = 1.0 - (n_weak / total) * 0.4
        return max(0.0, min(1.0, score)), f"{n_weak}/{total} weak joints"


class MaterialEfficiencyCheck(ManufacturingCheck):
    """材料效率检查

    规则: 零件体积利用率 ≥ 30% (避免空心/空隙过大)。
    """

    name = "material_efficiency"
    weight = 0.6

    def evaluate(
        self, body: MechanicalBody, catalog: Dict[str, PartSpec]
    ) -> Tuple[float, str]:
        parts = list(body.parts())
        if not parts:
            return 1.0, "no parts"

        n_inefficient = 0
        for part in parts:
            # 估算体积与包围盒体积比
            length = abs(part.params.get("length", 0.1))
            radius = abs(part.params.get("radius", 0.03))
            thickness = abs(part.params.get("thickness", 0.02))
            width = abs(part.params.get("width", thickness))
            height = abs(part.params.get("height", thickness))

            # 简化: shape factor
            shape = part.part_type
            if shape in ("cylinder", "hollow_cylinder"):
                volume = math.pi * radius ** 2 * length
                bbox_volume = (2 * radius) * (2 * radius) * length
            elif shape == "sphere":
                volume = 4 / 3 * math.pi * radius ** 3
                bbox_volume = (2 * radius) ** 3
            else:  # box
                volume = length * width * height
                bbox_volume = volume  # box shape factor = 1

            if bbox_volume > 0:
                utilization = volume / bbox_volume
                if utilization < 0.3:
                    n_inefficient += 1

        total = len(parts)
        score = 1.0 - (n_inefficient / total) * 0.2
        return max(0.0, min(1.0, score)), f"{n_inefficient}/{total} inefficient"


class RealtimeManufacturability:
    """实时制造性评估器

    嵌入进化循环，在适应度评估阶段同步计算制造性指标。

    Parameters
    ----------
    catalog : Dict[str, PartSpec]
        零件目录 (用于获取最小壁厚等参数)。
    checks : Optional[List[ManufacturingCheck]]
        要执行的检查列表。None = 全部默认检查。

    Usage:
        mfg = RealtimeManufacturability(catalog)
        score, details = mfg.evaluate(body)
        body.fitness_components['manufacturability'] = score
    """

    def __init__(
        self,
        catalog: Dict[str, PartSpec],
        checks: Optional[List[ManufacturingCheck]] = None,
    ) -> None:
        self.catalog = catalog
        self.checks: List[ManufacturingCheck] = checks or [
            WallThicknessCheck(),
            OverhangCheck(),
            InterferenceCheck(),
            JointStressCheck(),
            MaterialEfficiencyCheck(),
        ]

        self.n_evaluations = 0
        self._score_history: List[float] = []

    def evaluate(
        self, body: MechanicalBody
    ) -> Tuple[float, Dict[str, Tuple[float, str]]]:
        """评估单形态的制造性

        Returns
        -------
        overall_score : float
            0.0 (不可制造) ~ 1.0 (完美)
            乘法融合: 任何一环失败 → 整体严重下降
        details : Dict[str, Tuple[float, str]]
            {check_name: (score, detail_str)}
        """
        details: Dict[str, Tuple[float, str]] = {}

        # 乘法融合 (最严格)
        overall = 1.0
        for check in self.checks:
            score, detail = check.evaluate(body, self.catalog)
            details[check.name] = (score, detail)
            overall *= score

        # 加权乘法
        weighted = 1.0
        total_weight = 0.0
        for check in self.checks:
            w = check.weight
            s, _ = details[check.name]
            weighted *= max(s, 0.01) ** w
            total_weight += w

        if total_weight > 0:
            overall = weighted ** (1.0 / total_weight)

        self.n_evaluations += 1
        self._score_history.append(overall)

        return max(0.0, min(1.0, overall)), details

    def evaluate_batch(
        self, bodies: List[MechanicalBody]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """批量评估制造性

        Returns
        -------
        scores : ndarray (N,)
            每个形态的整体制造性分数。
        all_details : List[Dict]
            每个形态的详细评估结果。
        """
        scores = np.zeros(len(bodies))
        all_details = []

        for i, body in enumerate(bodies):
            score, details = self.evaluate(body)
            scores[i] = score
            all_details.append(details)

        return scores, all_details

    @property
    def stats(self) -> Dict:
        """统计信息"""
        n = max(self.n_evaluations, 1)
        hist = np.array(self._score_history) if self._score_history else np.zeros(1)
        return {
            "n_evaluations": self.n_evaluations,
            "mean_score": float(np.mean(hist)),
            "min_score": float(np.min(hist)),
            "max_score": float(np.max(hist)),
            "std_score": float(np.std(hist)),
            "pass_rate": float(np.mean(hist >= 0.5)),  # ≥ 0.5 视为通过
        }

    def reset_stats(self) -> None:
        """重置统计"""
        self.n_evaluations = 0
        self._score_history.clear()
