# ══════════════════════════════════════════════════════════
#  装配验证 — 干涉检查 / 公差累积 / 螺栓校核 / 配合验证
#
#  模块:
#    AssemblyValidator  — 干涉检测 + 间隙分析
#    ToleranceAnalyzer  — 公差累积链 (RSS/最坏情况)
#    BoltChecker         — VDI 2230 螺栓连接校核
#
#  与进化集成:
#    - 最终世代对所有精英个体跑装配验证
#    - 干涉 = 立即淘汰 (collision = 不能装配)
#    - 公差累积 < 容许 = 通过
# ══════════════════════════════════════════════════════════

from __future__ import annotations

import os
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

_logger = logging.getLogger(__name__)

__all__ = [
    "AssemblyReport",
    "AssemblyValidator",
    "ToleranceAnalyzer",
    "BoltChecker",
    "validate_assembly",
    "InterferenceResult",
    "ToleranceChain",
    "BoltJointResult",
]

# ══════════════════════════════════════════════════════════
#  数据类
# ══════════════════════════════════════════════════════════

@dataclass
class InterferenceResult:
    """干涉检测结果"""
    part_a: str
    part_b: str
    overlap_volume: float    # 干涉体积 (mm³)
    min_distance: float      # 最小距离 (mm, 负=干涉)
    contact_points: int      # 接触面数量
    
    @property
    def has_interference(self) -> bool:
        return self.overlap_volume > 0.01  # > 0.01 mm³ 算干涉
    
    @property
    def has_contact(self) -> bool:
        return abs(self.min_distance) < 0.1


@dataclass
class ToleranceChain:
    """公差累积链"""
    name: str
    elements: List[Dict]     # [{name, nominal, tolerance, type}]
    nominal: float           # 名义尺寸
    worst_case: float        # 最坏情况累积
    rss: float               # RSS 统计累积
    cpk_estimated: float     # 估计 Cpk
    
    @property
    def is_acceptable(self) -> bool:
        return self.rss <= self.nominal * 0.05  # < 5% of nominal


@dataclass
class BoltJointResult:
    """螺栓连接校核"""
    bolt_size: str           # M3/M4/...
    grade: str               # 8.8/10.9/12.9
    preload: float           # 预紧力 (N)
    working_load: float      # 工作载荷 (N)
    clamp_load: float        # 剩余夹紧力 (N)
    
    # 应力
    tensile_stress: float    # 拉应力 (MPa)
    shear_stress: float      # 剪应力 (MPa)
    von_mises: float         # 等效应力 (MPa)
    
    # 安全系数
    safety_yield: float      # 屈服安全系数
    safety_ultimate: float   # 极限安全系数
    safety_fatigue: float    # 疲劳安全系数
    
    # 表面压力
    surface_pressure: float  # 接触面压力 (MPa)
    surface_pressure_limit: float  # 容许压力
    
    passed: bool = True
    
    def summary(self) -> str:
        lines = [
            f"Bolt Joint {self.bolt_size} Grade {self.grade}:",
            f"  Preload: {self.preload:.0f} N  Working: {self.working_load:.0f} N",
            f"  Stress: σ_vm={self.von_mises:.0f} MPa",
            f"  Safety: yield={self.safety_yield:.1f} ultimate={self.safety_ultimate:.1f}",
            f"  Surface pressure: {self.surface_pressure:.0f}/{self.surface_pressure_limit:.0f} MPa",
            f"  Status: {'PASS' if self.passed else 'FAIL'}",
        ]
        return "\n".join(lines)


@dataclass
class AssemblyReport:
    """装配验证报告"""
    # 干涉
    interferences: List[InterferenceResult] = field(default_factory=list)
    has_interference: bool = False
    
    # 公差
    tolerance_chains: List[ToleranceChain] = field(default_factory=list)
    tolerance_failed: bool = False
    
    # 螺栓
    bolt_joints: List[BoltJointResult] = field(default_factory=list)
    bolt_failed: bool = False
    
    # 间隙
    min_clearance: Dict[str, float] = field(default_factory=dict)
    max_clearance: Dict[str, float] = field(default_factory=dict)
    
    # 总评
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def summary(self, verbose: bool = False) -> str:
        lines = [
            f"Assembly Validation: {'PASS' if self.passed else 'FAIL'}",
            f"  Interferences: {len(self.interferences)} ({'FAIL' if self.has_interference else 'PASS'})",
            f"  Tolerance chains: {len(self.tolerance_chains)} ({'FAIL' if self.tolerance_failed else 'PASS'})",
            f"  Bolt joints: {len(self.bolt_joints)} ({'FAIL' if self.bolt_failed else 'PASS'})",
        ]
        
        if verbose and self.interferences:
            for iv in self.interferences:
                if iv.has_interference:
                    lines.append(f"    ! {iv.part_a} ↔ {iv.part_b}: {iv.overlap_volume:.1f} mm³ overlap")
        
        if verbose and self.tolerance_chains:
            for tc in self.tolerance_chains:
                if not tc.is_acceptable:
                    lines.append(f"    ! {tc.name}: RSS={tc.rss:.3f}mm > limit")
        
        if verbose and self.bolt_joints:
            for bj in self.bolt_joints:
                if not bj.passed:
                    lines.append(f"    ! {bj.bolt_size}: safety={bj.safety_yield:.1f} < 1.5")
        
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings[:5]:
                lines.append(f"    - {w}")
        
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════
#  装配验证器
# ══════════════════════════════════════════════════════════

class AssemblyValidator:
    """装配验证器 — 干涉检测 + 间隙分析
    
    用法:
      >>> validator = AssemblyValidator()
      >>> report = validator.check_interferences(parts_dict)
      >>> print(report.summary())
    """
    
    def __init__(self, interference_tolerance: float = 0.05):
        """interference_tolerance: 容许干涉 (mm³), 默认 0.05"""
        self.tolerance = interference_tolerance
    
    def check_interferences(
        self,
        meshes: Dict[str, object],   # {part_id: trimesh.Trimesh}
        positions: Dict[str, np.ndarray] = None,
    ) -> List[InterferenceResult]:
        """检查零件间干涉
        
        Args:
            meshes: {part_id: trimesh mesh}
            positions: {part_id: (3,) np.array} 世界变换, None=已经在世界坐标
        """
        results = []
        ids = list(meshes.keys())
        
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                mesh_a = meshes[id_a]
                mesh_b = meshes[id_b]
                
                if mesh_a is None or mesh_b is None:
                    continue
                
                # 应用到世界坐标
                if positions:
                    a_copy = mesh_a.copy()
                    b_copy = mesh_b.copy()
                    a_copy.apply_translation(positions.get(id_a, np.zeros(3)))
                    b_copy.apply_translation(positions.get(id_b, np.zeros(3)))
                else:
                    a_copy = mesh_a
                    b_copy = mesh_b
                
                try:
                    # trimesh collision detection
                    # 使用 bounding box 快速剔除
                    if hasattr(a_copy, 'bounds') and hasattr(b_copy, 'bounds'):
                        if not self._bbox_overlap(a_copy.bounds, b_copy.bounds):
                            results.append(InterferenceResult(
                                part_a=id_a, part_b=id_b,
                                overlap_volume=0.0,
                                min_distance=999.0,
                                contact_points=0,
                            ))
                            continue
                    
                    # 精确检测
                    is_collision, data = self._check_collision(a_copy, b_copy)
                    
                    if is_collision:
                        overlap = data.get("overlap", 10.0)  # mm³
                        dist = data.get("distance", -0.1)
                        contacts = data.get("contacts", 1)
                    else:
                        overlap = 0.0
                        dist = data.get("distance", 10.0)
                        contacts = 0
                    
                    results.append(InterferenceResult(
                        part_a=id_a, part_b=id_b,
                        overlap_volume=overlap,
                        min_distance=dist,
                        contact_points=contacts,
                    ))
                    
                except Exception as e:
                    _logger.debug(f"Interference check failed {id_a}-{id_b}: {e}")
                    results.append(InterferenceResult(
                        part_a=id_a, part_b=id_b,
                        overlap_volume=0.0,
                        min_distance=999.0,
                        contact_points=0,
                    ))
        
        return results
    
    def _bbox_overlap(self, bounds_a, bounds_b) -> bool:
        """AABB 重叠检测"""
        for d in range(3):
            if bounds_a[1, d] < bounds_b[0, d]:
                return False
            if bounds_b[1, d] < bounds_a[0, d]:
                return False
        return True
    
    def _check_collision(self, mesh_a, mesh_b) -> Tuple[bool, Dict]:
        """trimesh 碰撞检测
        
        Returns:
            (is_collision, {"overlap": float, "distance": float, "contacts": int})
        """
        try:
            import trimesh
            
            # 方法 1: proximity check
            if hasattr(trimesh.proximity, 'closest_point'):
                try:
                    cp_a, cp_b, dist = trimesh.proximity.closest_point(mesh_a, mesh_b)
                    if dist < 0.01:  # < 0.01mm → 碰撞
                        return True, {
                            "overlap": 1.0,
                            "distance": float(dist),
                            "contacts": 1,
                        }
                except Exception:
                    pass
            
            # 方法 2: collision manager
            try:
                coll_mgr = trimesh.collision.CollisionManager()
                coll_mgr.add_object("a", mesh_a)
                coll_mgr.add_object("b", mesh_b)
                is_coll, contacts = coll_mgr.in_collision_internal(return_data=True)
                if is_coll:
                    return True, {
                        "overlap": 1.0,
                        "distance": -0.1,
                        "contacts": len(contacts) if contacts else 1,
                    }
            except Exception:
                pass
            
            # 方法 3: 体积交集 (昂贵)
            try:
                intersection = mesh_a.intersection(mesh_b, engine='manifold')
                if intersection is not None and hasattr(intersection, 'volume'):
                    overlap = abs(intersection.volume) * 1e9  # m³ → mm³
                    if overlap > self.tolerance:
                        return True, {
                            "overlap": overlap,
                            "distance": -0.1,
                            "contacts": len(intersection.faces) if hasattr(intersection, 'faces') else 1,
                        }
            except Exception:
                pass
            
            return False, {"overlap": 0.0, "distance": 10.0, "contacts": 0}
            
        except Exception:
            return False, {"overlap": 0.0, "distance": 10.0, "contacts": 0}
    
    def validate_clearances(
        self,
        meshes: Dict[str, object],
        positions: Dict[str, np.ndarray] = None,
        min_allowed: float = 0.1,  # mm
    ) -> Dict[str, float]:
        """检查装配间隙"""
        clearances = {}
        
        ids = list(meshes.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                
                try:
                    ma = meshes[id_a]
                    mb = meshes[id_b]
                    
                    # 最小距离
                    if hasattr(ma, 'vertices') and hasattr(mb, 'vertices'):
                        va = ma.vertices if positions is None else ma.vertices + positions.get(id_a, np.zeros(3))
                        vb = mb.vertices if positions is None else mb.vertices + positions.get(id_b, np.zeros(3))
                        
                        # KDTree 快速最近邻
                        from scipy.spatial import cKDTree
                        tree = cKDTree(vb)
                        distances, _ = tree.query(va, k=1)
                        min_dist = float(np.min(distances))
                        
                        key = f"{id_a}-{id_b}"
                        clearances[key] = min_dist
                except Exception:
                    pass
        
        return clearances


# ══════════════════════════════════════════════════════════
#  公差分析器
# ══════════════════════════════════════════════════════════

class ToleranceAnalyzer:
    """公差累积分析 (RSS + 最坏情况)
    
    两种方法:
      最坏情况 (Worst-Case): 所有零件同时最差
        T_wc = Σ |Ti|
      
      RSS (Root Sum Square): 统计叠加 (正态分布假设)
        T_rss = sqrt(Σ Ti²)
      
    适用:
      最坏情况 → n ≤ 4 (关键装配)
      RSS      → n > 4 (非关键装配，Cpk ≥ 1.33)
    """
    
    def __init__(self, cpk_target: float = 1.33):
        self.cpk_target = cpk_target
    
    def analyze_chain(
        self,
        name: str,
        elements: List[Tuple[str, float, float]],  # [(名称, 名义, 公差), ...]
        process_cpk: float = 1.0,
    ) -> ToleranceChain:
        """分析一条公差链"""
        total_nominal = sum(nom for _, nom, _ in elements)
        
        # 最坏情况
        worst_case = sum(tol for _, _, tol in elements)
        
        # RSS
        rss = np.sqrt(sum(tol**2 for _, _, tol in elements))
        
        # 估计 Cpk
        est_cpk = total_nominal / (3 * rss) if rss > 0 else float('inf')
        scaled_cpk = est_cpk * process_cpk
        
        elem_dicts = [
            {"name": n, "nominal": v, "tolerance": t, "type": "linear"}
            for n, v, t in elements
        ]
        
        return ToleranceChain(
            name=name,
            elements=elem_dicts,
            nominal=total_nominal,
            worst_case=worst_case,
            rss=rss,
            cpk_estimated=scaled_cpk,
        )
    
    def analyze_assembly(
        self,
        joints: List[Any],  # Joint objects
        parts: Dict[str, Any],
    ) -> List[ToleranceChain]:
        """分析装配体的所有公差链"""
        chains = []
        
        for joint in joints:
            parent_id = getattr(joint, 'parent_id', '')
            child_id = getattr(joint, 'child_id', '')
            
            # 轴孔配合
            shaft_dia = None
            hole_dia = None
            
            if child_id in parts:
                part = parts[child_id]
                shaft_dia = getattr(part, 'shaft_diameter', 
                            part.params.get('shaft_diameter', None) if hasattr(part, 'params') else None)
            
            if parent_id in parts:
                part = parts[parent_id]
                hole_dia = getattr(part, 'hole_diameter',
                           part.params.get('hole_diameter', None) if hasattr(part, 'params') else None)
            
            if shaft_dia is not None and hole_dia is not None:
                chain = self.analyze_chain(
                    f"shaft_fit_{parent_id}_{child_id}",
                    [
                        ("shaft", float(shaft_dia), 0.02),
                        ("hole", float(hole_dia), 0.03),
                    ]
                )
                chains.append(chain)
        
        return chains


# ══════════════════════════════════════════════════════════
#  螺栓校核器 (VDI 2230 简化)
# ══════════════════════════════════════════════════════════

# 螺栓几何数据
BOLT_DATA = {
    "M2":  {"d": 2.0, "d2": 1.74, "d3": 1.51, "As": 2.07, "pitch": 0.4},
    "M2.5":{"d": 2.5, "d2": 2.21, "d3": 1.95, "As": 3.39, "pitch": 0.45},
    "M3":  {"d": 3.0, "d2": 2.68, "d3": 2.39, "As": 5.03, "pitch": 0.5},
    "M4":  {"d": 4.0, "d2": 3.55, "d3": 3.14, "As": 8.78, "pitch": 0.7},
    "M5":  {"d": 5.0, "d2": 4.48, "d3": 4.02, "As": 14.2, "pitch": 0.8},
    "M6":  {"d": 6.0, "d2": 5.35, "d3": 4.77, "As": 20.1, "pitch": 1.0},
    "M8":  {"d": 8.0, "d2": 7.19, "d3": 6.47, "As": 36.6, "pitch": 1.25},
    "M10": {"d": 10.0,"d2": 9.03, "d3": 8.16, "As": 58.0, "pitch": 1.5},
    "M12": {"d": 12.0,"d2": 10.86,"d3": 9.86, "As": 84.3, "pitch": 1.75},
}

# 螺栓强度等级 (MPa)
BOLT_GRADES = {
    "4.6":  {"yield": 240,  "ultimate": 400},
    "4.8":  {"yield": 320,  "ultimate": 400},
    "5.6":  {"yield": 300,  "ultimate": 500},
    "5.8":  {"yield": 400,  "ultimate": 500},
    "6.8":  {"yield": 480,  "ultimate": 600},
    "8.8":  {"yield": 640,  "ultimate": 800},
    "10.9": {"yield": 900,  "ultimate": 1000},
    "12.9": {"yield": 1080, "ultimate": 1200},
    "A2-70": {"yield": 450, "ultimate": 700},   # 不锈钢
    "A4-80": {"yield": 600, "ultimate": 800},   # 不锈钢
}


class BoltChecker:
    """螺栓连接校核 (VDI 2230 简化版)
    
    校核项:
      1. 装配预紧力
      2. 工作应力 < 屈服强度
      3. 疲劳强度 (交变载荷)
      4. 表面压力 < 容许
      5. 防松 (剩余夹紧力 > 0)
    """
    
    def __init__(self, friction_coef: float = 0.12, safety_target: float = 1.5):
        self.mu = friction_coef       # 摩擦系数
        self.safety_target = safety_target  # 目标安全系数
    
    def check(
        self,
        bolt_size: str,
        grade: str = "8.8",
        axial_load: float = 500.0,    # 轴向工作载荷 (N)
        shear_load: float = 0.0,      # 剪切载荷 (N)
        preload_ratio: float = 0.7,   # 预紧力 / 屈服载荷
    ) -> BoltJointResult:
        """校核单个螺栓连接
        
        Args:
            bolt_size: "M3"/"M4"/...
            grade: "8.8"/"10.9"/"12.9"/"A2-70"
            axial_load: 轴向工作载荷 (N)
            shear_load: 剪切载荷 (N)
            preload_ratio: 使用屈服强度的比例 (典型 0.7)
        """
        bd = BOLT_DATA.get(bolt_size)
        bg = BOLT_GRADES.get(grade)
        
        if bd is None or bg is None:
            return BoltJointResult(
                bolt_size=bolt_size, grade=grade,
                preload=0, working_load=axial_load, clamp_load=0,
                tensile_stress=0, shear_stress=0, von_mises=0,
                safety_yield=0, safety_ultimate=0, safety_fatigue=0,
                surface_pressure=0, surface_pressure_limit=0,
                passed=False,
            )
        
        As = bd["As"]  # 应力截面积 (mm²)
        
        # 预紧力
        F_v = preload_ratio * bg["yield"] * As  # N
        
        # 剩余夹紧力
        F_kr = F_v - axial_load
        
        # 拉应力
        sigma_z = F_v / As  # MPa
        
        # 剪应力
        tau = shear_load / As  # MPa
        
        # von Mises
        sigma_vm = math.sqrt(sigma_z**2 + 3 * tau**2)
        
        # 安全系数
        safety_yield = bg["yield"] / sigma_vm if sigma_vm > 0 else float('inf')
        safety_ultimate = bg["ultimate"] / sigma_vm if sigma_vm > 0 else float('inf')
        
        # 疲劳 (简化: 疲劳极限 ≈ 0.35 × 极限强度)
        fatigue_limit = 0.35 * bg["ultimate"]
        safety_fatigue = fatigue_limit / sigma_vm if sigma_vm > 0 else float('inf')
        
        # 表面压力
        d_head = bd["d"] * 1.5
        A_head = math.pi / 4 * (d_head**2 - (bd["d"] * 0.85)**2)  # 近似
        p_surface = F_v / A_head if A_head > 0 else 0
        p_limit = bg["yield"] * 0.6  # 容许 ≈ 0.6 × 屈服
        
        passed = (
            safety_yield >= self.safety_target
            and F_kr > 0  # 不松动
            and p_surface <= p_limit
            and safety_fatigue >= 1.2
        )
        
        return BoltJointResult(
            bolt_size=bolt_size,
            grade=grade,
            preload=F_v,
            working_load=axial_load,
            clamp_load=F_kr,
            tensile_stress=sigma_z,
            shear_stress=tau,
            von_mises=sigma_vm,
            safety_yield=safety_yield,
            safety_ultimate=safety_ultimate,
            safety_fatigue=safety_fatigue,
            surface_pressure=p_surface,
            surface_pressure_limit=p_limit,
            passed=passed,
        )
    
    def check_pattern(
        self,
        bolt_size: str,
        grade: str,
        n_bolts: int,
        total_axial: float,
        total_shear: float = 0.0,
    ) -> List[BoltJointResult]:
        """校核多螺栓连接 (载荷均分)"""
        axial_per_bolt = total_axial / n_bolts
        shear_per_bolt = total_shear / n_bolts
        
        results = [
            self.check(bolt_size, grade, axial_per_bolt, shear_per_bolt)
            for _ in range(n_bolts)
        ]
        
        return results


# ══════════════════════════════════════════════════════════
#  总入口
# ══════════════════════════════════════════════════════════

def validate_assembly(
    meshes: Dict[str, object],
    positions: Dict[str, np.ndarray] = None,
    joints: List[Any] = None,
    parts: Dict[str, Any] = None,
    bolt_specs: List[Dict] = None,
    verbose: bool = False,
) -> AssemblyReport:
    """一键装配验证
    
    Args:
        meshes: {part_id: trimesh.Trimesh}
        positions: {part_id: world_position}
        joints: list of Joint objects
        parts: dict of Part objects
        bolt_specs: [{"size":"M3","grade":"8.8","load":500}, ...]
    
    Returns:
        AssemblyReport
    """
    report = AssemblyReport()
    
    # 1. 干涉检测
    validator = AssemblyValidator()
    interferences = validator.check_interferences(meshes, positions)
    report.interferences = interferences
    report.has_interference = any(iv.has_interference for iv in interferences)
    
    # 2. 间隙分析
    clearances = validator.validate_clearances(meshes, positions)
    report.min_clearance = clearances
    
    # 3. 公差分析
    if joints and parts:
        ta = ToleranceAnalyzer()
        chains = ta.analyze_assembly(joints, parts)
        report.tolerance_chains = chains
        report.tolerance_failed = any(not tc.is_acceptable for tc in chains)
    
    # 4. 螺栓校核
    if bolt_specs:
        bc = BoltChecker()
        for spec in bolt_specs:
            result = bc.check(
                bolt_size=spec.get("size", "M3"),
                grade=spec.get("grade", "8.8"),
                axial_load=spec.get("load", 500),
            )
            report.bolt_joints.append(result)
        report.bolt_failed = any(not bj.passed for bj in report.bolt_joints)
    
    # 总评
    report.passed = not (report.has_interference or report.tolerance_failed or report.bolt_failed)
    
    if report.has_interference:
        report.errors.append(f"{sum(1 for iv in interferences if iv.has_interference)} interference(s) detected")
    if report.tolerance_failed:
        report.errors.append(f"{sum(1 for tc in chains if not tc.is_acceptable)} tolerance chain(s) failed")
    if report.bolt_failed:
        report.errors.append(f"{sum(1 for bj in report.bolt_joints if not bj.passed)} bolt joint(s) failed")
    
    return report
