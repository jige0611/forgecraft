"""
关节公差系统 — FDM 打印间隙补偿

FDM 打印机会因挤出宽度/热收缩产生尺寸偏差。
本模块为每个关节计算所需的间隙补偿，使打印后零件可组装。

功能：
1. FDM 间隙计算 — 基于材料/喷嘴/层高
2. 关节公差应用 — 在 mesh 级别修正配合面
3. 配合等级 — 间隙配合(C11)/过渡配合(H7)/过盈配合(s6)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
__all__ = [
    "FDMProfile",
    "FDM_PROFILES",
    "FIT_CLASSES",
    "compute_joint_clearance",
    "apply_hinge_clearance",
    "compute_print_report",
    "auto_compensate_mesh",
]




@dataclass
class FDMProfile:
    """FDM 打印机参数配置

    关键间隙公式:
      min_clearance = extrusion_width * 1.5        (滑动配合下限)
      standard_clearance = extrusion_width * 2.0 + shrinkage*10  (标准间隙)
    """
    nozzle_diameter: float = 0.0004
    layer_height: float = 0.0002
    extrusion_width: float = 0.00045
    shrinkage: float = 0.005
    material: str = "PLA"

    @property
    def min_clearance(self) -> float:
        return self.extrusion_width * 1.5

    @property
    def standard_clearance(self) -> float:
        return self.extrusion_width * 2.0 + self.shrinkage * 10.0

    @property
    def loose_clearance(self) -> float:
        return self.extrusion_width * 3.0 + self.shrinkage * 15.0


DEFAULT_FDM = FDMProfile()

FDM_PROFILES = {
    "PLA_04mm": FDMProfile(nozzle_diameter=0.0004, layer_height=0.0002,
                            extrusion_width=0.00045, shrinkage=0.005, material="PLA"),
    "PLA_06mm": FDMProfile(nozzle_diameter=0.0006, layer_height=0.0003,
                            extrusion_width=0.00065, shrinkage=0.006, material="PLA"),
    "PETG_04mm": FDMProfile(nozzle_diameter=0.0004, layer_height=0.0002,
                             extrusion_width=0.00048, shrinkage=0.003, material="PETG"),
    "ABS_04mm": FDMProfile(nozzle_diameter=0.0004, layer_height=0.0002,
                            extrusion_width=0.00045, shrinkage=0.008, material="ABS"),
    "TPU_04mm": FDMProfile(nozzle_diameter=0.0004, layer_height=0.0002,
                            extrusion_width=0.00050, shrinkage=0.002, material="TPU"),
}

FIT_CLASSES = {
    "H7_f6": 0.00003,
    "H7_g6": 0.00005,
    "H7_h6": 0.00001,
    "C11_h11": 0.00015,
    "loose": 0.00030,
    "press_fit": -0.00005,
}


def compute_joint_clearance(
    fdm_profile: FDMProfile,
    joint_type: str,
    shaft_radius: float = 0.003,
    fit_class: str = "H7_f6",
) -> Tuple[float, Dict]:
    # ═══ Phase 1: Compute joint clearances ═══
    fit_clearance = FIT_CLASSES.get(fit_class, 0.00005)
    fdm_min = fdm_profile.min_clearance
    # 取 max(FDM最小间隙, ISO配合间隙 + 热补偿)
    effective = max(fdm_min, fit_clearance + fdm_profile.shrinkage * shaft_radius * 2)

    detail = {
        "fdm_min_clearance": fdm_min,
        "fit_clearance": fit_clearance,
        "thermal_compensation": fdm_profile.shrinkage * shaft_radius * 2,
        "effective_clearance": effective,
        "material": fdm_profile.material,
        "joint_type": joint_type,
        "shaft_radius": shaft_radius,
        "fit_class": fit_class,
    }

    return effective, detail


def apply_hinge_clearance(
    parent_mesh: trimesh.Trimesh,
    child_mesh: trimesh.Trimesh,
    joint_position: np.ndarray,
    joint_axis: np.ndarray,
    shaft_radius: float = 0.003,
    clearance: float = 0.0003,
    bearing_length: float = 0.008,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    parent = parent_mesh.copy()
    child = child_mesh.copy()

    shaft_r = shaft_radius + clearance / 2.0

    hole = trimesh.creation.cylinder(radius=shaft_r, height=bearing_length * 2.5, sections=24)

    joint_axis = joint_axis / (np.linalg.norm(joint_axis) + 1e-9)
    z_axis = np.array([0, 0, 1.0])
    angle = math.acos(np.clip(np.dot(z_axis, joint_axis), -1, 1))
    if angle > 1e-6 and angle < math.pi - 1e-6:
        rot_axis = np.cross(z_axis, joint_axis)
        rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-9)
        rot = trimesh.transformations.rotation_matrix(angle, rot_axis)
        hole.apply_transform(rot)

    hole.apply_translation(joint_position)

    try:
        child = child.difference(hole, engine="scad")
    except Exception:
        pass

    return parent, child


def compute_print_report(
    parts: List[dict],
    fdm_profile: FDMProfile = None,
) -> Dict:
    if fdm_profile is None:
        fdm_profile = DEFAULT_FDM

    joints = []
    clearance_info = []
    warnings = []

    for part in parts:
        params = part.get("params", {})
        thickness_checks = []
        for key in ["thickness", "radius", "height", "width"]:
            if key in params:
                val = params[key]
                thickness_checks.append((key, val))
                if val < fdm_profile.extrusion_width * 2:
                    warnings.append({
                        "part_id": part.get("part_id", "?"),
                        "part_type": part.get("part_type", "?"),
                        "issue": f"{key}={val*1000:.2f}mm < 最小壁厚 {fdm_profile.extrusion_width*2000:.2f}mm",
                        "severity": "error" if val < fdm_profile.extrusion_width else "warning",
                    })

    return {
        "profile": {
            "material": fdm_profile.material,
            "nozzle": fdm_profile.nozzle_diameter * 1000,
            "layer_height": fdm_profile.layer_height * 1000,
            "shrinkage": fdm_profile.shrinkage * 100,
        },
        "min_wall_thickness": fdm_profile.extrusion_width * 2000,
        "recommended_clearance": fdm_profile.standard_clearance * 1000,
        "warnings": warnings,
    }


def auto_compensate_mesh(
    mesh: trimesh.Trimesh,
    fdm_profile: FDMProfile = None,
) -> trimesh.Trimesh:
    if fdm_profile is None:
        fdm_profile = DEFAULT_FDM

    result = mesh.copy()
    factor = 1.0 + fdm_profile.shrinkage * 0.5
    center = result.centroid
    result.apply_translation(-center)
    result.vertices *= factor
    result.apply_translation(center)
    return result
