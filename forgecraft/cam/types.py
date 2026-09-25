"""
CAM 核心数据类型

定义切片、路径、排版、G-code 各阶段的数据结构。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ══════════════════════════════════════════════════════════
# 切片层
# ══════════════════════════════════════════════════════════

@dataclass
class SliceLayer:
    """单层切片数据"""
    z: float                                    # Z 高度 (mm)
    height: float                               # 层高 (mm)
    outer_contours: List[np.ndarray] = field(default_factory=list)   # 外轮廓 [(N,2)]
    inner_contours: List[np.ndarray] = field(default_factory=list)   # 内轮廓/孔洞 [(N,2)]
    is_bridge: bool = False
    bridge_regions: List[np.ndarray] = field(default_factory=list)


# ══════════════════════════════════════════════════════════
# 路径段
# ══════════════════════════════════════════════════════════

@dataclass
class PathSegment:
    """单条路径段"""
    points: np.ndarray          # (N, 2) 路径点
    is_extrude: bool = True     # True=挤出, False=空驶
    feedrate: float = 0.0       # 0 表示使用默认速度
    extruder_id: int = 0


@dataclass
class LayerToolpath:
    """单层完整路径"""
    z: float
    height: float
    perimeters: List[PathSegment] = field(default_factory=list)
    infill: List[PathSegment] = field(default_factory=list)
    skin: List[PathSegment] = field(default_factory=list)
    bridges: List[PathSegment] = field(default_factory=list)
    support: List[PathSegment] = field(default_factory=list)
    travel: List[PathSegment] = field(default_factory=list)


# ══════════════════════════════════════════════════════════
# 朝向
# ══════════════════════════════════════════════════════════

@dataclass
class OrientationResult:
    """零件朝向优化结果"""
    matrix: np.ndarray          # 4x4 变换矩阵
    support_volume: float = 0.0
    support_area: float = 0.0
    build_height: float = 0.0
    strategy: str = "min_support"
    score: float = 0.0


# ══════════════════════════════════════════════════════════
# 排版
# ══════════════════════════════════════════════════════════

@dataclass
class PlacedPart:
    """排版后的零件位置"""
    part_id: str
    mesh: "trimesh.Trimesh"
    orientation: np.ndarray     # 4x4 变换
    bbox: Tuple[float, float]   # (width, depth) 2D 包围盒
    pos_x: float = 0.0
    pos_y: float = 0.0
    material: str = "PLA"


@dataclass
class BuildPlateLayout:
    """排版结果"""
    plate_width: float
    plate_depth: float
    parts: List[PlacedPart] = field(default_factory=list)
    utilization: float = 0.0


# ══════════════════════════════════════════════════════════
# G-code
# ══════════════════════════════════════════════════════════

@dataclass
class GCodeSettings:
    """G-code 生成参数"""
    nozzle_diameter: float = 0.4
    layer_height: float = 0.2
    first_layer_height: float = 0.3
    extrusion_width: float = 0.45
    filament_diameter: float = 1.75
    print_speed: float = 60.0            # mm/s
    first_layer_speed: float = 25.0
    travel_speed: float = 150.0          # mm/s
    retract_distance: float = 5.0        # mm
    retract_speed: float = 40.0          # mm/s
    unretract_speed: float = 30.0
    bed_temp: float = 60.0
    nozzle_temp: float = 210.0
    fan_speed: int = 100
    fan_full_at_layer: int = 3
    dialect: str = "marlin"              # marlin | klipper | reprap
    extruder_count: int = 1
    skirt_loops: int = 2
    skirt_distance: float = 5.0
    brim_width: float = 0.0


# ══════════════════════════════════════════════════════════
# 估算
# ══════════════════════════════════════════════════════════

@dataclass
class PrintEstimate:
    """打印估算结果"""
    total_time_seconds: float = 0.0
    print_time_seconds: float = 0.0
    travel_time_seconds: float = 0.0
    filament_grams: Dict[str, float] = field(default_factory=dict)
    total_cost: float = 0.0
    material_cost: float = 0.0
    energy_cost: float = 0.0
    labor_cost: float = 0.0


@dataclass
class PrintabilityReport:
    """可打印性分析"""
    min_wall_thickness: float = 0.0
    min_wall_pass: bool = True
    overhang_angle_max: float = 0.0
    overhang_pass: bool = True
    bridge_length_max: float = 0.0
    bridge_pass: bool = True
    small_feature_min: float = 0.0
    small_feature_pass: bool = True
    first_layer_contact: float = 0.0
    first_layer_pass: bool = True
    warping_risk: str = "low"           # low | medium | high
    overall_verdict: str = "printable"  # printable | warning | unprintable
    issues: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════
# 流水线
# ══════════════════════════════════════════════════════════

@dataclass
class SlicedJob:
    """单零件完整切片作业"""
    part_id: str
    material: str
    layers: List[SliceLayer] = field(default_factory=list)
    toolpaths: List[LayerToolpath] = field(default_factory=list)
    gcode: str = ""


@dataclass
class FabricationPlan:
    """完整制造计划：从 MechanicalBody → 制造输出"""
    body_name: str = ""
    material_assignments: Dict[str, str] = field(default_factory=dict)
    orientations: Dict[str, OrientationResult] = field(default_factory=dict)
    layout: Optional[BuildPlateLayout] = None
    sliced_jobs: List[SlicedJob] = field(default_factory=list)
    total_cost: float = 0.0
    total_time_seconds: float = 0.0
    printability: Optional[PrintabilityReport] = None


__all__ = [
    "SliceLayer",
    "PathSegment",
    "LayerToolpath",
    "OrientationResult",
    "PlacedPart",
    "BuildPlateLayout",
    "GCodeSettings",
    "PrintEstimate",
    "PrintabilityReport",
    "SlicedJob",
    "FabricationPlan",
]
