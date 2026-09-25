"""
ForgeCraft CAM 包 — 从设计到制造的计算机辅助制造系统

功能:
  - 切片引擎 (固定/自适应层高、桥接检测)
  - 路径规划 (8 种填充图案 + 外壳 + 顶底面 + 桥接)
  - 零件朝向优化 (24 候选方向 × 3 评分策略)
  - 排版优化 (Skyline 2D 装箱 + GA 优化)
  - 多材料支持 (分配/清洗/调度/界面/相容)
  - G-code 生成 (Marlin/Klipper/RepRap)
  - 估算与仿真 (时间/成本/可打印性/热变形)

Usage:
    >>> from forgecraft.cam import CAMFabricationPipeline
    >>> pipeline = CAMFabricationPipeline()
    >>> plan = pipeline.run(best_body)
"""

from forgecraft.cam.types import (
    SliceLayer,
    PathSegment,
    LayerToolpath,
    OrientationResult,
    PlacedPart,
    BuildPlateLayout,
    GCodeSettings,
    PrintEstimate,
    PrintabilityReport,
    SlicedJob,
    FabricationPlan,
)

from forgecraft.cam.slicer import SlicingEngine, slice_mesh
from forgecraft.cam.toolpath import (
    ToolpathGenerator,
    PerimeterGenerator,
    SkinGenerator,
    BridgeGenerator,
    ToolpathOptimizer,
    generate_toolpath,
)
from forgecraft.cam.orienter import PartOrienter
from forgecraft.cam.build_plater import BuildPlater, SkylinePacker
from forgecraft.cam.gcode_writer import GCodeWriter
from forgecraft.cam.estimator import (
    PrintTimeEstimator,
    CostEstimator,
    PrintabilityAnalyzer,
    ThermalSimulator,
)
from forgecraft.cam.pipeline import CAMFabricationPipeline
from forgecraft.cam import infill
from forgecraft.cam import multimaterial

__all__ = [
    # Types
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
    # Engines
    "SlicingEngine",
    "slice_mesh",
    "ToolpathGenerator",
    "PerimeterGenerator",
    "SkinGenerator",
    "BridgeGenerator",
    "ToolpathOptimizer",
    "generate_toolpath",
    "PartOrienter",
    "BuildPlater",
    "SkylinePacker",
    "GCodeWriter",
    "PrintTimeEstimator",
    "CostEstimator",
    "PrintabilityAnalyzer",
    "ThermalSimulator",
    "CAMFabricationPipeline",
    # Sub-packages
    "infill",
    "multimaterial",
]
