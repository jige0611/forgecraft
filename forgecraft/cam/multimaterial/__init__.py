"""
多材料打印子包

功能:
  - MaterialAssigner : 零件 → 丝材映射
  - PurgeCalculator   : 换料清洗体积计算
  - ExtruderScheduler : 挤出机调度
  - InterfaceGenerator : 材料界面生成
  - MaterialCompatibilityMatrix : 材料相容性矩阵
"""

from forgecraft.cam.multimaterial.compatibility import (
    MaterialCompatibilityMatrix,
    COMPATIBILITY_MATRIX,
)
from forgecraft.cam.multimaterial.assigner import MaterialAssigner
from forgecraft.cam.multimaterial.purge import PurgeCalculator
from forgecraft.cam.multimaterial.scheduler import ExtruderScheduler
from forgecraft.cam.multimaterial.interface import InterfaceGenerator

__all__ = [
    "MaterialCompatibilityMatrix",
    "COMPATIBILITY_MATRIX",
    "MaterialAssigner",
    "PurgeCalculator",
    "ExtruderScheduler",
    "InterfaceGenerator",
]
