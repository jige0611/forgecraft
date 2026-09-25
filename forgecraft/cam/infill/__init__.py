"""
填充图案子包

支持的填充图案:
  - rectilinear : 直线填充，逐层旋转 90°
  - honeycomb   : 正六边形蜂窝
  - gyroid      : 三重周期极小曲面等高线
  - cubic       : 三维立方晶格投影
  - concentric  : 同心轮廓偏置
  - adaptive    : 应力场自适应密度
  - lightning   : 闪电填充 (仅内支撑)
  - solid       : 100% 实心
"""

from forgecraft.cam.infill.base import InfillPattern
from forgecraft.cam.infill.rectilinear import RectilinearInfill
from forgecraft.cam.infill.honeycomb import HoneycombInfill
from forgecraft.cam.infill.gyroid import GyroidInfill
from forgecraft.cam.infill.cubic import CubicInfill
from forgecraft.cam.infill.concentric import ConcentricInfill
from forgecraft.cam.infill.adaptive import AdaptiveInfill
from forgecraft.cam.infill.lightning import LightningInfill

__all__ = [
    "InfillPattern",
    "RectilinearInfill",
    "HoneycombInfill",
    "GyroidInfill",
    "CubicInfill",
    "ConcentricInfill",
    "AdaptiveInfill",
    "LightningInfill",
]
