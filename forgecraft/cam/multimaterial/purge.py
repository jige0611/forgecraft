"""
清洗塔计算器 — 换料时清洗体积与塔结构计算

清洗量 = 基础死腔体积 × 粘度比 × 安全系数
"""

import math
from typing import List, Tuple

import numpy as np

from forgecraft.cam.multimaterial.compatibility import MaterialCompatibilityMatrix

__all__ = ["PurgeCalculator", "PurgeTowerConfig"]


class PurgeTowerConfig:
    """清洗塔配置"""
    def __init__(
        self,
        tower_x: float = 200.0,
        tower_y: float = 200.0,
        tower_width: float = 15.0,
        tower_depth: float = 15.0,
        nozzle_diameter: float = 0.4,
        hotend_length: float = 30.0,
        safety_factor: float = 1.5,
    ):
        self.tower_x = tower_x
        self.tower_y = tower_y
        self.tower_width = tower_width
        self.tower_depth = tower_depth
        self.nozzle_diameter = nozzle_diameter
        self.hotend_length = hotend_length
        self.safety_factor = safety_factor


class PurgeCalculator:
    """清洗塔计算器

    Usage:
        calc = PurgeCalculator(config)
        volume = calc.purge_volume("PLA", "PETG")
        gcode = calc.generate_tower_gcode(total_volume, layer_height)
    """

    def __init__(self, config: PurgeTowerConfig = None):
        self.config = config or PurgeTowerConfig()
        self._compat = MaterialCompatibilityMatrix()

    def purge_volume(self, mat_from: str, mat_to: str) -> float:
        """计算从 mat_from 切换到 mat_to 所需的清洗体积 (mm³)

        公式: V = V_dead × η_ratio × safety × color_factor
        """
        # 死腔体积 = 喷嘴内腔 + 热端通道
        r = self.config.nozzle_diameter / 2.0
        dead_volume = math.pi * r * r * self.config.hotend_length

        # 粘度比
        data_from = self._compat.get_material_data(mat_from)
        data_to = self._compat.get_material_data(mat_to)
        visc_from = data_from.get("viscosity", 1.0)
        visc_to = data_to.get("viscosity", 1.0)
        viscosity_ratio = max(visc_from, visc_to) / min(visc_from, visc_to)

        # 相容性额外系数
        compat_factor = self._compat.purge_factor(mat_from, mat_to)

        purge_vol = dead_volume * viscosity_ratio * compat_factor * self.config.safety_factor

        # 深色→浅色额外翻倍
        if self._is_dark_to_light(mat_from, mat_to):
            purge_vol *= 2.0

        return max(purge_vol, dead_volume * 3)  # 至少清洗 3 倍死腔

    def generate_tower_gcode(
        self,
        total_purge_volume: float,
        layer_height: float,
        extrusion_width: float = 0.45,
    ) -> List[str]:
        """生成清洗塔 G-code

        Returns:
            G1 指令列表
        """
        tw = self.config.tower_width
        td = self.config.tower_depth
        tx = self.config.tower_x
        ty = self.config.tower_y

        # 每层清洗体积 = 塔截面积 × 层高
        layer_volume = tw * td * layer_height

        # 需要的塔层数
        num_layers = max(1, int(math.ceil(total_purge_volume / layer_volume)))

        lines = ["; === Purge Tower ==="]
        for i in range(num_layers):
            # 简单的矩形填充
            z = (i + 1) * layer_height
            lines.append(f"G0 X{tx:.3f} Y{ty:.3f} Z{z:.3f} F9000")
            lines.append(f"G1 X{tx + tw:.3f} Y{ty:.3f} F1500")
            lines.append(f"G1 X{tx + tw:.3f} Y{ty + td:.3f}")
            lines.append(f"G1 X{tx:.3f} Y{ty + td:.3f}")
            lines.append(f"G1 X{tx:.3f} Y{ty:.3f}")

        lines.append("; === End Purge Tower ===")
        return lines

    def _is_dark_to_light(self, mat_from: str, mat_to: str) -> bool:
        """判断是否从深色切换到浅色 (简化)"""
        dark_materials = {"ABS", "Nylon", "PC", "ASA"}
        light_materials = {"PLA", "PETG", "PVA", "HIPS"}
        return mat_from.upper() in dark_materials and mat_to.upper() in light_materials
