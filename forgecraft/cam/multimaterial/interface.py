"""
材料界面生成器 — 机械互锁 + 梯度界面

功能:
  - 机械互锁：两材料界面处生成燕尾榫结构
  - 梯度界面：过渡区逐层渐变材料比例
"""

import math
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["InterfaceGenerator"]


class InterfaceGenerator:
    """多材料界面生成器"""

    def __init__(
        self,
        tooth_width: float = 2.0,
        tooth_depth: float = 1.5,
        transition_layers: int = 5,
    ):
        self.tooth_width = tooth_width
        self.tooth_depth = tooth_depth
        self.transition_layers = transition_layers

    def generate_mechanical_interlock(
        self, interface_polygon: np.ndarray
    ) -> np.ndarray:
        """沿界面多边形生成燕尾榫互锁齿

        在界面边界上均匀分布三角形凹凸。
        """
        if len(interface_polygon) < 4:
            return interface_polygon

        perimeter = self._polygon_perimeter(interface_polygon)
        tooth_pitch = self.tooth_width * 2.0  # 每对齿的间距
        num_teeth = max(1, int(perimeter / tooth_pitch))

        new_verts = []

        for i in range(len(interface_polygon)):
            start = interface_polygon[i]
            end = interface_polygon[(i + 1) % len(interface_polygon)]
            seg_vec = end - start
            seg_len = np.linalg.norm(seg_vec)

            if seg_len < 1e-6:
                new_verts.append(start)
                continue

            seg_dir = seg_vec / seg_len
            normal = np.array([-seg_dir[1], seg_dir[0]])

            # 该段上齿的数量
            n_teeth = max(0, int(seg_len / tooth_pitch))
            if n_teeth == 0:
                new_verts.append(start)
                continue

            # 均匀分布齿
            for t in range(n_teeth):
                t_center = (t + 0.5) / n_teeth
                center = start + t_center * seg_vec

                # 锯齿顶点
                inner = center - normal * self.tooth_depth * 0.5
                outer = center + normal * self.tooth_depth * 0.5
                new_verts.append(inner)
                new_verts.append(outer)

            new_verts.append(end)

        return np.array(new_verts, dtype=np.float64)

    def generate_graded_interface(
        self, zone_width: float, num_layers: int = None
    ) -> List[float]:
        """生成梯度过渡区的材料比例序列

        Returns:
            [ratio_A] 列表，ratio_A 从 1.0 逐步降到 0.0
        """
        n = num_layers or self.transition_layers
        if n < 2:
            return [1.0]

        ratios = []
        for i in range(n):
            ratio = 1.0 - (i / (n - 1))
            ratios.append(ratio)

        return ratios

    def generate_interface_gcode(
        self,
        interface_polygon: np.ndarray,
        layer_height: float,
        extrusion_width: float = 0.45,
    ) -> List[str]:
        """生成界面打印 G-code (简化)"""
        interlock = self.generate_mechanical_interlock(interface_polygon)

        lines = ["; === Material Interface ==="]
        for i in range(len(interlock) - 1):
            a = interlock[i]
            b = interlock[i + 1]
            lines.append(f"G1 X{a[0]:.3f} Y{a[1]:.3f}")
            lines.append(f"G1 X{b[0]:.3f} Y{b[1]:.3f}")

        # 闭合
        if len(interlock) > 0:
            lines.append(
                f"G1 X{interlock[0][0]:.3f} Y{interlock[0][1]:.3f}"
            )
        lines.append("; === End Interface ===")

        return lines

    @staticmethod
    def _polygon_perimeter(polygon: np.ndarray) -> float:
        total = 0.0
        n = len(polygon)
        for i in range(n):
            total += np.linalg.norm(polygon[(i + 1) % n] - polygon[i])
        return total
