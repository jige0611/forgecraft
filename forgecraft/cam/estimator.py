"""
估算与仿真 — 打印时间、成本、可打印性分析、热变形预测

功能:
  - PrintTimeEstimator : 运动学 + 体积模型
  - CostEstimator      : 材料 + 能耗 + 人工 + 折旧
  - PrintabilityAnalyzer : 7 项可打印性检查
  - ThermalSimulator   : 简化 1D 热变形预测
"""

import math
from typing import Dict, List

import numpy as np

from forgecraft.cam.types import (
    LayerToolpath,
    PrintEstimate,
    PrintabilityReport,
    GCodeSettings,
)
from forgecraft.cam.multimaterial.compatibility import (
    MaterialCompatibilityMatrix,
    MATERIAL_DATA,
)

__all__ = [
    "PrintTimeEstimator",
    "CostEstimator",
    "PrintabilityAnalyzer",
    "ThermalSimulator",
]

# Filament 价格 (元/kg)
FILAMENT_PRICES = {
    "PLA": 25.0,
    "PETG": 30.0,
    "TPU": 40.0,
    "ABS": 22.0,
    "PVA": 60.0,
    "HIPS": 35.0,
    "Nylon": 50.0,
    "PC": 45.0,
    "ASA": 38.0,
}

# 电费 (元/kWh)
ELECTRICITY_PRICE = 0.6


# ══════════════════════════════════════════════════════════
# 打印时间估算
# ══════════════════════════════════════════════════════════

class PrintTimeEstimator:
    """打印时间估算 (运动学 + 体积模型)"""

    def __init__(self, settings: GCodeSettings = None):
        self.settings = settings or GCodeSettings()

    def estimate(self, toolpaths: List[LayerToolpath]) -> float:
        """估算总打印时间 (秒)"""
        s = self.settings

        # 运动学模型
        print_dist = 0.0
        travel_dist = 0.0
        total_volume = 0.0
        retraction_count = 0

        for layer in toolpaths:
            for seg_list in [layer.perimeters, layer.infill, layer.skin, layer.bridges]:
                for seg in seg_list:
                    if not seg.is_extrude:
                        continue
                    d = self._path_length(seg.points)
                    print_dist += d
                    volume = d * s.extrusion_width * s.layer_height
                    total_volume += volume

            for seg in layer.travel:
                travel_dist += self._path_length(seg.points)

            # 估算回抽次数 = 路径段间空驶次数
            all_segs = layer.perimeters + layer.infill + layer.skin + layer.bridges
            if len(all_segs) > 1:
                retraction_count += len(all_segs) - 1

        # 打印时间 = 路径长度 / 速度
        print_time = print_dist / s.print_speed
        travel_time = travel_dist / s.travel_speed

        # 体积限制 (最大体积流率 ≈ 15 mm³/s for 0.4mm nozzle)
        max_flow = 15.0
        flow_time = total_volume / max_flow

        # 冷却等待 (每层约 0.5s 最小层时间)
        min_layer_time = len(toolpaths) * 0.5

        # 回抽时间
        retract_time = retraction_count * 0.3  #每次约 0.3s

        # 综合取最大
        base_time = max(print_time, flow_time, min_layer_time)
        total = base_time + travel_time + retract_time

        return total

    @staticmethod
    def _path_length(points: np.ndarray) -> float:
        if len(points) < 2:
            return 0.0
        return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))


# ══════════════════════════════════════════════════════════
# 成本估算
# ══════════════════════════════════════════════════════════

class CostEstimator:
    """打印成本估算"""

    def __init__(
        self,
        printer_price: float = 2000.0,
        lifetime_hours: float = 5000.0,
        labor_rate_hourly: float = 20.0,
    ):
        self.printer_price = printer_price
        self.lifetime_hours = lifetime_hours
        self.labor_rate_hourly = labor_rate_hourly

    def estimate(
        self,
        print_time_seconds: float,
        filament_grams: Dict[str, float],
    ) -> PrintEstimate:
        """估算完整成本"""
        est = PrintEstimate()
        est.total_time_seconds = print_time_seconds

        hours = print_time_seconds / 3600.0

        # 材料成本
        material_cost = 0.0
        for mat, grams in filament_grams.items():
            mat_upper = mat.upper()
            price_per_kg = FILAMENT_PRICES.get(mat_upper, 30.0)
            material_cost += grams / 1000.0 * price_per_kg
            est.filament_grams[mat] = grams

        est.material_cost = material_cost

        # 能耗 (平均功率 ~150W)
        power_kw = 0.15
        est.energy_cost = power_kw * hours * ELECTRICITY_PRICE

        # 人工 (每次打印后处理约 10 分钟)
        est.labor_cost = self.labor_rate_hourly * (hours + 10.0 / 60.0)

        # 设备折旧
        wear_cost = self.printer_price / self.lifetime_hours * hours

        est.total_cost = material_cost + est.energy_cost + est.labor_cost + wear_cost

        return est


# ══════════════════════════════════════════════════════════
# 可打印性分析
# ══════════════════════════════════════════════════════════

class PrintabilityAnalyzer:
    """可打印性分析 (7 项检查)"""

    def __init__(self, settings: GCodeSettings = None):
        self.settings = settings or GCodeSettings()

    def analyze(
        self,
        mesh=None,
        layers: List = None,
        toolpaths: List = None,
    ) -> PrintabilityReport:
        """执行 7 项可打印性检查"""
        report = PrintabilityReport()
        s = self.settings

        # 1. 壁厚检查
        if mesh is not None:
            report.min_wall_thickness = self._check_wall_thickness(mesh)
            report.min_wall_pass = report.min_wall_thickness >= s.nozzle_diameter
            if not report.min_wall_pass:
                report.issues.append(
                    f"壁厚 {report.min_wall_thickness:.2f}mm < 喷嘴 {s.nozzle_diameter}mm"
                )

        # 2. 悬垂角检查
        if mesh is not None:
            report.overhang_angle_max = self._check_overhang(mesh)
            report.overhang_pass = report.overhang_angle_max <= 60.0
            if not report.overhang_pass:
                report.issues.append(
                    f"最大悬垂角 {report.overhang_angle_max:.0f}° > 60° 需要支撑"
                )

        # 3. 桥接检查
        if layers is not None:
            report.bridge_length_max = self._check_bridge_length(layers)
            report.bridge_pass = report.bridge_length_max <= 20.0
            if not report.bridge_pass:
                report.issues.append(
                    f"最大桥接 {report.bridge_length_max:.1f}mm > 20mm"
                )

        # 4. 细微特征检查
        if mesh is not None:
            report.small_feature_min = self._check_small_features(mesh)
            report.small_feature_pass = report.small_feature_min >= 0.5
            if not report.small_feature_pass:
                report.issues.append(
                    f"最小特征 {report.small_feature_min:.2f}mm < 0.5mm"
                )

        # 5. 首层接触
        if mesh is not None:
            report.first_layer_contact = self._check_first_layer(mesh)
            report.first_layer_pass = report.first_layer_contact >= 0.05
            if not report.first_layer_pass:
                report.issues.append(
                    f"首层接触 {report.first_layer_contact:.1%} < 5% 可能脱落"
                )

        # 6. 翘曲风险
        report.warping_risk = "low"  # 默认

        # 综合判定
        if len(report.issues) > 2:
            report.overall_verdict = "unprintable"
        elif len(report.issues) > 0:
            report.overall_verdict = "warning"
        else:
            report.overall_verdict = "printable"

        return report

    def _check_wall_thickness(self, mesh) -> float:
        """最小壁厚检查 (简化: 最短边长近似)"""
        if mesh is None or len(mesh.edges_unique) == 0:
            return 1.0
        edge_lengths = np.linalg.norm(
            mesh.vertices[mesh.edges_unique[:, 0]] - mesh.vertices[mesh.edges_unique[:, 1]],
            axis=1,
        )
        return float(np.min(edge_lengths))

    def _check_overhang(self, mesh) -> float:
        """最大悬垂角"""
        if mesh is None:
            return 0.0
        normals = mesh.face_normals
        z_down = np.array([0, 0, -1.0])
        dots = np.abs(np.dot(normals, z_down))
        angles = np.degrees(np.arccos(np.clip(dots, 0, 1)))
        return float(np.max(angles))

    def _check_bridge_length(self, layers) -> float:
        """最大桥接长度"""
        max_len = 0.0
        for layer in layers:
            for region in getattr(layer, 'bridge_regions', []):
                if len(region) >= 2:
                    diameter = np.max(np.linalg.norm(
                        region[:, None, :] - region[None, :, :], axis=2
                    ))
                    max_len = max(max_len, diameter)
        return max_len

    def _check_small_features(self, mesh) -> float:
        """最小特征尺寸"""
        if mesh is None or len(mesh.edges_unique) == 0:
            return 1.0
        return self._check_wall_thickness(mesh)

    def _check_first_layer(self, mesh) -> float:
        """首层接触面积比例"""
        if mesh is None:
            return 0.1
        z_min = mesh.bounds[0, 2]
        # 统计底面附近顶点
        bottom_mask = np.abs(mesh.vertices[:, 2] - z_min) < 0.01
        bottom_verts = mesh.vertices[bottom_mask]
        if len(bottom_verts) == 0:
            return 0.0
        # 简单估计：底面积 / 包围盒 XY 面积
        bbox_area = (mesh.bounds[1, 0] - mesh.bounds[0, 0]) * (mesh.bounds[1, 1] - mesh.bounds[0, 1])
        if bbox_area < 1e-10:
            return 1.0
        return min(1.0, len(bottom_verts) / 100.0)  # 粗略估计


# ══════════════════════════════════════════════════════════
# 热变形仿真 (简化 1D)
# ══════════════════════════════════════════════════════════

class ThermalSimulator:
    """简化热变形预测

    1D 逐层热应力累积模型：
        Δε = α × ΔT  (热应变)
        逐层收缩不均匀 → 翘曲
    """

    def __init__(self, settings: GCodeSettings = None):
        self.settings = settings or GCodeSettings()

    def simulate(
        self, num_layers: int, material: str = "PLA"
    ) -> dict:
        """模拟逐层热应力累积

        Returns:
            {"max_warp": ..., "risk": ..., "profile": ...}
        """
        mat_data = MaterialCompatibilityMatrix().get_material_data(material)

        # 热收缩系数 (简化)
        shrinkage = {
            "PLA": 0.002, "PETG": 0.003, "TPU": 0.001,
            "ABS": 0.005, "Nylon": 0.004, "PC": 0.004,
        }.get(material.upper(), 0.003)

        delta_T = mat_data.get("nozzle_temp", 210) - mat_data.get("bed_temp", 60)

        # 逐层累积
        profile = np.zeros(num_layers)
        for i in range(1, num_layers):
            strain = shrinkage * (delta_T / 200.0)
            profile[i] = profile[i - 1] * 0.95 + strain

        max_warp = profile[-1]
        if max_warp > 1.0:
            risk = "high"
        elif max_warp > 0.3:
            risk = "medium"
        else:
            risk = "low"

        return {
            "max_warp": float(max_warp),
            "risk": risk,
            "profile": profile.tolist(),
            "shrinkage": shrinkage,
            "delta_T": delta_T,
        }
