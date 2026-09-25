"""
CAM 制造流水线 — 从 MechanicalBody 到 G-code 的统一入口

功能:
  - 完整的 机械形态 → G-code 转换流水线
  - 自动材料分配、朝向优化、排版、切片、路径规划
  - 打印时间/成本估算
  - 可打印性分析

Usage:
    >>> from forgecraft.cam.pipeline import CAMFabricationPipeline
    >>> pipeline = CAMFabricationPipeline()
    >>> plan = pipeline.run(body)
    >>> print(plan.sliced_jobs[0].gcode)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import trimesh

from forgecraft.core.morphology import MechanicalBody, Part
from forgecraft.cam.types import (
    SliceLayer,
    SlicedJob,
    FabricationPlan,
    BuildPlateLayout,
    PlacedPart,
    GCodeSettings,
    PrintEstimate,
    PrintabilityReport,
    OrientationResult,
)
from forgecraft.cam.slicer import SlicingEngine
from forgecraft.cam.toolpath import ToolpathGenerator
from forgecraft.cam.orienter import PartOrienter
from forgecraft.cam.build_plater import BuildPlater
from forgecraft.cam.gcode_writer import GCodeWriter
from forgecraft.cam.estimator import (
    PrintTimeEstimator,
    CostEstimator,
    PrintabilityAnalyzer,
    ThermalSimulator,
)
from forgecraft.cam.multimaterial import (
    MaterialAssigner,
    ExtruderScheduler,
    MaterialCompatibilityMatrix,
)

_logger = logging.getLogger(__name__)

__all__ = ["CAMFabricationPipeline"]


class CAMFabricationPipeline:
    """CAM 制造流水线 — 统一入口

    整合所有 CAM 子系统，提供端到端的制造计划生成。

    Usage:
        pipeline = CAMFabricationPipeline(
            infill_pattern="gyroid",
            infill_density=0.15,
            strategy="min_support",
        )
        plan = pipeline.run(best_body)
        # plan.sliced_jobs[0].gcode → 可直接保存为 .gcode 文件
    """

    def __init__(
        self,
        infill_pattern: str = "rectilinear",
        infill_density: float = 0.2,
        perimeter_count: int = 3,
        top_layers: int = 4,
        bottom_layers: int = 3,
        layer_height: float = 0.2,
        first_layer_height: float = 0.3,
        extrusion_width: float = 0.45,
        plate_width: float = 220.0,
        plate_depth: float = 220.0,
        gap: float = 5.0,
        strategy: str = "min_support",
        default_material: str = "PLA",
        optimize_toolpath: bool = True,
    ):
        # 参数保存
        self.infill_pattern = infill_pattern
        self.infill_density = infill_density
        self.perimeter_count = perimeter_count
        self.top_layers = top_layers
        self.bottom_layers = bottom_layers
        self.layer_height = layer_height
        self.first_layer_height = first_layer_height
        self.extrusion_width = extrusion_width
        self.plate_width = plate_width
        self.plate_depth = plate_depth
        self.gap = gap
        self.strategy = strategy
        self.default_material = default_material
        self.optimize_toolpath = optimize_toolpath

        # 初始化子系统
        self.material_assigner = MaterialAssigner(default_material)
        self.orienter = PartOrienter(strategy=strategy)
        self.plater = BuildPlater(plate_width, plate_depth, gap, use_ga=True)
        self.compat = MaterialCompatibilityMatrix()

    def run(
        self,
        body: MechanicalBody,
        gcode_settings: GCodeSettings = None,
    ) -> FabricationPlan:
        """执行完整制造流水线

        Args:
            body: 机械形态 (进化后的最优个体)
            gcode_settings: G-code 生成参数 (None 则用默认)

        Returns:
            FabricationPlan: 完整的制造计划
        """
        plan = FabricationPlan()
        plan.body_name = getattr(body, 'name', 'unnamed')

        # ── 1. Part → Mesh ──
        parts_dict = body.parts_dict() if hasattr(body, 'parts_dict') else {}
        meshes: Dict[str, trimesh.Trimesh] = {}
        for pid, part in parts_dict.items():
            try:
                mesh = self._part_to_mesh(part)
                if mesh is not None and len(mesh.vertices) > 0:
                    meshes[pid] = mesh
            except Exception as e:
                _logger.warning(f"生成 mesh 失败 {pid}: {e}")

        if not meshes:
            _logger.error("没有可用的 mesh")
            return plan

        # ── 2. 材料分配 ──
        plan.material_assignments = {}
        for pid, part in parts_dict.items():
            if pid in meshes:
                plan.material_assignments[pid] = self.material_assigner.assign(part)

        # ── 3. 朝向优化 ──
        plan.orientations = {}
        for pid, mesh in meshes.items():
            result = self.orienter.optimize(mesh)
            plan.orientations[pid] = result

        # ── 4. 排版 ──
        placed_parts: List[PlacedPart] = []
        for pid, mesh in meshes.items():
            orient = plan.orientations[pid]
            rotated = mesh.copy()
            rotated.apply_transform(orient.matrix)

            # 2D 包围盒
            bbox_w = rotated.bounds[1, 0] - rotated.bounds[0, 0]
            bbox_d = rotated.bounds[1, 1] - rotated.bounds[0, 1]

            placed_parts.append(PlacedPart(
                part_id=pid,
                mesh=rotated,
                orientation=orient.matrix,
                bbox=(bbox_w, bbox_d),
                material=plan.material_assignments.get(pid, self.default_material),
            ))

        plan.layout = self.plater.layout(placed_parts)

        # ── 5. 切片 + 路径规划 + G-code ──
        slicer = SlicingEngine(
            layer_height=self.layer_height,
            first_layer_height=self.first_layer_height,
        )

        total_time = 0.0
        total_filament: Dict[str, float] = {}

        for placed in plan.layout.parts:
            mesh = placed.mesh

            # 切片
            layers = slicer.slice(mesh)
            if not layers:
                continue

            # 路径规划
            toolpath_gen = ToolpathGenerator(
                infill_pattern=self.infill_pattern,
                infill_density=self.infill_density,
                perimeter_count=self.perimeter_count,
                top_layers=self.top_layers,
                bottom_layers=self.bottom_layers,
                extrusion_width=self.extrusion_width,
                optimize=self.optimize_toolpath,
            )
            toolpaths = toolpath_gen.generate_all(layers)

            # G-code 生成
            gs = gcode_settings or GCodeSettings(
                layer_height=self.layer_height,
                first_layer_height=self.first_layer_height,
                extrusion_width=self.extrusion_width,
            )
            # 根据材料调整温度
            mat = placed.material
            gs.nozzle_temp = MaterialAssigner.get_nozzle_temp(mat)
            gs.bed_temp = MaterialAssigner.get_bed_temp(mat)

            writer = GCodeWriter(gs)
            gcode = writer.write_all(toolpaths, part_name=f"part_{placed.part_id}")

            # 时间估算
            estimator = PrintTimeEstimator(gs)
            job_time = estimator.estimate(toolpaths)
            total_time += job_time

            # 丝材估算
            filament_g = self._estimate_filament(mesh, mat)
            total_filament[mat] = total_filament.get(mat, 0.0) + filament_g

            plan.sliced_jobs.append(SlicedJob(
                part_id=placed.part_id,
                material=mat,
                layers=layers,
                toolpaths=toolpaths,
                gcode=gcode,
            ))

        plan.total_time_seconds = total_time

        # ── 6. 成本估算 ──
        cost_est = CostEstimator()
        estimate = cost_est.estimate(total_time, total_filament)
        plan.total_cost = estimate.total_cost

        # ── 7. 可打印性分析 ──
        analyzer = PrintabilityAnalyzer()
        # 取第一个有 mesh 的零件分析
        if placed_parts:
            plan.printability = analyzer.analyze(mesh=placed_parts[0].mesh)

        return plan

    def _estimate_filament(self, mesh: trimesh.Trimesh, material: str) -> float:
        """估算丝材重量 (g)"""
        volume_m3 = mesh.volume if hasattr(mesh, 'volume') else 0.0
        if volume_m3 <= 0:
            # 近似：包围盒体积 × 填充密度
            bbox = mesh.bounds
            volume_m3 = (
                (bbox[1, 0] - bbox[0, 0])
                * (bbox[1, 1] - bbox[0, 1])
                * (bbox[1, 2] - bbox[0, 2])
            ) * self.infill_density

        volume_cm3 = volume_m3 * 1e6
        density = self.material_assigner.get_filament_density(material)
        return volume_cm3 * density

    def _part_to_mesh(self, part: Part) -> Optional[trimesh.Trimesh]:
        """将 Part 转为 trimesh"""
        params = part.params
        ptype = part.part_type.lower()

        if ptype in ("base", "chassis", "plate", "mount", "motor_mount"):
            w = params.get("width", 0.05)
            d = params.get("depth", 0.05)
            h = params.get("height", 0.01)
            mesh = trimesh.creation.box(extents=[w, d, h])
        elif ptype in ("link", "limb", "arm", "leg"):
            r = params.get("radius", 0.01)
            h = params.get("length", 0.1)
            mesh = trimesh.creation.cylinder(radius=r, height=h, sections=16)
        elif ptype in ("foot", "wheel"):
            r = params.get("radius", 0.02)
            h = max(0.005, r * 0.3)
            mesh = trimesh.creation.cylinder(radius=r, height=h, sections=24)
        elif ptype in ("hinge", "ball_joint", "axle", "bearing"):
            r = params.get("radius", 0.008)
            mesh = trimesh.creation.icosphere(radius=r, subdivisions=2)
        elif ptype in ("motor", "actuator"):
            w = params.get("width", 0.03)
            d = params.get("depth", 0.03)
            h = params.get("height", 0.05)
            mesh = trimesh.creation.box(extents=[w, d, h])
        else:
            size = params.get("size", 0.03)
            mesh = trimesh.creation.box(extents=[size, size, size])

        if hasattr(part, 'position') and part.position is not None:
            mesh.apply_translation(part.position)
        return mesh

    def export_gcode(self, plan: FabricationPlan, output_dir: str = "."):
        """将 FabricationPlan 中的 G-code 导出到文件"""
        import os
        for job in plan.sliced_jobs:
            if job.gcode:
                filename = os.path.join(
                    output_dir, f"{plan.body_name}_{job.part_id}.gcode"
                )
                with open(filename, 'w') as f:
                    f.write(job.gcode)
                _logger.info(f"导出 G-code: {filename}")

    def summary(self, plan: FabricationPlan) -> str:
        """生成制造计划摘要"""
        lines = [
            "=" * 60,
            f"  ForgeCraft CAM 制造计划",
            "=" * 60,
            f"  零件名称: {plan.body_name}",
            f"  零件总数: {len(plan.sliced_jobs)}",
            f"  材料分配: {plan.material_assignments}",
        ]

        if plan.layout:
            lines.append(f"  排版利用率: {plan.layout.utilization:.1%}")

        lines.extend([
            f"  总打印时间: {plan.total_time_seconds / 3600:.1f} 小时",
            f"  估算成本: ¥{plan.total_cost:.2f}",
        ])

        if plan.printability:
            lines.append(f"  可打印性: {plan.printability.overall_verdict}")
            for issue in plan.printability.issues:
                lines.append(f"    ⚠ {issue}")

        lines.append("=" * 60)
        return "\n".join(lines)
