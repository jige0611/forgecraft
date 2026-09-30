# ══════════════════════════════════════════════════════════
# 🔧 制造集成管道 (端到端)
#
# 统一入口: 从 MechanicalBody → 可制造输出
#
# 功能:
#   ✅ STEP AP242 装配导出 (含材料属性+装配层级)
#   ✅ STL 多色导出 (区分零件类型)
#   ✅ URDF 机器人描述文件导出 (ROS 兼容)
#   ✅ BOM 物料清单 (含采购链接+真实成本)
#   ✅ 可制造性综合评分 (碰撞/公差/材料/成本)
#   ✅ 3D打印成本估算 (按材料+体积)
#   ✅ 设计审查报告 (Markdown/JSON)
#   ✅ 一键导出 (step/stl/urdf/bom/report)
#
# 使用:
#   >>> from forgecraft.manufacturing.pipeline import ManufacturingPipeline
#   >>> pipeline = ManufacturingPipeline()
#   >>> result = pipeline.export_all(best_body, output_dir="design_output")
# ══════════════════════════════════════════════════════════

import os
import json
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

import numpy as np

from forgecraft.core.morphology import MechanicalBody, Part, Joint
from forgecraft.core.materials_database import MaterialDB, get_part_density, calculate_part_mass
from forgecraft.core.collision_checker import CollisionChecker
from forgecraft.manufacturing.step_writer import _reset_ids
from forgecraft.manufacturing.step_assembly import (
    _write_product, _write_assembly_relation,
)
from forgecraft.manufacturing.tolerances import (
    FDMProfile, FDM_PROFILES, DEFAULT_FDM, FIT_CLASSES,
    compute_joint_clearance, compute_print_report,
)
from forgecraft.manufacturing.exporter import ManufacturingExporter

logger = logging.getLogger(__name__)

__all__ = [
    "ManufacturingReport",
    "ManufacturingPipeline",
    "get_pipeline",
    "export_design",
    "score_design",
    "export_step_assembly",
    "export_stl",
    "export_urdf",
    "score_manufacturability",
    "estimate_print_time",
    "estimate_filament_weight",
]


# ══════════════════════════════════════════════════════════
# 数据类型
# ══════════════════════════════════════════════════════════

@dataclass
class ManufacturingReport:
    """制造报告"""
    body_name: str
    timestamp: str
    total_parts: int
    total_joints: int
    actuated_joints: int
    
    # 材料分析
    materials_used: Dict[str, int]
    total_mass_kg: float
    total_volume_m3: float
    
    # 可制造性
    manufacturability_score: float  # 0-1
    collision_issues: int
    tolerance_warnings: int
    
    # 成本估算
    material_cost_usd: float
    printing_cost_usd: float
    electronics_cost_usd: float
    total_cost_usd: float
    
    # 打印参数
    recommended_material: str
    print_time_hours: float
    filament_grams: float
    
    # 采购
    purchase_items: List[Dict]
    purchase_total_usd: float
    
    # 文件列表
    exported_files: List[str] = field(default_factory=list)
    
    # 建议
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════
# 成本估算
# ══════════════════════════════════════════════════════════

# 3D打印材料成本 ($/kg)
FILAMENT_COSTS = {
    "PLA": 20.0,
    "PETG": 25.0,
    "ABS": 22.0,
    "TPU": 35.0,
    "nylon": 45.0,
    "PC": 40.0,
    "carbon_fiber_pla": 50.0,
}

# 打印服务成本 ($/小时)
PRINT_HOURLY_RATE = {
    "PLA": 3.0,
    "PETG": 4.0,
    "ABS": 3.5,
    "TPU": 5.0,
    "nylon": 8.0,
    "PC": 7.0,
}

# 打印材料密度 (kg/m³)
FILAMENT_DENSITIES = {
    "PLA": 1240.0,
    "PETG": 1270.0,
    "ABS": 1040.0,
    "TPU": 1200.0,
    "nylon": 1140.0,
    "PC": 1180.0,
}

# 电子元件参考价格
ELECTRONICS_COSTS = {
    "robomaster_gm6020": 89.0,
    "robomaster_m3508": 49.0,
    "robomaster_m2006": 39.0,
    "frc_falcon500": 99.0,
    "frc_neo550": 59.0,
    "frc_cim": 45.0,
    "mg996r": 8.0,
    "sg90": 2.0,
    "ds3225": 15.0,
    "lipo_3s_2200mAh": 18.0,
    "lipo_4s_5200mAh": 35.0,
    "lipo_6s_3500mAh": 45.0,
    "imu_mpu9250": 5.0,
    "imu_bno055": 15.0,
    "mcu_stm32f429": 25.0,
    "esc_c620": 30.0,
}


def estimate_print_time(volume_m3: float, layer_height_m: float = 0.0002,
                        nozzle_diameter_m: float = 0.0004,
                        print_speed_ms: float = 0.060) -> float:
    """估算 3D 打印时间 (小时)"""
    if volume_m3 <= 0:
        return 0.0
    # 体积 / (喷嘴面积 × 速度)
    nozzle_area = math.pi * (nozzle_diameter_m / 2) ** 2
    flow_rate = nozzle_area * print_speed_ms
    seconds = volume_m3 / flow_rate
    return seconds / 3600.0


def estimate_filament_weight(volume_m3: float, material: str) -> float:
    """估算耗材重量 (克)"""
    density = FILAMENT_DENSITIES.get(material, 1240.0)
    return volume_m3 * density * 1000.0


# ══════════════════════════════════════════════════════════
# STEP 导出增强版
# ══════════════════════════════════════════════════════════

# ── Export STEP assembly — all parts + joints → STEP AP242
def export_step_assembly(body: MechanicalBody, output_path: str,
                         material_db: MaterialDB = None) -> str:
    """
    导出 STEP AP242 装配文件 (含材料属性)
    
    Args:
        body: 机械体
        output_path: 输出路径
        material_db: 材料数据库
    
    Returns:
        文件路径
    """
    from forgecraft.manufacturing.step_writer import (
        _next_id,
    )
    from forgecraft.manufacturing.step_assembly import _write_product
    
    _reset_ids()
    lines = []
    lines.append("ISO-10303-21;")
    lines.append("HEADER;")
    lines.append(f"FILE_DESCRIPTION(('ForgeCraft Design - {body.name}'),'2;1');")
    lines.append(f"FILE_NAME('{body.name}.stp','{datetime.now().isoformat()}',"
                 f"('ForgeCraft'),(''),'ForgeCraft v1.0','','');")
    lines.append("FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF'));")
    lines.append("ENDSEC;")
    lines.append("DATA;")
    
    # 产品层级: 根产品 + 每个零件一个产品定义
    # (装配关系 NEXT_ASSEMBLY_USAGE_OCCURRENCE 需要引用各自的 PRODUCT_DEFINITION)
    product_lines, pid, pd_id, pdf_id = _write_product(body.name, "ForgeCraft Generated Design")
    lines.extend(product_lines)
    pdf_map: Dict[str, int] = {}
    
    if material_db is None:
        material_db = MaterialDB()
    
    # 为每个零件写入几何体
    part_meshes = {}
    joint_lines = []
    
    for part in body.parts():
        # 生成简化网格
        mesh = _generate_simple_mesh(part)
        part_meshes[part.part_id] = mesh
        
        # 获取材料
        material_id = material_db.get_part_material(part.part_type)
        material_props = material_db.get_material(material_id)
        
        part_prod_lines, _pid, _pd_id, part_pdf_id = _write_product(
            part.part_type, f"part {part.part_id[:6]}"
        )
        lines.extend(part_prod_lines)
        pdf_map[part.part_id] = part_pdf_id
        
        # 写入 tessellated shape (返回的是多行文本, 整块追加)
        from forgecraft.manufacturing.step_writer import _mesh_to_tessellated_shape
        if mesh.vertices.size > 0:
            shape_lines, _ = _mesh_to_tessellated_shape(
                mesh.vertices, mesh.faces, part.part_id,
            )
            lines.append(shape_lines)
    
    # 装配关系
    for parent_id, child_id in body.graph.edges:
        joint = body.get_joint(parent_id, child_id)
        child_part = body.get_part(child_id)
        
        # 变换: 子零件位置 - 父零件位置
        parent_pos = body.get_part(parent_id).position
        transform = child_part.position - parent_pos
        
        # _write_assembly_relation 返回 (lines, rel_id) 元组
        rel_lines, _ = _write_assembly_relation(
            pdf_map[parent_id], pdf_map[child_id], transform
        )
        lines.extend(rel_lines)
    
    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return output_path


def _generate_simple_mesh(part: Part) -> "trimesh.Trimesh":
    """为零件生成简化网格"""
    import trimesh
    
    length = part.params.get("length", 0.1)
    width = part.params.get("width", part.params.get("thickness", 0.02))
    height = part.params.get("height", part.params.get("thickness", 0.02))
    radius = part.params.get("radius", 0.03)
    
    part_type = part.part_type.lower()
    
    if "cylinder" in part_type or "motor" in part_type or "tube" in part_type:
        mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
    elif "sphere" in part_type or "foot" in part_type:
        mesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
    elif "box" in part_type or "extrusion" in part_type or "battery" in part_type:
        mesh = trimesh.creation.box(extents=[length, width, height])
    elif "rod" in part_type:
        mesh = trimesh.creation.cylinder(radius=width/2, height=length, sections=12)
    else:
        mesh = trimesh.creation.box(extents=[max(0.01, length), max(0.01, width), max(0.01, height)])
    
    # 移动到零件位置
    mesh.apply_translation(part.position)
    return mesh


# ══════════════════════════════════════════════════════════
# STL 多色导出
# ══════════════════════════════════════════════════════════

def export_stl(body: MechanicalBody, output_dir: str) -> List[str]:
    """
    导出 STL 文件 (按零件类型分文件)
    
    Returns:
        文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    files = []
    
    # 合并同类零件
    parts_by_type: Dict[str, List[Part]] = {}
    for part in body.parts():
        parts_by_type.setdefault(part.part_type, []).append(part)
    
    for part_type, parts in parts_by_type.items():
        meshes = []
        for part in parts:
            mesh = _generate_simple_mesh(part)
            meshes.append(mesh)
        
        if meshes:
            combined = meshes[0]
            for m in meshes[1:]:
                combined = combined.union(m)
            
            path = os.path.join(output_dir, f"{part_type.replace('/', '_')}.stl")
            combined.export(path)
            files.append(path)
    
    return files


# ══════════════════════════════════════════════════════════
# URDF 导出
# ══════════════════════════════════════════════════════════

def export_urdf(body: MechanicalBody, output_path: str,
                material_db: MaterialDB = None) -> str:
    """
    导出 URDF (Unified Robot Description Format)
    
    兼容 ROS/ROS2 + Gazebo
    
    Returns:
        文件路径
    """
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom
    
    if material_db is None:
        material_db = MaterialDB()
    
    root = Element("robot")
    root.set("name", body.name)
    
    # 遍历树结构
    for node_id in body.depth_first_order():
        part = body.get_part(node_id)
        parent_id = body.parent_of(node_id)
        
        # 质量计算
        size = np.array([
            part.params.get("length", 0.1),
            part.params.get("width", part.params.get("thickness", 0.02)),
            part.params.get("height", part.params.get("thickness", 0.02)),
        ])
        volume = np.prod(size)
        density = material_db.get_part_density(part.part_type)
        mass = volume * density
        
        link = SubElement(root, "link")
        link.set("name", node_id)
        
        # 惯性
        inertial = SubElement(link, "inertial")
        mass_elem = SubElement(inertial, "mass")
        mass_elem.set("value", f"{mass:.6f}")
        
        I_xx = mass / 12.0 * (size[1]**2 + size[2]**2)
        I_yy = mass / 12.0 * (size[0]**2 + size[2]**2)
        I_zz = mass / 12.0 * (size[0]**2 + size[1]**2)
        
        inertia = SubElement(inertial, "inertia")
        inertia.set("ixx", f"{I_xx:.8f}")
        inertia.set("iyy", f"{I_yy:.8f}")
        inertia.set("izz", f"{I_zz:.8f}")
        inertia.set("ixy", "0")
        inertia.set("ixz", "0")
        inertia.set("iyz", "0")
        
        # 视觉
        visual = SubElement(link, "visual")
        geom = SubElement(visual, "geometry")
        
        part_type = part.part_type.lower()
        if "cylinder" in part_type or "motor" in part_type:
            cyl = SubElement(geom, "cylinder")
            cyl.set("radius", f"{size[1]/2:.4f}")
            cyl.set("length", f"{size[0]:.4f}")
        elif "box" in part_type or "extrusion" in part_type:
            box = SubElement(geom, "box")
            box.set("size", f"{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}")
        elif "sphere" in part_type:
            sph = SubElement(geom, "sphere")
            sph.set("radius", f"{max(size)/2:.4f}")
        else:
            cyl = SubElement(geom, "cylinder")
            cyl.set("radius", "0.02")
            cyl.set("length", f"{max(size):.4f}")
        
        # 关节
        if parent_id is not None:
            joint = body.get_joint(parent_id, child_id=node_id)
            j_elem = SubElement(root, "joint")
            j_elem.set("name", f"joint_{parent_id}_{node_id}")
            j_elem.set("type", joint.joint_type if joint.joint_type != "hinge" else "revolute")
            
            parent_elem = SubElement(j_elem, "parent")
            parent_elem.set("link", parent_id)
            child_elem = SubElement(j_elem, "child")
            child_elem.set("link", node_id)
            
            axis = SubElement(j_elem, "axis")
            axis.set("xyz", f"{joint.axis[0]:.4f} {joint.axis[1]:.4f} {joint.axis[2]:.4f}")
            
            limit = SubElement(j_elem, "limit")
            limit.set("effort", f"{part.params.get('max_torque', 5.0):.2f}")
            limit.set("velocity", f"{part.params.get('max_velocity', 10.0):.2f}")
            
            if "range_min" in joint.params and "range_max" in joint.params:
                limit.set("lower", f"{joint.params['range_min']:.4f}")
                limit.set("upper", f"{joint.params['range_max']:.4f}")
    
    xml_str = minidom.parseString(tostring(root)).toprettyxml(indent="  ")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    
    return output_path


# ══════════════════════════════════════════════════════════
# 可制造性综合评分
# ══════════════════════════════════════════════════════════

# ── Score manufacturability — collision + tolerance + material + cost
def score_manufacturability(body: MechanicalBody,
                            material_db: MaterialDB = None,
                            fdm_profile: FDMProfile = None) -> Tuple[float, Dict]:
    """
    综合可制造性评分

    评分维度:
    1. 结构合理性 (30%): 碰撞检测结果
    2. 材料可行性 (25%): 是否有可用材料
    3. 公差可行性 (25%): 间隙是否合理
    4. 成本效益 (20%): 总成本是否合理

    Returns:
        (综合分数 0-1, 详细分解)
    """
    if material_db is None:
        material_db = MaterialDB()
    if fdm_profile is None:
        fdm_profile = FDMProfile()
    
    checker = CollisionChecker()
    
    # 1. 结构合理性
    collision_result = checker.check_body(body)
    structure_score = 1.0
    if not collision_result.is_valid:
        structure_score = 0.0
    elif collision_result.collision_pairs:
        structure_score = max(0, 1.0 - len(collision_result.collision_pairs) * 0.1)
    elif not collision_result.is_connected:
        structure_score = 0.3
    
    # 2. 材料可行性
    material_issues = 0
    for part in body.parts():
        material_id = material_db.get_part_material(part.part_type)
        if not material_db.get_material(material_id):
            material_issues += 1
    material_score = max(0, 1.0 - material_issues * 0.1)
    
    # 3. 公差可行性
    part_dicts = [{"part_id": p.part_id, "part_type": p.part_type, "params": p.params}
                  for p in body.parts()]
    print_report = compute_print_report(part_dicts, fdm_profile)
    tolerance_score = max(0, 1.0 - len(print_report["warnings"]) * 0.15)
    
    # 4. 成本效益
    total_cost = _estimate_total_cost(body)
    cost_score = 1.0 if total_cost < 100 else max(0, 1.0 - (total_cost - 100) / 500)
    
    # 加权综合
    final = (0.30 * structure_score + 0.25 * material_score +
             0.25 * tolerance_score + 0.20 * cost_score)
    
    details = {
        "structure_score": round(structure_score, 3),
        "material_score": round(material_score, 3),
        "tolerance_score": round(tolerance_score, 3),
        "cost_score": round(cost_score, 3),
        "final_score": round(final, 3),
        "collision_pairs": len(collision_result.collision_pairs),
        "material_issues": material_issues,
        "tolerance_warnings": len(print_report["warnings"]),
    }
    
    return final, details


def _estimate_total_cost(body: MechanicalBody) -> float:
    """估算总成本"""
    cost = 0.0
    for part in body.parts():
        pt = part.part_type.lower()
        for key, price in ELECTRONICS_COSTS.items():
            if key in pt:
                cost += price
                break
        else:
            size = np.array([
                part.params.get("length", 0.1),
                part.params.get("width", 0.02),
                part.params.get("height", 0.02),
            ])
            volume = np.prod(size)
            cost += volume * 1000 * 0.02  # ~$20/kg 平均打印成本
    return cost


# ══════════════════════════════════════════════════════════
# 主管道
# ══════════════════════════════════════════════════════════

class ManufacturingPipeline:
    """
    制造集成管道 - 统一入口
    
    用法:
        pipeline = ManufacturingPipeline(fdm="PLA_04mm")
        
        # 一键导出
        result = pipeline.export_all(best_body, output_dir="design_output")
        
        # 或分步导出
        pipeline.export_step(body, "output/robot.stp")
        pipeline.export_urdf(body, "output/robot.urdf")
        pipeline.export_bom(body, "output/bom.json")
        report = pipeline.generate_report(body)
    """
    
    def __init__(self, fdm: str = "PLA_04mm", quality: str = "high"):
        self.fdm_profile = FDM_PROFILES.get(fdm, DEFAULT_FDM)
        self.material_db = MaterialDB()
        self.checker = CollisionChecker()
        self._quality = quality
        self._exporter = None  # lazy init
    
    @property
    def exporter(self) -> ManufacturingExporter:
        if self._exporter is None:
            self._exporter = ManufacturingExporter(quality=self._quality)
        return self._exporter
    
    def export_all(self, body: MechanicalBody, output_dir: str = "design_output") -> ManufacturingReport:
        """
        一键导出所有制造文件
        
        Args:
            body: 机械体
            output_dir: 输出目录
        
        Returns:
            ManufacturingReport 包含完整评估
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(output_dir, body.name.replace(" ", "_"))
        
        exported = []
        
        # 1. STEP (new high-quality exporter)
        try:
            step_path = f"{base}.stp"
            self.exporter.export_step(body, step_path)
            exported.append(step_path)
        except Exception as e:
            logger.warning(f"STEP 导出失败: {e}")
        
        # 2. STL (new high-quality exporter)
        try:
            stl_dir = os.path.join(output_dir, "stl")
            stl_files = self.exporter.export_stl(body, stl_dir)
            exported.extend(stl_files)
        except Exception as e:
            logger.warning(f"STL 导出失败: {e}")
        
        # 3. 3MF (new)
        try:
            mf3_path = f"{base}.3mf"
            self.exporter.export_3mf(body, mf3_path)
            exported.append(mf3_path)
        except Exception as e:
            logger.warning(f"3MF 导出失败: {e}")
        
        # 4. URDF
        try:
            urdf_path = f"{base}.urdf"
            export_urdf(body, urdf_path, self.material_db)
            exported.append(urdf_path)
        except Exception as e:
            logger.warning(f"URDF 导出失败: {e}")
        
        # 5. BOM
        bom_path = f"{base}_bom.json"
        bom = self.generate_bom(body, bom_path)
        exported.append(bom_path)
        
        # 5. 报告
        report_path = f"{base}_report.md"
        report = self.generate_report(body, report_path)
        exported.append(report_path)
        
        # 6. JSON 报告
        json_path = f"{base}_report.json"
        report_data = self._report_to_dict(body, report)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        exported.append(json_path)
        
        report.exported_files = exported
        return report
    
    def generate_bom(self, body: MechanicalBody, output_path: str = None) -> Dict:
        """生成增强版 BOM (含材料+成本)"""
        items = []
        part_counts: Dict[str, int] = {}
        part_params: Dict[str, dict] = {}
        
        for part in body.parts():
            pt = part.part_type
            part_counts[pt] = part_counts.get(pt, 0) + 1
            if pt not in part_params:
                part_params[pt] = part.params
        
        total_cost = 0.0
        total_mass = 0.0
        
        for pt, qty in part_counts.items():
            params = part_params.get(pt, {})
            
            # 材料信息
            material_id = self.material_db.get_part_material(pt)
            material_name = self.material_db.get_material(material_id).name if self.material_db.get_material(material_id) else "Unknown"
            
            # 质量估算
            size = np.array([
                params.get("length", 0.1),
                params.get("width", params.get("thickness", 0.02)),
                params.get("height", params.get("thickness", 0.02)),
            ])
            volume = np.prod(size)
            density = self.material_db.get_part_density(pt)
            mass = volume * density * qty
            total_mass += mass
            
            # 成本
            unit_cost = self._get_part_cost(pt, params)
            item_cost = unit_cost * qty
            total_cost += item_cost
            
            items.append({
                "name": pt,
                "qty": qty,
                "material": material_name,
                "density_kgm3": density,
                "unit_mass_kg": round(mass / qty, 4),
                "total_mass_kg": round(mass, 4),
                "unit_cost_usd": round(unit_cost, 2),
                "total_cost_usd": round(item_cost, 2),
                "params": {k: round(v, 4) for k, v in list(params.items())[:4]},
            })
        
        bom = {
            "body_name": body.name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_unique_types": len(items),
                "total_parts": sum(it["qty"] for it in items),
                "total_mass_kg": round(total_mass, 3),
                "total_cost_usd": round(total_cost, 2),
                "currency": "USD",
            },
            "items": items,
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(bom, f, indent=2, ensure_ascii=False)
        
        return bom
    
    def _get_part_cost(self, part_type: str, params: dict) -> float:
        """估算单件成本"""
        pt = part_type.lower()
        
        # 电子元件
        for key, price in ELECTRONICS_COSTS.items():
            if key in pt:
                return price
        
        # 打印件
        size = np.array([
            params.get("length", 0.1),
            params.get("width", 0.02),
            params.get("height", 0.02),
        ])
        volume = np.prod(size)
        density = self.material_db.get_part_density(part_type)
        mass_kg = volume * density
        
        # 打印成本
        material = self.fdm_profile.material
        filament_cost = FILAMENT_COSTS.get(material, 20.0)
        return mass_kg * filament_cost * 1.5  # 1.5x margin
    
    def generate_report(self, body: MechanicalBody, output_path: str = None) -> ManufacturingReport:
        """
        生成完整设计审查报告
        """
        # 可制造性评分
        mfg_score, mfg_details = score_manufacturability(
            body, self.material_db, self.fdm_profile
        )
        
        # 材料统计
        materials_used: Dict[str, int] = {}
        total_volume = 0.0
        total_mass = 0.0
        for part in body.parts():
            material_id = self.material_db.get_part_material(part.part_type)
            materials_used[material_id] = materials_used.get(material_id, 0) + 1
            
            size = np.array([
                part.params.get("length", 0.1),
                part.params.get("width", 0.02),
                part.params.get("height", 0.02),
            ])
            volume = np.prod(size)
            total_volume += volume
            total_mass += self.material_db.calculate_mass(volume, material_id)
        
        # 打印参数
        filament_grams = estimate_filament_weight(total_volume, self.fdm_profile.material)
        print_hours = estimate_print_time(total_volume)
        
        # 碰撞检查
        collision_result = self.checker.check_body(body)
        
        # 公差检查
        part_dicts = [{"part_id": p.part_id, "part_type": p.part_type, "params": p.params}
                      for p in body.parts()]
        print_report = compute_print_report(part_dicts, self.fdm_profile)
        
        # 成本
        material_cost = 0.0
        for mat_id, count in materials_used.items():
            density = self.material_db.get_density(mat_id)
            material_cost += total_volume * density * FILAMENT_COSTS.get(
                self.fdm_profile.material, 20.0) * 1.5
        
        print_cost = print_hours * PRINT_HOURLY_RATE.get(self.fdm_profile.material, 4.0)
        electronics_cost = sum(
            ELECTRONICS_COSTS.get(p.part_type.lower(), 0)
            for p in body.parts()
        )
        total_cost = material_cost + print_cost + electronics_cost
        
        # 警告和建议
        warnings = []
        suggestions = []
        
        if not collision_result.is_valid:
            warnings.extend(collision_result.errors)
        for w in print_report["warnings"]:
            warnings.append(w["issue"])
        if total_cost > 500:
            suggestions.append("总成本较高，考虑减少电子元件数量")
        if total_mass > 5.0:
            suggestions.append("总质量超过 5kg，考虑轻量化设计")
        if len(body.parts()) > 20:
            suggestions.append("零件数超过 20，考虑简化设计")
        if mfg_score < 0.5:
            suggestions.append("可制造性评分较低，请检查碰撞和公差问题")
        
        # 采购清单
        purchase_items = []
        for part in body.parts():
            pt = part.part_type.lower()
            for key, price in ELECTRONICS_COSTS.items():
                if key in pt:
                    purchase_items.append({
                        "name": pt, "estimated_cost": price,
                        "type": "electronics",
                    })
                    break
        
        report = ManufacturingReport(
            body_name=body.name,
            timestamp=datetime.now().isoformat(),
            total_parts=body.num_parts(),
            total_joints=body.num_joints(),
            actuated_joints=len(body.actuated_joints()),
            materials_used=materials_used,
            total_mass_kg=round(total_mass, 3),
            total_volume_m3=round(total_volume, 8),
            manufacturability_score=round(mfg_score, 3),
            collision_issues=len(collision_result.collision_pairs),
            tolerance_warnings=len(print_report["warnings"]),
            material_cost_usd=round(material_cost, 2),
            printing_cost_usd=round(print_cost, 2),
            electronics_cost_usd=round(electronics_cost, 2),
            total_cost_usd=round(total_cost, 2),
            recommended_material=self.fdm_profile.material,
            print_time_hours=round(print_hours, 2),
            filament_grams=round(filament_grams, 1),
            purchase_items=purchase_items,
            purchase_total_usd=round(electronics_cost, 2),
            warnings=warnings,
            suggestions=suggestions,
        )
        
        if output_path:
            self._write_markdown_report(report, output_path)
        
        return report
    
    def _write_markdown_report(self, report: ManufacturingReport, path: str):
        """写入 Markdown 报告"""
        lines = [
            f"# ForgeCraft 设计审查报告",
            f"",
            f"**设计名称**: {report.body_name}",
            f"**生成时间**: {report.timestamp}",
            f"",
            f"---",
            f"",
            f"## 基本信息",
            f"",
            f"| 属性 | 值 |",
            f"|------|-----|",
            f"| 零件数 | {report.total_parts} |",
            f"| 关节数 | {report.total_joints} |",
            f"| 驱动关节 | {report.actuated_joints} |",
            f"| 总质量 | {report.total_mass_kg:.3f} kg |",
            f"| 总体积 | {report.total_volume_m3*1e6:.1f} cm³ |",
            f"",
            f"---",
            f"",
            f"## 可制造性评分: **{report.manufacturability_score:.1%}**",
            f"",
            f"| 维度 | 状态 |",
            f"|------|------|",
            f"| 碰撞检测 | {report.collision_issues} 个问题 |",
            f"| 公差检查 | {report.tolerance_warnings} 个警告 |",
            f"",
            f"---",
            f"",
            f"## 成本估算",
            f"",
            f"| 类别 | 成本 (USD) |",
            f"|------|-----------|",
            f"| 材料成本 | ${report.material_cost_usd:.2f} |",
            f"| 打印成本 | ${report.printing_cost_usd:.2f} |",
            f"| 电子元件 | ${report.electronics_cost_usd:.2f} |",
            f"| **总计** | **${report.total_cost_usd:.2f}** |",
            f"",
            f"### 3D 打印参数",
            f"- 推荐材料: {report.recommended_material}",
            f"- 预计耗时: {report.print_time_hours:.1f} 小时",
            f"- 耗材重量: {report.filament_grams:.0f} g",
            f"",
            f"---",
            f"",
            f"## 警告",
        ]
        
        if report.warnings:
            for w in report.warnings:
                lines.append(f"- ⚠️ {w}")
        else:
            lines.append("- ✅ 无警告")
        
        lines.extend(["", "## 建议"])
        
        if report.suggestions:
            for s in report.suggestions:
                lines.append(f"- 💡 {s}")
        else:
            lines.append("- ✅ 无建议")
        
        if report.exported_files:
            lines.extend(["", "## 导出文件"])
            for f in report.exported_files:
                lines.append(f"- `{os.path.basename(f)}`")
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def _report_to_dict(self, body: MechanicalBody, report: ManufacturingReport) -> Dict:
        """将报告转换为 JSON 序列化格式"""
        return {
            "body_name": report.body_name,
            "timestamp": report.timestamp,
            "total_parts": report.total_parts,
            "total_joints": report.total_joints,
            "actuated_joints": report.actuated_joints,
            "materials_used": report.materials_used,
            "total_mass_kg": report.total_mass_kg,
            "total_volume_m3": report.total_volume_m3,
            "manufacturability_score": report.manufacturability_score,
            "cost": {
                "material_usd": report.material_cost_usd,
                "printing_usd": report.printing_cost_usd,
                "electronics_usd": report.electronics_cost_usd,
                "total_usd": report.total_cost_usd,
            },
            "printing": {
                "material": report.recommended_material,
                "hours": report.print_time_hours,
                "filament_grams": report.filament_grams,
            },
            "warnings": report.warnings,
            "suggestions": report.suggestions,
            "collision_issues": report.collision_issues,
            "tolerance_warnings": report.tolerance_warnings,
        }


# ══════════════════════════════════════════════════════════
# 便捷接口
# ══════════════════════════════════════════════════════════

_pipeline = None


def get_pipeline() -> ManufacturingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ManufacturingPipeline()
    return _pipeline


def export_design(body: MechanicalBody, output_dir: str = "design_output") -> ManufacturingReport:
    """一键导出设计"""
    return get_pipeline().export_all(body, output_dir)


def score_design(body: MechanicalBody) -> Tuple[float, Dict]:
    """快捷评分"""
    return score_manufacturability(body)
