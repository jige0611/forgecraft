# ═══════════════════════════════════════════════════════════════
#  ForgeCraft URDF → Gazebo 验证流程
#  ═══════════════════════════════════════════════════════════════
#
#  功能：
#  - MJCF模型转换为标准URDF格式
#  - URDF验证（XML Schema + 物理约束）
#  - Gazebo兼容性检查
#  - ROS2集成准备
#
#  使用：
#    from urdf_export import URDFExporter, GazeboValidator
#    exporter = URDFExporter()
#    urdf_str = exporter.export(body, catalog)
#    validator = GazeboValidator(urdf_str)
#    report = validator.validate()
# ═══════════════════════════════════════════════════════════════

import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("ForgeCraft.URDF")


@dataclass
class URDFLink:
    """URDF Link"""
    name: str
    visual: Dict = None
    collision: Dict = None
    inertial: Dict = None
    
    def __post_init__(self):
        if self.visual is None:
            self.visual = {}
        if self.collision is None:
            self.collision = {}
        if self.inertial is None:
            self.inertial = {}


@dataclass 
class URDFJoint:
    """URDF Joint"""
    name: str
    parent: str
    child: str
    joint_type: str = "revolute"  # revolute, prismatic, fixed
    axis: List[float] = field(default_factory=lambda: [0, 0, 1])
    origin_xyz: List[float] = field(default_factory=lambda: [0, 0, 0])
    origin_rpy: List[float] = field(default_factory=lambda: [0, 0, 0])
    limit_lower: float = -3.14159
    limit_upper: float = 3.14159
    limit_effort: float = 100.0
    limit_velocity: float = 10.0


@dataclass
class ValidationReport:
    """验证报告"""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    
    def add_error(self, msg: str):
        self.valid = False
        self.errors.append(msg)
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    def add_info(self, msg: str):
        self.info.append(msg)
    
    @property
    def summary(self) -> str:
        lines = [
            f"Valid: {'YES' if self.valid else 'NO'}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings[:5]:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)


class URDFExporter:
    """MJCF → URDF 导出器"""
    
    # 材质颜色映射
    MATERIAL_COLORS = {
        "body": "0.2 0.4 0.8 1.0",      # 蓝色主体
        "wheel": "0.15 0.15 0.15 1.0",   # 深灰轮子
        "motor": "0.8 0.2 0.2 1.0",       # 红色电机
        "link": "0.6 0.6 0.6 1.0",        # 灰色连杆
        "foot": "0.2 0.6 0.2 1.0",        # 绿色足端
        "default": "0.7 0.7 0.7 1.0",
    }
    
    def __init__(self):
        self.links: List[URDFLink] = []
        self.joints: List[URDFJoint] = []
        self.materials: Dict[str, str] = {}
        self._link_names: set = set()  # 跟踪已创建的link名称
    
    def export_body(self, body, catalog: Dict) -> str:
        """将MechanicalBody导出为URDF字符串"""
        from forgecraft.simulation.builder import build_mjcf_model
        
        # 先构建MJCF获取结构信息
        mjcf_result = build_mjcf_model(body, catalog)
        
        if isinstance(mjcf_result, tuple):
            xml_string, joint_map, motor_map, _ = mjcf_result
        else:
            xml_string = mjcf_result
            joint_map = {}
            motor_map = {}
        
        # 解析MJCF并转换为URDF
        return self._mjcf_to_urdf(xml_string, body, joint_map, motor_map)
    
    def _mjcf_to_urdf(self, mjcf_xml: str, body, joint_map: dict, motor_map: dict) -> str:
        """MJCF XML → URDF XML 转换"""
        try:
            root = ET.fromstring(mjcf_xml)
        except ET.ParseError as e:
            logger.error(f"Failed to parse MJCF XML: {e}")
            return ""
        
        robot_name = getattr(body, 'name', 'forgecraft_robot')
        
        # 创建URDF根元素
        urdf_root = ET.Element("robot", name=robot_name)
        
        # 添加材料定义
        self._add_materials(urdf_root)
        
        # 解析worldbody创建links
        worldbody = root.find("worldbody")
        if worldbody is not None:
            self._parse_worldbody(worldbody, urdf_root, "base_link")
        
        # 只解析顶层bodies（递归方法会处理子body）
        for body_elem in root.findall("worldbody/body"):
            self._parse_body_with_joints(body_elem, urdf_root)
        
        # 也检查直接在mujoco下的顶层body
        if not self.links:
            for body_elem in root.findall("body"):
                self._parse_body_with_joints(body_elem, urdf_root)
        
        # 添加地面（可选）
        self._add_ground_plane(urdf_root)
        
        # 美化输出
        rough_string = ET.tostring(urdf_root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # 移除空行
        lines = [l for l in pretty_xml.split('\n') if l.strip()]
        return '\n'.join(lines)
    
    def _add_materials(self, root):
        """添加材质定义"""
        for mat_name, color in self.MATERIAL_COLORS.items():
            material = ET.SubElement(root, "material", name=mat_name)
            color_elem = ET.SubElement(material, "color", rgba=color)
    
    def _parse_worldbody(self, worldbody, urdf_root, parent_link: str):
        """解析worldbody中的几何体"""
        for geom in worldbody.findall("geom"):
            geom_type = geom.get("type", "box")
            name = geom.get("name", f"{parent_link}_geom")
            
            size = self._parse_size(geom.get("size", "0.05 0.05 0.05"))
            pos = geom.get("pos", "0 0 0").split()
            
            link = URDFLink(
                name=f"{parent_link}_visual",
                visual=self._create_visual(name, geom_type, size),
                collision=self._create_collision(geom_type, size),
                inertial=self._create_inertial(size)
            )
            
            self.links.append(link)
            self._add_link_to_urdf(urdf_root, link)
    
    def _parse_body_with_joints(self, body_elem, urdf_root):
        """解析body及其内部joints，传递正确的parent上下文"""
        body_name = body_elem.get("name", "link")
        
        # 创建link
        link = URDFLink(name=body_name)
        for geom in body_elem.findall("geom"):
            geom_type = geom.get("type", "box")
            size = self._parse_size(geom.get("size", "0.05"))
            link.visual = self._create_visual(f"{body_name}_vis", geom_type, size)
            link.collision = self._create_collision(geom_type, size)
            link.inertial = self._create_inertial(size)
        
        self.links.append(link)
        self._link_names.add(body_name)
        self._add_link_to_urdf(urdf_root, link)
        
        # 解析该body内的joints (parent=当前body_name)
        for joint_elem in body_elem.findall("joint"):
            self._parse_joint(joint_elem, urdf_root, parent_body_name=body_name)
        
        # 递归处理子body
        for child_body in body_elem.findall("body"):
            self._parse_body_with_joints(child_body, urdf_root)
    
    def _parse_joint(self, joint_elem, urdf_root, parent_body_name: str = None):
        """解析joint元素"""
        joint_name = joint_elem.get("name", "joint")
        joint_type_mjcf = joint_elem.get("type", "hinge")
        
        # MJCF→URDF类型映射
        type_map = {
            "hinge": "revolute",
            "slide": "prismatic",
            "ball": "continuous",
            "free": "fixed",
        }
        
        # 确定parent: 优先使用传入的parent_body_name，否则用第一个link或base_link
        if parent_body_name and parent_body_name in self._link_names:
            parent_link = parent_body_name
        elif self.links:
            parent_link = self.links[0].name
        else:
            parent_link = "base_link"
        
        # 确定child: 使用joint名称作为child link名（简化）
        child_link = f"{joint_name}_link"
        if child_link not in self._link_names:
            child_link = self._get_closest_link_name(joint_name)
        
        urdf_joint = URDFJoint(
            name=joint_name,
            parent=parent_link,
            child=child_link,
            joint_type=type_map.get(joint_type_mjcf, "revolute"),
            axis=self._parse_vec(joint_elem.get("axis", "0 0 1")),
            origin_xyz=self._parse_vec(joint_elem.get("pos", "0 0 0")),
        )
        
        # 范围限制
        range_elem = joint_elem.find("range")
        if range_elem is not None:
            limits = range_elem.text.split() if range_elem.text else ["-3.14", "3.14"]
            urdf_joint.limit_lower = float(limits[0]) if len(limits) > 0 else -3.14
            urdf_joint.limit_upper = float(limits[1]) if len(limits) > 1 else 3.14
        
        self.joints.append(urdf_joint)
        self._add_joint_to_urdf(urdf_root, urdf_joint)
    
    def _add_link_to_urdf(self, root, link: URDFLink):
        """添加link到URDF"""
        link_elem = ET.SubElement(root, "link", name=link.name)
        
        # Inertial
        if link.inertial:
            inertial = ET.SubElement(link_elem, "inertial")
            mass = ET.SubElement(inertial, "mass", value=str(link.inertial.get('mass', 1.0)))
            inertia = ET.SubElement(inertial, "inertia",
                                   ixx=str(link.inertial.get('ixx', 0.001)),
                                   ixy="0", ixz="0",
                                   iyy=str(link.inertial.get('iyy', 0.001)),
                                   iyz="0", izz=str(link.inertial.get('izz', 0.001)))
            origin = ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        
        # Visual
        if link.visual:
            visual = ET.SubElement(link_elem, "visual")
            origin = ET.SubElement(visual, "origin", 
                                  xyz=link.visual.get('xyz', '0 0 0'),
                                  rpy=link.visual.get('rpy', '0 0 0'))
            geometry = ET.SubElement(visual, "geometry")
            geom_type = link.visual.get('type', 'box')
            geom_data = link.visual.get('data', {})
            
            if geom_type == 'box':
                ET.SubElement(geometry, "size", **geom_data)
            elif geom_type == 'cylinder':
                ET.SubElement(geometry, "cylinder", **geom_data)
            elif geom_type == 'sphere':
                ET.SubElement(geometry, "sphere", **geom_data)
            elif geom_type == 'mesh':
                ET.SubElement(geometry, "mesh", **geom_data)
            
            material = ET.SubElement(visual, "material", 
                                    name=link.visual.get('material', 'default'))
        
        # Collision
        if link.collision:
            collision = ET.SubElement(link_elem, "collision")
            origin = ET.SubElement(collision, "origin",
                                  xyz=link.collision.get('xyz', '0 0 0'),
                                  rpy=link.collision.get('rpy', '0 0 0'))
            geometry = ET.SubElement(collision, "geometry")
            geom_type = link.collision.get('type', 'box')
            geom_data = link.collision.get('data', {})
            
            if geom_type == 'box':
                ET.SubElement(geometry, "box", **geom_data)
            elif geom_type == 'cylinder':
                ET.SubElement(geometry, "cylinder", **geom_data)
            elif geom_type == 'sphere':
                ET.SubElement(geometry, "sphere", **geom_data)
    
    def _add_joint_to_urdf(self, root, joint: URDFJoint):
        """添加joint到URDF"""
        joint_elem = ET.SubElement(root, "joint", name=joint.name, type=joint.joint_type)
        
        ET.SubElement(joint_elem, "parent", link=joint.parent)
        ET.SubElement(joint_elem, "child", link=joint.child)
        ET.SubElement(joint_elem, "origin", 
                     xyz=' '.join(map(str, joint.origin_xyz)),
                     rpy=' '.join(map(str, joint.origin_rpy)))
        ET.SubElement(joint_elem, "axis", xyz=' '.join(map(str, joint.axis)))
        
        if joint.joint_type != "fixed":
            limit = ET.SubElement(joint_elem, "limit",
                                 lower=str(joint.limit_lower),
                                 upper=str(joint.limit_upper),
                                 effort=str(joint.limit_effort),
                                 velocity=str(joint.limit_velocity))
    
    def _add_ground_plane(self, root):
        """添加地面平面"""
        ground = ET.SubElement(root, "link", name="ground_plane")
        visual = ET.SubElement(ground, "visual")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "plane", normal="0 0 1", size="10 10")
        material = ET.SubElement(visual, "material", name="ground_material")
        ET.SubElement(material, "color", rgba="0.8 0.8 0.8 1.0")
        
        # 固定关节连接地面
        fixed = ET.SubElement(root, "joint", name="ground_fix", type="fixed")
        ET.SubElement(fixed, "parent", link="world")
        ET.SubElement(fixed, "child", link="ground_plane")
        ET.SubElement(fixed, "origin", xyz="0 0 0", rpy="0 0 0")
    
    def _create_visual(self, name: str, geom_type: str, size: list) -> Dict:
        """创建visual数据"""
        data = {}
        if geom_type == "box":
            half_sizes = [float(s) * 2 for s in size]
            data = {"size": f"{half_sizes[0]} {half_sizes[1]} {half_sizes[2]}"}
        elif geom_type == "cylinder":
            radius = float(size[0]) if len(size) > 0 else 0.05
            half_len = float(size[1]) * 2 if len(size) > 1 else 0.1
            data = {"radius": str(radius), "length": str(half_len)}
        elif geom_type == "sphere":
            radius = float(size[0]) if size else 0.05
            data = {"radius": str(radius)}
        elif geom_type == "mesh":
            data = {"filename": f"package://forgecraft_models/meshes/{name}.stl"}
        
        return {
            "type": geom_type,
            "data": data,
            "xyz": "0 0 0",
            "rpy": "0 0 0",
            "material": "default"
        }
    
    def _create_collision(self, geom_type: str, size: list) -> Dict:
        """创建collision数据"""
        vis = self._create_visual("", geom_type, size)
        vis["material"] = None  # 碰撞不需要材质
        return vis
    
    def _get_closest_link_name(self, joint_name: str) -> str:
        """找到最接近的link名称"""
        if not self._link_names:
            return "base_link"
        # 简化：返回第一个link
        return list(self._link_names)[0] if self._link_names else "base_link"
    
    def _create_inertial(self, size: list) -> Dict:
        """估算惯性参数"""
        # 更合理的质量估算：基于尺寸，假设塑料/铝合金密度 ~500 kg/m³
        volume = 1.0
        if size:
            for s in size:
                volume *= abs(float(s)) * 2
        # 限制在合理范围：0.01kg ~ 10kg
        mass = max(min(volume * 500, 10.0), 0.01)
        
        return {
            "mass": round(mass, 3),
            "ixx": round(mass * 0.001, 6),
            "iyy": round(mass * 0.001, 6),
            "izz": round(mass * 0.001, 6),
        }
    
    def _parse_size(self, size_str: str) -> list:
        """解析size属性"""
        return [float(x) for x in size_str.split()]
    
    def _parse_vec(self, vec_str: str) -> list:
        """解析向量"""
        parts = vec_str.split()
        result = [float(x) for x in parts]
        while len(result) < 3:
            result.append(0.0)
        return result[:3]


class GazeboValidator:
    """Gazebo兼容性验证器"""
    
    def __init__(self, urdf_string: str):
        self.urdf_string = urdf_string
        self.report = ValidationReport()
    
    def validate(self) -> ValidationReport:
        """执行完整验证"""
        self._validate_xml_syntax()
        self._validate_required_elements()
        self._validate_links()
        self._validate_joints()
        self._validate_physics_constraints()
        self._check_gazebo_compatibility()
        
        return self.report
    
    def _validate_xml_syntax(self):
        """验证XML语法"""
        try:
            ET.fromstring(self.urdf_string)
            self.report.add_info("XML syntax: Valid")
        except ET.ParseError as e:
            self.report.add_error(f"XML parse error: {e}")
    
    def _validate_required_elements(self):
        """检查必需元素"""
        try:
            root = ET.fromstring(self.urdf_string)
            
            if root.tag != "robot":
                self.report.add_error(f"Root element must be <robot>, got <{root.tag}>")
            else:
                robot_name = root.get("name", "")
                if not robot_name:
                    self.report.add_warning("Robot has no name attribute")
                else:
                    self.report.add_info(f"Robot name: {robot_name}")
                
                links = root.findall("link")
                joints = root.findall("joint")
                
                if not links:
                    self.report.add_error("No <link> elements found")
                else:
                    self.report.add_info(f"Links: {len(links)}")
                
                if not joints:
                    self.report.add_warning("No <joint> elements found (static model?)")
                else:
                    self.report.add_info(f"Joints: {len(joints)}")
                    
        except Exception as e:
            pass  # 已在语法检查中处理
    
    def _validate_links(self):
        """验证links"""
        try:
            root = ET.fromstring(self.urdf_string)
            link_names = set()
            
            for link in root.findall("link"):
                name = link.get("name", "")
                if not name:
                    self.report.add_error("Found unnamed link")
                    continue
                    
                if name in link_names:
                    self.report.add_error(f"Duplicate link name: {name}")
                link_names.add(name)
                
                # 检查inertial
                inertial = link.find("inertial")
                if inertial is None:
                    self.report.add_warning(f"Link '{name}' missing inertial properties")
                else:
                    mass = inertial.find("mass")
                    if mass is None:
                        self.report.add_warning(f"Link '{name}' missing mass")
                    else:
                        m_val = float(mass.get("value", 0))
                        if m_val <= 0:
                            self.report.add_error(f"Link '{name}' has invalid mass: {m_val}")
                
                # 检查visual
                visual = link.find("visual")
                if visual is None:
                    self.report.add_warning(f"Link '{name}' has no visual (invisible)")
                    
        except Exception:
            pass
    
    def _validate_joints(self):
        """验证joints"""
        try:
            root = ET.fromstring(self.urdf_string)
            link_names = set(l.get("name", "") for l in root.findall("link") if l.get("name"))
            joint_names = set()
            
            for joint in root.findall("joint"):
                jname = joint.get("name", "")
                if not jname:
                    self.report.add_error("Found unnamed joint")
                    continue
                
                if jname in joint_names:
                    self.report.add_error(f"Duplicate joint name: {jname}")
                joint_names.add(jname)
                
                parent = joint.find("parent")
                child = joint.find("child")
                
                if parent is None or child is None:
                    self.report.add_error(f"Joint '{jname}' missing parent/child")
                    continue
                
                parent_link = parent.get("link", "")
                child_link = child.get("link", "")
                
                if parent_link and parent_link not in link_names:
                    self.report.add_error(f"Joint '{jname}' references unknown parent: {parent_link}")
                    
                if child_link and child_link not in link_names:
                    self.report.add_error(f"Joint '{jname}' references unknown child: {child_link}")
                
                # 检查limit
                jtype = joint.get("type", "")
                if jtype not in ("fixed", "continuous"):
                    limit = joint.find("limit")
                    if limit is None:
                        self.report.add_warning(f"Joint '{jname}' ({jtype}) missing limits")
                        
        except Exception:
            pass
    
    def _validate_physics_constraints(self):
        """物理约束验证"""
        try:
            root = ET.fromstring(self.urdf_string)
            
            total_mass = 0.0
            for link in root.findall("link"):
                inertial = link.find("inertial")
                if inertial is not None:
                    mass = inertial.find("mass")
                    if mass is not None:
                        total_mass += float(mass.get("value", 0))
            
            if total_mass > 0:
                self.report.add_info(f"Total estimated mass: {total_mass:.2f} kg")
                
                if total_mass > 100:
                    self.report.add_warning(f"Very heavy robot ({total_mass:.1f} kg)")
                elif total_mass < 0.01:
                    self.report.add_warning(f"Very light robot ({total_mass*1000:.1f} g)")
            
            # 检查关节范围
            for joint in root.findall("joint"):
                limit = joint.find("limit")
                if limit is not None:
                    lower = float(limit.get("lower", 0))
                    upper = float(limit.get("upper", 0))
                    if lower >= upper:
                        self.report.add_error(f"Joint '{joint.get('name')}': lower >= upper limit")
                        
        except Exception:
            pass
    
    def _check_gazebo_compatibility(self):
        """Gazebo特定兼容性检查"""
        gazebo_issues = [
            "Ensure all meshes have proper STL files in package path",
            "Check transmission elements for ROS control integration",
            "Verify friction coefficients in Gazebo plugin config",
            "Consider adding <gazebo> tags for sensor simulation",
        ]
        
        self.report.add_info("Gazebo compatibility notes:")
        for issue in gazebo_issues:
            self.report.add_info(f"  - {issue}")


def export_robot_to_urdf(body, catalog, output_path: str = None) -> Tuple[str, ValidationReport]:
    """
    完整导出流程：生成URDF + 验证
    
    Returns:
        (urdf_string, validation_report)
    """
    exporter = URDFExporter()
    urdf_string = exporter.export_body(body, catalog)
    
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(urdf_string)
        logger.info(f"URDF exported to: {path}")
    
    # 验证
    validator = GazeboValidator(urdf_string)
    report = validator.validate()
    
    return urdf_string, report


if __name__ == "__main__":
    # 测试导出流程
    print("="*60)
    print("  URDF Export & Validation Test")
    print("="*60 + "\n")
    
    from locomotion_templates import LocomotionTemplateGenerator
    from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
    
    catalog = load_catalog()
    gen = LocomotionTemplateGenerator(seed=42)
    pop = gen.generate_population(size=1)
    body = pop[0]
    
    print(f"Exporting robot: {body.name}")
    urdf_str, report = export_robot_to_urdf(body, catalog, "test_output/test_robot.urdf")
    
    print(f"\n{report.summary}")
    print(f"\nURDF length: {len(urdf_str)} chars")
