from typing import Dict, List, Optional, Tuple
import logging
import os
import sys

import numpy as np

from forgecraft.core.catalog import DEFAULT_CATALOG
from forgecraft.core.morphology import MechanicalBody, Part
from forgecraft.config import PartSpec

# 提高递归深度限制，防止深层形态图构建时溢出
sys.setrecursionlimit(10000)

_logger = logging.getLogger(__name__)

__all__ = ["build_mjcf_model", "_part_geom", "_joint_xml", "_motor_actuator",
           "build_mjcf_with_convex_collision"]


# ── tendon 用计数器 ───────────────────────────────────────
_tendon_counter = [0]


# ── Part geometry to MJCF XML element (box/cylinder/sphere based on part_type)
# mesh_file: 可选 STL 路径，提供时使用 <geom type="mesh"> 代替简单原语
def _part_geom(part: Part, spec: PartSpec, has_touch: bool = False, mesh_file: str = None) -> str:
    if mesh_file and os.path.isfile(mesh_file):
        mass = part.params.get("mass", spec.mass)
        friction = part.params.get("friction_primary", spec.friction[0] if spec.friction else 0.6)
        mesh_name = os.path.splitext(os.path.basename(mesh_file))[0]
        rgba = " ".join(map(str, spec.color))
        geom = f'<geom type="mesh" mesh="{mesh_name}" rgba="{rgba}" mass="{mass:.4f}"'
        if friction != 0.6:
            geom = geom.replace('"/>', f'" friction="{friction:.2f} 0.01 0.01"/>')
        if has_touch:
            geom = geom.replace('"/>', f'" name="touch_{part.part_id}"/>')
        if not geom.endswith('/>'):
            geom += '/>'
        return geom
    length = part.params.get("length", 0.1)
    radius = part.params.get("radius", 0.03)
    thickness = part.params.get("thickness", 0.02)
    width = part.params.get("width", thickness)
    height = part.params.get("height", thickness)
    mass = part.params.get("mass", spec.mass)
    friction = part.params.get("friction_primary", spec.friction[0] if spec.friction else 0.6)
    density = part.params.get("density", spec.density)

    if spec.shape == "box":
        size_str = f"{length/2:.4f} {height:.4f} {width:.4f}"
        geom = f'<geom type="box" size="{size_str}" rgba="{" ".join(map(str, spec.color))}" mass="{mass:.4f}"'
    elif spec.shape == "cylinder":
        geom = f'<geom type="cylinder" size="{radius:.4f} {length/2:.4f}" rgba="{" ".join(map(str, spec.color))}" mass="{mass:.4f}"'
    elif spec.shape == "sphere":
        geom = f'<geom type="sphere" size="{radius:.4f}" rgba="{" ".join(map(str, spec.color))}" mass="{mass:.4f}"'
    elif spec.shape == "hollow_cylinder":
        outer = part.params.get("outer_diameter", 0.04) / 2.0
        l = part.params.get("length", 0.2)
        geom = f'<geom type="cylinder" size="{outer:.4f} {l/2:.4f}" rgba="{" ".join(map(str, spec.color))}" mass="{mass:.4f}"'
    elif spec.shape == "helical_spring":
        coil_d = part.params.get("coil_diameter", 0.03) / 2.0
        free_len = part.params.get("free_length", length)
        geom = f'<geom type="cylinder" size="{coil_d:.4f} {free_len/2:.4f}" rgba="{" ".join(map(str, spec.color))}" mass="{mass:.4f}"'
    elif spec.shape == "hemisphere_shell":
        r = part.params.get("radius", radius)
        geom = f'<geom type="sphere" size="{r:.4f}" rgba="{" ".join(map(str, spec.color))}" mass="{mass:.4f}"'
    else:
        size_str = f"{length/2:.4f} {thickness:.4f} {length/2:.4f}"
        geom = f'<geom type="box" size="{size_str}" rgba="{" ".join(map(str, spec.color))}" mass="{mass:.4f}"'

    if friction != 0.6:
        geom = geom.replace('"/>', f'" friction="{friction:.2f} 0.01 0.01"/>')
    if has_touch:
        geom = geom.replace('"/>', f'"')
        geom += f' name="touch_{part.part_id}"'
    if not geom.endswith('/>'):
        geom += '/>'
    return geom


def _joint_xml(jname: str, joint_type: str, axis: np.ndarray, params: dict, parent_pos: np.ndarray, child_pos: np.ndarray) -> Optional[str]:
    if joint_type == "fixed":
        return None

    rel_pos = child_pos - parent_pos
    pos_str = f"{rel_pos[0]:.4f} {rel_pos[1]:.4f} {rel_pos[2]:.4f}"
    axis_str = f"{axis[0]:.4f} {axis[1]:.4f} {axis[2]:.4f}"

    range_str = ""
    if "range_min" in params and "range_max" in params:
        range_str = f' range="{params["range_min"]:.4f} {params["range_max"]:.4f}"'

    damping = params.get("damping", 0.5)

    if joint_type == "hinge":
        return f'<joint name="{jname}" type="hinge" pos="{pos_str}" axis="{axis_str}"{range_str} damping="{damping:.4f}"/>'
    elif joint_type == "ball":
        return f'<joint name="{jname}" type="ball" pos="{pos_str}" damping="{damping:.4f}"/>'
    elif joint_type == "slide":
        return f'<joint name="{jname}" type="slide" pos="{pos_str}" axis="{axis_str}"{range_str} damping="{damping:.4f}"/>'
    elif joint_type == "free":
        return f'<joint name="{jname}" type="free" pos="{pos_str}"/>'
    else:
        _logger.warning("不支持的关节类型 '%s' (关节: %s)，已跳过。支持的: hinge/ball/slide/free/fixed", joint_type, jname)
        return None


# ── Motor actuator XML: position/velocity/torque motor per actuated joint
def _motor_actuator(joint_name: str, params: dict) -> str:
    torque = params.get("max_torque", 5.0)
    velocity = params.get("max_velocity", 10.0)
    return f'<motor name="act_{joint_name}" joint="{joint_name}" gear="1" ctrllimited="true" ctrlrange="-1 1" forcelimited="true" forcerange="{-torque:.4f} {torque:.4f}"/>'


def _spring_tendon(body: MechanicalBody, node_id: str, part: Part, spec: PartSpec) -> Optional[str]:
    """为 helical_spring 零件生成 MuJoCo tendon 元素，模拟弹性储能。"""
    _tendon_counter[0] += 1
    tn = _tendon_counter[0]
    stiffness = part.params.get("stiffness", 1000.0)
    damping = part.params.get("damping", 10.0)
    max_deformation = part.params.get("max_deformation", 0.05)
    free_len = part.params.get("free_length", part.params.get("length", 0.15))

    child = body.children_of(node_id)
    child_body = child[0] if child else None
    child_name = child_body if child_body else f"spring_anchor_{tn}"
    parent_name = node_id

    if not child_body:
        return None

    return (
        f'\n    <tendon name="spring_{tn}">\n'
        f'      <spatial name="sp_{tn}" limited="true" range="0 {free_len + max_deformation:.4f}" '
        f'stiffness="{stiffness:.1f}" damping="{damping:.1f}"\n'
        f'               body1="{parent_name}" body2="{child_name}" '
        f'site1="{parent_name}" site2="{child_name}"/>\n'
        f'    </tendon>'
    )


def _build_body_recursive(
    body: MechanicalBody,
    node_id: str,
    parent_pos: np.ndarray,
    depth: int,
    joint_counter: list,
    catalog: Dict[str, PartSpec],
    lines: list,
    joint_map: dict,
    motor_map: dict,
    sensor_info: dict,
    visited: Optional[set] = None,
) -> None:
    # 深度保护：防止环形图或深度异常导致无限递归
    MAX_DEPTH = 100
    if depth > MAX_DEPTH:
        _logger.warning(f"_build_body_recursive: max depth {MAX_DEPTH} exceeded at node {node_id}, truncating")
        lines.append(f"{'    ' * (depth + 1)}<!-- [TRUNCATED: depth > {MAX_DEPTH}] -->")
        return

    if visited is None:
        visited = set()
    if node_id in visited:
        _logger.warning(f"_build_body_recursive: cycle detected at node {node_id}, skipping")
        return
    visited.add(node_id)

    indent = "    " * (depth + 1)
    part = body.get_part(node_id)
    spec = catalog.get(part.part_type)

    if depth == 0:
        pos_str = f"{part.position[0]:.4f} {part.position[1]:.4f} {part.position[2]:.4f}"
        lines.append(f'{indent}<body name="{node_id}" pos="{pos_str}">')
        # 根 body 必须带自由关节, 否则整体被焊死在世界坐标系:
        # 自由度只剩内部关节, framepos 传感器的 COM 恒定,
        # `_com_z < fall_height` 永不触发, displacement / speed / velocity 恒为 0。
        lines.append(f"{indent}  <freejoint/>")
    else:
        parent = body.parent_of(node_id)
        joint = body.get_joint(parent, node_id)
        jname = f"j_{parent}_{node_id}"
        joint_counter[0] += 1
        joint_map[jname] = (parent, node_id)

        joint_str = _joint_xml(jname, joint.joint_type, joint.axis, joint.params, parent_pos, part.position)
        if joint_str:
            lines.append(f'{indent}<body name="{node_id}" pos="0 0 0">')
            lines.append(f"{indent}  {joint_str}")
        else:
            rel_pos = part.position - parent_pos
            pos_str = f"{rel_pos[0]:.4f} {rel_pos[1]:.4f} {rel_pos[2]:.4f}"
            lines.append(f'{indent}<body name="{node_id}" pos="{pos_str}">')

    if spec:
        has_touch = getattr(spec, 'has_touch', False)
        has_imu = getattr(spec, 'has_imu', False)

        # V2: 脚型零件自动启用 touch sensor
        is_foot = spec.shape in ("hemisphere_shell",) or \
                  any(kw in part.part_type for kw in ("foot", "pad", "spike"))
        if is_foot:
            has_touch = True

        lines.append(f"{indent}  {_part_geom(part, spec, has_touch=has_touch)}")
        if has_touch:
            lines.append(f'{indent}  <site name="{node_id}" type="sphere" size="0.005"/>')
            sensor_info["touch_nodes"].append(node_id)
        if has_imu:
            lines.append(f'{indent}  <site name="imu_site_{node_id}" type="box" size="0.005 0.005 0.005"/>')
            sensor_info["imu_nodes"].append(node_id)

        # V2: 弹簧零件生成 tendon
        if spec.shape == "helical_spring":
            tendon_xml = _spring_tendon(body, node_id, part, spec)
            if tendon_xml:
                sensor_info.setdefault("tendon_lines", []).append(tendon_xml)

    children = body.children_of(node_id)
    for child_id in children:
        _build_body_recursive(body, child_id, part.position, depth + 1, joint_counter, catalog, lines, joint_map, motor_map, sensor_info, visited=visited)

    lines.append(f"{indent}</body>")


# ── Build MuJoCo XML from MechanicalBody — geometry → joints → tendons → sensors → actuators
def build_mjcf_model(
    body: MechanicalBody,
    catalog: Optional[Dict[str, PartSpec]] = None,
) -> Tuple[str, dict, dict, dict]:
    # 防御: 空 body 或无效 catalog
    if body.root_id is None or body.num_parts() == 0:
        from forgecraft.core.generator import BodyGenerator
        _logger.warning("build_mjcf_model called with empty body, generating minimal fallback")
        body = BodyGenerator().generate_random_body("fallback", min_parts=3, max_parts=4)

    catalog = catalog or dict(DEFAULT_CATALOG)
    if not catalog:
        _logger.error("build_mjcf_model: catalog is empty")
        catalog = dict(DEFAULT_CATALOG)

    try:
        return _build_mjcf_impl(body, catalog)
    except Exception as e:
        _logger.error(f"build_mjcf_model failed: {e}")
        # 返回最小合法 XML (单零件 + 自由关节, 确保仿真可启动)
        fallback = _build_fallback_xml()
        return fallback, {}, {}, {"touch_nodes": [], "imu_nodes": []}


def _build_fallback_xml() -> str:
    return '''<mujoco model="forgecraft_fallback">
  <compiler angle="radian" coordinate="local"/>
  <worldbody>
    <light directional="true" pos="0 0 5" dir="0 0 -1"/>
    <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1"/>
    <body name="fallback" pos="0 0 0.5">
      <freejoint/>
      <geom name="fallback_geom" type="box" size="0.05 0.05 0.05" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>'''


def _build_mjcf_impl(
    body: MechanicalBody,
    catalog: Dict[str, PartSpec],
) -> Tuple[str, dict, dict, dict]:
    lines = []
    joint_counter = [0]
    joint_map = {}
    motor_map = {}
    sensor_info = {"touch_nodes": [], "imu_nodes": []}
    _tendon_counter[0] = 0  # 每次构建重置

    lines.append('<mujoco model="forgecraft_body">')
    lines.append('  <compiler angle="radian" coordinate="local"/>')
    lines.append('  <size nconmax="2000" njmax="20000" nstack="100000000"/>')
    lines.append("")
    lines.append("  <option>")
    lines.append('    <flag contact="enable" gravity="enable"/>')
    lines.append("  </option>")
    lines.append("")
    lines.append("  <default>")
    lines.append('    <geom friction="0.6 0.01 0.01" condim="3"/>')
    lines.append('    <joint damping="0.5" armature="0.01"/>')
    lines.append('    <motor ctrlrange="-1 1"/>')
    lines.append("  </default>")
    lines.append("")
    lines.append("  <asset>")
    lines.append('    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>')
    lines.append('    <texture name="texplane" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="512" height="512"/>')
    lines.append('    <material name="matplane" reflectance="0.0" texture="texplane" texrepeat="1 1" texuniform="true"/>')
    lines.append("  </asset>")
    lines.append("")
    lines.append("  <worldbody>")
    lines.append('    <light directional="true" cutoff="30" exponent="2" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1" pos="0 0 5" dir="0 0 -1"/>')
    lines.append('    <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1" material="matplane" friction="0.8 0.01 0.01"/>')

    if body.root_id is not None:
        root_pos = body.get_part(body.root_id).position
        _build_body_recursive(body, body.root_id, root_pos, 0, joint_counter, catalog, lines, joint_map, motor_map, sensor_info)

    lines.append("  </worldbody>")
    lines.append("")

    actuated_joints = []
    for u, v in body.graph.edges:
        child_part = body.get_part(v)
        if child_part.params.get("actuated", 0.0) > 0.5:
            jname = f"j_{u}_{v}"
            actuated_joints.append((jname, child_part))

    if actuated_joints:
        lines.append("  <actuator>")
        for jname, child_part in actuated_joints:
            lines.append(f"    {_motor_actuator(jname, child_part.params)}")
            motor_map[jname] = jname
        lines.append("  </actuator>")
        lines.append("")

    # V2: 弹簧 tendon 元素
    tendon_lines = sensor_info.get("tendon_lines", [])
    if tendon_lines:
        lines.append("  <tendon>")
        for tl in tendon_lines:
            lines.append(tl)
        lines.append("  </tendon>")
        lines.append("")

    lines.append("  <sensor>")
    lines.append('    <framepos name="com" objtype="body" objname="' + str(body.root_id) + '"/>')
    lines.append('    <framelinvel name="com_vel" objtype="body" objname="' + str(body.root_id) + '"/>')
    lines.append('    <frameangacc name="com_acc" objtype="body" objname="' + str(body.root_id) + '"/>')

    for jname, (parent_id, child_id) in joint_map.items():
        if body.get_part(child_id).params.get("actuated", 0.0) > 0.5:
            lines.append(f'    <jointpos name="jp_{jname}" joint="{jname}"/>')
            lines.append(f'    <jointvel name="jv_{jname}" joint="{jname}"/>')

    for jname in motor_map:
        lines.append(f'    <actuatorfrc name="af_{jname}" actuator="act_{jname}"/>')

    for node_id in sensor_info["touch_nodes"]:
        lines.append(f'    <touch name="touch_{node_id}" site="{node_id}"/>')

    for node_id in sensor_info["imu_nodes"]:
        lines.append(f'    <framequat name="imu_quat_{node_id}" objtype="body" objname="{node_id}"/>')
        lines.append(f'    <accelerometer name="imu_acc_{node_id}" site="imu_site_{node_id}"/>')
        lines.append(f'    <gyro name="imu_gyro_{node_id}" site="imu_site_{node_id}"/>')

    lines.append("  </sensor>")
    lines.append("")

    lines.append("</mujoco>")

    return "\n".join(lines), joint_map, motor_map, sensor_info


# ═══════════════════════════════════════════════════════════
#  带凸分解碰撞体的 MJCF 构建
# ═══════════════════════════════════════════════════════════

def build_mjcf_with_convex_collision(
    body: MechanicalBody,
    catalog: Optional[Dict[str, PartSpec]] = None,
    collision_dir: str = "collision_meshes",
    mesh_quality: str = "high",
    max_hulls: int = 3,
) -> Tuple[str, dict, dict, dict]:
    """用凸分解碰撞体 mesh 构建 MuJoCo XML + 生成 STL
    
    Returns:
        (mjcf_xml, joint_map, motor_map, sensor_info)
    """
    from forgecraft.manufacturing.mesh_engine import MeshEngine
    from forgecraft.simulation.convex_collision import mesh_to_collision_geoms

    catalog = catalog or dict(DEFAULT_CATALOG)
    # Convert CatalogSpec → PartSpec if needed
    if catalog:
        first_val = next(iter(catalog.values())) if catalog else None
        if first_val is not None and hasattr(first_val, 'to_part_spec'):
            catalog = {k: v.to_part_spec() for k, v in catalog.items()}
    engine = MeshEngine(mesh_quality)
    os.makedirs(collision_dir, exist_ok=True)

    # 生成每个零件的凸分解碰撞体
    collision_meshes = {}
    for part in body.parts():
        spec = catalog.get(part.part_type)
        if spec is None:
            continue
        mesh = engine.build_part(spec, dict(part.params))
        hulls = mesh_to_collision_geoms(mesh, mesh_quality, max_hulls)
        part_files = []
        for i, hull in enumerate(hulls):
            stl_name = f"{part.part_id}_collision_{i}.stl"
            stl_path = os.path.join(collision_dir, stl_name)
            hull.export(stl_path)
            mesh_name = os.path.splitext(stl_name)[0]
            part_files.append((hull, stl_path, mesh_name))
        collision_meshes[part.part_id] = part_files

    # 构建基础 XML
    xml, joint_map, motor_map, sensor_info = build_mjcf_model(body, catalog)

    # 插入 mesh asset 声明
    asset_lines = []
    for files in collision_meshes.values():
        for _, stl_path, mesh_name in files:
            asset_lines.append(
                f'    <mesh name="{mesh_name}" file="{os.path.basename(stl_path)}"/>'
            )
    if asset_lines:
        asset_block = "  <asset>\n" + "\n".join(asset_lines) + "\n  </asset>\n"
        xml = xml.replace("<worldbody>", asset_block + "  <worldbody>")

    # 替换每个零件的简单原语 geom 为 mesh geom
    for part in body.parts():
        if part.part_id not in collision_meshes:
            continue
        files = collision_meshes[part.part_id]
        if not files:
            continue
        _, _, mesh_name = files[0]
        spec = catalog.get(part.part_type)
        rgba = " ".join(map(str, spec.color)) if spec else "0.5 0.5 0.5 1.0"
        mass = part.params.get("mass", spec.mass if spec else 0.1)
        
        # 替换: 找该 body 块中的第一个 geom 行
        body_start = f'<body name="{part.part_id}"'
        pos = xml.find(body_start)
        if pos < 0:
            continue
        chunk = xml[pos:pos + 500]
        geom_start = chunk.find('<geom type=')
        if geom_start < 0:
            continue
        geom_end = chunk.find('/>', geom_start) + 2
        old_geom = chunk[geom_start:geom_end]
        new_geom = f'<geom type="mesh" mesh="{mesh_name}" rgba="{rgba}" mass="{mass:.4f}"/>'
        xml = xml.replace(old_geom, new_geom, 1)
    
    return xml, joint_map, motor_map, sensor_info
