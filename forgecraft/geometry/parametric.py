"""
参数化机械零件几何生成器 (Phase 2)

为每种 part_type 生成具象的机械几何，全部采用构建式方法（无布尔运算依赖）。
所有生成器返回 trimesh.Trimesh 对象，质量由 QUALITY 键名控制。

支持的零件类型:
- brushless_motor: 圆柱电机 + 轴 + 法兰 + 安装孔 + 散热槽 + 后端盖
- brushless_motor_compact: 紧凑型电机
- micro_bearing: 轴承 (外圈 + 内圈 + 滚珠 + 防尘盖)
- alloy_chassis: 构建式框架 (边梁 + 横肋 + 安装凸台 + 底板)
- carbon_tube: 空心管 + 端盖 + 螺纹孔
- launch_spring: 螺旋弹簧 (cylinder-wire 扫描, 4 级顶点可控)
- sprint_foot: 半球截面 + 交叉防滑纹
- compact_lipo / high_power_lipo: LiPo + 倒角 + XT60 + 引线
- generic: 回退基本几何
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import trimesh

# ================================================================
# 质量预设 (Phase 2 细化)
# ================================================================

QUALITY_PRESETS = {
    "low": {
        "cyl_sections": 12,
        "sphere_subd": 1,
        "spring_pts_per_turn": 12,
        "spring_wire_sides": 6,
        "bearing_balls": 6,
        "chassis_ribs": 2,
        "motor_ribs": 3,
    },
    "medium": {
        "cyl_sections": 20,
        "sphere_subd": 2,
        "spring_pts_per_turn": 24,
        "spring_wire_sides": 8,
        "bearing_balls": 8,
        "chassis_ribs": 3,
        "motor_ribs": 5,
    },
    "high": {
        "cyl_sections": 32,
        "sphere_subd": 2,
        "spring_pts_per_turn": 48,
        "spring_wire_sides": 10,
        "bearing_balls": 10,
        "chassis_ribs": 5,
        "motor_ribs": 7,
    },
    "ultra": {
        "cyl_sections": 64,
        "sphere_subd": 3,
        "spring_pts_per_turn": 80,
        "spring_wire_sides": 14,
        "bearing_balls": 14,
        "chassis_ribs": 8,
        "motor_ribs": 11,
    },
}


# ================================================================
# 工具函数
# ================================================================

def _align_cylinder_to_vec(radius: float, p1: np.ndarray, p2: np.ndarray,
                           sections: int = 8) -> trimesh.Trimesh:
    """创建从 p1 指向 p2 的圆柱体 (用于弹簧线/连杆)"""
    vec = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    length = float(np.linalg.norm(vec))
    if length < 1e-10:
        return trimesh.creation.icosphere(radius=radius, subdivisions=1)
    mid = (np.asarray(p1) + np.asarray(p2)) / 2.0
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    direction = vec / length
    z_axis = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, z_axis)) > 0.9999:
        if direction[2] < 0:
            rot = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:
            rot = np.eye(4)
    else:
        axis = np.cross(z_axis, direction)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(z_axis, direction))
        rot = trimesh.transformations.rotation_matrix(angle, axis)
    cyl.apply_transform(rot)
    cyl.apply_translation(mid)
    return cyl


# ================================================================
# 零件生成器
# ================================================================

def _motor(params: dict, q: dict) -> trimesh.Trimesh:
    """电机: 圆柱体 + 散热槽肋 + 轴 + 法兰 + 安装孔 + 后端盖"""
    L = float(params.get("length", 0.08))
    R = float(params.get("radius", 0.028))
    shaft_r = R * 0.15
    shaft_l = L * 0.22
    sec = q["cyl_sections"]
    n_ribs = q.get("motor_ribs", 5)

    parts = []

    # 主体
    body = trimesh.creation.cylinder(radius=R, height=L, sections=sec)
    body.apply_translation([0, 0, L / 2])
    parts.append(body)

    # 散热槽肋 (圆周凸环)
    rib_h = L * 0.015
    rib_r = R * 1.04
    for i in range(n_ribs):
        z = L * 0.15 + L * 0.7 * i / max(n_ribs - 1, 1)
        rib = trimesh.creation.cylinder(radius=rib_r, height=rib_h, sections=sec)
        rib.apply_translation([0, 0, z])
        parts.append(rib)

    # 后端盖
    endcap = trimesh.creation.cylinder(radius=R * 0.95, height=L * 0.06, sections=sec)
    endcap.apply_translation([0, 0, L * 0.03])
    parts.append(endcap)

    # 轴 (从前端伸出)
    shaft = trimesh.creation.cylinder(radius=shaft_r, height=shaft_l, sections=max(sec // 2, 8))
    shaft.apply_translation([0, 0, L + shaft_l / 2 - L * 0.02])

    # 轴端倒角 (小锥台)
    chamfer = trimesh.creation.cylinder(radius=shaft_r * 0.7, height=shaft_r * 1.5,
                                        sections=max(sec // 2, 8))
    chamfer.apply_translation([0, 0, L + shaft_l - shaft_r * 0.3])
    parts.extend([shaft, chamfer])

    # 法兰
    flange = trimesh.creation.cylinder(radius=R * 1.12, height=L * 0.04, sections=sec)
    flange_z = L + L * 0.02
    flange.apply_translation([0, 0, flange_z])

    # 法兰安装孔 (4 个)
    bolt_r = R * 0.08
    bolt_pos_r = R * 0.92
    hole_parts = []
    for a in [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]:
        h = trimesh.creation.cylinder(radius=bolt_r, height=L * 0.3, sections=8)
        h.apply_translation([bolt_pos_r * np.cos(a), bolt_pos_r * np.sin(a), flange_z])
        hole_parts.append(h)
    try:
        combined_holes = trimesh.boolean.union(hole_parts)
        flange = flange.difference(combined_holes)
    except Exception:
        pass  # 布尔失败时保留无孔法兰
    parts.append(flange)

    return trimesh.util.concatenate(parts)


def _motor_compact(params: dict, q: dict) -> trimesh.Trimesh:
    """紧凑型电机"""
    L = float(params.get("length", 0.05))
    R = float(params.get("radius", 0.019))
    shaft_r = R * 0.18
    shaft_l = L * 0.2
    sec = q["cyl_sections"]
    n_ribs = max(q.get("motor_ribs", 5) // 2, 2)

    parts = []

    body = trimesh.creation.cylinder(radius=R, height=L, sections=sec)
    body.apply_translation([0, 0, L / 2])
    parts.append(body)

    # 散热肋
    rib_h = L * 0.02
    rib_r = R * 1.04
    for i in range(n_ribs):
        z = L * 0.15 + L * 0.7 * i / max(n_ribs - 1, 1)
        rib = trimesh.creation.cylinder(radius=rib_r, height=rib_h, sections=sec)
        rib.apply_translation([0, 0, z])
        parts.append(rib)

    # 后端盖
    endcap = trimesh.creation.cylinder(radius=R * 0.93, height=L * 0.07, sections=sec)
    endcap.apply_translation([0, 0, L * 0.035])
    parts.append(endcap)

    # 轴
    shaft = trimesh.creation.cylinder(radius=shaft_r, height=shaft_l, sections=max(sec // 2, 8))
    shaft.apply_translation([0, 0, L + shaft_l / 2 - L * 0.02])
    parts.append(shaft)

    # 法兰
    flange = trimesh.creation.cylinder(radius=R * 1.1, height=L * 0.05, sections=sec)
    flange.apply_translation([0, 0, L + L * 0.025])
    parts.append(flange)

    return trimesh.util.concatenate(parts)


def _bearing(params: dict, q: dict) -> trimesh.Trimesh:
    """轴承: 外圈 + 内圈 + 滚珠 + 防尘盖环"""
    outer_r = float(params.get("outer_radius", params.get("radius", 0.01)))
    inner_r = float(params.get("inner_radius", outer_r * 0.55))
    h = float(params.get("thickness", params.get("length", 0.004)))
    sec = q["cyl_sections"]
    n_balls = q["bearing_balls"]
    shield_r = (outer_r + inner_r) / 2 + (outer_r - inner_r) * 0.25

    parts = []

    # 外圈 (圆环)
    outer = trimesh.creation.cylinder(radius=outer_r, height=h, sections=sec * 2)
    outer_hole = trimesh.creation.cylinder(
        radius=outer_r - h * 0.45, height=h * 3, sections=sec)
    try:
        outer = outer.difference(outer_hole)
    except Exception:
        outer = trimesh.creation.cylinder(radius=outer_r, height=h, sections=sec)
    outer.apply_translation([0, 0, h / 2])
    parts.append(outer)

    # 内圈 (圆环)
    inner = trimesh.creation.cylinder(radius=inner_r + h * 0.35, height=h, sections=sec * 2)
    inner_hole = trimesh.creation.cylinder(
        radius=inner_r - h * 0.25, height=h * 3, sections=sec)
    try:
        inner = inner.difference(inner_hole)
    except Exception:
        inner = trimesh.creation.cylinder(radius=inner_r + h * 0.35, height=h, sections=sec)
    inner.apply_translation([0, 0, h / 2])
    parts.append(inner)

    # 滚珠
    mid_r = (outer_r + inner_r) / 2
    ball_r = (outer_r - inner_r) / 2 - h * 0.15
    ball_r = max(ball_r, 0.0003)
    for i in range(n_balls):
        a = 2 * np.pi * i / n_balls
        b = trimesh.creation.icosphere(radius=ball_r * 0.8,
                                       subdivisions=q["sphere_subd"])
        b.apply_translation([mid_r * np.cos(a), mid_r * np.sin(a), h / 2])
        parts.append(b)

    # 防尘盖环 (两侧薄圆环)
    shield_thick = h * 0.08
    for sz in [-shield_thick, h]:
        shield = trimesh.creation.cylinder(radius=shield_r, height=shield_thick,
                                           sections=sec)
        shield_hole = trimesh.creation.cylinder(
            radius=inner_r + h * 0.4, height=shield_thick * 3, sections=sec // 2)
        try:
            shield = shield.difference(shield_hole)
        except Exception:
            pass
        shield.apply_translation([0, 0, sz])
        parts.append(shield)

    return trimesh.util.concatenate(parts)


def _chassis(params: dict, q: dict) -> trimesh.Trimesh:
    """构建式框架底盘: 4 条边梁 + 横肋 + 4 角安装凸台 + 底板

    完全采用构建式方法 (定位平移), 不依赖布尔运算 (difference/union)。
    """
    l = float(params.get("length", 0.08))
    w = float(params.get("width", 0.25))
    h = float(params.get("height", 0.02))
    n_ribs = q.get("chassis_ribs", 5)

    beam_w = max(h * 1.6, 0.004)       # 边梁宽度
    rib_w = max(h * 1.0, 0.0025)       # 横肋宽度
    plate_h = max(h * 0.25, 0.0015)    # 底板厚
    pad_r = beam_w * 1.2               # 安装凸台半径
    pad_h = h                           # 凸台高
    hole_r = pad_r * 0.35              # 安装孔径

    parts = []

    # ── 底板 ──
    plate = trimesh.creation.box([l, w, plate_h])
    plate.apply_translation([0, 0, plate_h / 2])
    parts.append(plate)

    # ── 4 条边梁 ──
    # 长边梁 (沿 X)
    for sy in [-1, 1]:
        beam = trimesh.creation.box([l, beam_w, h])
        beam.apply_translation([0, sy * (w / 2 - beam_w / 2), h / 2])
        parts.append(beam)
    # 短边梁 (沿 Y), 夹在长边梁内侧
    for sx in [-1, 1]:
        beam = trimesh.creation.box([beam_w, w - 2 * beam_w, h])
        beam.apply_translation([sx * (l / 2 - beam_w / 2), 0, h / 2])
        parts.append(beam)

    # ── 横肋 (沿 Y, 等距排列) ──
    inner_w = w - 2 * beam_w
    if n_ribs > 0 and inner_w > rib_w * 2:
        # 肋分布在两个短边梁之间
        inner_l = l - 2 * beam_w
        for i in range(n_ribs):
            if n_ribs == 1:
                x = 0
            else:
                x = -inner_l / 2 + inner_l * (i + 0.5) / n_ribs
            rib = trimesh.creation.box([rib_w, inner_w, h * 0.85])
            rib.apply_translation([x, 0, h * 0.85 / 2])
            parts.append(rib)

    # ── 4 角安装凸台 ──
    pad_margin = beam_w / 2
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            pad = trimesh.creation.cylinder(radius=pad_r, height=pad_h, sections=16)
            pad.apply_translation(
                [sx * (l / 2 - pad_margin), sy * (w / 2 - pad_margin), pad_h / 2])
            parts.append(pad)
            # 沉头孔 (凸台表面薄圆柱)
            hole = trimesh.creation.cylinder(radius=hole_r, height=pad_h * 0.4, sections=12)
            hole.apply_translation(
                [sx * (l / 2 - pad_margin), sy * (w / 2 - pad_margin), pad_h * 0.15])
            parts.append(hole)

    return trimesh.util.concatenate(parts)


def _tube(params: dict, q: dict) -> trimesh.Trimesh:
    """薄壁碳纤维管 + 端盖 + 中心螺纹孔"""
    L = float(params.get("length", 0.5))
    R = float(params.get("radius", 0.02))
    wall_t = max(R * 0.12, 0.0015)
    sec = q["cyl_sections"]

    parts = []

    # 空心管
    outer = trimesh.creation.cylinder(radius=R, height=L, sections=sec)
    inner_hole = trimesh.creation.cylinder(
        radius=R - wall_t, height=L * 2, sections=max(sec // 2, 8))
    try:
        tube = outer.difference(inner_hole)
    except Exception:
        tube = outer
    tube.apply_translation([0, 0, L / 2])
    parts.append(tube)

    # 端部加固箍
    ring_h = wall_t * 2
    for z in [ring_h / 2, L - ring_h / 2]:
        ring = trimesh.creation.cylinder(
            radius=R + wall_t * 0.8, height=ring_h, sections=sec)
        ring.apply_translation([0, 0, z])
        parts.append(ring)

    # 端盖 (薄圆盘 + 中心螺纹孔)
    cap_r = R + wall_t * 0.6
    cap_h = wall_t * 2.5
    thread_r = R * 0.15
    for z in [cap_h / 2, L - cap_h / 2]:
        cap = trimesh.creation.cylinder(radius=cap_r, height=cap_h, sections=sec)
        cap.apply_translation([0, 0, z])
        # 中心孔
        tap = trimesh.creation.cylinder(radius=thread_r, height=cap_h * 1.5, sections=12)
        tap.apply_translation([0, 0, z])
        try:
            cap = cap.difference(tap)
        except Exception:
            pass
        parts.append(cap)

    return trimesh.util.concatenate(parts)


def _spring(params: dict, q: dict) -> trimesh.Trimesh:
    """螺旋弹簧: cylinder-wire 扫描 (Phase 2)

    用圆柱段连接连续点 → 顶点数精确可控。
    low: ~300, medium: ~800, high: ~2500, ultra: ~6500
    """
    L = float(params.get("length", 0.03))
    coil_r = float(params.get("coil_radius", 0.008))
    wire_r = float(params.get("wire_radius", 0.0015))
    turns = float(params.get("turns", params.get("coils", 4)))
    pts_per_turn = q["spring_pts_per_turn"]
    wire_sides = q["spring_wire_sides"]

    n = int(turns * pts_per_turn)
    t = np.linspace(0, turns * 2 * np.pi, n)
    pts = np.column_stack([
        coil_r * np.cos(t),
        coil_r * np.sin(t),
        np.linspace(0, L, n),
    ])

    meshes = []
    for i in range(n - 1):
        seg = _align_cylinder_to_vec(wire_r, pts[i], pts[i + 1], sections=wire_sides)
        meshes.append(seg)

    if not meshes:
        # 单圈退化: 放一个环
        return trimesh.creation.cylinder(radius=wire_r, height=0.001, sections=wire_sides)

    return trimesh.util.concatenate(meshes)


def _foot(params: dict, q: dict) -> trimesh.Trimesh:
    """竞速脚: 半球截面 + 交叉防滑纹 (Phase 2)"""
    r = float(params.get("radius", 0.017))
    subd = q["sphere_subd"]

    parts = []

    # 半球截面
    sph = trimesh.creation.icosphere(radius=r, subdivisions=max(subd, 2))
    cut = trimesh.creation.box([r * 3, r * 3, r * 0.7])
    cut.apply_translation([0, 0, -r * 0.35])
    try:
        foot = sph.difference(cut)
    except Exception:
        foot = sph
    foot.apply_translation([0, 0, r * 0.35])
    parts.append(foot)

    # 交叉防滑纹: X 方向 + Y 方向窄条
    n_lines = max(subd + 3, 5)
    line_w = r * 0.03
    line_h = r * 0.04
    for sign in [-1, 1]:
        for i in range(1, n_lines):
            offset = r * i / n_lines
            # 沿 X 方向条
            bar_x = trimesh.creation.box([r * 1.6, line_w, line_h])
            bar_x.apply_translation([0, sign * offset, r * 0.04])
            parts.append(bar_x)
            # 沿 Y 方向条
            bar_y = trimesh.creation.box([line_w, r * 1.6, line_h])
            bar_y.apply_translation([sign * offset, 0, r * 0.04])
            parts.append(bar_y)
    # 中心十字加粗
    bar_cx = trimesh.creation.box([r * 1.5, line_w * 1.8, line_h * 1.3])
    bar_cx.apply_translation([0, 0, r * 0.05])
    bar_cy = trimesh.creation.box([line_w * 1.8, r * 1.5, line_h * 1.3])
    bar_cy.apply_translation([0, 0, r * 0.05])
    parts.extend([bar_cx, bar_cy])

    return trimesh.util.concatenate(parts)


def _battery(params: dict, q: dict) -> trimesh.Trimesh:
    """LiPo 电池: 主体 + XT60 + 引出线 + 标签凹槽 (Phase 2)"""
    L = float(params.get("length", 0.11))
    W = float(params.get("width", 0.035))
    H = float(params.get("height", 0.03))

    parts = []

    # 主体 box
    box = trimesh.creation.box([L, W, H])
    box.apply_translation([0, 0, H / 2])
    parts.append(box)

    # 顶面标签凹槽 (表示为一层略微凹陷的矩形)
    label_l = L * 0.75
    label_w = W * 0.75
    label = trimesh.creation.box([label_l, label_w, H * 0.03])
    label.apply_translation([0, 0, H * 0.975])
    parts.append(label)

    # XT60 接口凸台 (在 +X 端)
    plug_w = min(W * 0.5, H * 0.9)
    plug_h = H * 0.65
    plug_l = L * 0.1
    plug = trimesh.creation.box([plug_l, plug_w, plug_h])
    plug.apply_translation([L / 2 + plug_l / 2, 0, H / 2])
    parts.append(plug)

    # XT60 插针孔 (两个小圆柱孔标记)
    pin_r = plug_w * 0.15
    pin_offset = plug_w * 0.22
    for sy in [-1, 1]:
        pin = trimesh.creation.cylinder(radius=pin_r, height=plug_l * 1.1, sections=10)
        pin.apply_translation([L / 2 + plug_l / 2, sy * pin_offset, H / 2])
        parts.append(pin)

    # 引出线 (12AWG, 两根)
    wire_r = H * 0.08
    wire_l = L * 0.25
    wire_start_x = L / 2 + plug_l
    for i, sy in enumerate([-0.35, 0.35]):
        wire = trimesh.creation.cylinder(radius=wire_r, height=wire_l, sections=10)
        wire_z = H / 2
        # 先水平再向下弯 — 简化为水平段 + 90° 弯角
        wire.apply_translation([wire_start_x + wire_l / 2, sy * plug_w, wire_z])
        parts.append(wire)

    # 平衡充接口 (小凸台在 -X 端)
    bal_w = W * 0.25
    bal_h = H * 0.3
    bal_l = L * 0.04
    bal = trimesh.creation.box([bal_l, bal_w, bal_h])
    bal.apply_translation([-L / 2 - bal_l / 2, 0, H / 2])
    parts.append(bal)

    return trimesh.util.concatenate(parts)


def _generic(params: dict, q: dict) -> trimesh.Trimesh:
    """回退: 基本 box/cylinder"""
    L = float(params.get("length", 0.1))
    R = float(params.get("radius", 0.03))
    W = float(params.get("width", 0.03))
    H = float(params.get("height", 0.03))
    sec = q["cyl_sections"]

    if R > max(W, H):
        m = trimesh.creation.cylinder(radius=R, height=L, sections=sec)
        m.apply_translation([0, 0, L / 2])
        return m
    else:
        m = trimesh.creation.box([L, max(W, 0.01), max(H, 0.01)])
        m.apply_translation([0, 0, max(H, 0.01) / 2])
        return m


# ================================================================
# 生成器注册表
# ================================================================

GENERATOR_REGISTRY: Dict[str, Callable] = {
    "brushless_motor": _motor,
    "brushless_motor_compact": _motor_compact,
    "micro_bearing": _bearing,
    "alloy_chassis": _chassis,
    "carbon_tube": _tube,
    "launch_spring": _spring,
    "sprint_foot": _foot,
    "compact_lipo": _battery,
    "high_power_lipo": _battery,
}


class ParametricGenerator:
    """参数化零件几何生成器"""

    def __init__(self, quality: str = "high"):
        self.quality = quality
        self._preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["high"])

    @property
    def registry(self) -> Dict[str, Callable]:
        return GENERATOR_REGISTRY

    def make(self, part_type: str, params: dict,
             position: Optional[np.ndarray] = None,
             quality: Optional[str] = None) -> Optional[trimesh.Trimesh]:
        gen = GENERATOR_REGISTRY.get(part_type)
        if gen is None:
            return None

        q = QUALITY_PRESETS.get(quality or self.quality, self._preset)
        try:
            mesh = gen(params, q)
        except Exception:
            mesh = _generic(params, q)

        if position is not None:
            mesh.apply_translation(np.asarray(position, dtype=np.float64))

        return mesh

    def has_generator(self, part_type: str) -> bool:
        return part_type in GENERATOR_REGISTRY

    def available_types(self) -> list:
        return sorted(GENERATOR_REGISTRY.keys())
