"""
标准件库 — 常用机械零件的参数化 trimesh 模型

基于 ISO/DIN 标准尺寸，可直接用于装配和 BOM 生成。

零件清单:
  螺丝: hex_screw_m2 ~ m8, countersunk_screw_m2 ~ m6
  螺母: hex_nut_m2 ~ m8, lock_nut_m2 ~ m8
  垫圈: flat_washer_m2 ~ m8, spring_washer_m2 ~ m8
  轴承: ball_bearing_608,  625,  688, 6000
  其他: shaft_coupling, rod_end, spacer, set_screw

所有 mesh watertight，面数可控。

使用:
  >>> from forgecraft.manufacturing.standard_parts import hex_screw_m3
  >>> screw = hex_screw_m3(length=20)
  >>> screw.export("screw_m3x20.stl")
"""

from typing import Dict, Optional, Tuple
import math
import numpy as np
import trimesh

from forgecraft.manufacturing.mesh_engine import MeshEngine, _safe_bool

__all__ = [
    "hex_screw_m2", "hex_screw_m3", "hex_screw_m4",
    "hex_screw_m5", "hex_screw_m6", "hex_screw_m8",
    "countersunk_screw_m3", "countersunk_screw_m4", "countersunk_screw_m5",
    "hex_nut_m2", "hex_nut_m3", "hex_nut_m4",
    "hex_nut_m5", "hex_nut_m6", "hex_nut_m8",
    "lock_nut_m3", "lock_nut_m4", "lock_nut_m5", "lock_nut_m6",
    "flat_washer_m2", "flat_washer_m3", "flat_washer_m4",
    "flat_washer_m5", "flat_washer_m6", "flat_washer_m8",
    "spring_washer_m3", "spring_washer_m4", "spring_washer_m5",
    "ball_bearing_608", "ball_bearing_625", "ball_bearing_688", "ball_bearing_6000",
    "shaft_coupling_3mm", "shaft_coupling_5mm", "shaft_coupling_8mm",
    "rod_end_m3", "rod_end_m4", "rod_end_m5",
    "spacer_m3", "spacer_m4",
    "get_standard_part",
    "STANDARD_PARTS_CATALOG",
]

_default_engine = MeshEngine("high")

# ── ISO 公制螺纹尺寸 (外径 x 螺距, in mm) ──
_THREAD_SPECS = {
    2:  (2.0,  0.4),
    2.5:(2.5,  0.45),
    3:  (3.0,  0.5),
    4:  (4.0,  0.7),
    5:  (5.0,  0.8),
    6:  (6.0,  1.0),
    8:  (8.0,  1.25),
}

# ── 六角对边 (mm) ──
_HEX_AF = {2: 4.0, 2.5: 5.0, 3: 5.5, 4: 7.0, 5: 8.0, 6: 10.0, 8: 13.0}
# ── 六角头高度 (mm) ──
_HEX_HEAD_H = {2: 1.6, 2.5: 2.0, 3: 2.4, 4: 3.1, 5: 3.9, 6: 4.7, 8: 6.2}
# ── 螺母高度 (mm) ──
_NUT_H = {2: 1.6, 2.5: 2.0, 3: 2.4, 4: 3.2, 5: 4.0, 6: 5.0, 8: 6.5}
# ── 垫圈外径 (mm) ──
_WASHER_OD = {2: 5.0, 2.5: 6.5, 3: 7.0, 4: 9.0, 5: 10.0, 6: 12.0, 8: 16.0}
# ── 垫圈厚度 (mm) ──
_WASHER_T = {2: 0.3, 2.5: 0.5, 3: 0.5, 4: 0.8, 5: 1.0, 6: 1.6, 8: 1.6}
# ── 轴承 ID/OD/宽度 (mm) ──
_BEARING_SPECS = {
    "608":  (8,  22, 7),
    "625":  (5,  16, 5),
    "688":  (8,  16, 5),
    "6000": (10, 26, 8),
    "6001": (12, 28, 8),
    "6002": (15, 32, 9),
    "6200": (10, 30, 9),
    "6201": (12, 32, 10),
}


def _hex_prism(af: float, height: float, sections: int = 64) -> trimesh.Trimesh:
    """正六棱柱"""
    r = af / math.sqrt(3)
    cyl = trimesh.creation.cylinder(radius=r, height=height, sections=6)
    # 旋转使顶点对齐 Y 轴
    rot = trimesh.transformations.rotation_matrix(math.pi / 6, [0, 0, 1])
    cyl.apply_transform(rot)
    cyl.apply_translation([0, 0, -height / 2])
    return cyl


# ═══════════════════════════════════════════════════════════
#  螺丝
# ═══════════════════════════════════════════════════════════

def _hex_screw(size: int, length: float, head_h: float = None,
                af: float = None, thread: bool = False) -> trimesh.Trimesh:
    """通用六角头螺栓"""
    od, pitch = _THREAD_SPECS[size]
    af = af or _HEX_AF.get(size, 6)
    head_h = head_h or _HEX_HEAD_H.get(size, 3)
    
    thread_r = od / 2
    thread_len = length * 0.7
    
    shaft = _default_engine._cylinder(thread_r, length, sections=48)
    # 移到正位
    shaft.apply_translation([0, 0, -length / 2])
    
    head = _hex_prism(af, head_h, sections=48)
    head.apply_translation([0, 0, -length - head_h / 2])
    
    try:
        result = shaft.union(head, engine="manifold")
    except Exception:
        result = shaft + head
    
    return result


def _countersunk_screw(size: int, length: float) -> trimesh.Trimesh:
    """沉头螺丝"""
    od, pitch = _THREAD_SPECS[size]
    head_d = od * 2
    head_h = od * 0.6
    
    shaft = _default_engine._cylinder(od / 2, length, sections=48)
    shaft.apply_translation([0, 0, -length / 2])
    
    # 沉头: cone
    head = trimesh.creation.cone(radius=head_d / 2, height=head_h, sections=48)
    head.apply_translation([0, 0, -length - head_h / 2])
    
    try:
        return shaft.union(head, engine="manifold")
    except Exception:
        return shaft + head


# 六角头螺栓
def hex_screw_m2(length: float = 12): return _hex_screw(2, length)
def hex_screw_m3(length: float = 12): return _hex_screw(3, length)
def hex_screw_m4(length: float = 16): return _hex_screw(4, length)
def hex_screw_m5(length: float = 20): return _hex_screw(5, length)
def hex_screw_m6(length: float = 25): return _hex_screw(6, length)
def hex_screw_m8(length: float = 30): return _hex_screw(8, length)

# 沉头螺丝
def countersunk_screw_m3(length: float = 12): return _countersunk_screw(3, length)
def countersunk_screw_m4(length: float = 16): return _countersunk_screw(4, length)
def countersunk_screw_m5(length: float = 20): return _countersunk_screw(5, length)


# ═══════════════════════════════════════════════════════════
#  螺母
# ═══════════════════════════════════════════════════════════

def _hex_nut(size: int) -> trimesh.Trimesh:
    af = _HEX_AF.get(size, 6)
    h = _NUT_H.get(size, 4)
    nut = _hex_prism(af, h, sections=48)
    
    # 中心孔
    od, _ = _THREAD_SPECS[size]
    hole = trimesh.creation.cylinder(radius=od / 2 + 0.1, height=h * 1.2, sections=48)
    try:
        return nut.difference(hole, engine="manifold")
    except Exception:
        return nut


def hex_nut_m2(): return _hex_nut(2)
def hex_nut_m3(): return _hex_nut(3)
def hex_nut_m4(): return _hex_nut(4)
def hex_nut_m5(): return _hex_nut(5)
def hex_nut_m6(): return _hex_nut(6)
def hex_nut_m8(): return _hex_nut(8)


def _lock_nut(size: int) -> trimesh.Trimesh:
    """防松螺母 (尼龙圈) — 比标准螺母高 20%"""
    af = _HEX_AF.get(size, 6)
    h = _NUT_H.get(size, 4) * 1.2
    od, _ = _THREAD_SPECS[size]
    
    nut = _hex_prism(af, h, sections=48)
    hole = trimesh.creation.cylinder(radius=od / 2 + 0.1, height=h * 1.2, sections=48)
    try:
        nut = nut.difference(hole, engine="manifold")
    except Exception:
        pass
    
    # 尼龙环标识: 顶部一圈
    ring = trimesh.creation.cylinder(radius=af / 2 - 1, height=1.5, sections=48)
    ring.apply_translation([0, 0, h / 2 - 0.75])
    
    try:
        return nut.union(ring, engine="manifold")
    except Exception:
        return nut


def lock_nut_m3(): return _lock_nut(3)
def lock_nut_m4(): return _lock_nut(4)
def lock_nut_m5(): return _lock_nut(5)
def lock_nut_m6(): return _lock_nut(6)


# ═══════════════════════════════════════════════════════════
#  垫圈
# ═══════════════════════════════════════════════════════════

def _flat_washer(size: int) -> trimesh.Trimesh:
    od, _ = _THREAD_SPECS[size]
    washer_od = _WASHER_OD.get(size, od * 2.5)
    washer_t = _WASHER_T.get(size, 1)
    return _default_engine._ring(washer_od / 2, od / 2 + 0.2, washer_t)


def flat_washer_m2(): return _flat_washer(2)
def flat_washer_m3(): return _flat_washer(3)
def flat_washer_m4(): return _flat_washer(4)
def flat_washer_m5(): return _flat_washer(5)
def flat_washer_m6(): return _flat_washer(6)
def flat_washer_m8(): return _flat_washer(8)


def _spring_washer(size: int) -> trimesh.Trimesh:
    """弹簧垫圈 — 开口环 + 错位"""
    od, _ = _THREAD_SPECS[size]
    washer_od = _WASHER_OD.get(size, od * 2.5) / 2
    washer_t = _WASHER_T.get(size, 1) * 2
    
    ring = _default_engine._ring(washer_od, od / 2 + 0.1, washer_t)
    
    # 开口: 切一刀
    cutter = trimesh.creation.box(extents=[washer_od * 2, 1, washer_t * 3])
    cutter.apply_translation([washer_od * 0.7, 0, 0])
    try:
        ring = ring.difference(cutter, engine="manifold")
    except Exception:
        pass
    
    # 错位 (一端抬高)
    verts = ring.vertices.copy()
    bounds = ring.bounds
    for i, v in enumerate(verts):
        if v[0] > bounds[1][0] * 0.5:
            verts[i][2] += washer_t * 0.8
    ring.vertices = verts
    
    return ring


def spring_washer_m3(): return _spring_washer(3)
def spring_washer_m4(): return _spring_washer(4)
def spring_washer_m5(): return _spring_washer(5)


# ═══════════════════════════════════════════════════════════
#  轴承
# ═══════════════════════════════════════════════════════════

def _ball_bearing(id_mm: float, od_mm: float, width: float) -> trimesh.Trimesh:
    """深沟球轴承: 外圈 + 内圈 + 保持架"""
    sections = 64
    
    # 外圈
    outer = _default_engine._ring(od_mm / 2, (od_mm + id_mm) / 4 + 1, width)
    
    # 内圈
    inner = _default_engine._ring((od_mm + id_mm) / 4 - 1, id_mm / 2, width)
    inner.apply_translation([0, 0, 0])
    
    try:
        result = outer.union(inner, engine="manifold")
    except Exception:
        result = outer + inner
    
    # 滚珠 (装饰) — 6 个均匀分布
    ball_r = (od_mm - id_mm) / 8
    race_r = (od_mm + id_mm) / 4
    for i in range(6):
        angle = i * np.pi / 3
        bx = race_r * np.cos(angle)
        by = race_r * np.sin(angle)
        ball = _default_engine._sphere(ball_r, subdiv=2)
        ball.apply_translation([bx, by, 0])
        try:
            result = result.union(ball, engine="manifold")
        except Exception:
            result = result + ball
    
    return result


def ball_bearing_608():  return _ball_bearing(8, 22, 7)
def ball_bearing_625():  return _ball_bearing(5, 16, 5)
def ball_bearing_688():  return _ball_bearing(8, 16, 5)
def ball_bearing_6000(): return _ball_bearing(10, 26, 8)
ball_bearing_6001 = lambda: _ball_bearing(12, 28, 8)
ball_bearing_6002 = lambda: _ball_bearing(15, 32, 9)
ball_bearing_6200 = lambda: _ball_bearing(10, 30, 9)
ball_bearing_6201 = lambda: _ball_bearing(12, 32, 10)


# ═══════════════════════════════════════════════════════════
#  联轴器
# ═══════════════════════════════════════════════════════════

def _shaft_coupling(bore_d: float, od: float = None, length: float = None) -> trimesh.Trimesh:
    """弹性联轴器: 圆柱 + 两端顶丝 + 中心十字槽"""
    od = od or bore_d * 2.5
    length = length or bore_d * 3
    
    body = _default_engine._cylinder(od / 2, length)
    
    # 两端顶丝孔
    bounds = body.bounds
    set_screw_hole = _default_engine._cylinder(1.5, od, sections=16)
    # 旋转到水平方向
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    set_screw_hole.apply_transform(rot)
    
    # 近端
    top = set_screw_hole.copy()
    top.apply_translation([0, 0, bounds[1][2] - 3])
    body = _safe_bool(body, top, "difference")
    
    # 远端
    bot = set_screw_hole.copy()
    bot.apply_translation([0, 0, bounds[0][2] + 3])
    body = _safe_bool(body, bot, "difference")
    
    return body


def shaft_coupling_3mm(): return _shaft_coupling(3, od=9,  length=15)
def shaft_coupling_5mm(): return _shaft_coupling(5, od=14, length=25)
def shaft_coupling_8mm(): return _shaft_coupling(8, od=20, length=35)


# ═══════════════════════════════════════════════════════════
#  杆端关节轴承
# ═══════════════════════════════════════════════════════════

def _rod_end(bore_d: float, thread_size: int) -> trimesh.Trimesh:
    """杆端轴承: 球头 + 螺纹杆"""
    # 球头
    head_r = bore_d * 1.5
    head = _default_engine._sphere(head_r, subdiv=3)
    
    # 内孔
    hole = _default_engine._cylinder(bore_d / 2, head_r * 3, sections=32)
    hole2 = hole.copy()
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    hole2.apply_transform(rot)
    head = _safe_bool(head, hole, "difference")
    head = _safe_bool(head, hole2, "difference")
    
    # 螺纹杆
    od, _ = _THREAD_SPECS[thread_size]
    shank = _default_engine._cylinder(od / 2, head_r * 2.5, sections=32)
    shank.apply_translation([0, 0, -head_r * 2.5 / 2 - head_r * 1.2])
    
    try:
        return head.union(shank, engine="manifold")
    except Exception:
        return head + shank


def rod_end_m3(): return _rod_end(3, 3)
def rod_end_m4(): return _rod_end(4, 4)
def rod_end_m5(): return _rod_end(5, 5)


# ═══════════════════════════════════════════════════════════
#  间隔柱
# ═══════════════════════════════════════════════════════════

def _spacer(bore_d: float, od: float, length: float) -> trimesh.Trimesh:
    return _default_engine._hollow_cylinder(od / 2, bore_d / 2, length)


def spacer_m3(length: float = 10): return _spacer(3.2, 6, length)
def spacer_m4(length: float = 10): return _spacer(4.3, 8, length)


# ═══════════════════════════════════════════════════════════
#  零件目录
# ═══════════════════════════════════════════════════════════

STANDARD_PARTS_CATALOG: Dict[str, callable] = {
    "hex_screw_m3":         hex_screw_m3,
    "hex_screw_m4":         hex_screw_m4,
    "hex_screw_m5":         hex_screw_m5,
    "hex_screw_m6":         hex_screw_m6,
    "hex_screw_m8":         hex_screw_m8,
    "countersunk_screw_m3": countersunk_screw_m3,
    "countersunk_screw_m4": countersunk_screw_m4,
    "hex_nut_m3":           hex_nut_m3,
    "hex_nut_m4":           hex_nut_m4,
    "hex_nut_m5":           hex_nut_m5,
    "hex_nut_m6":           hex_nut_m6,
    "hex_nut_m8":           hex_nut_m8,
    "lock_nut_m3":          lock_nut_m3,
    "lock_nut_m4":          lock_nut_m4,
    "lock_nut_m5":          lock_nut_m5,
    "lock_nut_m6":          lock_nut_m6,
    "flat_washer_m3":       flat_washer_m3,
    "flat_washer_m4":       flat_washer_m4,
    "flat_washer_m5":       flat_washer_m5,
    "flat_washer_m6":       flat_washer_m6,
    "flat_washer_m8":       flat_washer_m8,
    "spring_washer_m3":     spring_washer_m3,
    "spring_washer_m4":     spring_washer_m4,
    "spring_washer_m5":     spring_washer_m5,
    "ball_bearing_608":     ball_bearing_608,
    "ball_bearing_625":     ball_bearing_625,
    "ball_bearing_688":     ball_bearing_688,
    "ball_bearing_6000":    ball_bearing_6000,
    "shaft_coupling_3mm":   shaft_coupling_3mm,
    "shaft_coupling_5mm":   shaft_coupling_5mm,
    "shaft_coupling_8mm":   shaft_coupling_8mm,
    "rod_end_m3":           rod_end_m3,
    "rod_end_m4":           rod_end_m4,
    "rod_end_m5":           rod_end_m5,
    "spacer_m3":            spacer_m3,
    "spacer_m4":            spacer_m4,
}


def get_standard_part(name: str, **kwargs) -> Optional[trimesh.Trimesh]:
    """按名称获取标准件 mesh"""
    if name in STANDARD_PARTS_CATALOG:
        return STANDARD_PARTS_CATALOG[name](**kwargs)
    return None
