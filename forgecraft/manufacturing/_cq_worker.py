"""
CadQuery 工作进程 — 独立进程，在 Python 3.12 + cadquery 下运行

通过 stdin JSON 接收零件规格，输出真 B-Rep STEP/STL 文件。

不依赖 forgecraft 任何模块，纯 cadquery + Python 标准库。

协议:
  stdin:  {"parts": [...], "output_dir": "...", "format": "step"|"stl"|"both"}
  stderr: 进度信息
  stdout:  {"files": [...], "errors": [...]}
"""

import sys
import os
import json
import math

try:
    import cadquery as cq
except ImportError:
    print(json.dumps({"errors": ["cadquery not installed"]}))
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
#  几何构建
# ═══════════════════════════════════════════════════════════

def build_cylinder(radius, length, sections=64):
    """圆柱体 (沿 Z 轴，中心在原点)"""
    return (
        cq.Workplane("XY")
        .circle(radius)
        .extrude(length, both=True)
    )

def build_box(lx, ly, lz):
    """长方体 (X=length, Y=width, Z=height)"""
    return cq.Workplane("XY").box(lx, ly, lz)

def build_sphere(radius):
    """球体"""
    return cq.Workplane("XY").sphere(radius)

def build_hemisphere(radius):
    """上半球 (Z >= 0)"""
    sph = cq.Workplane("XY").sphere(radius)
    box = cq.Workplane("XY").box(radius*3, radius*3, radius*1.5)
    return sph.cut(box)

def build_hollow_cylinder(outer_r, inner_r, length):
    """空心圆柱"""
    outer = build_cylinder(outer_r, length)
    inner = build_cylinder(inner_r, length * 1.1)
    return outer.cut(inner)

def build_ring(outer_r, inner_r, height):
    """圆环"""
    return build_hollow_cylinder(outer_r, inner_r, height)

def build_helical_spring(coil_r, height, wire_r, turns=8):
    """螺旋弹簧 — 螺旋扫掠"""
    path = cq.Workplane("XZ")
    pts = []
    n = turns * 32
    for i in range(n + 1):
        t = i * 2 * math.pi * turns / n
        z = -height/2 + i * height / n
        pts.append((coil_r * math.cos(t), z, coil_r * math.sin(t)))
    
    wire = cq.Workplane("XY").circle(wire_r)
    return (
        cq.Workplane("XY")
        .spline(pts)
        .close()
        .sweep(wire)
    )

def add_chamfer_ends(solid, radius):
    """两端倒角"""
    chamfer_h = min(1.5, radius * 0.25)
    if chamfer_h < 0.2:
        return solid
    try:
        solid = solid.faces(">Z").chamfer(chamfer_h)
    except Exception:
        pass
    try:
        solid = solid.faces("<Z").chamfer(chamfer_h)
    except Exception:
        pass
    return solid

def add_flanges(solid, body_r, flange_r, flange_h):
    """双端法兰面"""
    try:
        top_flange = build_cylinder(flange_r, flange_h)
        top_flange = top_flange.translate((0, 0, body_r if 'length' not in dir() else 0))
        bot_flange = build_cylinder(flange_r, flange_h)
        solid = solid.union(top_flange).union(bot_flange)
    except Exception:
        pass
    return solid

def add_center_bore(solid, shaft_d, total_h):
    """中心轴孔"""
    drill = build_cylinder(shaft_d / 2, total_h * 1.1)
    return solid.cut(drill)

def add_mounting_holes(solid, pattern_r, hole_d, depth):
    """4孔安装阵列 (±X, ±Y)"""
    positions = [(pattern_r, 0), (-pattern_r, 0), (0, pattern_r), (0, -pattern_r)]
    for dx, dy in positions:
        drill = build_cylinder(hole_d / 2, depth + 1)
        drill = drill.translate((dx, dy, 0))
        try:
            solid = solid.cut(drill)
        except Exception:
            pass
    return solid


# ═══════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════

def build_part(part_def):
    """从零件定义构建 cadquery solid"""
    shape = part_def.get("geometry_type", "cylinder")
    params = part_def.get("params", {})
    actuated = part_def.get("actuated", False)
    
    length = float(params.get("length", 100))
    radius = float(params.get("radius", 20))
    width = float(params.get("width", length * 0.5))
    height = float(params.get("height", radius * 2))
    
    # ── 基础几何 ──
    if shape == "cylinder":
        solid = build_cylinder(radius, length)
    elif shape == "box":
        solid = build_box(length, width, height)
    elif shape == "sphere":
        solid = build_sphere(radius)
    elif shape == "hemisphere":
        solid = build_hemisphere(radius)
    elif shape == "hemisphere_shell":
        outer = build_hemisphere(radius)
        wall = float(params.get("wall_thickness", max(0.2, radius * 0.15)))
        inner = build_hemisphere(max(0.1, radius - wall))
        inner = inner.translate((0, 0, wall * 0.5))
        solid = outer.cut(inner)
    elif shape in ("helical_spring", "helix"):
        turns = int(params.get("turns", 8))
        wire_r = float(params.get("wire_radius", params.get("radius", 3)))
        solid = build_helical_spring(radius, length, wire_r, turns)
    elif shape == "hollow_cylinder":
        wall = float(params.get("wall_thickness", max(0.2, radius * 0.2)))
        inner_r = max(0.1, radius - wall)
        solid = build_hollow_cylinder(radius, inner_r, length)
    elif shape == "hollow_square":
        wall = float(params.get("wall_thickness", max(0.2, width * 0.15)))
        outer = build_box(length, width, height)
        inner = build_box(
            max(0.1, length - 2*wall),
            max(0.1, width - 2*wall),
            height * 2
        )
        inner = inner.translate((0, 0, height * 0.5))
        solid = outer.cut(inner)
    elif shape == "ring":
        inner_r = float(params.get("inner_radius", radius * 0.7))
        solid = build_ring(radius, inner_r, height)
    elif shape == "rectangular_prism":
        solid = build_box(length, width, height)
    elif shape == "cup_shaped":
        solid = build_cylinder(radius, length)
        cutter = build_sphere(radius * 2)
        cutter = cutter.translate((0, 0, length/2 - radius))
        solid = solid.cut(cutter)
    elif shape in ("chip_qfn", "module_lga", "pcb_rectangle"):
        solid = build_box(length, width, max(0.5, height))
    else:
        solid = build_cylinder(radius, length)
    
    # ── L2: 端部倒角 ──
    if not actuated and shape in ("cylinder", "box", "hollow_cylinder"):
        solid = add_chamfer_ends(solid, radius)
    
    # ── L3: 法兰 + 轴孔 + 安装孔 ──
    if actuated:
        flange_r = max(radius * 1.4, radius + 3)
        flange_h = min(4.0, max(1.5, length * 0.08))
        
        # 法兰
        try:
            top = build_cylinder(flange_r, flange_h * 2)
            top = top.translate((0, 0, length/2 + flange_h))
            solid = solid.union(top)
        except Exception:
            pass
        try:
            bot = build_cylinder(flange_r, flange_h * 2)
            bot = bot.translate((0, 0, -length/2 - flange_h))
            solid = solid.union(bot)
        except Exception:
            pass
        
        # 轴孔
        shaft_d = float(params.get("shaft_diameter", max(2.0, radius * 0.8)))
        total_h = length + flange_h * 4
        drill = build_cylinder(shaft_d / 2, total_h * 1.1)
        solid = solid.cut(drill)
        
        # 安装孔 x4
        bolt_r = flange_r * 0.6
        hole_d = float(params.get("mounting_hole_diameter", 3.2))
        positions = [(bolt_r, 0), (-bolt_r, 0), (0, bolt_r), (0, -bolt_r)]
        for dx, dy in positions:
            hole = build_cylinder(hole_d / 2, flange_h * 3)
            hole = hole.translate((dx, dy, length/2 + flange_h))
            try:
                solid = solid.cut(hole)
            except Exception:
                pass
    
    # 移到世界坐标
    pos = part_def.get("position", [0, 0, 0])
    if any(pos):
        solid = solid.translate(tuple(float(p) for p in pos))
    
    return solid


def main():
    raw = sys.stdin.read()
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"errors": [f"JSON parse: {e}"]}))
        sys.exit(1)
    
    parts_def = request.get("parts", [])
    output_dir = request.get("output_dir", "cadquery_output")
    fmt = request.get("format", "step")
    name = request.get("name", "ForgeCraft_Body")
    
    os.makedirs(output_dir, exist_ok=True)
    files = []
    errors = []
    
    if not parts_def:
        errors.append("No parts in request")
        print(json.dumps({"files": [], "errors": errors}))
        return
    
    # 构建装配体
    assembly = cq.Assembly(name=name)
    
    for i, part_def in enumerate(parts_def):
        pname = part_def.get("part_id", f"part_{i}")
        try:
            solid = build_part(part_def)
            # 获取颜色
            color = part_def.get("color", [150, 150, 150])
            rgba = tuple(c / 255 if c > 1 else c for c in color[:3]) + (1.0,)
            assembly.add(solid, name=pname, color=cq.Color(*rgba))
        except Exception as e:
            errors.append(f"{pname}: {e}")
            continue
    
    if errors and len(assembly.children) == 0:
        print(json.dumps({"files": [], "errors": errors}))
        return
    
    # 导出
    if fmt in ("step", "both"):
        step_path = os.path.join(output_dir, f"{name}.step")
        try:
            assembly.save(step_path)
            files.append(step_path)
        except Exception as e:
            errors.append(f"STEP export: {e}")
    
    if fmt in ("stl", "both"):
        stl_dir = os.path.join(output_dir, "stl")
        os.makedirs(stl_dir, exist_ok=True)
        for i, part_def in enumerate(parts_def):
            pname = part_def.get("part_id", f"part_{i}")
            try:
                solid = build_part(part_def)
                stl_path = os.path.join(stl_dir, f"{pname}.stl")
                cq.exporters.export(solid, stl_path, tolerance=0.01, angularTolerance=0.1)
                files.append(stl_path)
            except Exception as e:
                errors.append(f"STL {pname}: {e}")
    
    print(json.dumps({"files": files, "errors": errors}))


if __name__ == "__main__":
    main()
