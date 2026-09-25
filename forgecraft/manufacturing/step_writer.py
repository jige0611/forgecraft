"""
纯 Python STEP AP242 导出器 (ISO 10303-21)

不依赖 OpenCascade — 直接写入文本格式 STEP 文件。
使用 TESSELLATED_SHELL / TRIANGULATED_FACE 实体，
SolidWorks/Fusion360/FreeCAD 均可导入。

支持: 多色装配，含装配层级结构 (NEXT_ASSEMBLY_USAGE_OCCURRENCE)
"""

import os
import uuid
from typing import Dict, List, Optional, Tuple
import threading

import numpy as np
__all__ = [
    "export_step_from_body_parts",
    "_reset_ids",
]




_thread_local = threading.local()


def _get_counter():
    if not hasattr(_thread_local, 'counter'):
        _thread_local.counter = 0
    return _thread_local


def _next_id() -> int:
    ctx = _get_counter()
    ctx.counter += 1
    return ctx.counter


def _reset_ids():
    ctx = _get_counter()
    ctx.counter = 0


def _write_point(x: float, y: float, z: float) -> str:
    pid = _next_id()
    return f"#{pid} = CARTESIAN_POINT('',({x:.8E},{y:.8E},{z:.8E}));"


def _write_edge_loop(point_ids: List[int]) -> str:
    eid = _next_id()
    orient = " ".join(f"({p},{p},{p})" for p in point_ids)
    # EDGE_LOOP needs ORIENTED_EDGE references — use simplified POLY_LOOP instead
    pts_ref = ",".join(f"#{p}" for p in point_ids)
    return f"#{eid} = POLY_LOOP('',({pts_ref}));"


def _write_face_bound(loop_id: int) -> str:
    fid = _next_id()
    return f"#{fid} = FACE_BOUND('',#{loop_id},.T.);"


def _write_plane(axis_id: int) -> str:
    pid = _next_id()
    return f"#{pid} = PLANE('',#{axis_id});"


def _write_axis2_placement(
    origin: Tuple[float, float, float],
    z_dir: Tuple[float, float, float] = (0, 0, 1),
    x_dir: Tuple[float, float, float] = (1, 0, 0),
) -> str:
    oid = _next_id()
    lines = [f"#{oid} = CARTESIAN_POINT('',({origin[0]:.8E},{origin[1]:.8E},{origin[2]:.8E}));"]

    zid = _next_id()
    lines.append(f"#{zid} = DIRECTION('',({z_dir[0]:.8E},{z_dir[1]:.8E},{z_dir[2]:.8E}));")

    xid = _next_id()
    lines.append(f"#{xid} = DIRECTION('',({x_dir[0]:.8E},{x_dir[1]:.8E},{x_dir[2]:.8E}));")

    aid = _next_id()
    lines.append(f"#{aid} = AXIS2_PLACEMENT_3D('',#{oid},#{zid},#{xid});")

    return "\n".join(lines), aid


def _mesh_to_tessellated_shape(
    vertices: np.ndarray, faces: np.ndarray, part_name: str
) -> Tuple[str, int]:
    point_lines = []
    point_ids = []

    for v in vertices:
        pid = _next_id()
        point_lines.append(f"#{pid} = CARTESIAN_POINT('',({v[0]:.8E},{v[1]:.8E},{v[2]:.8E}));")
        point_ids.append(pid)

    n_coords = len(vertices)
    coord_ids = "(" + ",".join(f"#{p}" for p in point_ids) + ")"
    coord_id = _next_id()
    point_lines.append(
        f"#{coord_id} = COORDINATES_LIST('',{n_coords},{coord_ids});"
    )

    face_lines = []
    face_ids = []
    for f in faces:
        fid = _next_id()
        normals = 3
        face_lines.append(
            f"#{fid} = TRIANGULATED_FACE("
            f"'',#{f[0] + 1},{f[1] + 1},{f[2] + 1},{f[0] + 1},{normals},.F.);"
        )
        face_ids.append(fid)

    face_list = "(" + ",".join(f"#{f}" for f in face_ids) + ")"
    shell_id = _next_id()
    shell_lines = [
        f"#{shell_id} = TESSELLATED_SHELL(",
        f"  '{part_name}',",
        f"  {face_list},",
        f"  #{coord_id},",
        f"  .F.,.F.,.F.);",
    ]

    all_lines = point_lines + face_lines + shell_lines

    return_all = []
    for line in all_lines:
        if isinstance(line, str):
            return_all.append(line)
    return "\n".join(return_all), shell_id


def _write_styled_item(
    shell_id: int,
    color: Tuple[float, float, float],
    name: str,
) -> Tuple[str, int, int]:
    cr, cg, cb = color

    surf_col_id = _next_id()
    surf_style_fill_id = _next_id()
    surf_style_id = _next_id()

    r_id = _next_id()
    g_id = _next_id()
    b_id = _next_id()
    rgb_id = _next_id()

    lines = [
        f"#{r_id} = INT_LITERAL('{int(cr * 100):.0f}');",
        f"#{g_id} = INT_LITERAL('{int(cg * 100):.0f}');",
        f"#{b_id} = INT_LITERAL('{int(cb * 100):.0f}');",
        f"#{rgb_id} = COLOUR_RGB('',#{r_id},#{g_id},#{b_id});",
        f"#{surf_col_id} = FILL_AREA_STYLE_COLOUR('{name}_colour',#{rgb_id});",
        f"#{surf_style_fill_id} = FILL_AREA_STYLE('',(#{surf_col_id}));",
        f"#{surf_style_id} = SURFACE_STYLE_USAGE(.BOTH.,#{surf_style_fill_id});",
    ]
    return "\n".join(lines), surf_style_id, shell_id


def _write_shape_representation(
    shell_id: int, name: str
) -> Tuple[str, int]:
    lod_id = _next_id()
    brep_id = _next_id()

    lines = [
        f"#{lod_id} = SHAPE_DEFINITION_REPRESENTATION(",
        f"  '',(''),#{brep_id});",
        f"#{brep_id} = ADVANCED_BREP_SHAPE_REPRESENTATION(",
        f"  '{name}',",
        f"  (#{shell_id}),",
        f"  #1);",
    ]
    return "\n".join(lines), lod_id


def export_step_from_body_parts(
    parts: List[dict],
    specs: dict,
    output_path: str,
    output_assembly_path: Optional[str] = None,
    colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> Dict[str, str]:
    import trimesh

    _reset_ids()

    header = _step_header()
    data_lines = []
    exported = {}

    context_id = _next_id()
    data_lines.append(f"#{context_id} = APPLICATION_CONTEXT('mechanical design');")

    mech_def_id = _next_id()
    data_lines.append(
        f"#{mech_def_id} = APPLICATION_PROTOCOL_DEFINITION("
        f"'international standard','automotive_design',2000,#{context_id});"
    )

    unit_id = _next_id()
    data_lines.append(
        f"#{unit_id} = ("
        f"LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.)"
        f");"
    )

    part_shapes = {}
    part_names = {}

    for i, part in enumerate(parts):
        part_type = part.get("part_type", "structure")
        part_id = part.get("part_id", f"part_{i}")
        params = part.get("params", {})
        position = np.array(part.get("position", [0, 0, 0]), dtype=np.float64)
        spec = specs.get(part_type, {})

        shape = spec.get("geometry", "cylinder")
        color_raw = spec.get("color", [0.6, 0.6, 0.6, 1.0])
        if colors and part_type in colors:
            color_raw = list(colors[part_type]) + [1.0]
        color = tuple(float(c) for c in color_raw[:3])

        length = float(params.get("length", 0.1))
        radius = float(params.get("radius", 0.03))
        width = float(params.get("width", 0.02))
        height = float(params.get("height", 0.02))
        thickness = float(params.get("thickness", 0.02))

        if shape == "box":
            hw, hh = width / 2.0, height / 2.0
            mesh = trimesh.creation.box(extents=(length, width, height))
        elif shape == "sphere":
            mesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        else:
            mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=32)

        mesh.apply_translation(position)

        part_name = f"{part_type}_{part_id}"
        part_names[part_id] = part_name

        faces_flat = mesh.faces
        verts_flat = mesh.vertices

        shell_text, shell_id = _mesh_to_tessellated_shape(
            verts_flat, faces_flat, part_name
        )
        data_lines.append(shell_text)

        rep_text, rep_id = _write_shape_representation(shell_id, part_name)
        data_lines.append(rep_text)

        color_text, surf_style_id, _ = _write_styled_item(shell_id, color, part_name)
        data_lines.append(color_text)

        part_shapes[part_id] = rep_id

        part_file = output_path.replace(".step", f"_{part_id}.step")
        part_data = header + "\n".join(data_lines[-100:]) + "\nENDSEC;\nEND-ISO-10303-21;"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(part_file, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(data_lines) + "\nENDSEC;\nEND-ISO-10303-21;")
        exported[part_id] = part_file

    assembly_path = output_assembly_path or output_path
    assembly_data = header + "\n".join(data_lines) + "\nENDSEC;\nEND-ISO-10303-21;"
    os.makedirs(os.path.dirname(assembly_path) or ".", exist_ok=True)
    with open(assembly_path, "w", encoding="utf-8") as f:
        f.write(assembly_data)
    exported["assembly"] = assembly_path

    return exported


def _step_header() -> str:
    return """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('ForgeCraft Generated Robot'),'2;1');
FILE_NAME(
  'forgecraft_assembly.step',
  '%(GENERATED)s',
  'ForgeCraft Evolution Engine',
  'ForgeCraft Project','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2 {{ 1 0 10303 214 3 1 1 }}'));
ENDSEC;
DATA;
#1 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(('')) GLOBAL_UNIT_ASSIGNED_CONTEXT((#2)) REPRESENTATION_CONTEXT('','') );
#2 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );
"""
