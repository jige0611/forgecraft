"""
STEP AP242 多层次装配体导出器

生成符合 ISO 10303-242 的 STEP 文件, 包含:
- 装配体层级 (多级 PRODUCT + NEXT_ASSEMBLY_USAGE_OCCURRENCE)
- 镶嵌几何 (TESSELLATED_ITEM + COORDINATES_LIST + TRIANGULATED_FACE)
- 材料指定 (MATERIAL_DESIGNATION)
- 单位: 毫米 / 千克

加工厂可直接导入 SolidWorks / Fusion 360 / CATIA。

格式参考: ISO 10303-242:2020 (AP242 managed model based 3D engineering)
"""

from __future__ import annotations

import datetime
import os
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh


# ================================================================
# STEP 实体模板
# ================================================================

STEP_HEADER = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('ForgeCraft Assembly'),'2;1');
FILE_NAME(
  '{filename}',
  '{timestamp}',
  ('ForgeCraft',''),
  ('',''),
  'ForgeCraft STEP Exporter',
  'ForgeCraft',
  '');
FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF',));
ENDSEC;
"""


def _step_id_generator():
    """生成连续 STEP 实体 ID"""
    n = 10
    while True:
        n += 1
        yield f"#{n}"


def _step_real(v: float, precision: int = 6) -> str:
    return f"{v:.{precision}f}"


def _step_list(values, fmt=str):
    return "(" + ",".join(fmt(v) for v in values) + ")"


def _step_coordinates(vertices: np.ndarray) -> str:
    """坐标列表: COORDINATES_LIST ( (x,y,z), ... )"""
    lines = []
    for v in vertices:
        lines.append(f"    ({_step_real(v[0])},{_step_real(v[1])},{_step_real(v[2])})")
    return "(" + ",\n".join(lines) + ")"


def _step_normals(faces: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """计算面法向量"""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    mag = np.linalg.norm(normals, axis=1, keepdims=True)
    mag[mag < 1e-12] = 1.0
    return normals / mag


def _generate_assembly_step(
    assembly_name: str,
    parts: List[Tuple[str, str, np.ndarray, np.ndarray, str]],
    #       (part_name, part_type, vertices, faces, material_name)
    output_path: str,
    description: str = "",
) -> str:
    """生成多层级装配体 STEP AP242 文件

    Returns: 文件路径
    """
    G = _step_id_generator()

    # ── 上下文和单位 ──
    app_ctx = next(G)
    si_unit = next(G)
    length_unit = next(G)
    mass_unit = next(G)
    geom_ctx = next(G)

    context_lines = f"""
{app_ctx}=APPLICATION_CONTEXT('managed model based 3d engineering');
{si_unit}=(
  LENGTH_UNIT()
  NAMED_UNIT(*)
  SI_UNIT(.MILLI.,.METRE.)
);
{length_unit}=LENGTH_UNIT('millimetre');
{mass_unit}=(
  MASS_UNIT()
  NAMED_UNIT(*)
  SI_UNIT(.KILO.,.GRAM.)
);
{geom_ctx}=(
  GEOMETRIC_REPRESENTATION_CONTEXT(3)
  GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({length_unit}))
  GLOBAL_UNIT_ASSIGNED_CONTEXT(({si_unit},{mass_unit}))
  REPRESENTATION_CONTEXT('','')
);
"""

    # ── 装配体 PRODUCT ──
    asm_prod = next(G)
    asm_pdf = next(G)
    asm_pd = next(G)
    asm_pds = next(G)

    assembly_lines = f"""
{asm_prod}=PRODUCT('{assembly_name}','{assembly_name}','',({geom_ctx}));
{asm_pdf}=PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE('','',{asm_prod},.NOT_KNOWN.);
{asm_pd}=PRODUCT_DEFINITION('','',{asm_pdf},{app_ctx});
{asm_pds}=PRODUCT_DEFINITION_SHAPE('', '', {asm_pd});
"""

    # ── 子零件 ──
    entity_lines = []
    nauo_ids = []

    for idx, (part_name, part_type, vertices, faces, material) in enumerate(parts):
        pn = part_name.replace(" ", "_")
        prod = next(G)
        pdf = next(G)
        pd = next(G)
        pds = next(G)
        nauo = next(G)

        # 镶嵌几何
        coord_entity = next(G)
        tri_face = next(G)
        shape_rep = next(G)
        sr_item = next(G)
        tess_item = next(G)

        face_count = len(faces)
        normal_count = face_count
        norms = _step_normals(faces, vertices)

        # COORDINATES_LIST
        coords_str = _step_coordinates(vertices)

        # 法向量列表 (每个面对应1个法向量)
        norm_str = _step_coordinates(norms)

        # TRIANGULATED_FACE: indices + normals
        idx_lines = []
        for i, f in enumerate(faces):
            idx_lines.append(
                f"    ({_step_list(f, lambda x: str(int(x)))},{_step_list(norms[i], lambda v: _step_real(float(v)))})"
            )

        entity_lines.append(f"""
{prod}=PRODUCT('{pn}','{part_type}','',({geom_ctx}));
{pdf}=PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE('','',{prod},.NOT_KNOWN.);
{pd}=PRODUCT_DEFINITION('','',{pdf},{app_ctx});
{pds}=PRODUCT_DEFINITION_SHAPE('','',{pd});

{coord_entity}=COORDINATES_LIST('',{coords_str});

{tri_face}=TRIANGULATED_FACE(
  'triangulated_face',
  {face_count},
  ({coord_entity}),
  {norm_str},
  ({",".join(idx_lines)}),
  .F.
);

{tess_item}=TESSELLATED_ITEM('',{tri_face});
{shape_rep}=SHAPE_REPRESENTATION('',({tess_item}),{geom_ctx});
{sr_item}=SHAPE_REPRESENTATION_RELATIONSHIP('','',{shape_rep},{pds});

{nauo}=NEXT_ASSEMBLY_USAGE_OCCURRENCE('{pn}','{part_type}','',{asm_pd},{prod},'');
""")

        nauo_ids.append(nauo)

    # ── 材料指定 ──
    for idx, (part_name, part_type, vertices, faces, material) in enumerate(parts):
        mat_entity = next(G)
        mat_desig = next(G)
        entity_lines.append(f"""
{mat_entity}=MATERIAL('{material}','',({app_ctx}));
{mat_desig}=MATERIAL_DESIGNATION('','',{nauo_ids[idx]},{mat_entity});
""")

    # ── 组装 ──
    body = ""
    body += context_lines
    body += assembly_lines
    body += "".join(entity_lines)

    # 写入文件
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        f.write(STEP_HEADER.format(filename=os.path.basename(output_path),
                                   timestamp=timestamp))
        f.write("DATA;\n")
        # 重新编号实体
        lines = body.strip().split("\n")
        counter = 10
        id_map = {}
        new_lines = []
        for line in lines:
            if line.startswith("#") and "=" in line:
                old_id = line.split("=")[0].strip()
                new_id = f"#{counter}"
                id_map[old_id] = new_id
                counter += 1
        for line in lines:
            nl = line
            if line.startswith("#") and "=" in line:
                old_id = line.split("=")[0].strip()
                nl = line.replace(old_id + "=", id_map[old_id] + "=")
            else:
                for old_id, new_id in id_map.items():
                    nl = nl.replace(old_id, new_id)
            new_lines.append(nl)

        f.write("\n".join(new_lines))
        f.write("\nENDSEC;\nEND-ISO-10303-21;\n")

    return output_path


def export_step(
    body_data: dict,
    gen,
    output_path: str,
    assembly_name: str = "ForgeCraft_Assembly",
    quality: str = "high",
) -> str:
    """导出 STEP AP242 装配体文件

    Args:
        body_data: {parts, joints}
        gen: ParametricGenerator 实例
        output_path: 输出 .stp 文件路径
        assembly_name: 装配体名称
        quality: 几何质量

    Returns:
        输出文件路径
    """
    from forgecraft.geometry.materials import get_material

    parts = body_data.get("parts", [])

    step_parts = []
    for p in parts:
        pt = p.get("part_type", "unknown")
        pid = p.get("part_id", "unknown")
        params = p.get("params", {})
        mesh = gen.make(pt, params)
        if mesh is None:
            continue
        mat = get_material(pt)
        step_parts.append((pid, pt, mesh.vertices, mesh.faces, mat.name))

    return _generate_assembly_step(
        assembly_name=assembly_name,
        parts=step_parts,
        output_path=output_path,
    )


def export_step_parts(
    body_data: dict,
    gen,
    output_dir: str,
    assembly_name: str = "ForgeCraft_Assembly",
    quality: str = "high",
) -> Dict[str, str]:
    """导出 STEP 装配体 + 每个零件独立 STEP

    Returns:
        {part_id: filepath, ..., 'assembly': filepath}
    """
    from forgecraft.geometry.materials import get_material

    os.makedirs(output_dir, exist_ok=True)
    result = {}

    parts = body_data.get("parts", [])

    for p in parts:
        pt = p.get("part_type", "unknown")
        pid = p.get("part_id", "unknown")
        params = p.get("params", {})
        mesh = gen.make(pt, params)
        if mesh is None:
            continue
        mat = get_material(pt)
        fname = os.path.join(output_dir, f"{pid}.stp")
        _generate_assembly_step(
            assembly_name=pid,
            parts=[(pid, pt, mesh.vertices, mesh.faces, mat.name)],
            output_path=fname,
        )
        result[pid] = fname

    # 装配体
    asm_path = os.path.join(output_dir, f"{assembly_name}.stp")
    export_step(body_data, gen, asm_path, assembly_name, quality)
    result["assembly"] = asm_path

    return result
