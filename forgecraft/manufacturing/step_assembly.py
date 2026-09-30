"""
STEP AP242 装配增强 — 支持装配层级 + 材料属性

在当前 tessellated shell 基础上:
  1. 添加 NEXT_ASSEMBLY_USAGE_OCCURRENCE 装配关系
  2. 添加 MATERIAL_DESIGNATION 材料标注
  3. 添加 PRODUCT_DEFINITION_FORMATION 产品层级

纯 Python 实现, 不依赖 OpenCascade。
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

from forgecraft.manufacturing.step_writer import (
    _next_id, _reset_ids, _mesh_to_tessellated_shape,
)
from forgecraft.core.morphology import MechanicalBody

_logger = logging.getLogger(__name__)

__all__ = [
    "_write_product",
    "_write_assembly_relation",
    "export_assembly_step",
]




def _write_product(product_name: str, description: str = "") -> List[str]:
    pid = _next_id()
    pd_id = _next_id()
    pdf_id = _next_id()
    lines = [
        f"#{pid} = PRODUCT('{product_name}','{description}','',(#));",
        f"#{pd_id} = PRODUCT_DEFINITION('','',#{pid},#);",
        f"#{pdf_id} = PRODUCT_DEFINITION_FORMATION('','',#{pd_id});",
    ]
    return lines, pid, pd_id, pdf_id


def _write_assembly_relation(
    parent_pdf: int, child_pdf: int,
    transform: Optional[np.ndarray] = None,
) -> List[str]:
    """生成 NEXT_ASSEMBLY_USAGE_OCCURRENCE。"""
    rel_id = _next_id()
    n_id = _next_id()
    old_pid = _next_id()
    lines = [
        f"#{rel_id} = NEXT_ASSEMBLY_USAGE_OCCURRENCE('','','','',#{parent_pdf},#{child_pdf},$);",
    ]
    return lines, rel_id


def _write_material(name: str, density: float = 1000.0) -> List[str]:
    """生成 MATERIAL_DESIGNATION。"""
    md_id = _next_id()
    mat_id = _next_id()
    prop_id = _next_id()
    lines = [
        f"#{mat_id} = MATERIAL('{name}','');",
        f"#{prop_id} = PRODUCT_RELATED_PRODUCT_CATEGORY('material','',(#));",
        f"#{md_id} = MATERIAL_DESIGNATION('',#{mat_id},#);",
    ]
    return lines, md_id


def export_assembly_step(
    body: MechanicalBody,
    output_path: str,
    part_meshes: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> str:
    """
    导出带装配层级的 AP242 STEP 文件。

    Args:
        body: 机械体
        output_path: 输出路径
        part_meshes: {part_id: (vertices, faces)} 可选预计算网格

    Returns:
        输出文件路径
    """
    _reset_ids()
    lines = []

    # HEADER
    lines.append("ISO-10303-21;")
    lines.append("HEADER;")
    lines.append("FILE_DESCRIPTION(('ForgeCraft Assembly'),'2;1');")
    lines.append(f"FILE_NAME('{body.name}','','','','','','');")
    lines.append("FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF'));")
    lines.append("ENDSEC;")
    lines.append("DATA;")

    # 根产品
    root_parts = []
    for node_id in body.depth_first_order():
        part = body.get_part(node_id)
        parent = body.parent_of(node_id)
        root_parts.append((node_id, part, parent))

    # 为每个零件生成产品定义
    product_map = {}
    for node_id, part, parent in root_parts:
        prod_lines, pid, pd_id, pdf_id = _write_product(
            part.part_type, f"part {node_id[:6]}"
        )
        lines.extend(prod_lines)
        product_map[node_id] = {
            "product_id": pid,
            "pd_id": pd_id,
            "pdf_id": pdf_id,
            "part": part,
        }

    # 装配关系
    for node_id, part, parent in root_parts:
        if parent and parent in product_map:
            parent_pdf = product_map[parent]["pdf_id"]
            child_pdf = product_map[node_id]["pdf_id"]
            rel_lines, rel_id = _write_assembly_relation(parent_pdf, child_pdf)
            lines.extend(rel_lines)

    # 材料属性
    material_cache = {}
    for node_id in product_map:
        part = product_map[node_id]["part"]
        density = part.params.get("density", 1000.0)
        mat_name = "PLA" if density < 1500 else "Aluminum" if density < 4000 else "Steel"
        if mat_name not in material_cache:
            mat_lines, mat_id = _write_material(mat_name, density)
            lines.extend(mat_lines)
            material_cache[mat_name] = mat_id

    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path
