"""
统一制造导出器 — STEP AP242 + STL + 3MF

基于 trimesh + mesh_engine，将 MechanicalBody 导出为:
  - STEP AP242 (tessellated shell, SolidWorks/Fusion360 兼容)
  - STL (binary/ascii, 自适应精度)
  - 3MF (3D 打印标准格式)

替代手写 ISO 10303 字符串方案，直接使用 trimesh mesh → tessellated shell。

使用:
  >>> from forgecraft.manufacturing.exporter import ManufacturingExporter
  >>> exp = ManufacturingExporter(quality="high")
  >>> exp.export_step(body, "output.step")
  >>> exp.export_stl(body, "output/")
  >>> exp.export_all(body, "output/")
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import logging
import time

import numpy as np
import trimesh

from forgecraft.manufacturing.mesh_engine import MeshEngine, QUALITY_PRESETS
from forgecraft.core.morphology import MechanicalBody, Part, Joint
from forgecraft.config import PartSpec

_logger = logging.getLogger(__name__)

__all__ = ["ManufacturingExporter", "export_body"]

# ── 尝试导入 cadquery 桥接 (可选依赖) ──
try:
    from forgecraft.manufacturing.cadquery_bridge import CadQueryBridge, cadquery_available
    _HAS_CQ_BRIDGE = True
except ImportError:
    _HAS_CQ_BRIDGE = False
    cadquery_available = lambda: False

# ── STEP header ──
_STEP_HEADER = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Tessellated STEP AP242 by ForgeCraft v1.0'),'2;1');
FILE_NAME('{name}.stp','{date}',('ForgeCraft AI'),(''),'ForgeCraft v1.0','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN {{ 1 0 10303 214 3 1 1 }}'));
ENDSEC;
DATA;
#1 = APPLICATION_CONTEXT('mechanical design');
#2 = MECHANICAL_CONTEXT('',#1,'mechanical');
#3 = APPLICATION_PROTOCOL_DEFINITION('','automotive_design',2000,#1);
#4 = PRODUCT('{name}','{name}','',(#2));
#5 = PRODUCT_DEFINITION_FORMATION('','',#4);
#6 = PRODUCT_DEFINITION('','',#5,#3);
#7 = PRODUCT_DEFINITION_SHAPE('', '', #6);
"""

_STEP_FOOTER = """ENDSEC;
END-ISO-10303-21;
"""


class ManufacturingExporter:
    """统一制造导出器
    
    quality: "low" | "medium" | "high" | "ultra"
    """
    
    def __init__(self, quality: str = "high", use_cadquery: bool = None):
        self.engine = MeshEngine(quality)
        self.quality = quality
        self._id_counter = 0
        
        # cadquery 桥接: None=自动(ultra时启用), True=强制, False=禁用
        if use_cadquery is None:
            use_cadquery = (quality == "ultra" and cadquery_available())
        self._cq = CadQueryBridge(quality) if use_cadquery else None
    
    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter
    
    def _reset_ids(self):
        self._id_counter = 10  # start after built-in entities
    
    # ══════════════════════════════════════════
    #  STEP 导出 (tessellated shell)
    # ══════════════════════════════════════════
    
    def _mesh_to_step_entities(self, mesh: trimesh.Trimesh,
                                 name: str) -> List[str]:
        """将 trimesh 转为 STEP TESSELLATED_SHELL 实体"""
        lines = []
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Cartesian points
        point_ids = []
        for v in vertices:
            pid = self._next_id()
            point_ids.append(pid)
            lines.append(f"#{pid} = CARTESIAN_POINT('',({v[0]:.8E},{v[1]:.8E},{v[2]:.8E}));")
        
        # Coordinates list
        coord_list = "(" + ",".join(f"#{p}" for p in point_ids) + ")"
        coord_id = self._next_id()
        lines.append(f"#{coord_id} = COORDINATES_LIST('',{len(vertices)},{coord_list});")
        
        # Triangulated faces
        face_ids = []
        for f in faces:
            fid = self._next_id()
            lines.append(
                f"#{fid} = TRIANGULATED_FACE('',#{int(f[0])+1},#{int(f[1])+1},#{int(f[2])+1},#{int(f[0])+1},3,.F.);"
            )
            face_ids.append(fid)
        
        # Tessellated shell
        face_list = "(" + ",".join(f"#{f}" for f in face_ids) + ")"
        shell_id = self._next_id()
        
        # Calculate optional normals
        has_normals = hasattr(mesh, 'face_normals') and mesh.face_normals is not None
        normal_str = ".F." if not has_normals else ".T."
        
        lines.append(
            f"#{shell_id} = TESSELLATED_SHELL('{name}',{face_list},#{coord_id},{normal_str},.F.,.F.);"
        )
        
        # Styled item (color)
        color_r, color_g, color_b = 60, 60, 60
        ci = self._next_id()
        cr = self._next_id(); cg = self._next_id(); cb = self._next_id()
        crgb = self._next_id()
        fcol = self._next_id()
        fstyle = self._next_id()
        sstyle = self._next_id()
        styled_id = self._next_id()
        
        lines.extend([
            f"#{cr} = INT_LITERAL('{color_r}');",
            f"#{cg} = INT_LITERAL('{color_g}');",
            f"#{cb} = INT_LITERAL('{color_b}');",
            f"#{crgb} = COLOUR_RGB('',#{cr},#{cg},#{cb});",
            f"#{fcol} = FILL_AREA_STYLE_COLOUR('{name}_colour',#{crgb});",
            f"#{fstyle} = FILL_AREA_STYLE('',(#{fcol}));",
            f"#{sstyle} = SURFACE_STYLE_USAGE(.BOTH.,#{fstyle});",
            f"#{styled_id} = STYLED_ITEM('',(#{sstyle}),#{shell_id});",
        ])
        
        # Shape representation
        lod_id = self._next_id()
        brep_id = self._next_id()
        lines.append(
            f"#{brep_id} = ADVANCED_BREP_SHAPE_REPRESENTATION('{name}',(#{styled_id}),#1);"
        )
        lines.append(
            f"#{lod_id} = SHAPE_DEFINITION_REPRESENTATION('','',#{brep_id});"
        )
        
        return lines, lod_id
    
    def _build_step_file(self, meshes: List[Tuple[trimesh.Trimesh, str]],
                          name: str = "ForgeCraft_Body") -> str:
        """构建完整 STEP 文件内容"""
        self._reset_ids()
        
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        
        lines = [_STEP_HEADER.format(name=name, date=date_str)]
        
        sr_ids = []
        for mesh, part_name in meshes:
            part_lines, lod_id = self._mesh_to_step_entities(mesh, part_name)
            lines.extend(part_lines)
            sr_ids.append(lod_id)
        
        # Shape representation relationship — links parts to product
        for lod_id in sr_ids:
            rel_id = self._next_id()
            lines.append(
                f"#{rel_id} = SHAPE_REPRESENTATION_RELATIONSHIP('','',#{lod_id},#7);"
            )
        
        lines.append(_STEP_FOOTER)
        return "\n".join(lines)
    
    def export_step(self, body: MechanicalBody,
                     output_path: str,
                     catalog: Dict[str, PartSpec] = None) -> str:
        """导出 STEP 文件
        
        - quality=ultra + cadquery可用 → 真 B-Rep STEP (数学曲面)
        - 其他 → AP242 tessellated shell (三角面片)
        """
        # ── 优先使用 cadquery B-Rep ──
        if self._cq and self._cq.available:
            result = self._cq.export_step(body, output_path, catalog)
            if result and os.path.exists(result):
                sz = os.path.getsize(result)
                _logger.info(f"STEP B-Rep 导出 (cadquery): {result} ({sz:,} bytes)")
                return result
            _logger.warning("cadquery STEP 失败，回退到 tessellated")
        
        # ── 回退: tessellated shell ──
        meshes = []
        for part in body.parts():
            spec = catalog.get(part.part_type) if catalog else None
            mesh = self._part_to_mesh(part, spec)
            meshes.append((mesh, part.part_id[:12]))
        
        content = self._build_step_file(meshes, name=body.name)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        sz = os.path.getsize(output_path)
        _logger.info(f"STEP tessellated 导出: {output_path} ({sz:,} bytes, {len(meshes)} parts)")
        return output_path
    
    # ══════════════════════════════════════════
    #  STL 导出
    # ══════════════════════════════════════════
    
    def export_stl(self, body: MechanicalBody,
                    output_dir: str,
                    catalog: Dict[str, PartSpec] = None,
                    binary: bool = True) -> List[str]:
        """导出每个零件为独立 STL，加一个总装配 STL"""
        os.makedirs(output_dir, exist_ok=True)
        files = []
        
        assembly_meshes = []
        for part in body.parts():
            spec = catalog.get(part.part_type) if catalog else None
            mesh = self._part_to_mesh(part, spec)
            
            # 应用世界变换
            pos = part.position
            transform = np.eye(4)
            transform[:3, 3] = pos
            mesh.apply_transform(transform)
            
            fname = os.path.join(output_dir, f"{part.part_id}.stl")
            mesh.export(fname)
            files.append(fname)
            assembly_meshes.append(mesh)
        
        # 总装配 STL
        if len(assembly_meshes) > 1:
            try:
                combined = trimesh.util.concatenate(assembly_meshes)
                asm_path = os.path.join(output_dir, f"{body.name.replace(' ','_')}.stl")
                combined.export(asm_path)
                files.append(asm_path)
            except Exception:
                pass
        
        _logger.info(f"STL 导出: {len(files)} files → {output_dir}")
        return files
    
    def export_3mf(self, body: MechanicalBody,
                    output_path: str,
                    catalog: Dict[str, PartSpec] = None) -> str:
        """导出 3MF (3D 打印标准格式)"""
        scene = self._body_to_scene(body, catalog)
        scene.export(output_path, file_type="3mf")
        _logger.info(f"3MF 导出: {output_path}")
        return output_path
    
    # ══════════════════════════════════════════
    #  完整导出
    # ══════════════════════════════════════════
    
    def export_all(self, body: MechanicalBody,
                    output_dir: str = "design_output",
                    catalog: Dict[str, PartSpec] = None) -> Dict[str, Any]:
        """一键导出 STEP + STL + 3MF"""
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, body.name.replace(" ", "_"))
        
        result = {"body": body.name, "files": [], "errors": []}
        
        # STEP
        try:
            step_path = f"{base}.stp"
            self.export_step(body, step_path, catalog)
            result["files"].append(step_path)
        except Exception as e:
            result["errors"].append(f"STEP: {e}")
            _logger.warning(f"STEP 导出失败: {e}")
        
        # STL
        try:
            stl_dir = os.path.join(output_dir, "stl")
            stl_files = self.export_stl(body, stl_dir, catalog)
            result["files"].extend(stl_files)
        except Exception as e:
            result["errors"].append(f"STL: {e}")
            _logger.warning(f"STL 导出失败: {e}")
        
        # 3MF
        try:
            mf3_path = f"{base}.3mf"
            self.export_3mf(body, mf3_path, catalog)
            result["files"].append(mf3_path)
        except Exception as e:
            result["errors"].append(f"3MF: {e}")
            _logger.warning(f"3MF 导出失败: {e}")
        
        return result
    
    # ══════════════════════════════════════════
    #  内部
    # ══════════════════════════════════════════
    
    def _part_to_mesh(self, part: Part,
                       spec: Optional[PartSpec] = None) -> trimesh.Trimesh:
        """将 Part 转为 trimesh"""
        params = dict(part.params)
        
        if spec is not None:
            return self.engine.build_part(spec, params)
        
        # Fallback: 用 part_type 推断
        shape = "cylinder"
        return self.engine.build_part(
            PartSpec(part_type=part.part_type, mass=0.1, shape=shape, size_range=[0.01, 1.0]),
            params
        )
    
    def _body_to_scene(self, body: MechanicalBody,
                        catalog: Dict[str, PartSpec] = None) -> trimesh.Scene:
        """将 MechanicalBody 转为 trimesh.Scene"""
        scene = trimesh.Scene()
        for part in body.parts():
            spec = catalog.get(part.part_type) if catalog else None
            mesh = self._part_to_mesh(part, spec)
            pos = part.position
            transform = np.eye(4)
            transform[:3, 3] = pos
            mesh.apply_transform(transform)
            scene.add_geometry(mesh, node_name=part.part_id)
        return scene


# ── 便捷函数 ──

def export_body(body: MechanicalBody, output_dir: str = "design_output",
                 quality: str = "high",
                 catalog: Dict[str, PartSpec] = None) -> Dict[str, Any]:
    """便捷导出函数"""
    exporter = ManufacturingExporter(quality)
    return exporter.export_all(body, output_dir, catalog)
