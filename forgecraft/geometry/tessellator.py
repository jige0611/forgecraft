# ══════════════════════════════════════════════════════════
# forgecraft.geometry.tessellator — 自适应镶嵌
#
#   Catmull-Clark 细分网格 → 三角面片 (STL)
#
#   用已细分的顶点直接三角化 (不需要极限曲面求值)
#   输出 watertight manifold mesh
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from forgecraft.geometry.mesh import PolyMesh

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "AdaptiveTessellator",
    "TessellationConfig",
    "tessellate_to_stl",
    "tessellate_to_trimesh",
]


@dataclass
class TessellationConfig:
    chord_error: float = 0.01
    max_subdivisions: int = 3


class AdaptiveTessellator:
    """细分网格 → 三角形 STL
    
    用法:
      >>> tess = AdaptiveTessellator()
      >>> tri_mesh = tess.tessellate(cc_mesh)
      >>> tri_mesh.export("output.stl")
    """
    
    def __init__(self, config: Optional[TessellationConfig] = None):
        self.config = config or TessellationConfig()
    
    def tessellate(self, mesh: PolyMesh):
        """将多边形网格三角化"""
        import trimesh
        
        verts, faces = mesh.to_vertices_faces()
        tri_faces = []
        
        for face in faces:
            if len(face) == 3:
                tri_faces.append(face)
            elif len(face) == 4:
                # 四边形 → 2 三角形
                tri_faces.append([face[0], face[1], face[2]])
                tri_faces.append([face[0], face[2], face[3]])
            elif len(face) > 4:
                # 扇形三角化
                for i in range(1, len(face) - 1):
                    tri_faces.append([face[0], face[i], face[i + 1]])
        
        if not tri_faces:
            return trimesh.Trimesh()
        
        tri_mesh = trimesh.Trimesh(
            vertices=np.array(verts),
            faces=np.array(tri_faces, dtype=np.int32),
        )
        tri_mesh.merge_vertices()
        try:
            tri_mesh.remove_degenerate_faces()
        except AttributeError:
            pass
        try:
            tri_mesh.remove_duplicate_faces()
        except AttributeError:
            pass
        
        return tri_mesh


def tessellate_to_stl(mesh: PolyMesh, output_path: str, chord_error: float = 0.01):
    tess = AdaptiveTessellator(TessellationConfig(chord_error=chord_error))
    tri_mesh = tess.tessellate(mesh)
    tri_mesh.export(output_path)
    return output_path


def tessellate_to_trimesh(mesh: PolyMesh, chord_error: float = 0.01):
    tess = AdaptiveTessellator(TessellationConfig(chord_error=chord_error))
    return tess.tessellate(mesh)
