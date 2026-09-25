# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_quadrature — 积分规则
#
#   正则面: Gauss-Legendre 在 [0,1]²
#   非常面: 预细分缓存 → 映射到正则子面 → Gauss 积分
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import QuadPoint, IrregularFaceCache

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "gauss_legendre_1d",
    "gauss_legendre_2d",
    "cc_quadrature",
    "cc_quadrature_points",
]


# ══════════════════════════════════════════════════════════
#  Gauss-Legendre 积分点
# ══════════════════════════════════════════════════════════

def gauss_legendre_1d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """1D Gauss-Legendre 积分点 (在 [0,1] 上)
    
    Args:
        n: 积分点数
    
    Returns:
        (points, weights) 各 (n,)
    """
    # 标准 [-1,1] → [0,1] 映射
    pts_std, wts_std = np.polynomial.legendre.leggauss(n)
    pts = (pts_std + 1.0) / 2.0
    wts = wts_std / 2.0
    return pts, wts


def gauss_legendre_2d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """2D Gauss-Legendre 积分点 (在 [0,1]² 上)
    
    Args:
        n: 每方向积分点数
    
    Returns:
        (points_n2x2, weights_n2)
    """
    pts_1d, wts_1d = gauss_legendre_1d(n)
    n2 = n * n
    points = np.zeros((n2, 2))
    weights = np.zeros(n2)
    
    k = 0
    for i in range(n):
        for j in range(n):
            points[k, 0] = pts_1d[i]
            points[k, 1] = pts_1d[j]
            weights[k] = wts_1d[i] * wts_1d[j]
            k += 1
    
    return points, weights


# ══════════════════════════════════════════════════════════
#  CC 面积分
# ══════════════════════════════════════════════════════════

def cc_quadrature(
    mesh: PolyMesh,
    face_id: int,
    order: int = 3,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> List[QuadPoint]:
    """为 CC 面生成积分点列表
    
    正则面: 直接 Gauss-Legendre
    非常面: 通过 IrregularFaceCache 映射到正则子面
    
    Args:
        mesh: CC 控制网格
        face_id: 面索引
        order: Gauss 积分阶数
        irregular_cache: 非常面预细分缓存
    """
    verts = mesh.face_vertices(face_id)
    is_regular = (len(verts) == 4 and all(mesh.vertex_valence(v) == 4 for v in verts))
    
    pts_2d, wts = gauss_legendre_2d(order)
    quad_points = []
    
    if is_regular:
        for k in range(len(wts)):
            qp = QuadPoint(
                u=pts_2d[k, 0],
                v=pts_2d[k, 1],
                weight=wts[k],
                detJ=1.0,      # 后面由 Jacobian 计算
                sub_face_id=-1,
            )
            quad_points.append(qp)
    else:
        # 非常面: 确保缓存已建立
        if irregular_cache is not None:
            irregular_cache.build(mesh, face_id)
        
        for k in range(len(wts)):
            u, v = pts_2d[k, 0], pts_2d[k, 1]
            
            if irregular_cache is not None:
                result = irregular_cache.get_regular_subface(face_id, u, v)
                if result is not None:
                    _, local_u, local_v = result
                    qp = QuadPoint(
                        u=u, v=v,
                        weight=wts[k],
                        sub_face_id=face_id,
                        local_u=local_u,
                        local_v=local_v,
                    )
                    quad_points.append(qp)
                    continue
            
            # 回退: 直接用参数坐标 (非正则面也尝试直接积分)
            qp = QuadPoint(u=u, v=v, weight=wts[k])
            quad_points.append(qp)
    
    return quad_points


def cc_quadrature_points(
    mesh: PolyMesh,
    order: int = 3,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> List[Tuple[int, List[QuadPoint]]]:
    """为所有面生成积分点
    
    Returns:
        [(face_id, [QuadPoint, ...]), ...]
    """
    result = []
    for f in range(mesh.n_faces):
        qps = cc_quadrature(mesh, f, order, irregular_cache)
        result.append((f, qps))
    return result
