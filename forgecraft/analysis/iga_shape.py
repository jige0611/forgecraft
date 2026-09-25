# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_shape — CC 基函数求值
#
#   Catmull-Clark 细分面的 IGA 形函数:
#     N_i(u,v): 控制顶点对极限曲面位置的贡献系数
#     ∂N_i/∂x:  形函数对物理坐标的导数 (用于 B 矩阵)
#
#   正则面 (16 控制点): 双三次 B 样条基函数
#   非常面: 递归细分 → 映射到正则子面 → B 样条
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numba

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.geometry.evaluator import (
    _uniform_bspline_basis,
    _uniform_bspline_deriv,
    evaluate_bicubic_bspline,
)
from forgecraft.analysis._iga_base import ShapeResult

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "cc_shape_functions",
    "cc_element_control_points",
    "is_face_regular",
    "compute_jacobian",
]


# ══════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════

def is_face_regular(mesh: PolyMesh, face_id: int) -> bool:
    """检查面是否正则 (四边形 + 所有顶点度=4)"""
    verts = mesh.face_vertices(face_id)
    if len(verts) != 4:
        return False
    return all(mesh.vertex_valence(v) == 4 for v in verts)


def cc_element_control_points(mesh: PolyMesh, face_id: int) -> Tuple[np.ndarray, bool]:
    """提取面的 4×4 控制点网格 (1 环邻域)
    
    Returns:
        (control_4x4x3, is_regular)
        如果非常面且无法提取，返回 (zeros, False)
    """
    from forgecraft.geometry.evaluator import LimitEvaluator
    
    ev = LimitEvaluator(max_subdivisions=0)
    return ev._extract_patch(mesh, face_id)


def compute_jacobian(control: np.ndarray, u: float, v: float) -> Tuple[np.ndarray, float]:
    """计算参数空间到物理空间的 Jacobian
    
    Args:
        control: (4, 4, 3) 控制点
        u, v: 参数 ∈ [0,1]
    
    Returns:
        (J_3x2, det_J_pseudo)
        J = [∂P/∂u, ∂P/∂v]
        det_J_pseudo = sqrt(det(J^T J))
    """
    Nu = _uniform_bspline_basis(u, 3)
    Nv = _uniform_bspline_basis(v, 3)
    dNu = _uniform_bspline_deriv(u, 3)
    dNv = _uniform_bspline_deriv(v, 3)
    
    dP_du = np.zeros(3)
    dP_dv = np.zeros(3)
    
    for i in range(4):
        for j in range(4):
            dP_du += dNu[i] * Nv[j] * control[i, j]
            dP_dv += Nu[i] * dNv[j] * control[i, j]
    
    J = np.column_stack([dP_du, dP_dv])  # (3, 2)
    
    # 伪行列式: sqrt(det(J^T J))
    JTJ = J.T @ J
    det_pseudo = np.sqrt(max(0, np.linalg.det(JTJ)))
    
    return J, det_pseudo


# ══════════════════════════════════════════════════════════
#  CC 形函数求值 (热路径)
# ══════════════════════════════════════════════════════════

def cc_shape_functions(
    mesh: PolyMesh,
    face_id: int,
    u: float,
    v: float,
    irregular_cache=None,
) -> ShapeResult:
    """CC 基函数在 (u,v) 处求值
    
    返回 ShapeResult 包含:
      - N: (16,) 基函数值 (正则面) 或更多 (非常面)
      - dN_du, dN_dv: 参数空间导数
      - dN_dx: 物理空间导数 (用于 B 矩阵)
      - J, detJ: Jacobian
    
    Args:
        mesh: CC 控制网格
        face_id: 面索引
        u, v: 参数 ∈ [0, 1]
        irregular_cache: IrregularFaceCache 或 None
    """
    # 提取控制点
    control, is_regular = cc_element_control_points(mesh, face_id)
    
    # 非常面: 尝试通过缓存映射到正则子面
    if not is_regular and irregular_cache is not None:
        result = irregular_cache.get_regular_subface(face_id, u, v)
        if result is not None:
            control, u, v = result
    
    # 检查控制点是否有效
    if control.shape != (4, 4, 3) or np.allclose(control, 0):
        return _empty_shape_result()
    
    return _compute_shape_bspline(control, u, v)


def _compute_shape_bspline(control: np.ndarray, u: float, v: float) -> ShapeResult:
    """双三次 B 样条形函数求值 (核心计算)
    
    control: (4, 4, 3) 控制点网格
    """
    Nu = _uniform_bspline_basis(u, 3)
    Nv = _uniform_bspline_basis(v, 3)
    dNu = _uniform_bspline_deriv(u, 3)
    dNv = _uniform_bspline_deriv(v, 3)
    
    # 基函数值: N_{ij}(u,v) = B_i(u) * B_j(v)
    # 展平为 (16,) 数组
    N = np.zeros(16)
    dN_du_vals = np.zeros((16, 3))
    dN_dv_vals = np.zeros((16, 3))
    
    # Jacobian: dP/du, dP/dv
    dP_du = np.zeros(3)
    dP_dv = np.zeros(3)
    
    k = 0
    for i in range(4):
        for j in range(4):
            N[k] = Nu[i] * Nv[j]
            dN_du_vals[k] = dNu[i] * Nv[j] * np.ones(3)
            dN_dv_vals[k] = Nu[i] * dNv[j] * np.ones(3)
            
            dP_du += dNu[i] * Nv[j] * control[i, j]
            dP_dv += Nu[i] * dNv[j] * control[i, j]
            k += 1
    
    J = np.column_stack([dP_du, dP_dv])
    JTJ = J.T @ J
    det_pseudo = np.sqrt(max(1e-15, np.linalg.det(JTJ)))
    
    # 物理空间导数: dN/dx = J (J^T J)^{-1} [dN/du, dN/dv]
    JTJ_inv = np.linalg.inv(JTJ)
    J_pinv = JTJ_inv @ J.T  # (2, 3) 伪逆
    
    dN_dx = np.zeros((16, 3))
    for k in range(16):
        grad_param = np.array([dN_du_vals[k, 0], dN_dv_vals[k, 0]])
        dN_dx[k] = J_pinv.T @ grad_param
    
    return ShapeResult(
        N=N,
        dN_du=dN_du_vals,
        dN_dv=dN_dv_vals,
        dN_dx=dN_dx,
        J=J,
        detJ=det_pseudo,
        n_control=16,
    )


def _empty_shape_result() -> ShapeResult:
    """返回空形函数结果"""
    return ShapeResult(
        N=np.zeros(16),
        dN_du=np.zeros((16, 3)),
        dN_dv=np.zeros((16, 3)),
        dN_dx=np.zeros((16, 3)),
        J=np.zeros((3, 2)),
        detJ=0.0,
        n_control=16,
    )
