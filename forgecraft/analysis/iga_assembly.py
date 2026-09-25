# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_assembly — 并行组装
#
#   流式 CSR 稀疏矩阵组装
#   并行逐面计算 Ke/Me → triplet 数组 → coo_matrix → csr_matrix
#   支持刚度矩阵、质量矩阵、力向量
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    DOFMap, QuadPoint, IrregularFaceCache, IGASettings, IGAError,
)
from forgecraft.analysis.iga_quadrature import cc_quadrature, cc_quadrature_points

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "StreamingCSRAssembler",
    "assemble_stiffness",
    "assemble_mass",
    "assemble_stiffness_parallel",
    "assemble_force_vector",
]


# ══════════════════════════════════════════════════════════
#  流式 CSR 组装器
# ══════════════════════════════════════════════════════════

class StreamingCSRAssembler:
    """流式 CSR 稀疏矩阵组装器
    
    预分配 triplet 数组 (rows, cols, vals)，避免重复 resizing。
    
    用法:
        assembler = StreamingCSRAssembler(n_dof, estimated_nnz)
        for face_id in range(n_faces):
            Ke, element_dofs = integrator.stiffness(...)
            assembler.add_element(element_dofs, Ke)
        K_csr = assembler.to_csr()
    """
    
    def __init__(self, n_dof: int, estimated_nnz: int = 0):
        self.n_dof = n_dof
        
        if estimated_nnz <= 0:
            # 每单元 48 DOF → 最多 48×48=2304 条目
            estimated_nnz = n_dof * 80  # 保守估计每个 DOF ~80 个非零
        
        self._capacity = estimated_nnz * 2
        self._rows = np.empty(self._capacity, dtype=np.int32)
        self._cols = np.empty(self._capacity, dtype=np.int32)
        self._vals = np.empty(self._capacity, dtype=np.float64)
        self._count = 0
    
    def _ensure_capacity(self, additional: int):
        """确保有足够容量"""
        needed = self._count + additional
        if needed <= self._capacity:
            return
        
        new_cap = max(needed, int(self._capacity * 1.5))
        self._rows = np.resize(self._rows, new_cap)
        self._cols = np.resize(self._cols, new_cap)
        self._vals = np.resize(self._vals, new_cap)
        self._capacity = new_cap
    
    def add_element(
        self,
        element_dofs: List[int],
        Ke: np.ndarray,
    ):
        """添加单元刚度/质量矩阵
        
        Args:
            element_dofs: 单元全局 DOF 编号列表
            Ke: (n_local, n_local) 单元矩阵
        """
        n_local = len(element_dofs)
        
        if n_local == 0 or Ke.shape[0] <= 1:
            return
        
        nnz = n_local * n_local
        self._ensure_capacity(nnz)
        
        start = self._count
        end = start + nnz
        
        dofs = np.array(element_dofs, dtype=np.int32)
        rows = np.repeat(dofs, n_local)
        cols = np.tile(dofs, n_local)
        
        self._rows[start:end] = rows
        self._cols[start:end] = cols
        self._vals[start:end] = Ke.ravel()
        self._count = end
    
    def add_vector(
        self,
        element_dofs: List[int],
        Fe: np.ndarray,
        F_global: np.ndarray,
    ):
        """将单元载荷向量加到全局力向量
        
        Args:
            element_dofs: 单元全局 DOF 编号
            Fe: (n_local,) 单元载荷向量
            F_global: (n_dof,) 全局力向量 (原地修改)
        """
        for i, d in enumerate(element_dofs):
            F_global[d] += Fe[i]
    
    def to_coo(self) -> coo_matrix:
        """转换为 COO 稀疏矩阵 (去重求和)"""
        return coo_matrix(
            (self._vals[:self._count],
             (self._rows[:self._count], self._cols[:self._count])),
            shape=(self.n_dof, self.n_dof),
        )
    
    def to_csr(self) -> csr_matrix:
        """转换为 CSR 稀疏矩阵 (去重求和并压缩)"""
        K_coo = self.to_coo()
        # 去重求和
        K_csr = K_coo.tocsr()
        return K_csr
    
    def reset(self):
        """重置计数器 (保留容量)"""
        self._count = 0
    
    def clear(self):
        """清空所有数据"""
        self._count = 0
        self._rows = np.empty(0, dtype=np.int32)
        self._cols = np.empty(0, dtype=np.int32)
        self._vals = np.empty(0, dtype=np.float64)
        self._capacity = 0


# ══════════════════════════════════════════════════════════
#  刚度矩阵组装
# ══════════════════════════════════════════════════════════

def assemble_stiffness(
    mesh: PolyMesh,
    integrator,
    material,
    dof_map: DOFMap,
    settings: IGASettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> csr_matrix:
    """顺序组装全局刚度矩阵
    
    Args:
        mesh: CC 控制网格
        integrator: ElementIntegrator 实例
        material: 材料模型
        dof_map: DOF 映射
        settings: IGA 设置
        irregular_cache: 非常面缓存
    
    Returns:
        K: (n_dof, n_dof) CSR 稀疏刚度矩阵
    """
    n_dof = dof_map.n_dof
    n_faces = mesh.n_faces
    
    quad_list = cc_quadrature_points(mesh, settings.quadrature_order, irregular_cache)
    
    assembler = StreamingCSRAssembler(n_dof)
    
    for face_id in range(n_faces):
        qps = quad_list[face_id][1]
        
        try:
            Ke, element_dofs = integrator.stiffness(
                mesh, face_id, material, qps, irregular_cache,
            )
        except Exception as e:
            _logger.debug(f"Face {face_id}: stiffness assembly failed: {e}")
            continue
        
        if len(element_dofs) > 0 and Ke.shape[0] > 1:
            # 将控制顶点 ID 转换为全局 DOF ID
            dof_ids = dof_map.element_dofs(element_dofs)
            assembler.add_element(dof_ids.tolist(), Ke)
    
    K = assembler.to_csr()
    return K


def assemble_stiffness_parallel(
    mesh: PolyMesh,
    integrator,
    material,
    dof_map: DOFMap,
    settings: IGASettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
    n_jobs: int = -1,
) -> csr_matrix:
    """并行组装全局刚度矩阵
    
    使用 joblib 并行独立面计算 Ke，
    然后用汇集 triplet 数组组装 CSR 矩阵。
    
    Args:
        mesh: CC 控制网格
        integrator: ElementIntegrator 实例
        material: 材料模型
        dof_map: DOF 映射
        settings: IGA 设置
        irregular_cache: 非常面缓存
        n_jobs: 并行作业数 (-1 = 所有 CPU)
    
    Returns:
        K: (n_dof, n_dof) CSR 稀疏刚度矩阵
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        _logger.warning("joblib 不可用，回退到顺序组装")
        return assemble_stiffness(
            mesh, integrator, material, dof_map, settings, irregular_cache,
        )
    
    n_dof = dof_map.n_dof
    n_faces = mesh.n_faces
    
    quad_list = cc_quadrature_points(mesh, settings.quadrature_order, irregular_cache)
    
    if n_jobs < 1:
        import os
        n_jobs = max(1, os.cpu_count() or 4)
    
    with Parallel(n_jobs=n_jobs, verbose=settings.verbosity) as parallel:
        results = parallel(
            delayed(_compute_element_ke)(
                face_id, mesh, integrator, material, quad_list[face_id][1],
                irregular_cache,
            )
            for face_id in range(n_faces)
        )
    
    assembler = StreamingCSRAssembler(n_dof)
    
    for face_id, (Ke, element_dofs) in enumerate(results):
        if Ke is not None and len(element_dofs) > 0 and Ke.shape[0] > 1:
            # 并行模式下也需要转换
            dof_ids = dof_map.element_dofs(element_dofs)
            assembler.add_element(dof_ids.tolist(), Ke)
    
    return assembler.to_csr()


def _compute_element_ke(
    face_id: int,
    mesh,
    integrator,
    material,
    qps: list,
    irregular_cache,
) -> Tuple[int, Tuple[Optional[np.ndarray], list]]:
    """单个面的刚度计算 (用于并行)"""
    try:
        Ke, element_dofs = integrator.stiffness(
            mesh, face_id, material, qps, irregular_cache,
        )
        return face_id, (Ke, element_dofs)
    except Exception:
        return face_id, (None, [])


# ══════════════════════════════════════════════════════════
#  质量矩阵组装
# ══════════════════════════════════════════════════════════

def assemble_mass(
    mesh: PolyMesh,
    integrator,
    material,
    dof_map: DOFMap,
    settings: IGASettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> csr_matrix:
    """组装全局一致质量矩阵
    
    Args:
        mesh: CC 控制网格
        integrator: ElementIntegrator 实例
        material: 材料模型
        dof_map: DOF 映射
        settings: IGA 设置
        irregular_cache: 非常面缓存
    
    Returns:
        M: (n_dof, n_dof) CSR 稀疏质量矩阵
    """
    n_dof = dof_map.n_dof
    n_faces = mesh.n_faces
    
    quad_list = cc_quadrature_points(mesh, settings.quadrature_order, irregular_cache)
    
    assembler = StreamingCSRAssembler(n_dof)
    
    for face_id in range(n_faces):
        qps = quad_list[face_id][1]
        
        try:
            Me, element_dofs = integrator.mass(
                mesh, face_id, material, qps, irregular_cache,
            )
        except Exception as e:
            _logger.debug(f"Face {face_id}: mass assembly failed: {e}")
            continue
        
        if len(element_dofs) > 0 and Me.shape[0] > 1:
            dof_ids = dof_map.element_dofs(element_dofs)
            assembler.add_element(dof_ids.tolist(), Me)
    
    M = assembler.to_csr()
    return M


# ══════════════════════════════════════════════════════════
#  力向量组装
# ══════════════════════════════════════════════════════════

def assemble_force_vector(
    mesh: PolyMesh,
    dof_map: DOFMap,
    neumann_bcs: list,
    point_loads: list,
    settings: IGASettings,
) -> np.ndarray:
    """组装全局力向量 (面力 + 点载荷)
    
    Args:
        mesh: CC 控制网格
        dof_map: DOF 映射
        neumann_bcs: NeumannBC 列表
        point_loads: PointLoad 列表
        settings: IGA 设置
    
    Returns:
        F: (n_dof,) 全局力向量
    """
    from forgecraft.analysis.iga_boundary import (
        apply_neumann_load, apply_point_loads,
    )
    
    F = np.zeros(dof_map.n_dof)
    
    if neumann_bcs:
        apply_neumann_load(mesh, dof_map, neumann_bcs, F, settings.quadrature_order)
    
    if point_loads:
        apply_point_loads(point_loads, F)
    
    return F
