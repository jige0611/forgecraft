# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_boundary — 边界条件
#
#   DirichletBC:  位移边界 (罚函数法 + 拉格朗日乘子)
#   NeumannBC:    面力边界 (压力 / 牵引力)
#   PointLoad:    点载荷
#   apply_dirichlet_penalty: 罚函数法施加 Dirichlet BC
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable

import numpy as np

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import DOFMap, IGAError

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "DirichletBC",
    "NeumannBC",
    "PointLoad",
    "apply_dirichlet_penalty",
    "apply_neumann_load",
    "apply_point_load",
]


# ══════════════════════════════════════════════════════════
#  边界条件数据结构
# ══════════════════════════════════════════════════════════

@dataclass
class DirichletBC:
    """Dirichlet 位移边界条件
    
    用法:
        # 固定底面所有自由度
        bc = DirichletBC(
            dof_ids=[0, 1, 2, 3, 4, 5, ...],
            values=0.0,
            label="fixed_bottom"
        )
        
        # 用函数指定位移
        bc = DirichletBC.from_function(
            mesh=mesh,
            dof_map=dof_map,
            predicate=lambda v_idx, x, y, z: z < 0.01,
            value_func=lambda x, y, z: (0.001, 0, 0),
        )
    """
    dof_ids: np.ndarray      # (n_dof_bc,) 受约束的全局 DOF 编号
    values: np.ndarray       # (n_dof_bc,) 指定位移值 (或标量广播)
    label: str = ""
    
    def __post_init__(self):
        self.dof_ids = np.asarray(self.dof_ids, dtype=np.int64)
        val = np.asarray(self.values)
        if val.ndim == 0:
            self.values = np.full(len(self.dof_ids), float(val))
        else:
            self.values = val.astype(np.float64)
        
        if len(self.dof_ids) != len(self.values):
            raise IGAError(
                f"DirichletBC: dof_ids ({len(self.dof_ids)}) and values "
                f"({len(self.values)}) must have same length"
            )
    
    @classmethod
    def from_function(
        cls,
        mesh: PolyMesh,
        dof_map: DOFMap,
        predicate: Callable,
        value: float = 0.0,
        value_func: Optional[Callable] = None,
        label: str = "",
    ) -> DirichletBC:
        """根据几何谓词自动查找约束自由度
        
        Args:
            mesh: CC 控制网格
            dof_map: DOF 映射
            predicate: (vertex_id, x, y, z) -> bool
            value: 统一位移值 (或用 value_func)
            value_func: (x, y, z) -> (ux, uy, uz) 或标量
            label: 标签
        """
        dof_ids = []
        values = []
        
        for vi in range(mesh.n_vertices):
            x, y, z = mesh.vertices[vi]
            if predicate(vi, x, y, z):
                dx, dy, dz = dof_map.vertex_dofs(vi)
                dof_ids.extend([dx, dy, dz])
                
                if value_func is not None:
                    v = value_func(x, y, z)
                    if isinstance(v, (int, float)):
                        values.extend([v, v, v])
                    else:
                        values.extend(list(v))
                else:
                    values.extend([value, value, value])
        
        return cls(
            dof_ids=np.array(dof_ids, dtype=np.int64),
            values=np.array(values, dtype=np.float64),
            label=label,
        )
    
    @classmethod
    def from_vertex_set(
        cls,
        vertex_ids: List[int],
        dof_map: DOFMap,
        value: float = 0.0,
        dof_mask: Tuple[bool, bool, bool] = (True, True, True),
        label: str = "",
    ) -> DirichletBC:
        """从顶点集合创建 BC
        
        Args:
            vertex_ids: 顶点 ID 列表
            dof_map: DOF 映射
            value: 位移值
            dof_mask: (lock_x, lock_y, lock_z) 约束哪些方向
            label: 标签
        """
        dof_ids = []
        values = []
        
        mx, my, mz = dof_mask
        
        for vi in vertex_ids:
            dx, dy, dz = dof_map.vertex_dofs(vi)
            if mx:
                dof_ids.append(dx)
                values.append(value)
            if my:
                dof_ids.append(dy)
                values.append(value)
            if mz:
                dof_ids.append(dz)
                values.append(value)
        
        return cls(
            dof_ids=np.array(dof_ids, dtype=np.int64),
            values=np.array(values, dtype=np.float64),
            label=label,
        )
    
    @property
    def n_constrained(self) -> int:
        return len(self.dof_ids)


@dataclass
class NeumannBC:
    """Neumann 面力边界条件 (压力 / 牵引力)
    
    用法:
        # 压力载荷 (法向)
        bc = NeumannBC(
            face_ids=[0, 1, 2],
            traction=(0, 0, -1e6),  # Pa, 面力矢量
            label="pressure"
        )
    """
    face_ids: List[int]          # 受载面列表
    traction: Tuple[float, float, float]  # (tx, ty, tz) 牵引力矢量 (Pa)
    label: str = ""
    
    def __post_init__(self):
        self.traction = np.array(self.traction, dtype=np.float64)
    
    @classmethod
    def pressure(
        cls,
        face_ids: List[int],
        pressure: float,          # 正值 = 法向压力
        mesh: PolyMesh,
        label: str = "pressure",
    ) -> NeumannBC:
        """创建法向压力载荷
        
        自动计算每个面的外法向量。
        """
        # 使用第一个面计算法向
        v0, v1, v2, v3 = mesh.face_vertices(face_ids[0])
        p0 = np.array(mesh.vertices[v0])
        p1 = np.array(mesh.vertices[v1])
        p2 = np.array(mesh.vertices[v2])
        
        e1 = p1 - p0
        e2 = p2 - p0
        normal = np.cross(e1, e2)
        n = normal / np.linalg.norm(normal)
        
        traction = tuple(-pressure * n)
        
        return cls(face_ids=face_ids, traction=traction, label=label)
    
    def force_vector(
        self,
        mesh: PolyMesh,
        dof_map: DOFMap,
        quadrature_order: int = 3,
    ) -> np.ndarray:
        """计算 Neumann BC 的等效节点力向量
        
        每面遍历积分点，对形函数加权得到等效节点力。
        """
        from forgecraft.analysis.iga_shape import cc_shape_functions
        from forgecraft.analysis.iga_quadrature import gauss_legendre_2d
        
        F = np.zeros(dof_map.n_dof)
        pts_2d, wts = gauss_legendre_2d(quadrature_order)
        t = self.traction
        
        for face_id in self.face_ids:
            verts = mesh.face_vertices(face_id)
            if len(verts) != 4:
                continue
            
            # 该面控制点的全局索引
            all_verts = list(verts)
            seen = set(verts)
            for v in verts:
                ring = mesh.vertex_ring(v)
                for r in ring:
                    if r not in seen and len(all_verts) < 16:
                        all_verts.append(r)
                        seen.add(r)
            while len(all_verts) < 16:
                all_verts.append(all_verts[-1])
            all_verts = all_verts[:16]
            
            for k in range(len(wts)):
                u, v = pts_2d[k, 0], pts_2d[k, 1]
                shape = cc_shape_functions(mesh, face_id, u, v)
                
                if shape.detJ < 1e-15:
                    continue
                
                factor = wts[k] * shape.detJ
                wN = factor * shape.N  # (16,)
                
                for node_i in range(16):
                    vi = all_verts[node_i]
                    dx, dy, dz = dof_map.vertex_dofs(vi)
                    F[dx] += wN[node_i] * t[0]
                    F[dy] += wN[node_i] * t[1]
                    F[dz] += wN[node_i] * t[2]
        
        return F


@dataclass
class PointLoad:
    """点载荷 / 集中力
    
    用法:
        # 在顶点施加集中力
        load = PointLoad(vertex_id=5, force=(0, 0, -1000), dof_map=dof_map)
    """
    vertex_id: int
    force: Tuple[float, float, float]  # (Fx, Fy, Fz) N
    dof_map: DOFMap
    
    _dof_ids: np.ndarray = field(init=False)
    _values: np.ndarray = field(init=False)
    
    def __post_init__(self):
        dx, dy, dz = self.dof_map.vertex_dofs(self.vertex_id)
        self._dof_ids = np.array([dx, dy, dz], dtype=np.int64)
        self._values = np.array(self.force, dtype=np.float64)
    
    def assemble(self, F: np.ndarray):
        """将点载荷加到全局力向量"""
        for i, d in enumerate(self._dof_ids):
            F[d] += self._values[i]


# ══════════════════════════════════════════════════════════
#  Dirichlet BC 施加
# ══════════════════════════════════════════════════════════

def apply_dirichlet_penalty(
    K: np.ndarray,
    F: np.ndarray,
    bc: DirichletBC,
    penalty: float = 1e12,
):
    """罚函数法施加 Dirichlet BC
    
    修改全局刚度矩阵和力向量:
        K_jj += penalty
        F_j  += penalty * u_bar_j
    
    Args:
        K: (n_dof, n_dof) 全局刚度矩阵 (原地修改)
        F: (n_dof,) 全局力向量 (原地修改)
        bc: DirichletBC
        penalty: 罚因子 (默认 1e12)
    """
    for i, d in enumerate(bc.dof_ids):
        K[d, d] += penalty
        F[d] += penalty * bc.values[i]


def apply_dirichlet_elimination(
    K: np.ndarray,
    F: np.ndarray,
    bc: DirichletBC,
    u: np.ndarray,
) -> np.ndarray:
    """直接消除法施加 Dirichlet BC
    
    对约束 DOF 行/列置 1，右侧置指定位移。
    适用于小规模问题 (direct solver)。
    
    Args:
        K: (n_dof, n_dof) 全局刚度矩阵 (原地修改)
        F: (n_dof,) 全局力向量 (原地修改)
        u: (n_dof,) 位移向量 (原地修改)
        bc: DirichletBC
    
    Returns:
        u (修改后)
    """
    n = K.shape[0]
    fixed = set(bc.dof_ids)
    free = [i for i in range(n) if i not in fixed]
    
    # 先填充约束 DOF 位移
    for i, d in enumerate(bc.dof_ids):
        u[d] = bc.values[i]
    
    # 消除: F_free -= K_free,fixed @ u_fixed
    for d in bc.dof_ids:
        for i in free:
            F[i] -= K[i, d] * u[d]
    
    # 置约束行/列为单位
    for d in bc.dof_ids:
        K[d, :] = 0.0
        K[:, d] = 0.0
        K[d, d] = 1.0
        F[d] = u[d]
    
    return free


# ══════════════════════════════════════════════════════════
#  Neumann BC 和点载荷施加
# ══════════════════════════════════════════════════════════

def apply_neumann_load(
    mesh: PolyMesh,
    dof_map: DOFMap,
    bc_list: List[NeumannBC],
    F: np.ndarray,
    quadrature_order: int = 3,
):
    """将所有 Neumann BC 施加到力向量
    
    Args:
        mesh: CC 控制网格
        dof_map: DOF 映射
        bc_list: NeumannBC 列表
        F: (n_dof,) 全局力向量 (原地修改)
        quadrature_order: 积分阶数
    """
    for bc in bc_list:
        F_bc = bc.force_vector(mesh, dof_map, quadrature_order)
        F += F_bc


def apply_point_load(
    load: PointLoad,
    F: np.ndarray,
):
    """将点载荷施加到力向量
    
    Args:
        load: PointLoad
        F: (n_dof,) 全局力向量 (原地修改)
    """
    load.assemble(F)


def apply_point_loads(loads: List[PointLoad], F: np.ndarray):
    """施加多个点载荷"""
    for load in loads:
        load.assemble(F)
