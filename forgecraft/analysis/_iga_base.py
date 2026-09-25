# ══════════════════════════════════════════════════════════
# forgecraft.analysis._iga_base — IGA 基础类型
#
#   共用类型和结构，消除模块间循环导入。
#   所有 dataclass、DOF 编号、缓存结构集中于此。
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


__all__ = [
    "IGASettings",
    "DOFMap",
    "QuadPoint",
    "StiffnessCache",
    "IrregularFaceCache",
    "ShapeResult",
    "StaticResult",
    "ModalResult",
    "TransientResult",
    "StressField",
    "IGAError",
]


# ══════════════════════════════════════════════════════════
#  全局配置
# ══════════════════════════════════════════════════════════

@dataclass
class IGASettings:
    """IGA 求解全局配置
    
    用法:
      >>> settings = IGASettings(quadrature_order=3, solver="direct")
      >>> result = solve_static(mesh, material, bc_list, loads, settings)
    """
    quadrature_order: int = 3
    irregular_subdivisions: int = 4   # 非常面预细分次数
    solver: str = "direct"            # "direct" | "cg" | "amg"
    parallel: bool = True
    use_gpu: bool = False
    n_threads: int = 0                # 0 = 自动检测 CPU 核心数
    tolerance: float = 1e-10          # 迭代求解器收敛容差
    max_iter: int = 5000              # 迭代求解器最大迭代数
    shell_theory: str = "KL"          # "KL" (Kirchhoff-Love) | "RM" (Reissner-Mindlin)
    verbosity: int = 1                # 0=静默, 1=进度, 2=详细


# ══════════════════════════════════════════════════════════
#  DOF 编号
# ══════════════════════════════════════════════════════════

@dataclass
class DOFMap:
    """控制顶点 → 全局自由度编号映射
    
    每个控制顶点 3 个 DOF: (u_x, u_y, u_z)
    全局 DOF 总数 = n_vertices × 3
    """
    n_vertices: int
    n_dof: int                          # = n_vertices * 3
    _vertex_to_dof: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    
    @classmethod
    def from_mesh(cls, mesh) -> DOFMap:
        """从 PolyMesh 构建 DOF 映射"""
        nv = mesh.n_vertices
        nd = nv * 3
        v2d = {v: (v * 3, v * 3 + 1, v * 3 + 2) for v in range(nv)}
        return cls(n_vertices=nv, n_dof=nd, _vertex_to_dof=v2d)
    
    def vertex_dofs(self, vertex_id: int) -> Tuple[int, int, int]:
        """返回顶点 (v_x, v_y, v_z) 的全局 DOF 编号"""
        return self._vertex_to_dof[vertex_id]
    
    def element_dofs(self, vertex_ids: List[int]) -> np.ndarray:
        """返回单元 (控制点列表) 的全局 DOF 编号数组
        Returns: (n_control * 3,) 的 int64 数组
        """
        dofs = np.empty(len(vertex_ids) * 3, dtype=np.int64)
        for i, v in enumerate(vertex_ids):
            dx, dy, dz = self._vertex_to_dof[v]
            dofs[i * 3] = dx
            dofs[i * 3 + 1] = dy
            dofs[i * 3 + 2] = dz
        return dofs
    
    def element_dof_count(self, n_control_points: int) -> int:
        """单元的自由度数"""
        return n_control_points * 3


# ══════════════════════════════════════════════════════════
#  积分点
# ══════════════════════════════════════════════════════════

@dataclass
class QuadPoint:
    """一个积分点 (带权重和 Jacobian 因子)"""
    u: float
    v: float
    weight: float
    detJ: float = 1.0         # Jacobian 行列式
    sub_face_id: int = -1     # 非常面: 映射到的正则子面
    local_u: float = 0.0      # 正则子面上的局部参数
    local_v: float = 0.0


# ══════════════════════════════════════════════════════════
#  缓存结构
# ══════════════════════════════════════════════════════════

@dataclass
class StiffnessCache:
    """单元刚度阵缓存 (避免重复计算不变单元)"""
    face_id: int
    Ke: np.ndarray             # (n_dof, n_dof) 单元刚度矩阵
    Me: Optional[np.ndarray] = None  # (n_dof, n_dof) 单元质量矩阵


class IrregularFaceCache:
    """非常面预细分层次缓存
    
    对不规则面一次性细分，缓存子面的正则控制点块。
    后续积分点查找只需 O(log n) 映射到已存在的正则子面。
    """
    
    def __init__(self, max_subdivisions: int = 4):
        self.max_subdivisions = max_subdivisions
        # face_id → {sub_face_index → (control_points_4x4x3, parent_u_range, parent_v_range)}
        self._cache: Dict[int, List[Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]]] = {}
    
    def build(self, mesh, face_id: int):
        """为不规则面构建预细分层次"""
        from forgecraft.geometry.catmull_clark import CatmullClark
        
        if face_id in self._cache:
            return
        
        sub_faces: List[Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]] = []
        
        cc = CatmullClark()
        current_mesh = mesh
        
        for level in range(self.max_subdivisions):
            result = cc.subdivide(current_mesh)
            child_mesh = result.mesh
            
            # 父面 face_id 对应 4 个子面
            for si in range(2):
                for sj in range(2):
                    child_face = face_id * 4 + si * 2 + sj
                    if child_face >= child_mesh.n_faces:
                        continue
                    
                    # 提取子面的 4×4 控制点块 (如果是正则面)
                    from forgecraft.geometry.evaluator import LimitEvaluator
                    ev = LimitEvaluator(max_subdivisions=0)
                    control, is_regular = ev._extract_patch(child_mesh, child_face)
                    
                    if is_regular:
                        u_range = (si / 2.0, (si + 1) / 2.0)
                        v_range = (sj / 2.0, (sj + 1) / 2.0)
                        sub_faces.append((control, u_range, v_range))
            
            current_mesh = child_mesh
        
        self._cache[face_id] = sub_faces
    
    def get_regular_subface(self, face_id: int, u: float, v: float
                            ) -> Optional[Tuple[np.ndarray, float, float]]:
        """查找 (u,v) 落在哪个正则子面上
        
        Returns:
            (control_4x4x3, local_u, local_v) or None
        """
        if face_id not in self._cache:
            return None
        
        for control, (u0, u1), (v0, v1) in self._cache[face_id]:
            if u0 <= u <= u1 and v0 <= v <= v1:
                local_u = (u - u0) / (u1 - u0)
                local_v = (v - v0) / (v1 - v0)
                return (control, local_u, local_v)
        
        return None
    
    def clear(self):
        self._cache.clear()


# ══════════════════════════════════════════════════════════
#  形函数结果
# ══════════════════════════════════════════════════════════

@dataclass
class ShapeResult:
    """CC 基函数在参数 (u,v) 处的求值结果
    
    N_i:   基函数值 (标量)
    dN_dx: 基函数对物理坐标的导数 (∂Nᵢ/∂x, ∂Nᵢ/∂y, ∂Nᵢ/∂z)
    J:     Jacobian 矩阵 (∂P/∂u, ∂P/∂v) → (3, 2)
    detJ:  Jacobian 行列式
    """
    N: np.ndarray                    # (n_control,) 基函数值
    dN_du: np.ndarray                # (n_control, 3) ∂N/∂u
    dN_dv: np.ndarray                # (n_control, 3) ∂N/∂v
    dN_dx: np.ndarray                # (n_control, 3) ∂N/∂x (物理空间导数)
    J: np.ndarray                    # (3, 2) Jacobian
    detJ: float
    n_control: int


# ══════════════════════════════════════════════════════════
#  求解结果
# ══════════════════════════════════════════════════════════

@dataclass
class StaticResult:
    """静力分析结果"""
    displacements: np.ndarray         # (n_dof,) 节点位移
    reactions: Optional[np.ndarray] = None  # 支反力 (if BC applied)
    strain_energy: float = 0.0
    stress: Optional[StressField] = None
    iterations: int = 0
    converged: bool = True
    wall_time: float = 0.0


@dataclass
class ModalResult:
    """模态分析结果"""
    frequencies: np.ndarray           # (n_modes,) 自然频率 (Hz)
    angular_frequencies: np.ndarray   # (n_modes,) 圆频率 (rad/s)
    mode_shapes: np.ndarray           # (n_dof, n_modes) 振型
    participation_factors: np.ndarray  # 振型参与系数
    effective_mass: np.ndarray        # 有效模态质量
    converged: bool = True


@dataclass
class TransientResult:
    """瞬态动力学结果"""
    time: np.ndarray                  # (n_steps,) 时间点
    displacement_history: np.ndarray  # (n_dof, n_steps)
    velocity_history: np.ndarray      # (n_dof, n_steps)
    acceleration_history: np.ndarray  # (n_dof, n_steps)
    kinetic_energy: np.ndarray        # (n_steps,)
    strain_energy: np.ndarray         # (n_steps,)
    dt: float
    n_steps: int


@dataclass
class StressField:
    """应力场"""
    von_mises: np.ndarray             # (n_control_points,) von Mises 应力
    principal_stresses: np.ndarray    # (n_control_points, 3) σ₁, σ₂, σ₃
    stress_components: np.ndarray     # (n_quad_points, 6) σ_xx..τ_xy
    quad_points_positions: np.ndarray  # (n_quad_points, 3) 积分点位置


# ══════════════════════════════════════════════════════════
#  异常
# ══════════════════════════════════════════════════════════

class IGAError(Exception):
    """IGA 求解异常"""
    pass
