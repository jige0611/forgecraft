# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_topology_types — 相位场拓扑优化类型
#
#   集中所有数据类型定义, 消除模块间循环导入。
#   相位场密度 φ(x) ∈ [φ_min, 1], IGA 基函数 C¹ 连续。
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    # 插值函数
    "InterpolationFunction",
    "g_simp", "dg_simp",
    "g_ramp", "dg_ramp",
    "g_polynomial", "dg_polynomial",
    # 设置
    "TopologySettings",
    "TopologyMaterial",
    # 约束
    "TopologyConstraint",
    "ConstraintType",
    # 结果
    "TopologyOptimizationResult",
    # 历史
    "TopologyIterationData",
]

# ══════════════════════════════════════════════════════════
#  材料插值函数 (JIT 兼容)
# ══════════════════════════════════════════════════════════

class InterpolationFunction:
    """材料插值函数协议

    g(φ): [0,1] → [φ_min, 1]  退化材料模量
    g'(φ): dg/dφ                灵敏度分析用

    三种实现:
      - SIMP:  g(φ) = φ_min + (1-φ_min)·φ^p
      - RAMP:  g(φ) = φ / (1 + q·(1-φ))
      - 多项式: g(φ) = φ³
    """

    @staticmethod
    def get(name: str, **params):
        """工厂方法"""
        if name.lower() == "simp":
            p = params.get("p", 3.0)
            phi_min = params.get("phi_min", 1e-6)
            return SIMPInterpolation(p, phi_min)
        elif name.lower() == "ramp":
            q = params.get("q", 3.0)
            phi_min = params.get("phi_min", 1e-6)
            return RAMPInterpolation(q, phi_min)
        elif name.lower() == "polynomial":
            return PolynomialInterpolation()
        else:
            raise ValueError(f"Unknown interpolation: {name}")

    def g(self, phi: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def dg(self, phi: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SIMPInterpolation(InterpolationFunction):
    """SIMP 插值: g(φ) = φ_min + (1-φ_min)·φ^p"""

    def __init__(self, p: float = 3.0, phi_min: float = 1e-6):
        self.p = float(p)
        self.phi_min = float(phi_min)

    def g(self, phi: np.ndarray) -> np.ndarray:
        return self.phi_min + (1.0 - self.phi_min) * np.power(phi, self.p)

    def dg(self, phi: np.ndarray) -> np.ndarray:
        return self.p * (1.0 - self.phi_min) * np.power(phi, self.p - 1.0)


class RAMPInterpolation(InterpolationFunction):
    """RAMP 插值: g(φ) = φ_min + (1-φ_min)·φ/(1+q(1-φ))"""

    def __init__(self, q: float = 3.0, phi_min: float = 1e-6):
        self.q = float(q)
        self.phi_min = float(phi_min)

    def g(self, phi: np.ndarray) -> np.ndarray:
        denom = 1.0 + self.q * (1.0 - phi)
        return self.phi_min + (1.0 - self.phi_min) * phi / denom

    def dg(self, phi: np.ndarray) -> np.ndarray:
        denom = 1.0 + self.q * (1.0 - phi)
        return (1.0 - self.phi_min) * (1.0 + self.q) / (denom * denom)


class PolynomialInterpolation(InterpolationFunction):
    """多项式插值: g(φ) = φ³"""

    def g(self, phi: np.ndarray) -> np.ndarray:
        return np.power(phi, 3.0)

    def dg(self, phi: np.ndarray) -> np.ndarray:
        return 3.0 * np.power(phi, 2.0)


# ══════════════════════════════════════════════════════════
#  JIT 友好标量函数 (供 numba 热路径使用)
# ══════════════════════════════════════════════════════════

def g_simp(phi: float, p: float = 3.0, phi_min: float = 1e-6) -> float:
    """SIMP 插值标量版"""
    return phi_min + (1.0 - phi_min) * (phi ** p)


def dg_simp(phi: float, p: float = 3.0, phi_min: float = 1e-6) -> float:
    """SIMP 插值导数标量版"""
    if phi <= 0.0:
        return 0.0
    return p * (1.0 - phi_min) * (phi ** (p - 1.0))


def g_ramp(phi: float, q: float = 3.0, phi_min: float = 1e-6) -> float:
    """RAMP 插值标量版"""
    denom = 1.0 + q * (1.0 - phi)
    return phi_min + (1.0 - phi_min) * phi / denom


def dg_ramp(phi: float, q: float = 3.0, phi_min: float = 1e-6) -> float:
    """RAMP 插值导数标量版"""
    denom = 1.0 + q * (1.0 - phi)
    return (1.0 - phi_min) * (1.0 + q) / (denom * denom)


def g_polynomial(phi: float) -> float:
    """多项式插值标量版"""
    return phi ** 3.0


def dg_polynomial(phi: float) -> float:
    """多项式插值导数标量版"""
    return 3.0 * phi ** 2.0


# ══════════════════════════════════════════════════════════
#  双阱势 (相场界面能)
# ══════════════════════════════════════════════════════════

def w_double_well(phi: float) -> float:
    """双阱势 w(φ) = φ²(1-φ)²"""
    return (phi ** 2.0) * ((1.0 - phi) ** 2.0)


def dw_double_well(phi: float) -> float:
    """双阱势一阶导数"""
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def d2w_double_well(phi: float) -> float:
    """双阱势二阶导数"""
    return 2.0 * (1.0 - 6.0 * phi + 6.0 * phi ** 2.0)


# ══════════════════════════════════════════════════════════
#  设置
# ══════════════════════════════════════════════════════════

@dataclass
class TopologySettings:
    """相位场拓扑优化全局设置

    核心参数:
      - vol_frac:           目标体积分数 (0~1)
      - epsilon:            界面厚度 (类比断裂相场的 length_scale)
      - p_init/p_max:       SIMP 惩罚指数连续范围
      - kappa:              界面能缩放系数 (控制实体/空洞过渡带宽度)

    连续方案 (Continuation):
      逐步增大惩罚指数 p, 收缩界面 ε, 避免陷入局部极小。
    """

    # ── 目标 ──
    vol_frac: float = 0.3                         # 目标体积分数
    phi_min: float = 1e-6                         # 密度下限 (防奇异)

    # ── 材料插值 ──
    interpolation: str = "SIMP"                   # "SIMP" | "RAMP" | "polynomial"
    p_init: float = 1.0                           # SIMP 惩罚起始值
    p_max: float = 3.0                            # SIMP 惩罚终值
    p_step: float = 0.25                          # 每次连续增量

    # ── 相场界面 ──
    epsilon: float = 0.02                         # 界面厚度
    epsilon_decay: float = 0.95                   # 收缩因子
    kappa: float = 1.0                            # 界面能缩放

    # ── 连续方案 ──
    continue_p: bool = True                       # 是否递增惩罚
    continue_epsilon: bool = True                 # 是否收缩界面
    continuation_freq: int = 10                   # 每隔多少迭代更新参数

    # ── 优化器 ──
    optimizer: str = "OC"                         # "OC" | "augmented_lagrangian"
    max_iter: int = 100                           # 最大外循环
    design_tol: float = 1e-3                      # 设计变量收敛容差
    obj_tol: float = 1e-4                         # 目标函数相对收敛容差
    move_limit: float = 0.2                       # OC 移动限制

    # ── 体积约束 ──
    constraint_method: str = "augmented_lagrangian"  # "augmented_lagrangian" | "direct"
    lagrangian_multiplier: float = 0.0            # 拉格朗日乘子初值
    penalty_param: float = 1.0                    # 增广拉格朗日罚参数 ρ
    penalty_growth: float = 1.2                   # 每次增长因子
    penalty_max: float = 1e6                      # 罚参数上限

    # ── 求解器 ──
    quadrature_order: int = 3
    max_iter_inner: int = 500
    tolerance: float = 1e-10
    penalty_bc: float = 1e12                      # Dirichlet BC 罚因子
    verbosity: int = 1

    # ── 滤波 ──
    density_filter_radius: float = 0.0            # 额外密度滤波半径 (0=仅 ε 正则化)

    @property
    def p_current(self) -> float:
        """当前惩罚指数 (内部可变状态)"""
        return getattr(self, '_p_current', self.p_init)

    @p_current.setter
    def p_current(self, value: float):
        self._p_current = value


@dataclass
class TopologyMaterial:
    """拓扑优化材料 (含插值参数)

    结构材料 + SIMP/RAMP 插值配置。
    优化过程中材料模量通过 g(φ) 退化。
    """

    E: float = 210e9                  # 杨氏模量 [Pa]
    nu: float = 0.3                   # 泊松比
    rho: float = 7800.0               # 密度 [kg/m³]
    plasticity: bool = False          # 是否考虑塑性

    # 插值覆盖 (None 则使用 TopologySettings)
    interpolation: Optional[str] = None  # "SIMP" | "RAMP" | "polynomial"
    p: Optional[float] = None            # 覆盖 TopologySettings.p


# ══════════════════════════════════════════════════════════
#  约束定义
# ══════════════════════════════════════════════════════════

class ConstraintType:
    """约束类型枚举"""
    VOLUME = "volume"
    COMPLIANCE = "compliance"
    STRESS = "stress"
    DISPLACEMENT = "displacement"
    MANUFACTURING = "manufacturing"


@dataclass
class TopologyConstraint:
    """优化约束定义

    用法:
      >>> vol_c = TopologyConstraint.volume(target=0.3)
      >>> stress_c = TopologyConstraint.stress(target=250e6, p_norm=8.0)
      >>> disp_c = TopologyConstraint.displacement(target=0.001, dof_id=42)
    """

    constraint_type: str
    target: float
    tolerance: float = 1e-3

    # 应力约束专用
    p_norm: float = 8.0              # p-norm 聚合指数
    sigma_yield: float = 250e6       # 屈服强度 [Pa]

    # 位移约束专用
    dof_id: Optional[int] = None     # 被约束的 DOF 编号
    direction: Optional[str] = None   # "x" | "y" | "z" | None (=magnitude)

    @classmethod
    def volume(cls, target: float = 0.3, tolerance: float = 1e-3) -> TopologyConstraint:
        return cls(constraint_type=ConstraintType.VOLUME, target=target, tolerance=tolerance)

    @classmethod
    def compliance(cls, target: float, tolerance: float = 1e-3) -> TopologyConstraint:
        return cls(constraint_type=ConstraintType.COMPLIANCE, target=target, tolerance=tolerance)

    @classmethod
    def stress(cls, target: float, p_norm: float = 8.0, tolerance: float = 1e-3) -> TopologyConstraint:
        return cls(constraint_type=ConstraintType.STRESS, target=target,
                   tolerance=tolerance, p_norm=p_norm)

    @classmethod
    def displacement(cls, target: float, dof_id: int, tolerance: float = 1e-3) -> TopologyConstraint:
        return cls(constraint_type=ConstraintType.DISPLACEMENT, target=target,
                   tolerance=tolerance, dof_id=dof_id)


# ══════════════════════════════════════════════════════════
#  结果与历史
# ══════════════════════════════════════════════════════════

@dataclass
class TopologyIterationData:
    """单次迭代历史"""
    iteration: int
    compliance: float
    volume_fraction: float
    phi_change: float                     # ||φ_new - φ_old||∞
    compliance_change: float              # |C_new - C_old| / C_new
    lagrange_multiplier: float
    penalty_param: float
    sim_p: float
    epsilon: float
    n_design_updates: int = 0             # 设计更新内迭代数


@dataclass
class TopologyOptimizationResult:
    """拓扑优化完整结果"""

    # 优化设计
    phi_field: np.ndarray                 # (n_vertices,) 最终相场
    optimized_density: np.ndarray         # (n_vertices,) 材料密度 = g(φ)
    displacement: Optional[np.ndarray] = None  # (n_dof,) 最终位移

    # 收敛信息
    iterations: int = 0
    converged: bool = False
    termination_reason: str = ""           # "converged" | "max_iter" | "diverged"

    # 历史
    compliance_history: List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)
    iteration_data: List[TopologyIterationData] = field(default_factory=list)

    # 后处理
    solid_vertices: Optional[np.ndarray] = None     # φ ≥ 0.5 的顶点索引
    solid_volume: float = 0.0                       # 实体体积
    solid_mass: float = 0.0                         # 实体质量
    mass_reduction: float = 0.0                     # 减重百分比

    # 性能
    wall_time: float = 0.0
    n_total_linear_solves: int = 0

    def summary(self) -> str:
        """生成结果摘要"""
        lines = [
            "═" * 48,
            "  Phase-Field Topology Optimization Result",
            "═" * 48,
            f"  Iterations:         {self.iterations}",
            f"  Converged:          {self.converged} ({self.termination_reason})",
            f"  Final compliance:   {self.compliance_history[-1]:.4e}" if self.compliance_history else "",
            f"  Final volume frac:  {self.volume_history[-1]:.4f}" if self.volume_history else "",
            f"  Mass reduction:     {self.mass_reduction:.1%}",
            f"  Solid mass:         {self.solid_mass:.4f} kg",
            f"  Wall time:          {self.wall_time:.2f}s",
            f"  Linear solves:      {self.n_total_linear_solves}",
            "═" * 48,
        ]
        return "\n".join(line for line in lines if line)
