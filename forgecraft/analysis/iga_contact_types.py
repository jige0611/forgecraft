# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_contact_types — 接触/断裂共用类型
#
#   ContactSettings:    接触求解设置
#   ContactPair:        候选接触对
#   GapResult:          间隙计算结果
#   FrictionModel:      Coulomb 摩擦模型
#   CohesiveLaw:        内聚力牵引力-分离律
#   PhaseFieldSettings: 相场断裂设置
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "ContactSettings",
    "ContactPair",
    "GapResult",
    "FrictionModel",
    "CohesiveLaw",
    "PhaseFieldSettings",
    "CrackInfo",
]


# ══════════════════════════════════════════════════════════
#  接触设置
# ══════════════════════════════════════════════════════════

@dataclass
class ContactSettings:
    """接触求解设置
    
    用法:
      >>> cs = ContactSettings(method="augmented_lagrange", friction_coefficient=0.3)
    """
    method: str = "penalty"              # "penalty" | "lagrange" | "augmented_lagrange"
    penalty_normal: float = 1e10         # 法向罚因子 ε_N
    penalty_tangential: float = 1e9      # 切向罚因子 ε_T (摩擦)
    friction_model: str = "coulomb"      # "coulomb" | "none"
    friction_coefficient: float = 0.3    # μ
    augmented_tolerance: float = 1e-6    # Uzawa 收敛容差
    augmented_max_iter: int = 20         # Uzawa 最大迭代
    search_method: str = "aabb"          # "aabb" | "bsphere"
    gap_tolerance: float = 1e-8          # 间隙容差
    quadrature_order_contact: int = 3    # 接触面 Gauss 积分阶数
    mortar: bool = False                 # 是否启用 mortar 接触


# ══════════════════════════════════════════════════════════
#  接触对
# ══════════════════════════════════════════════════════════

@dataclass
class ContactPair:
    """候选接触对 (从面积分点-主面投影)
    
    每个活跃接触对代表一个从面积分点与主面上最近点的配对。
    """
    slave_face: int          # 从面 ID
    master_face: int         # 主面 ID
    slave_uv: np.ndarray     # (2,) 从面参数坐标 (积分点)
    master_uv: np.ndarray    # (2,) 主面最近点参数坐标 (Newton 投影结果)
    slave_position: np.ndarray  # (3,) 从面积分点物理坐标
    master_position: np.ndarray # (3,) 主面最近点物理坐标
    normal: np.ndarray       # (3,) 主面法向量 (在最近点处)
    gap: float               # 带符号间隙 (负 = 穿透)
    active: bool = True      # 是否活跃接触
    lambda_n: float = 0.0    # 拉格朗日乘子 (法向) — 增广拉格朗日用
    lambda_t: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 乘子 (切向)
    slip: np.ndarray = field(default_factory=lambda: np.zeros(2))      # 累积滑移量


# ══════════════════════════════════════════════════════════
#  间隙计算结果
# ══════════════════════════════════════════════════════════

@dataclass
class GapResult:
    """最近点投影 (Closest Point Projection) 的完整结果"""
    position: np.ndarray          # (3,) 主面上最近点物理坐标
    normal: np.ndarray            # (3,) 主面法向量 (单位, 指向外)
    parametric_coords: np.ndarray # (2,) 主面参数坐标 (u, v)
    gap: float                    # 带符号间隙: (x_slave - x_master) · n
    distance: float               # 绝对距离 ||x_slave - x_master||
    converged: bool               # Newton 迭代是否收敛
    iterations: int               # 迭代次数


# ══════════════════════════════════════════════════════════
#  Coulomb 摩擦模型
# ══════════════════════════════════════════════════════════

@dataclass
class FrictionModel:
    """Coulomb 摩擦模型 (正则化)
    
    摩擦准则:  Φ(t_T, p_N) = ||t_T|| - μ·max(0, -p_N) ≤ 0
    
    正则化策略: 平滑 stick-slip 过渡
      slip_factor = ||t_trial|| / (ε_T + μ·|p_N|)
      当 slip_factor < 1: stick (弹性)
      当 slip_factor ≥ 1: slip (滑动)
    """
    mu: float                     # Coulomb 摩擦系数
    regularization: float = 1e-4  # 正则化参数 (避免 stick-slip 梯度不连续)
    
    def slip_function(self, t_trial: np.ndarray, p_norm: float) -> float:
        """返回滑移指示值 Φ / (ε_τ + μ|p|)"""
        norm_t = np.linalg.norm(t_trial)
        denom = self.regularization + self.mu * p_norm
        return norm_t / max(denom, 1e-15)
    
    def consistent_tangent(
        self, t_trial: np.ndarray, p_norm: float,
    ) -> np.ndarray:
        """一致切线刚度 (3×3, 局部坐标)
        
        返回 ∂t_contact/∂Δg 用于 Newton-Raphson 收敛。
        """
        norm_t = np.linalg.norm(t_trial)
        
        if norm_t < 1e-15:
            return np.eye(3)
        
        n_hat = t_trial / norm_t
        slip_val = self.slip_function(t_trial, p_norm)
        
        # 正则化: 平滑过渡
        if slip_val < 1.0:
            # Stick: 弹性切向刚度
            # D_T = ε_T · I
            return np.eye(3)
        else:
            # Slip: 塑性 + Coulomb 软化
            # D_T = μ|p|/||t_trial|| · (I - n⊗n)
            factor = self.mu * p_norm / norm_t
            D_iso = np.eye(3) - np.outer(n_hat, n_hat)
            return factor * D_iso


# ══════════════════════════════════════════════════════════
#  内聚力牵引力-分离律
# ══════════════════════════════════════════════════════════

@dataclass
class CohesiveLaw:
    """双线性牵引力-分离律 (Traction-Separation Law)
    
         σ
         ↑
      σ_max ───●━━━┓
         │    ╱    ┃  G_c = σ_max·δ_f / 2
         │   ╱     ┃
         │  ╱      ┃
         │ ╱       ┃
         │╱        ┃
         ●─────────●──→ δ
         0   δ_0   δ_f
    
    - δ ≤ δ_0:  弹性加载 (罚刚度 K_0 = σ_max/δ_0)
    - δ_0 < δ ≤ δ_f: 线性软化 (损伤演化)
    - δ > δ_f:  完全失效 (traction = 0)
    """
    sigma_max: float          # 界面强度 (Pa)
    Gc: float                 # 断裂韧性 (J/m²)
    delta_0: float            # 损伤起始分离量 (m)
    delta_f: float            # 完全失效分离量 (m)
    penalty_stiffness: float = 1e12  # 未损伤界面罚刚度
    law_type: str = "bilinear"       # "bilinear" | "exponential" | "trapezoidal"
    
    @classmethod
    def from_energy(cls, sigma_max: float, Gc: float, penalty: float = 1e12) -> CohesiveLaw:
        """从强度+断裂能构造双线性律"""
        delta_0 = sigma_max / penalty
        delta_f = 2.0 * Gc / sigma_max
        return cls(
            sigma_max=sigma_max,
            Gc=Gc,
            delta_0=delta_0,
            delta_f=delta_f,
            penalty_stiffness=penalty,
        )
    
    def damage(self, delta: np.ndarray, delta_max_history: float) -> Tuple[float, float]:
        """损伤变量和当前有效分离量
        
        Args:
            delta: (3,) 分离向量 [δ_n, δ_t1, δ_t2]
            delta_max_history: 历史最大有效分离量
        
        Returns:
            (D, delta_eff) 损伤变量和有效分离量
        """
        # 有效分离量 (混合模式)
        delta_n = max(0.0, delta[0])  # 法向分离 (仅拉伸致损)
        delta_t = np.sqrt(delta[1]**2 + delta[2]**2)  # 切向分离
        
        # 能量-等效分离量
        delta_eff = np.sqrt(delta_n**2 + delta_t**2)
        
        if delta_eff <= delta_max_history or delta_eff <= self.delta_0:
            D = 0.0
        elif delta_eff >= self.delta_f:
            D = 1.0
        else:
            # 双线性软化
            D = (self.delta_f * (delta_eff - self.delta_0)) / (
                delta_eff * (self.delta_f - self.delta_0)
            )
        
        return D, max(delta_eff, delta_max_history)
    
    def traction(
        self, delta: np.ndarray, D: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """返回牵引力和切线刚度
        
        Args:
            delta: (3,) 分离向量
            D: 当前损伤
        
        Returns:
            (t, D_tangent) 牵引力和 (3, 3) 切线刚度
        """
        K0 = self.penalty_stiffness
        t_elastic = K0 * delta
        
        if D <= 0.0:
            return t_elastic, K0 * np.eye(3)
        elif D >= 1.0:
            return np.zeros(3), np.zeros((3, 3))
        
        # 损伤后
        t = (1.0 - D) * t_elastic
        
        # 切线刚度 (损伤演化贡献)
        D_tangent = (1.0 - D) * K0 * np.eye(3)
        
        # 添加损伤软化项
        if delta[0] > 0:  # 仅法向
            dD_ddelta = self.delta_f * self.delta_0 / (
                (delta[0] - self.delta_0)**2 * self.delta_f
            ) if delta[0] > self.delta_0 else 0.0
            D_tangent[0, 0] -= dD_ddelta * K0 * delta[0]
        
        return t, D_tangent


# ══════════════════════════════════════════════════════════
#  相场断裂设置
# ══════════════════════════════════════════════════════════

@dataclass
class PhaseFieldSettings:
    """相场断裂模型设置
    
    AT2 退化函数: g(d) = (1-d)² + η
    
    裂纹正则化长度 ℓ 的选择:
      ℓ 应 ≥ 2×h_e (网格尺寸)，确保裂纹带内有足够 Gauss 点
    
    断裂韧性 Gc 的来源:
      - 标准材料: Gc ≈ 1-100 J/m² (脆性)
      - 韧性材料: Gc ≈ 10³-10⁵ J/m² (韧性)
    """
    Gc: float = 100.0             # 临界能量释放率 (J/m²)
    length_scale: float = 0.02    # 正则化长度 ℓ
    eta: float = 1e-6             # 退化函数残差 (避免数值奇性)
    model: str = "AT2"            # "AT2" (Ambrosio-Tortorelli 二阶)
    irreversibility: str = "history"  # "history" | "penalty" | "damage_constraint"
    
    # 求解设置
    stagger_tolerance: float = 1e-4   # 交错迭代容差
    stagger_max_iter: int = 100       # 交错迭代最大次数
    use_degradation_split: bool = True  # 是否使用拉-压分解 (仅拉伸退化)
    monolithic: bool = False          # True = 整体求解 (需高内存)

    def g(self, d: float) -> float:
        """AT2 退化函数"""
        return (1.0 - d)**2 + self.eta


@dataclass
class CrackInfo:
    """裂纹几何信息 (用于可视化)"""
    positions: np.ndarray    # (n, 3) 裂纹表面点
    normals: np.ndarray      # (n, 3) 裂纹法向量
    damage_field: np.ndarray # (n_vertices,) 顶点相场值
    crack_area: float        # 裂纹总面积 (近似)
    crack_length: float      # 裂纹长度 (3D→投影)
