# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_material — 材料本构模型
#
#   各向同性 / 正交各向异性 / 超弹性
#   返回 Voigt 记法的 6×6 本构矩阵 D (3D)
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numba


__all__ = [
    "LinearIsotropic",
    "LinearOrthotropic",
    "NeoHookean",
    "compliance_to_stiffness",
]


# ══════════════════════════════════════════════════════════
#  线性各向同性
# ══════════════════════════════════════════════════════════

@dataclass
class LinearIsotropic:
    """线性各向同性材料 (Hooke 定律)
    
    Args:
        E:   杨氏模量 (Pa)
        nu:  泊松比
        rho: 密度 (kg/m³)
    
    用法:
      >>> steel = LinearIsotropic(E=210e9, nu=0.3, rho=7800)
      >>> D = steel.D
    """
    E: float
    nu: float
    rho: float = 0.0
    
    # 工程常数
    G: float = None       # 剪切模量 G = E / (2*(1+nu))
    lam: float = None     # Lamé 第一参数
    mu: float = None      # Lamé 第二参数 (= G)
    K: float = None       # 体积模量
    D: np.ndarray = None  # 6×6 本构矩阵 (缓存)
    
    def __post_init__(self):
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.mu = self.G
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))
        self.D = _build_isotropic_D(self.E, self.nu)
    
    @property
    def density(self) -> float:
        return self.rho


def _build_isotropic_D(E: float, nu: float) -> np.ndarray:
    """构建 6×6 各向同性本构矩阵 (Voigt 记法)
    
    σ = D ε, 其中:
      σ = [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_zx]ᵀ
      ε = [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_zx]ᵀ
    
    D = E / ((1+ν)(1-2ν)) ×
        [1-ν   ν    ν    0         0         0    ]
        [ ν   1-ν   ν    0         0         0    ]
        [ ν    ν   1-ν   0         0         0    ]
        [ 0    0    0   (1-2ν)/2   0         0    ]
        [ 0    0    0    0        (1-2ν)/2   0    ]
        [ 0    0    0    0         0        (1-2ν)/2]
    """
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    G = (1.0 - 2.0 * nu) / 2.0
    
    D = np.zeros((6, 6))
    b = 1.0 - nu
    
    D[0, 0] = b;  D[0, 1] = nu; D[0, 2] = nu
    D[1, 0] = nu; D[1, 1] = b;  D[1, 2] = nu
    D[2, 0] = nu; D[2, 1] = nu; D[2, 2] = b
    D[3, 3] = G
    D[4, 4] = G
    D[5, 5] = G
    
    D *= c
    return D


# ══════════════════════════════════════════════════════════
#  线性正交各向异性
# ══════════════════════════════════════════════════════════

@dataclass
class LinearOrthotropic:
    """线性正交各向异性 (9 参数, 3D 打印常见)
    
    Args:
        E1, E2, E3:   三个主方向的杨氏模量 (Pa)
        nu12, nu23, nu31: 泊松比
        G12, G23, G31:    剪切模量 (Pa)
        rho:              密度 (kg/m³)
    """
    E1: float; E2: float; E3: float
    nu12: float; nu23: float; nu31: float
    G12: float; G23: float; G31: float
    rho: float = 0.0
    
    D: np.ndarray = None  # 6×6 本构矩阵 (缓存)
    
    def __post_init__(self):
        self.D = _build_orthotropic_D(
            self.E1, self.E2, self.E3,
            self.nu12, self.nu23, self.nu31,
            self.G12, self.G23, self.G31,
        )
    
    @property
    def density(self) -> float:
        return self.rho


def _build_orthotropic_D(E1, E2, E3, nu12, nu23, nu31, G12, G23, G31) -> np.ndarray:
    """构建 6×6 正交各向异性本构矩阵"""
    # 用柔度矩阵求逆得到刚度矩阵
    S = np.zeros((6, 6))
    
    S[0, 0] = 1.0 / E1
    S[1, 1] = 1.0 / E2
    S[2, 2] = 1.0 / E3
    
    S[0, 1] = S[1, 0] = -nu12 / E1
    S[1, 2] = S[2, 1] = -nu23 / E2
    S[0, 2] = S[2, 0] = -nu31 / E3
    
    S[3, 3] = 1.0 / G12
    S[4, 4] = 1.0 / G23
    S[5, 5] = 1.0 / G31
    
    return np.linalg.inv(S)


def compliance_to_stiffness(S: np.ndarray) -> np.ndarray:
    """柔度矩阵 → 刚度矩阵"""
    return np.linalg.inv(S)


# ══════════════════════════════════════════════════════════
#  超弹性 Neo-Hookean
# ══════════════════════════════════════════════════════════

@dataclass
class NeoHookean:
    """Neo-Hookean 超弹性 (可压缩)
    
    应变能密度: W = C10*(I₁-3) + K/2*(J-1)²
    
    Args:
        C10: 材料常数 (Pa), C10 = μ/2
        K:   体积模量 (Pa)
        rho: 密度 (kg/m³)
    
    用法:
      >>> rubber = NeoHookean(C10=0.5e6, K=100e6, rho=1100)
      >>> sigma, D_tangent = rubber.stress_tangent(F)
    """
    C10: float
    K: float
    rho: float = 0.0
    
    @property
    def mu(self) -> float:
        """剪切模量"""
        return 2.0 * self.C10
    
    @property
    def density(self) -> float:
        return self.rho
    
    def _cauchy_stress(self, F: np.ndarray) -> np.ndarray:
        """计算 Cauchy 应力 (Voigt 6 分量)，不计算切线刚度
        
        Args:
            F: (3, 3) 变形梯度
        
        Returns:
            sigma_6: (6,) Voigt 记法 Cauchy 应力
        """
        J = np.linalg.det(F)
        
        # 等容部分
        b_bar = J**(-2.0/3.0) * (F @ F.T)
        tr_b = np.trace(b_bar)
        
        # Kirchhoff 应力
        tau_iso = self.mu / J**(2.0/3.0) * (F @ F.T - (tr_b / 3.0) * np.eye(3))
        tau_vol = self.K * (J - 1.0) * J * np.eye(3)
        tau = tau_iso + tau_vol
        
        # Cauchy 应力 σ = τ / J
        sigma = tau / J
        
        return np.array([
            sigma[0, 0], sigma[1, 1], sigma[2, 2],
            sigma[0, 1], sigma[1, 2], sigma[0, 2],
        ])
    
    def stress_tangent(self, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算 Cauchy 应力和切线刚度
        
        Args:
            F: (3, 3) 变形梯度
        
        Returns:
            (sigma_6, D_6x6) Voigt 记法的应力和切线刚度
        """
        sigma_6 = self._cauchy_stress(F)
        
        # 切线刚度 (数值差分，使用 _cauchy_stress 避免递归)
        eps = 1e-6
        D_tangent = np.zeros((6, 6))
        I6 = np.eye(6)
        
        for i in range(6):
            Fp = _voigt_to_deformation_gradient(F, I6[i] * eps / 2.0)
            Fm = _voigt_to_deformation_gradient(F, -I6[i] * eps / 2.0)
            
            sp = self._cauchy_stress(Fp)
            sm = self._cauchy_stress(Fm)
            
            D_tangent[:, i] = (sp - sm) / eps
        
        return sigma_6, D_tangent


def _voigt_to_deformation_gradient(F: np.ndarray, dE_voigt: np.ndarray) -> np.ndarray:
    """Voigt 应变增量 → 变形梯度更新
    
    dE = [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_zx]
    F_new = (I + dε) @ F
    """
    dE = np.array([
        [dE_voigt[0],           dE_voigt[3] / 2.0, dE_voigt[5] / 2.0],
        [dE_voigt[3] / 2.0, dE_voigt[1],           dE_voigt[4] / 2.0],
        [dE_voigt[5] / 2.0, dE_voigt[4] / 2.0, dE_voigt[2]],
    ])
    return (np.eye(3) + dE) @ F
