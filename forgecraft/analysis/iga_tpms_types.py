# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_tpms_types — TPMS 晶格类型系统
#
#   梯度 TPMS (三周期极小曲面) 晶格:
#     - 8 种标准 TPMS (Gyroid, Schwarz P/D, Diamond, ...)
#     - 空间梯度参数: 晶胞尺寸 ω(x), 壁厚 t(x)
#     - 应力自适应映射策略
#     - CC 控制顶点参数化 → C¹ 连续光滑梯度
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = [
    "TPMSType",
    "GradingStrategy",
    "TPMSParameters",
    "TPMSGradingSettings",
    "TPMSLatticeResult",
    "TPMS_SURFACE_FUNCTIONS",
    "TPMS_DEFAULTS",
]


# ══════════════════════════════════════════════════════════
#  TPMS 类型枚举
# ══════════════════════════════════════════════════════════

class TPMSType(Enum):
    """TPMS 曲面类型

    每种 TPMS 由隐式方程 f(x,y,z) = C 定义。
    实体区域: |f(x,y,z)| < t/2 (壁厚控制)
    """
    GYROID = auto()          # sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = C
    SCHWARZ_P = auto()       # cos(x) + cos(y) + cos(z) = C
    SCHWARZ_D = auto()       # cos(x)cos(y)cos(z) - sin(x)sin(y)sin(z) = C
    DIAMOND = auto()         # sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z) + ...
    SPLIT_P = auto()         # 1.1(sin(2x)sin(z)cos(y) + sin(2y)sin(x)cos(z) + sin(2z)sin(y)cos(x)) - 0.2(...)
    LIDINOID = auto()        # 0.5(sin(2x)sin(z)cos(y) + sin(2y)sin(x)cos(z) + sin(2z)sin(y)cos(x)) - ...
    NEOVIUS = auto()         # 3(cos(x)+cos(y)+cos(z)) + 4cos(x)cos(y)cos(z) = C
    IWP = auto()             # cos(x)cos(y) + cos(y)cos(z) + cos(z)cos(x) = C

    def __str__(self) -> str:
        return self.name


# ══════════════════════════════════════════════════════════
#  梯度策略枚举
# ══════════════════════════════════════════════════════════

class GradingStrategy(Enum):
    """应力 → 晶格参数的映射策略"""
    STRESS_PROPORTIONAL = auto()   # ω ∝ σ_vm^p  (幂律)
    ENERGY_DENSITY = auto()         # ω ∝ ψe^p   (应变能密度)
    VON_MISES_THRESHOLD = auto()    # σ > σ_th → 密, σ < σ_th → 疏 (分段)
    MANUAL_FIELD = auto()           # 用户自定义标量场

    def __str__(self) -> str:
        return self.name


# ══════════════════════════════════════════════════════════
#  每顶点 TPMS 参数
# ══════════════════════════════════════════════════════════

@dataclass
class TPMSParameters:
    """单个控制顶点的 TPMS 晶格参数

    TPMS 水平集方程:
      f(ωx·x, ωy·y, ωz·z) - offset = 0

    实体区域:
      |f(ωx·x, ωy·y, ωz·z) - offset| < thickness / 2

    Args:
        cell_size: 晶胞尺寸 (各向同性) 或 (cell_x, cell_y, cell_z)
        thickness: 壁厚 (实体壳厚度, 0 表示无限薄曲面)
        offset: 水平集偏移 (改变固/空比)
        frequency: 频率 ω = 2π / cell_size, 可选 (自动从 cell_size 计算)
    """
    cell_size: float = 1.0
    thickness: float = 0.1
    offset: float = 0.0
    frequency: Optional[Tuple[float, float, float]] = None

    @classmethod
    def uniform(cls, cell_size: float = 1.0, thickness: float = 0.1,
                offset: float = 0.0) -> TPMSParameters:
        """创建均匀参数"""
        return cls(cell_size=cell_size, thickness=thickness, offset=offset)

    @property
    def wx(self) -> float:
        if self.frequency is not None:
            return self.frequency[0]
        return 2.0 * np.pi / max(self.cell_size, 1e-10)

    @property
    def wy(self) -> float:
        if self.frequency is not None:
            return self.frequency[1]
        return 2.0 * np.pi / max(self.cell_size, 1e-10)

    @property
    def wz(self) -> float:
        if self.frequency is not None:
            return self.frequency[2]
        return 2.0 * np.pi / max(self.cell_size, 1e-10)


# ══════════════════════════════════════════════════════════
#  梯度设置
# ══════════════════════════════════════════════════════════

@dataclass
class TPMSGradingSettings:
    """应力自适应梯度映射设置

    应力场 σ(x) → 晶格参数 (cell_size, thickness):

      cell_size(x) = CS_min + (CS_max - CS_min) · (σ(x)/σ_ref)^exponent
      thickness(x) = T_min  + (T_max  - T_min)  · (σ(x)/σ_ref)^thickness_exponent

    高应力区域 → 小晶胞 + 厚壁 → 更密/更强
    低应力区域 → 大晶胞 + 薄壁 → 更疏/更轻

    Args:
        tpms_type: TPMS 类型
        strategy: 应力映射策略
        cell_size_range: (min, max) 晶胞尺寸范围 (m)
        thickness_range: (min, max) 壁厚范围 (m)
        stress_exponent: 应力 → 晶胞尺寸幂律指数
        thickness_exponent: 应力 → 壁厚幂律指数
        stress_reference: 参考应力 (用于归一化, None=自动 max)
        smoothing_radius: Helmholtz 平滑半径 (0=不平滑)
        volume_target: 目标体积分数 (None=不约束)
        min_density: 最小相对密度 (避免断开)
        max_density: 最大相对密度
        iterations: 迭代优化轮数 (模式 B)
        convergence_tol: 迭代收敛容差
    """
    tpms_type: TPMSType = TPMSType.GYROID
    strategy: GradingStrategy = GradingStrategy.STRESS_PROPORTIONAL

    # 晶格参数范围
    cell_size_range: Tuple[float, float] = (0.5, 2.0)
    thickness_range: Tuple[float, float] = (0.02, 0.15)
    offset_range: Tuple[float, float] = (-0.2, 0.2)

    # 应力 → 参数映射
    stress_exponent: float = 1.0
    thickness_exponent: float = 1.0
    stress_reference: Optional[float] = None  # None = 自动使用 max(σ)

    # 平滑
    smoothing_radius: float = 0.0  # Helmholtz R

    # 体积约束
    volume_target: Optional[float] = None
    min_density: float = 0.1
    max_density: float = 0.9

    # 迭代
    iterations: int = 1
    convergence_tol: float = 0.01

    # 其他
    verbosity: int = 0

    def __post_init__(self):
        cs_min, cs_max = self.cell_size_range
        if cs_min <= 0 or cs_max < cs_min:
            raise ValueError(f"cell_size_range 无效: {self.cell_size_range}")
        t_min, t_max = self.thickness_range
        if t_min < 0 or t_max < t_min:
            raise ValueError(f"thickness_range 无效: {self.thickness_range}")


# ══════════════════════════════════════════════════════════
#  结果数据类
# ══════════════════════════════════════════════════════════

@dataclass
class TPMSLatticeResult:
    """TPMS 晶格生成结果

    Attributes:
        success: 是否成功
        tpms_type: 使用的 TPMS 类型
        strategy: 使用的梯度策略
        cell_size_field: (n_vertices,) 晶胞尺寸场
        thickness_field: (n_vertices,) 壁厚场
        stress_field: (n_vertices,) 输入应力场 (von Mises)
        relative_density_field: (n_vertices,) 相对密度场
        volume_fraction: 总体积分数
        average_cell_size: 平均晶胞尺寸
        iterations: 实际迭代次数
        convergence_history: 收敛历史
        mesh_vertices: 生成的三角形网格顶点 (N,3)
        mesh_faces: 生成的三角形网格面片 (M,3)
        wall_time: 计算耗时 (秒)
    """
    success: bool = False
    tpms_type: TPMSType = TPMSType.GYROID
    strategy: GradingStrategy = GradingStrategy.STRESS_PROPORTIONAL

    # 场数据
    cell_size_field: Optional[np.ndarray] = None
    thickness_field: Optional[np.ndarray] = None
    stress_field: Optional[np.ndarray] = None
    relative_density_field: Optional[np.ndarray] = None

    volume_fraction: float = 0.0
    average_cell_size: float = 0.0

    # 收敛
    iterations: int = 0
    convergence_history: List[float] = field(default_factory=list)

    # 几何输出
    mesh_vertices: Optional[np.ndarray] = None
    mesh_faces: Optional[np.ndarray] = None

    wall_time: float = 0.0

    def summary(self) -> str:
        lines = [
            f"TPMS Lattice: {self.tpms_type.name}",
            f"  Strategy: {self.strategy.name}",
            f"  Volume: {self.volume_fraction:.2%}",
            f"  Avg cell: {self.average_cell_size:.3f}",
            f"  Iterations: {self.iterations}",
            f"  Time: {self.wall_time:.3f}s",
        ]
        if self.mesh_vertices is not None:
            nv = len(self.mesh_vertices)
            nf = len(self.mesh_faces) if self.mesh_faces is not None else 0
            lines.append(f"  Mesh: {nv} verts, {nf} tris")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════
#  TPMS 曲面函数注册表
# ══════════════════════════════════════════════════════════

TPMS_SURFACE_FUNCTIONS: Dict[TPMSType, str] = {
    TPMSType.GYROID:     "sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)",
    TPMSType.SCHWARZ_P:  "cos(x) + cos(y) + cos(z)",
    TPMSType.SCHWARZ_D:  "cos(x)*cos(y)*cos(z) - sin(x)*sin(y)*sin(z)",
    TPMSType.DIAMOND:    "sin(x)*sin(y)*sin(z) + sin(x)*cos(y)*cos(z) + cos(x)*sin(y)*cos(z) + cos(x)*cos(y)*sin(z)",
    TPMSType.SPLIT_P:    "1.1*(sin(2*x)*sin(z)*cos(y)+sin(2*y)*sin(x)*cos(z)+sin(2*z)*sin(y)*cos(x)) - 0.2*(cos(2*x)*cos(2*y)+cos(2*y)*cos(2*z)+cos(2*z)*cos(2*x))",
    TPMSType.LIDINOID:   "0.5*(sin(2*x)*sin(z)*cos(y)+sin(2*y)*sin(x)*cos(z)+sin(2*z)*sin(y)*cos(x)) - 0.5*(cos(2*x)*cos(2*y)+cos(2*y)*cos(2*z)+cos(2*z)*cos(2*x))",
    TPMSType.NEOVIUS:    "3*(cos(x)+cos(y)+cos(z)) + 4*cos(x)*cos(y)*cos(z)",
    TPMSType.IWP:        "cos(x)*cos(y) + cos(y)*cos(z) + cos(z)*cos(x)",
}

# Gibson-Ashby 等效模量参数: E_eff/E_solid = C * (ρ_rel)^n
TPMS_DEFAULTS = {
    TPMSType.GYROID:     {"C": 0.5, "n": 2.0, "description": "弯曲主导, 各向同性, 高能量吸收"},
    TPMSType.SCHWARZ_P:  {"C": 0.3, "n": 2.0, "description": "弯曲主导, 简单立方对称"},
    TPMSType.SCHWARZ_D:  {"C": 0.4, "n": 2.0, "description": "弯曲主导, 钻石立方对称"},
    TPMSType.DIAMOND:    {"C": 0.4, "n": 2.0, "description": "弯曲主导, 高比表面积"},
    TPMSType.SPLIT_P:    {"C": 0.35, "n": 2.0, "description": "弯曲主导, 各向异性可控"},
    TPMSType.LIDINOID:   {"C": 0.35, "n": 2.0, "description": "弯曲主导, 高渗透性"},
    TPMSType.NEOVIUS:    {"C": 0.6, "n": 2.0, "description": "接近拉伸主导, 高强度"},
    TPMSType.IWP:        {"C": 0.45, "n": 2.0, "description": "弯曲主导, 闭孔结构"},
}
