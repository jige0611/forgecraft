# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_multiphysics_types — 多物理场基础类型
#
#   四场耦合 (结构 + 热 + 流 + 电磁) 的全部 dataclass 和设置。
#   集中于此消除模块间循环导入。
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    # 标量 DOF 映射
    "ScalarDOFMap",
    # 热场
    "ThermalSettings",
    "ThermalMaterial",
    "ThermalBC",
    "TemperatureBC",
    "HeatFluxBC",
    "ConvectionBC",
    "RadiationBC",
    "ThermalResult",
    # 流体场
    "FluidSettings",
    "FluidMaterial",
    "FluidBC",
    "VelocityBC",
    "PressureBC",
    "FluidResult",
    # 电磁场
    "EMSettings",
    "EMMaterial",
    "EMBC",
    "VoltageBC",
    "CurrentDensityBC",
    "EMResult",
    # 多物理场耦合
    "MultiphysicsSettings",
    "MultiphysicsResult",
    "CouplingStatus",
]


# ══════════════════════════════════════════════════════════
#  标量 DOF 映射 (温度 / 压力 / 电势共用)
# ══════════════════════════════════════════════════════════

@dataclass
class ScalarDOFMap:
    """标量场的 DOF 映射 (每个顶点 1 个自由度)

    用于温度 T、流体压力 p、电势 phi 等标量场。
    与 DOFMap (3 DOF/顶点, 用于位移) 互补。

    用法:
      >>> sdof = ScalarDOFMap.from_mesh(mesh)
      >>> sdof.n_dof  # == mesh.n_vertices
    """

    n_vertices: int
    n_dof: int  # == n_vertices

    @classmethod
    def from_mesh(cls, mesh) -> ScalarDOFMap:
        """从 PolyMesh 构建标量 DOF 映射"""
        return cls(n_vertices=mesh.n_vertices, n_dof=mesh.n_vertices)

    def vertex_dof(self, vertex_id: int) -> int:
        """顶点 ID → 标量 DOF 编号 (恒等映射)"""
        return vertex_id

    def element_dofs(self, vertex_ids: List[int]) -> np.ndarray:
        """控制点列表 → 标量 DOF 编号数组"""
        return np.array(vertex_ids, dtype=np.int64)

    def element_dof_count(self, n_control_points: int) -> int:
        """单元标量自由度数"""
        return n_control_points


# ══════════════════════════════════════════════════════════
#  热场类型
# ══════════════════════════════════════════════════════════

@dataclass
class ThermalSettings:
    """热传导求解设置

    控制方程: ρ·cp·∂T/∂t = ∇·(k∇T) + Q
    """

    solver: str = "direct"  # "direct" | "cg"
    steady_state: bool = True  # True = 稳态, False = 瞬态
    time_step: float = 0.01  # 瞬态时间步长 [s]
    n_steps: int = 100  # 瞬态总步数
    theta: float = 1.0  # 时间积分: 0=Euler显式, 0.5=CN, 1=Euler隐式
    quadrature_order: int = 3
    nonlinear: bool = False  # 温度相关热导率 k(T)
    radiation: bool = False  # 是否包含辐射边界
    tolerance: float = 1e-10
    max_iter: int = 500
    penalty: float = 1e12  # Dirichlet BC 罚因子
    verbosity: int = 0  # 0=静默, 1=进度, 2=详细


@dataclass
class ThermalMaterial:
    """热材料属性

    Args:
        conductivity:      热导率 k [W/(m·K)]
        specific_heat:     比热容 cp [J/(kg·K)]
        density:           密度 ρ [kg/m³]
        thermal_expansion: 热膨胀系数 α [1/K]
    """

    conductivity: float  # k
    specific_heat: float = 0.0  # cp (瞬态需要)
    density: float = 0.0  # ρ (瞬态需要)
    thermal_expansion: float = 0.0  # α

    @property
    def diffusivity(self) -> float:
        """热扩散率 α = k/(ρ·cp) [m²/s]"""
        if self.density > 0 and self.specific_heat > 0:
            return self.conductivity / (self.density * self.specific_heat)
        return 0.0


class ThermalBC:
    """热边界条件基类"""
    pass


@dataclass
class TemperatureBC(ThermalBC):
    """温度边界 (Dirichlet)

    用法:
      >>> bc = TemperatureBC(vertex_ids=[0, 1, 2], values=300.0, label="cold_wall")
      >>> bc = TemperatureBC.from_function(mesh, predicate=lambda v,x,y,z: z<0.01, value=373.0)
    """

    vertex_ids: np.ndarray  # (n,) 受约束顶点
    values: np.ndarray  # (n,) 温度值 [K]
    label: str = ""

    def __post_init__(self):
        self.vertex_ids = np.asarray(self.vertex_ids, dtype=np.int64)
        val = np.asarray(self.values)
        if val.ndim == 0:
            self.values = np.full(len(self.vertex_ids), float(val))
        else:
            self.values = val.astype(np.float64)

    @classmethod
    def from_function(cls, mesh, predicate, value=300.0, label=""):
        """根据几何谓词自动查找约束顶点"""
        vids = []
        vals = []
        for vi in range(mesh.n_vertices):
            x, y, z = mesh.vertices[vi]
            if predicate(vi, x, y, z):
                vids.append(vi)
                v = value(x, y, z) if callable(value) else value
                vals.append(float(v))
        return cls(
            vertex_ids=np.array(vids, dtype=np.int64),
            values=np.array(vals, dtype=np.float64),
            label=label,
        )

    @property
    def n_constrained(self) -> int:
        return len(self.vertex_ids)


@dataclass
class HeatFluxBC(ThermalBC):
    """热流边界 (Neumann)

    q = -k·∇T·n = q_prescribed [W/m²]
    正值 = 热流入, 负值 = 热流出
    """

    face_ids: List[int]  # 受载面
    flux: float  # 热流密度 [W/m²] (正=流入)
    label: str = ""

    def __post_init__(self):
        self.flux = float(self.flux)


@dataclass
class ConvectionBC(ThermalBC):
    """对流边界 (Robin)

    -k·∇T·n = h·(T - T_inf)
    """

    face_ids: List[int]  # 受载面
    h: float  # 对流换热系数 [W/(m²·K)]
    T_inf: float  # 环境温度 [K]
    label: str = ""


@dataclass
class RadiationBC(ThermalBC):
    """辐射边界 (非线性)

    -k·∇T·n = ε·σ·(T⁴ - T_inf⁴)
    其中 σ = 5.67e-8 W/(m²·K⁴)

    线性化: ε·σ·(T³+T²·T_inf+T·T_inf²+T_inf³)·(T-T_inf)
    """

    face_ids: List[int]
    emissivity: float  # ε ∈ [0, 1]
    T_inf: float  # 环境辐射温度 [K]
    label: str = ""

    @property
    def stefan_boltzmann(self) -> float:
        return 5.67e-8


@dataclass
class ThermalResult:
    """热分析结果"""

    temperature: np.ndarray  # (n_vertices,) 温度场 [K]
    heat_flux: Optional[np.ndarray] = None  # (n_faces, n_qp, 3) 热流矢量
    iterations: int = 0
    converged: bool = True
    wall_time: float = 0.0


# ══════════════════════════════════════════════════════════
#  流体场类型
# ══════════════════════════════════════════════════════════

@dataclass
class FluidSettings:
    """流体求解设置

    两种层次:
      - thin_film:  Reynolds 方程 (标量压力 p)
      - stokes:     Stokes 方程 (速度 u + 压力 p 混合格式)
    """

    formulation: str = "thin_film"  # "thin_film" | "stokes" | "navier_stokes"
    solver: str = "direct"  # "direct" | "cg" | "minres"
    quadrature_order: int = 3
    stabilized: bool = True  # SUPG/PSPG 稳定化 (对流占优)
    nonlinear: bool = False  # Navier-Stokes 对流项
    max_iter: int = 100
    tolerance: float = 1e-8
    penalty: float = 1e12
    # 薄膜流特有
    film_thickness: float = 1e-4  # 膜厚 h [m]
    sliding_velocity: float = 0.0  # 壁面滑移速度 U [m/s]
    verbosity: int = 0


@dataclass
class FluidMaterial:
    """流体材料属性"""

    viscosity: float  # 动力粘度 μ [Pa·s]
    density: float = 1000.0  # 密度 ρ [kg/m³]
    thermal_conductivity: float = 0.0  # 热导率 (共轭传热用)
    specific_heat: float = 0.0  # 比热容 (共轭传热用)

    @property
    def kinematic_viscosity(self) -> float:
        """运动粘度 ν = μ/ρ [m²/s]"""
        return self.viscosity / self.density if self.density > 0 else 0.0


class FluidBC:
    """流体边界条件基类"""
    pass


@dataclass
class VelocityBC(FluidBC):
    """速度边界 (Dirichlet for velocity)

    用法:
      >>> bc = VelocityBC(vertex_ids=[0,1,2], values=(0,0,0), label="no_slip")
    """

    vertex_ids: np.ndarray  # (n,) 受约束顶点
    values: np.ndarray  # (n, 3) 速度分量 [m/s]
    label: str = ""

    def __post_init__(self):
        self.vertex_ids = np.asarray(self.vertex_ids, dtype=np.int64)
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.ndim == 1:
            self.values = np.tile(self.values, (len(self.vertex_ids), 1))
        if self.values.shape[0] == 1:
            self.values = np.tile(self.values, (len(self.vertex_ids), 1))

    @classmethod
    def no_slip(cls, vertex_ids, label=""):
        """无滑移边界 (v = 0)"""
        vals = np.zeros((len(vertex_ids), 3))
        return cls(vertex_ids=np.array(vertex_ids), values=vals, label=label)


@dataclass
class PressureBC(FluidBC):
    """压力边界 (Neumann for pressure)"""

    face_ids: List[int]
    pressure: float  # 压力 [Pa]
    label: str = ""


@dataclass
class FluidResult:
    """流体求解结果"""

    velocity: Optional[np.ndarray] = None  # (n_vertices, 3) 速度 [m/s]
    pressure: Optional[np.ndarray] = None  # (n_vertices,) 压力 [Pa]
    iterations: int = 0
    converged: bool = True
    wall_time: float = 0.0


# ══════════════════════════════════════════════════════════
#  电磁场类型
# ══════════════════════════════════════════════════════════

@dataclass
class EMSettings:
    """电磁场求解设置"""

    formulation: str = "electrostatic"  # "electrostatic" | "magnetostatic"
    solver: str = "direct"  # "direct" | "cg"
    frequency: float = 0.0  # 谐波频率 [Hz] (0=静态)
    quadrature_order: int = 3
    tolerance: float = 1e-10
    max_iter: int = 500
    penalty: float = 1e12
    verbosity: int = 0


@dataclass
class EMMaterial:
    """电磁材料属性"""

    permittivity: float = 8.854e-12  # 介电常数 ε [F/m]
    permeability: float = 1.257e-6  # 磁导率 μ [H/m]
    conductivity: float = 0.0  # 电导率 σ [S/m]

    @classmethod
    def vacuum(cls):
        """真空电磁属性"""
        return cls(permittivity=8.854e-12, permeability=1.257e-6, conductivity=0.0)

    @classmethod
    def copper(cls):
        """铜的电磁属性"""
        return cls(
            permittivity=8.854e-12,
            permeability=1.257e-6,
            conductivity=5.8e7,
        )


class EMBC:
    """电磁边界条件基类"""
    pass


@dataclass
class VoltageBC(EMBC):
    """电压边界 (Dirichlet for φ)"""

    vertex_ids: np.ndarray
    values: np.ndarray  # (n,) 电势 [V]
    label: str = ""

    def __post_init__(self):
        self.vertex_ids = np.asarray(self.vertex_ids, dtype=np.int64)
        val = np.asarray(self.values)
        if val.ndim == 0:
            self.values = np.full(len(self.vertex_ids), float(val))
        else:
            self.values = val.astype(np.float64)

    @property
    def n_constrained(self) -> int:
        return len(self.vertex_ids)


@dataclass
class CurrentDensityBC(EMBC):
    """电流密度边界 (Neumann for φ)

    J·n = J_prescribed [A/m²]
    """

    face_ids: List[int]
    current_density: float  # [A/m²] (正=流入)
    label: str = ""


@dataclass
class EMResult:
    """电磁场求解结果"""

    potential: np.ndarray  # (n_vertices,) 电势 [V]
    e_field: Optional[np.ndarray] = None  # (n_vertices, 3) 电场 E = -∇φ [V/m]
    d_field: Optional[np.ndarray] = None  # (n_vertices, 3) 电位移 D = εE [C/m²]
    current_density: Optional[np.ndarray] = None  # (n_vertices, 3) J = σE [A/m²]
    joule_heat: Optional[np.ndarray] = None  # (n_vertices,) Q_joule = σ|E|² [W/m³]
    iterations: int = 0
    converged: bool = True
    wall_time: float = 0.0


# ══════════════════════════════════════════════════════════
#  多物理场耦合
# ══════════════════════════════════════════════════════════

@dataclass
class MultiphysicsSettings:
    """多物理场耦合全局设置

    控制耦合策略、收敛容差和各场间相互作用开关。
    """

    # 耦合策略
    coupling_scheme: str = "staggered"  # "staggered" | "monolithic"
    max_coupling_iter: int = 20
    coupling_tolerance: float = 1e-6

    # 松弛
    relaxation: str = "aitken"  # "none" | "fixed" | "aitken" | "anderson"
    relaxation_param: float = 0.5  # 固定松弛参数
    anderson_memory: int = 5  # Anderson 加速的记忆深度

    # 求解顺序 (块 Gauss-Seidel)
    coupling_order: Tuple[str, ...] = ("thermal", "fluid", "structural", "em")

    # 场间耦合开关
    thermal_structural: bool = True  # 热膨胀
    fluid_structural: bool = False  # 流固耦合 FSI
    thermal_fluid: bool = False  # 共轭传热
    em_thermal: bool = False  # 焦耳热
    em_structural: bool = False  # 洛伦兹力

    # 载荷步 (准静态递进)
    n_load_steps: int = 1
    load_step_factor: float = 1.0

    verbosity: int = 0


@dataclass
class CouplingStatus:
    """单次耦合迭代的状态"""

    iteration: int
    delta_temperature: float = 0.0
    delta_displacement: float = 0.0
    delta_velocity: float = 0.0
    delta_potential: float = 0.0
    relaxation_factor: float = 1.0
    converged: bool = False


@dataclass
class MultiphysicsResult:
    """多物理场耦合求解结果"""

    structural: Any = None  # StaticResult
    thermal: Optional[ThermalResult] = None
    fluid: Optional[FluidResult] = None
    em: Optional[EMResult] = None
    coupling_history: List[CouplingStatus] = field(default_factory=list)
    converged: bool = False
    total_coupling_iterations: int = 0
    wall_time: float = 0.0

    def summary(self) -> str:
        """生成结果摘要"""
        lines = ["MultiphysicsResult:"]
        if self.structural is not None:
            lines.append(
                f"  Structural: strain_energy={getattr(self.structural, 'strain_energy', 'N/A')}"
            )
        if self.thermal is not None:
            lines.append(
                f"  Thermal: max_T={self.thermal.temperature.max():.2f} K, "
                f"min_T={self.thermal.temperature.min():.2f} K"
            )
        if self.fluid is not None and self.fluid.pressure is not None:
            lines.append(
                f"  Fluid: max_p={self.fluid.pressure.max():.2f} Pa"
            )
        if self.em is not None:
            lines.append(
                f"  EM: max_phi={self.em.potential.max():.2f} V"
            )
        lines.append(f"  Converged: {self.converged}")
        lines.append(f"  Coupling iterations: {self.total_coupling_iterations}")
        lines.append(f"  Wall time: {self.wall_time:.2f}s")
        return "\n".join(lines)
