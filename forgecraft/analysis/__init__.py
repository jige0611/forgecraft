# ══════════════════════════════════════════════════════════
# forgecraft.analysis — 工程分析模块
#
# 八大子系统:
#   fea.py       — 有限元分析 (FEA): 应力/位移/安全系数/屈曲/疲劳
#   topology.py  — 拓扑优化 (SIMP): 最小柔度设计, 30-60% 减重
#   assembly.py  — 装配验证: 干涉检查/公差累积/螺栓校核/配合验证
#   iga_*.py     — IGA 等几何分析 (12 文件): 静力/模态/瞬态/自适应
#   iga_contact_*.py / iga_fracture_*.py — 接触+断裂 (5 文件): 碰撞/摩擦/相场/内聚力
#   iga_multiphysics_*.py — 多物理场耦合 (6 文件): 热/流/电磁 + 交错/整体耦合
#   iga_topology_*.py     — 相位场拓扑优化 (4 文件): IGA 原生, 无棋盘格, 自伴随灵敏度
#   iga_tpms_*.py         — 梯度 TPMS 晶格 (4 文件): 应力自适应, 8 种曲面, 迭代优化
# ══════════════════════════════════════════════════════════

from forgecraft.analysis.fea import (
    FEAEngine,
    FEAResult,
    fea_available,
    compute_structural_safety,
    estimate_mass,
)
from forgecraft.analysis.topology import (
    TopologyOptimizer,
    TopologyResult,
    topology_optimize,
)
from forgecraft.analysis.assembly import (
    AssemblyValidator,
    AssemblyReport,
    ToleranceAnalyzer,
    BoltChecker,
    validate_assembly,
)

# IGA 子模块
from forgecraft.analysis._iga_base import (
    IGASettings, DOFMap, QuadPoint, StiffnessCache,
    IrregularFaceCache, ShapeResult,
    StaticResult, ModalResult, TransientResult, StressField, IGAError,
)
from forgecraft.analysis.iga_material import (
    LinearIsotropic, LinearOrthotropic, NeoHookean, compliance_to_stiffness,
)
from forgecraft.analysis.iga_shape import (
    cc_shape_functions, cc_element_control_points, is_face_regular, compute_jacobian,
)
from forgecraft.analysis.iga_quadrature import (
    gauss_legendre_1d, gauss_legendre_2d, cc_quadrature, cc_quadrature_points,
)
from forgecraft.analysis.iga_element import (
    ElementIntegrator, MembraneIntegrator,
)
from forgecraft.analysis.iga_boundary import (
    DirichletBC, NeumannBC, PointLoad,
    apply_dirichlet_penalty, apply_dirichlet_elimination,
    apply_neumann_load, apply_point_load, apply_point_loads,
)
from forgecraft.analysis.iga_assembly import (
    StreamingCSRAssembler,
    assemble_stiffness, assemble_mass,
    assemble_stiffness_parallel, assemble_force_vector,
)
from forgecraft.analysis.iga_solver import (
    solve_static, solve_linear, jacobi_preconditioner, compute_reactions,
)
from forgecraft.analysis.iga_post import (
    compute_stress, compute_von_mises, compute_principal_stresses,
    export_vtk, export_vtk_displacement,
)
from forgecraft.analysis.iga_dynamic import (
    solve_modal, solve_transient, hrz_lumped_mass, rayleigh_damping,
)
from forgecraft.analysis.iga_adaptivity import (
    ZZErrorEstimator, h_adaptive_loop, compute_error_indicators, mark_elements,
)

# IGA 接触子模块
from forgecraft.analysis.iga_contact_types import (
    ContactSettings, ContactPair, GapResult, FrictionModel,
    CohesiveLaw, PhaseFieldSettings, CrackInfo,
)
from forgecraft.analysis.iga_contact_search import (
    AABBTree, AABBNode,
    find_closest_point, closest_point_newton, compute_gap,
    find_contact_pairs, build_contact_pairs,
)
from forgecraft.analysis.iga_contact_enforce import (
    assemble_contact_penalty, assemble_friction_contribution,
    solve_penalty_contact, solve_augmented_lagrange,
    contact_force_residual, compute_contact_energy,
)
from forgecraft.analysis.iga_fracture_cohesive import (
    CohesiveInterfaceIntegrator,
    cohesive_stiffness, cohesive_traction,
    insert_cohesive_elements, compute_effective_separation,
)
from forgecraft.analysis.iga_fracture_phasefield import (
    PhaseFieldSolver,
    solve_phasefield,
    assemble_phasefield_system,
    compute_elastic_energy_density as compute_elastic_energy_field,
    assemble_degraded_stiffness,
    extract_crack_geometry,
)
from forgecraft.analysis.iga_contact_assembly import (
    assemble_with_contact_fracture,
    ContactFractureSolver,
    solve_contact_fracture,
    assemble_total_system,
)

# IGA 多物理场子模块
from forgecraft.analysis.iga_multiphysics_types import (
    ScalarDOFMap,
    ThermalSettings, ThermalMaterial,
    TemperatureBC, HeatFluxBC, ConvectionBC, RadiationBC, ThermalResult,
    FluidSettings, FluidMaterial,
    VelocityBC, PressureBC, FluidResult,
    EMSettings, EMMaterial,
    VoltageBC, CurrentDensityBC, EMResult,
    MultiphysicsSettings, MultiphysicsResult, CouplingStatus,
)
from forgecraft.analysis.iga_thermal import (
    assemble_thermal_conductivity, assemble_thermal_capacity,
    assemble_thermal_load, apply_temperature_bc_penalty,
    solve_thermal_steady, solve_thermal_transient,
    compute_thermal_strain, compute_thermal_expansion_force,
)
from forgecraft.analysis.iga_fluid import (
    assemble_reynolds_stiffness, assemble_reynolds_load,
    solve_thin_film,
    assemble_stokes_viscous, assemble_stokes_pressure_gradient,
    assemble_stokes_divergence, solve_stokes,
    compute_fsi_force,
)
from forgecraft.analysis.iga_electromagnetic import (
    assemble_electrostatic_stiffness, assemble_electrostatic_load,
    apply_voltage_bc_penalty, solve_electrostatic,
    assemble_magnetostatic_stiffness, solve_magnetostatic,
    compute_e_field, compute_joule_heat, compute_lorentz_force,
)
from forgecraft.analysis.iga_multiphysics_coupling import (
    MultiphysicsCoupler, solve_multiphysics,
    aitken_relaxation, anderson_acceleration,
)
from forgecraft.analysis.iga_multiphysics_assembly import (
    MultiphysicsSolver,
    assemble_monolithic_system, solve_monolithic,
)

# IGA 相位场拓扑优化子模块
from forgecraft.analysis.iga_topology_types import (
    InterpolationFunction,
    g_simp as g_simp_interp, dg_simp as dg_simp_interp,
    g_ramp as g_ramp_interp, dg_ramp as dg_ramp_interp,
    g_polynomial as g_polynomial_interp, dg_polynomial as dg_polynomial_interp,
    TopologySettings, TopologyMaterial,
    TopologyConstraint, ConstraintType,
    TopologyOptimizationResult, TopologyIterationData,
)
from forgecraft.analysis.iga_topology_phasefield import (
    assemble_phasefield_laplacian,
    assemble_phasefield_stiffness,
    assemble_phasefield_driving_force,
    assemble_degraded_stiffness_topology as degrade_for_topology,
    compute_strain_energy_density as compute_strain_energy_topo,
    solve_phasefield_field,
    compute_volume_fraction,
    apply_density_filter,
)
from forgecraft.analysis.iga_topology_sensitivity import (
    compute_compliance_sensitivity,
    compute_volume_sensitivity,
    compute_stress_constraint_sensitivity,
    compute_displacement_constraint_sensitivity,
    adjoint_solve,
    filter_sensitivity,
)
from forgecraft.analysis.iga_topology_optimizer import (
    TopologyOptimizer as PhaseFieldTopologyOptimizer,
    topology_optimize as topology_phasefield_optimize,
)

# IGA 梯度 TPMS 晶格子模块
from forgecraft.analysis.iga_tpms_types import (
    TPMSType, GradingStrategy,
    TPMSParameters, TPMSGradingSettings, TPMSLatticeResult,
    TPMS_SURFACE_FUNCTIONS, TPMS_DEFAULTS,
)
from forgecraft.analysis.iga_tpms_core import (
    tpms_level_set, tpms_level_set_gradient,
    tpms_is_solid, tpms_is_solid_graded,
    compute_relative_density, compute_tpms_volume_fraction,
    effective_youngs_modulus, effective_shear_modulus,
    cell_size_to_frequency, frequency_to_cell_size,
)
from forgecraft.analysis.iga_tpms_grading import (
    compute_von_mises_field, compute_stress_driving_field,
    map_stress_to_cell_size, map_stress_to_thickness,
    smooth_grading_field, normalize_grading_for_volume,
    compute_tpms_parameters_field, estimate_average_density,
)
from forgecraft.analysis.iga_tpms_generator import (
    TPMSLatticeGenerator,
    tpms_lattice_generate, tpms_lattice_iterative,
    marching_cubes_tpms, sample_tpms_on_grid,
)

__all__ = [
    # FEA
    "FEAEngine",
    "FEAResult",
    "fea_available",
    "compute_structural_safety",
    "estimate_mass",
    # Topology
    "TopologyOptimizer",
    "TopologyResult",
    "topology_optimize",
    # Assembly
    "AssemblyValidator",
    "AssemblyReport",
    "ToleranceAnalyzer",
    "BoltChecker",
    "validate_assembly",
    # IGA Base
    "IGASettings", "DOFMap", "QuadPoint", "StiffnessCache",
    "IrregularFaceCache", "ShapeResult",
    "StaticResult", "ModalResult", "TransientResult", "StressField", "IGAError",
    # IGA Material
    "LinearIsotropic", "LinearOrthotropic", "NeoHookean", "compliance_to_stiffness",
    # IGA Shape
    "cc_shape_functions", "cc_element_control_points", "is_face_regular", "compute_jacobian",
    # IGA Quadrature
    "gauss_legendre_1d", "gauss_legendre_2d", "cc_quadrature", "cc_quadrature_points",
    # IGA Element
    "ElementIntegrator", "MembraneIntegrator",
    # IGA Boundary
    "DirichletBC", "NeumannBC", "PointLoad",
    "apply_dirichlet_penalty", "apply_dirichlet_elimination",
    "apply_neumann_load", "apply_point_load", "apply_point_loads",
    # IGA Assembly
    "StreamingCSRAssembler",
    "assemble_stiffness", "assemble_mass",
    "assemble_stiffness_parallel", "assemble_force_vector",
    # IGA Solver
    "solve_static", "solve_linear", "jacobi_preconditioner", "compute_reactions",
    # IGA Post
    "compute_stress", "compute_von_mises", "compute_principal_stresses",
    "export_vtk", "export_vtk_displacement",
    # IGA Dynamic
    "solve_modal", "solve_transient", "hrz_lumped_mass", "rayleigh_damping",
    # IGA Adaptivity
    "ZZErrorEstimator", "h_adaptive_loop", "compute_error_indicators", "mark_elements",
    # IGA Contact Types
    "ContactSettings", "ContactPair", "GapResult", "FrictionModel",
    "CohesiveLaw", "PhaseFieldSettings", "CrackInfo",
    # IGA Contact Search
    "AABBTree", "AABBNode",
    "find_closest_point", "closest_point_newton", "compute_gap",
    "find_contact_pairs", "build_contact_pairs",
    # IGA Contact Enforce
    "assemble_contact_penalty", "assemble_friction_contribution",
    "solve_penalty_contact", "solve_augmented_lagrange",
    "contact_force_residual", "compute_contact_energy",
    # IGA Fracture Cohesive
    "CohesiveInterfaceIntegrator",
    "cohesive_stiffness", "cohesive_traction",
    "insert_cohesive_elements", "compute_effective_separation",
    # IGA Fracture Phasefield
    "PhaseFieldSolver",
    "solve_phasefield",
    "assemble_phasefield_system",
    "compute_elastic_energy_field",
    "assemble_degraded_stiffness",
    "extract_crack_geometry",
    # IGA Contact Assembly
    "assemble_with_contact_fracture",
    "ContactFractureSolver",
    "solve_contact_fracture",
    "assemble_total_system",
    # --- 多物理场 ---
    # Multi-physics Types
    "ScalarDOFMap",
    "ThermalSettings", "ThermalMaterial",
    "TemperatureBC", "HeatFluxBC", "ConvectionBC", "RadiationBC",
    "ThermalResult",
    "FluidSettings", "FluidMaterial",
    "VelocityBC", "PressureBC",
    "FluidResult",
    "EMSettings", "EMMaterial",
    "VoltageBC", "CurrentDensityBC",
    "EMResult",
    "MultiphysicsSettings", "MultiphysicsResult", "CouplingStatus",
    # Multi-physics Thermal
    "assemble_thermal_conductivity", "assemble_thermal_capacity",
    "assemble_thermal_load", "apply_temperature_bc_penalty",
    "solve_thermal_steady", "solve_thermal_transient",
    "compute_thermal_strain", "compute_thermal_expansion_force",
    # Multi-physics Fluid
    "assemble_reynolds_stiffness", "assemble_reynolds_load",
    "solve_thin_film",
    "assemble_stokes_viscous", "assemble_stokes_pressure_gradient",
    "assemble_stokes_divergence", "solve_stokes",
    "compute_fsi_force",
    # Multi-physics EM
    "assemble_electrostatic_stiffness", "assemble_electrostatic_load",
    "apply_voltage_bc_penalty", "solve_electrostatic",
    "assemble_magnetostatic_stiffness", "solve_magnetostatic",
    "compute_e_field", "compute_joule_heat", "compute_lorentz_force",
    # Multi-physics Coupling
    "MultiphysicsCoupler", "solve_multiphysics",
    "aitken_relaxation", "anderson_acceleration",
    # Multi-physics Assembly
    "MultiphysicsSolver",
    "assemble_monolithic_system", "solve_monolithic",
    # --- 相位场拓扑优化 ---
    # Topology Types
    "InterpolationFunction",
    "TopologySettings", "TopologyMaterial",
    "TopologyConstraint", "ConstraintType",
    "TopologyOptimizationResult", "TopologyIterationData",
    # Topology Phase-Field
    "assemble_phasefield_laplacian",
    "assemble_phasefield_stiffness",
    "assemble_phasefield_driving_force",
    "degrade_for_topology",
    "compute_strain_energy_topo",
    "solve_phasefield_field",
    "compute_volume_fraction",
    "apply_density_filter",
    # Topology Sensitivity
    "compute_compliance_sensitivity",
    "compute_volume_sensitivity",
    "compute_stress_constraint_sensitivity",
    "compute_displacement_constraint_sensitivity",
    "adjoint_solve",
    "filter_sensitivity",
    # Topology Optimizer
    "PhaseFieldTopologyOptimizer",
    "topology_phasefield_optimize",
    # --- 梯度 TPMS 晶格 ---
    # TPMS Types
    "TPMSType", "GradingStrategy",
    "TPMSParameters", "TPMSGradingSettings", "TPMSLatticeResult",
    "TPMS_SURFACE_FUNCTIONS", "TPMS_DEFAULTS",
    # TPMS Core
    "tpms_level_set", "tpms_level_set_gradient",
    "tpms_is_solid", "tpms_is_solid_graded",
    "compute_relative_density", "compute_tpms_volume_fraction",
    "effective_youngs_modulus", "effective_shear_modulus",
    "cell_size_to_frequency", "frequency_to_cell_size",
    # TPMS Grading
    "compute_von_mises_field", "compute_stress_driving_field",
    "map_stress_to_cell_size", "map_stress_to_thickness",
    "smooth_grading_field", "normalize_grading_for_volume",
    "compute_tpms_parameters_field", "estimate_average_density",
    # TPMS Generator
    "TPMSLatticeGenerator",
    "tpms_lattice_generate", "tpms_lattice_iterative",
    "marching_cubes_tpms", "sample_tpms_on_grid",
]
