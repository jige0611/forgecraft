# ══════════════════════════════════════════════════════════
# forgecraft.core — 核心模块
# 形态生成、材料数据库、碰撞检测、关节优化
# ══════════════════════════════════════════════════════════

from forgecraft.core.morphology import MechanicalBody, Part, Joint

from forgecraft.core.materials_database import (
    MaterialDB,
    MaterialProperties,
    get_material_db,
    get_part_material,
    get_part_density,
    calculate_part_mass,
)

from forgecraft.core.collision_checker import (
    CollisionChecker,
    CollisionResult,
    get_collision_checker,
    is_body_valid,
    filter_valid_bodies,
)

from forgecraft.core.joint_optimizer import (
    JointOptimizer,
    JointOptimizationResult,
    get_joint_optimizer,
    optimize_body_joints,
)

from forgecraft.core.generator import (
    BodyGenerator,
)

from forgecraft.core.parallel_generator import (
    ParallelGeometryGenerator,
    SynchronousGenerator,
    create_generator,
)

from forgecraft.core.smart_cache import (
    SmartCacheManager,
    GPUMemoryManager,
)

from forgecraft.core.plugin_system import (
    Plugin,
    PluginManager,
    ConfigManager,
    DependencyInjector,
)

from forgecraft.core.flexible_components import (
    SpringElement,
    SpringProperties,
    SpringLibrary,
    DamperElement,
    DamperProperties,
    TendonSystem,
    TendonSegment,
    FlexibleConnector,
    SoftBodySimulator,
)

from forgecraft.core.validation import (
    validate_positive,
    validate_range,
    validate_probability,
    validate_int_positive,
    validate_not_empty,
    validate_evolution_config,
    validate_rl_config,
    validate_sim_config,
    clamp_float,
    safe_divide,
    safe_mean,
)

__all__ = [
    # Morphology
    "MechanicalBody", "Part", "Joint",
    # Materials
    "MaterialDB", "MaterialProperties",
    "get_material_db", "get_part_material",
    "get_part_density", "calculate_part_mass",
    # Collision
    "CollisionChecker", "CollisionResult",
    "get_collision_checker", "is_body_valid", "filter_valid_bodies",
    # Joint Optimization
    "JointOptimizer", "JointOptimizationResult",
    "get_joint_optimizer", "optimize_body_joints",
    # Generation
    "BodyGenerator",
    "ParallelGeometryGenerator", "SynchronousGenerator",
    "create_generator",
    # Smart Cache
    "SmartCacheManager", "GPUMemoryManager",
    # Plugin System
    "Plugin", "PluginManager", "ConfigManager", "DependencyInjector",
    # Flexible Components
    "SpringElement", "SpringProperties", "SpringLibrary",
    "DamperElement", "DamperProperties",
    "TendonSystem", "TendonSegment",
    "FlexibleConnector", "SoftBodySimulator",
    # Validation
    "validate_positive", "validate_range", "validate_probability",
    "validate_int_positive", "validate_not_empty",
    "validate_evolution_config", "validate_rl_config", "validate_sim_config",
    "clamp_float", "safe_divide", "safe_mean",
]
