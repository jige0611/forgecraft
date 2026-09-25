# ══════════════════════════════════════════════════════════
# forgecraft.simulation — 物理仿真模块
# MuJoCo 场景构建、地形课程学习、多后端支持
# ══════════════════════════════════════════════════════════

from forgecraft.simulation.builder import build_mjcf_model, build_mjcf_with_convex_collision
from forgecraft.simulation.terrain_curriculum import (
    TerrainStage,
    get_terrain_stage,
    generate_terrain_mjcf,
    setup_terrain_curriculum,
)
from forgecraft.simulation.genesis_backend import (
    GenesisSimBackend,
    GenesisNotAvailableError,
    genesis_available,
    require_genesis,
)
from forgecraft.simulation.backend_registry import (
    BackendRegistry,
    get_backend_status,
    create_simulation_backend,
)

__all__ = [
    "build_mjcf_model",
    "TerrainStage",
    "get_terrain_stage",
    "generate_terrain_mjcf",
    "setup_terrain_curriculum",
    # 多后端
    "GenesisSimBackend",
    "GenesisNotAvailableError",
    "genesis_available",
    "require_genesis",
    "BackendRegistry",
    "get_backend_status",
    "create_simulation_backend",
]
