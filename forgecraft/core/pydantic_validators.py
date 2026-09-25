"""
Pydantic 验证层 — 替换手写 validation.py

将 config.py 的 dataclass 转换为 Pydantic BaseModel:
  - 运行时类型检查 (Field)
  - 范围/正数/概率约束 (Field gt/ge/le constraints)
  - 自定义 __post_init__ → model_validator
  - 保持与原 dataclass 接口兼容 (model_dump() → dict)

用法:
    from forgecraft.core.pydantic_validators import SimModel, EvolutionModel
    
    config = EvolutionModel(population_size=50)  # 自动验证
    config.model_dump()  # → dict 传给原 dataclass
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ══════════════════════════════════════════════════════════
# SimConfig
# ══════════════════════════════════════════════════════════

class SimModel(BaseModel):
    """物理仿真配置 — Pydantic 验证"""
    gravity: float = Field(default=-9.81, description="重力加速度 (m/s²)")
    timestep: float = Field(default=0.005, gt=0, le=0.1, description="仿真步长 (s)")
    friction: float = Field(default=0.6, ge=0, le=2.0, description="全局摩擦系数")
    max_steps: int = Field(default=500, ge=1, le=10000, description="最大仿真步数")
    substeps: int = Field(default=10, ge=1, le=100, description="MuJoCo 内部子步数")


# ══════════════════════════════════════════════════════════
# RLConfig
# ══════════════════════════════════════════════════════════

class RLModel(BaseModel):
    """强化学习配置 — Pydantic 验证"""
    # 网络
    hidden_dim: int = Field(default=128, ge=8, le=2048)
    gnn_layers: int = Field(default=3, ge=1, le=12)
    gnn_hidden: int = Field(default=64, ge=8, le=512)
    morph_embed_dim: int = Field(default=64, ge=8, le=512)

    # 优化器
    actor_lr: float = Field(default=3e-4, gt=0, le=0.1)
    critic_lr: float = Field(default=1e-3, gt=0, le=0.1)
    lr_decay: bool = True
    lr_final_factor: float = Field(default=0.05, gt=0, le=1.0)

    # PPO
    gamma: float = Field(default=0.99, ge=0.8, le=0.999)
    lam: float = Field(default=0.95, ge=0.5, le=1.0)
    clip_ratio: float = Field(default=0.2, ge=0.01, le=0.5)
    target_kl: float = Field(default=0.03, gt=0, le=1.0)
    entropy_coef: float = Field(default=0.02, ge=0, le=1.0)
    train_iters: int = Field(default=100, ge=1, le=10000)
    batch_size: int = Field(default=256, ge=1, le=65536)
    ppo_epochs: int = Field(default=12, ge=1, le=100)
    value_loss_coef: float = Field(default=0.5, ge=0, le=10.0)
    max_grad_norm: float = Field(default=0.5, gt=0, le=100.0)
    adaptive_entropy: bool = True
    entropy_min: float = Field(default=0.008, ge=0, le=1.0)
    entropy_max: float = Field(default=0.05, ge=0, le=1.0)

    @model_validator(mode="after")
    def check_entropy_range(self):
        if self.entropy_min >= self.entropy_max:
            raise ValueError(f"entropy_min ({self.entropy_min}) must be < entropy_max ({self.entropy_max})")
        return self


# ══════════════════════════════════════════════════════════
# EvolutionConfig
# ══════════════════════════════════════════════════════════

class EvolutionModel(BaseModel):
    """进化算法配置 — Pydantic 验证"""
    population_size: int = Field(default=50, ge=2, le=10000)
    generations: int = Field(default=200, ge=1, le=100000)
    elite_count: int = Field(default=8, ge=0)
    mutation_rate: float = Field(default=0.35, ge=0, le=1.0)
    crossover_rate: float = Field(default=0.6, ge=0, le=1.0)
    topo_mutation_prob: float = Field(default=0.18, ge=0, le=1.0)
    param_mutation_prob: float = Field(default=0.35, ge=0, le=1.0)
    param_mutation_scale: float = Field(default=0.12, gt=0, le=1.0)
    selection_pressure: float = Field(default=2.5, ge=1.0, le=20.0)

    @model_validator(mode="after")
    def check_elites_vs_population(self):
        if self.elite_count >= self.population_size:
            raise ValueError(
                f"elite_count ({self.elite_count}) must be < population_size ({self.population_size})"
            )
        return self


# ══════════════════════════════════════════════════════════
# PartSpec
# ══════════════════════════════════════════════════════════

class PartSpecModel(BaseModel):
    """零件规格 — Pydantic 验证"""
    part_type: str = Field(min_length=1, max_length=64)
    mass: float = Field(gt=0, le=100.0)
    shape: str = Field(min_length=1, max_length=32)
    size_range: List[float] = Field(min_length=1, max_length=2)

    can_actuate: bool = False
    max_torque: float = Field(default=0.0, ge=0)
    color: List[float] = Field(default_factory=lambda: [0.6, 0.6, 0.6, 1.0])
    param_ranges: Dict[str, List[float]] = Field(default_factory=dict)
    mass_range: List[float] = Field(default_factory=list)
    density: float = Field(default=1000.0, gt=0)
    has_touch: bool = False
    has_imu: bool = False
    friction: List[float] = Field(default_factory=lambda: [0.6, 0.01, 0.01])
    joint_type: str = "hinge"
    joint_axis: List[float] = Field(default_factory=lambda: [0.0, 1.0, 0.0])
    joint_range_min: float = Field(default=-1.5, ge=-10.0, le=10.0)
    joint_range_max: float = Field(default=1.5, ge=-10.0, le=10.0)
    joint_damping: float = Field(default=0.5, ge=0, le=100.0)

    @field_validator("size_range")
    @classmethod
    def check_size_range_ordered(cls, v: List[float]):
        if len(v) == 2 and v[0] > v[1]:
            raise ValueError(f"size_range must be [min, max], got {v}")
        return v

    @field_validator("color")
    @classmethod
    def check_color_rgba(cls, v: List[float]):
        if len(v) != 4 or any(c < 0 or c > 1 for c in v):
            raise ValueError(f"color must be RGBA in [0,1], got {v}")
        return v

    @field_validator("joint_axis")
    @classmethod
    def check_joint_axis_length(cls, v: List[float]):
        if len(v) != 3:
            raise ValueError(f"joint_axis must be 3D, got len={len(v)}")
        return v


# ══════════════════════════════════════════════════════════
# RewardComponent
# ══════════════════════════════════════════════════════════

class RewardComponentModel(BaseModel):
    """奖励分量 — Pydantic 验证"""
    name: str = Field(min_length=1, max_length=64)
    weight: float = Field(default=1.0, ge=-100.0, le=100.0)
    expression: str = ""
    description: str = ""


# ══════════════════════════════════════════════════════════
# TaskConfig
# ══════════════════════════════════════════════════════════

class TaskModel(BaseModel):
    """任务配置 — Pydantic 验证"""
    reward_components: List[RewardComponentModel] = Field(default_factory=list)
    fitness_formula: str = ""
    fitness_components: List[str] = Field(default_factory=list)
    max_episode_steps: int = Field(default=500, ge=1, le=100000)
    fall_height: float = Field(default=0.05, ge=0)
    terrain: str = "flat"
    domain: str = "robot"
    terminal_fall_penalty: float = Field(default=-2.0, le=0)
    terminal_survival_bonus: float = Field(default=1.0, ge=0)


# ══════════════════════════════════════════════════════════
# 工具函数 — 从 Pydantic model 回退到 dataclass
# ══════════════════════════════════════════════════════════

def to_sim_config(model: SimModel, SimConfig):
    """SimModel → 原 SimConfig dataclass"""
    return SimConfig(
        gravity=model.gravity, timestep=model.timestep,
        friction=model.friction, max_steps=model.max_steps,
        substeps=model.substeps,
    )


def to_rl_config(model: RLModel, RLConfig):
    """RLModel → 原 RLConfig dataclass"""
    return RLConfig(
        hidden_dim=model.hidden_dim, gnn_layers=model.gnn_layers,
        gnn_hidden=model.gnn_hidden, morph_embed_dim=model.morph_embed_dim,
        actor_lr=model.actor_lr, critic_lr=model.critic_lr,
        lr_decay=model.lr_decay, lr_final_factor=model.lr_final_factor,
        gamma=model.gamma, lam=model.lam, clip_ratio=model.clip_ratio,
        target_kl=model.target_kl, entropy_coef=model.entropy_coef,
        train_iters=model.train_iters, batch_size=model.batch_size,
        ppo_epochs=model.ppo_epochs, value_loss_coef=model.value_loss_coef,
        max_grad_norm=model.max_grad_norm,
        adaptive_entropy=model.adaptive_entropy,
        entropy_min=model.entropy_min, entropy_max=model.entropy_max,
    )


def to_evolution_config(model: EvolutionModel, EvolutionConfig):
    """EvolutionModel → 原 EvolutionConfig dataclass"""
    return EvolutionConfig(
        population_size=model.population_size, generations=model.generations,
        elite_count=model.elite_count, mutation_rate=model.mutation_rate,
        crossover_rate=model.crossover_rate,
        topo_mutation_prob=model.topo_mutation_prob,
        param_mutation_prob=model.param_mutation_prob,
        param_mutation_scale=model.param_mutation_scale,
        selection_pressure=model.selection_pressure,
    )


def to_part_spec(model: PartSpecModel, PartSpec):
    """PartSpecModel → 原 PartSpec dataclass"""
    return PartSpec(
        part_type=model.part_type, mass=model.mass, shape=model.shape,
        size_range=model.size_range, can_actuate=model.can_actuate,
        max_torque=model.max_torque, color=model.color,
        param_ranges=model.param_ranges, mass_range=model.mass_range,
        density=model.density, has_touch=model.has_touch, has_imu=model.has_imu,
        friction=model.friction, joint_type=model.joint_type,
        joint_axis=model.joint_axis,
        joint_range_min=model.joint_range_min,
        joint_range_max=model.joint_range_max,
        joint_damping=model.joint_damping,
    )
