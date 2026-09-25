"""ForgeCraft 配置数据类

所有可调参数集中定义，支持类型安全 + 运行时验证。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "SimConfig",
    "RLConfig",
    "EvolutionConfig",
    "PartSpec",
    "RewardComponent",
    "TaskConfig",
]


@dataclass(slots=True)
class SimConfig:
    """物理仿真配置"""
    gravity: float = -9.81      # 重力加速度 (m/s²)
    timestep: float = 0.005     # 仿真步长 (s)
    friction: float = 0.6       # 全局摩擦系数
    max_steps: int = 500        # 最大仿真步数
    substeps: int = 10          # MuJoCo 内部子步数

    def __post_init__(self):
        from forgecraft.core.validation import validate_sim_config
        validate_sim_config(self)


@dataclass
class RLConfig:
    """强化学习配置"""
    # 网络
    hidden_dim: int = 128       # 隐藏层维度
    gnn_layers: int = 3         # GNN 层数
    gnn_hidden: int = 64        # GNN 隐藏维度
    morph_embed_dim: int = 64   # 形态嵌入维度

    # 优化器
    actor_lr: float = 3e-4      # Actor 学习率
    critic_lr: float = 1e-3     # Critic 学习率
    lr_decay: bool = True       # 是否使用学习率衰减
    lr_final_factor: float = 0.05  # 衰减最终比例

    # PPO
    gamma: float = 0.99         # 折扣因子
    lam: float = 0.95           # GAE lambda
    clip_ratio: float = 0.2     # PPO clip 比率
    target_kl: float = 0.03     # 目标 KL 散度
    entropy_coef: float = 0.02  # 熵系数
    train_iters: int = 100      # 训练迭代次数
    batch_size: int = 256       # 批大小
    ppo_epochs: int = 12        # PPO epoch 数
    value_loss_coef: float = 0.5  # 价值损失系数
    max_grad_norm: float = 0.5  # 梯度裁剪阈值
    adaptive_entropy: bool = True  # 自适应熵
    entropy_min: float = 0.008  # 熵下限
    entropy_max: float = 0.05   # 熵上限

    def __post_init__(self):
        from forgecraft.core.validation import validate_rl_config
        validate_rl_config(self)


@dataclass
class EvolutionConfig:
    """进化算法配置"""
    population_size: int = 50      # 种群规模
    generations: int = 200         # 进化代数
    elite_count: int = 8           # 精英保留数
    mutation_rate: float = 0.35    # 变异率
    crossover_rate: float = 0.6    # 交叉率
    topo_mutation_prob: float = 0.18   # 拓扑变异概率
    param_mutation_prob: float = 0.35  # 参数变异概率
    param_mutation_scale: float = 0.12 # 参数变异幅度
    selection_pressure: float = 2.5    # 选择压力 (1=均匀, 越高越精英)

    def __post_init__(self):
        from forgecraft.core.validation import validate_evolution_config
        validate_evolution_config(self)


@dataclass
class PartSpec:
    """零件规格"""
    part_type: str          # 零件类型标识
    mass: float             # 质量 (kg)
    shape: str              # 形状 ("box", "cylinder", "sphere")
    size_range: List[float] # 尺寸范围 [min, max]

    # 可选属性
    can_actuate: bool = False
    max_torque: float = 0.0
    color: List[float] = field(default_factory=lambda: [0.6, 0.6, 0.6, 1.0])
    param_ranges: Dict[str, List[float]] = field(default_factory=dict)
    mass_range: List[float] = field(default_factory=lambda: [])
    density: float = 1000.0
    has_touch: bool = False
    has_imu: bool = False
    friction: List[float] = field(default_factory=lambda: [0.6, 0.01, 0.01])
    joint_type: str = "hinge"
    joint_axis: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    joint_range_min: float = -1.5
    joint_range_max: float = 1.5
    joint_damping: float = 0.5

    def get_range(self, key: str, default_min: float = 0.01, default_max: float = 0.3) -> List[float]:
        """获取参数范围"""
        if key in self.param_ranges:
            return self.param_ranges[key]
        return [self.size_range[0] if self.size_range else default_min,
                self.size_range[1] if len(self.size_range) > 1 else default_max]

    def clamp_param(self, key: str, value: float) -> float:
        """安全裁剪参数值到合法范围"""
        lo, hi = self.get_range(key)
        return max(lo, min(hi, value))


@dataclass
class RewardComponent:
    """奖励分量"""
    name: str
    weight: float = 1.0
    expression: str = ""
    description: str = ""


@dataclass(slots=True)
class TaskConfig:
    """任务配置"""
    reward_components: List[RewardComponent] = field(default_factory=list)
    fitness_formula: str = ""
    fitness_components: List[str] = field(default_factory=list)
    max_episode_steps: int = 500
    fall_height: float = 0.05
    terrain: str = "flat"
    domain: str = "robot"
    terminal_fall_penalty: float = -2.0
    terminal_survival_bonus: float = 1.0

    def get_weight(self, name: str) -> float:
        """获取指定奖励分量的权重"""
        for c in self.reward_components:
            if c.name == name:
                return c.weight
        return 0.0

    def has_component(self, name: str) -> bool:
        """检查是否包含指定奖励分量"""
        return any(c.name == name for c in self.reward_components)

    def to_legacy(self):
        """转换为旧版配置格式 (兼容性)"""
        return type('LegacyTaskConfig', (), {
            'speed_weight': self.get_weight('displacement'),
            'energy_weight': abs(self.get_weight('energy')),
            'upright_weight': self.get_weight('upright'),
            'alive_bonus': self.get_weight('alive'),
            'max_episode_steps': self.max_episode_steps,
        })()
