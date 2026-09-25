# ══════════════════════════════════════════════════════════
# 🔧 输入验证与防御性编程工具
#
# 用于确保跨模块调用的类型安全与合法性检查。
# 所有公共入口函数应使用这些验证器。
#
# 设计原则:
#   ✅ Fail fast — 在边界尽早检测非法输入
#   ✅ Clear messages — 错误信息包含参数名和期望值
#   ✅ Zero cost when valid — 生产环境可禁用 (VALIDATE_INPUTS=False)
#   ✅ Type-safe — 完整类型注解
# ══════════════════════════════════════════════════════════

import os
from typing import Any, Dict, List, Optional, Sequence, Type, Union
import logging

import numpy as np

_logger = logging.getLogger(__name__)

# 全局开关：生产环境设为 False 可跳过验证
VALIDATE_INPUTS = os.environ.get("FORGECRAFT_VALIDATE", "1") == "1"


# ══════════════════════════════════════════════════════════
# 基本类型验证
# ══════════════════════════════════════════════════════════

def validate_positive(value: float, name: str = "value") -> None:
    """验证值为正数"""
    if VALIDATE_INPUTS and value <= 0:
        raise ValueError(f"{name} 必须 > 0，当前: {value}")


def validate_non_negative(value: float, name: str = "value") -> None:
    """验证值为非负数"""
    if VALIDATE_INPUTS and value < 0:
        raise ValueError(f"{name} 必须 >= 0，当前: {value}")


def validate_range(value: float, lo: float, hi: float, name: str = "value") -> None:
    """验证值在闭区间内"""
    if VALIDATE_INPUTS and not (lo <= value <= hi):
        raise ValueError(
            f"{name} 必须在 [{lo}, {hi}] 范围内，当前: {value}"
        )


def validate_probability(value: float, name: str = "probability") -> None:
    """验证概率值在 [0, 1]"""
    validate_range(value, 0.0, 1.0, name)


def validate_int_positive(value: int, name: str = "value") -> None:
    """验证整数为正"""
    if VALIDATE_INPUTS and (not isinstance(value, int) or value <= 0):
        raise ValueError(f"{name} 必须为正整数，当前: {value}")


def validate_int_range(value: int, lo: int, hi: int, name: str = "value") -> None:
    """验证整数在闭区间内"""
    if VALIDATE_INPUTS and (not isinstance(value, int) or not (lo <= value <= hi)):
        raise ValueError(
            f"{name} 必须是 [{lo}, {hi}] 范围内的整数，当前: {value}"
        )


# ══════════════════════════════════════════════════════════
# 容器验证
# ══════════════════════════════════════════════════════════

def validate_not_empty(seq: Sequence, name: str = "sequence") -> None:
    """验证序列非空"""
    if VALIDATE_INPUTS and len(seq) == 0:
        raise ValueError(f"{name} 不能为空")


def validate_all_positive(seq: Sequence[float], name: str = "sequence") -> None:
    """验证序列所有元素为正"""
    if VALIDATE_INPUTS:
        for i, v in enumerate(seq):
            if v <= 0:
                raise ValueError(f"{name}[{i}] 必须 > 0，当前: {v}")


def validate_same_length(a: Sequence, b: Sequence,
                         name_a: str = "a", name_b: str = "b") -> None:
    """验证两个序列长度相等"""
    if VALIDATE_INPUTS and len(a) != len(b):
        raise ValueError(
            f"{name_a} 和 {name_b} 长度必须相同，"
            f"当前: len({name_a})={len(a)}, len({name_b})={len(b)}"
        )


def validate_dict_keys(d: Dict, required: List[str], name: str = "dict") -> None:
    """验证字典包含必需键"""
    if VALIDATE_INPUTS:
        missing = [k for k in required if k not in d]
        if missing:
            raise KeyError(f"{name} 缺少必需键: {missing}")


# ══════════════════════════════════════════════════════════
# 类型断言
# ══════════════════════════════════════════════════════════

def check_type(value: Any, expected_type: Type, name: str = "value") -> None:
    """运行时类型检查"""
    if VALIDATE_INPUTS and not isinstance(value, expected_type):
        raise TypeError(
            f"{name} 类型应为 {expected_type.__name__}，"
            f"实际: {type(value).__name__}"
        )


def check_instance_any(value: Any, allowed_types: List[Type], name: str = "value") -> None:
    """运行时多类型检查"""
    if VALIDATE_INPUTS and not any(isinstance(value, t) for t in allowed_types):
        names = [t.__name__ for t in allowed_types]
        raise TypeError(
            f"{name} 类型应为 {' 或 '.join(names)}，"
            f"实际: {type(value).__name__}"
        )


# ══════════════════════════════════════════════════════════
# NumPy 专用验证
# ══════════════════════════════════════════════════════════

def validate_ndarray(arr: np.ndarray, ndim: int = None,
                     shape: tuple = None, name: str = "array") -> None:
    """验证 NumPy 数组维度和形状"""
    if not VALIDATE_INPUTS:
        return

    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} 必须是 np.ndarray，实际: {type(arr).__name__}")

    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} 维度应为 {ndim}D，实际: {arr.ndim}D")

    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} 形状应为 {shape}，实际: {arr.shape}")


def validate_finite(arr: np.ndarray, name: str = "array") -> None:
    """验证数组无 NaN/Inf"""
    if VALIDATE_INPUTS and not np.all(np.isfinite(arr)):
        n_nan = np.sum(np.isnan(arr))
        n_inf = np.sum(np.isinf(arr))
        raise ValueError(f"{name} 包含 {n_nan} 个 NaN，{n_inf} 个 Inf")


# ══════════════════════════════════════════════════════════
# 配置验证
# ══════════════════════════════════════════════════════════

def validate_evolution_config(config) -> None:
    """验证进化配置的合法性"""
    if not VALIDATE_INPUTS:
        return

    validate_int_positive(config.population_size, "population_size")
    validate_int_positive(config.elite_count, "elite_count")
    validate_range(config.mutation_rate, 0.0, 1.0, "mutation_rate")
    validate_range(config.crossover_rate, 0.0, 1.0, "crossover_rate")
    validate_range(config.selection_pressure, 1.0, 10.0, "selection_pressure")

    if config.elite_count >= config.population_size:
        raise ValueError(
            f"elite_count ({config.elite_count}) 必须 < "
            f"population_size ({config.population_size})"
        )


def validate_rl_config(config) -> None:
    """验证 RL 配置的合法性"""
    if not VALIDATE_INPUTS:
        return

    validate_positive(config.actor_lr, "actor_lr")
    validate_positive(config.critic_lr, "critic_lr")
    validate_range(config.gamma, 0.0, 1.0, "gamma")
    validate_range(config.lam, 0.0, 1.0, "lam")
    validate_range(config.clip_ratio, 0.0, 1.0, "clip_ratio")
    validate_positive(config.entropy_coef, "entropy_coef")
    validate_int_positive(config.ppo_epochs, "ppo_epochs")
    validate_int_positive(config.batch_size, "batch_size")
    validate_int_positive(config.hidden_dim, "hidden_dim")

    if config.actor_lr > 1.0 or config.critic_lr > 1.0:
        raise ValueError(
            f"学习率过高: actor={config.actor_lr}, critic={config.critic_lr}"
        )


def validate_sim_config(config) -> None:
    """验证仿真配置的合法性"""
    if not VALIDATE_INPUTS:
        return

    validate_positive(config.timestep, "timestep")
    validate_range(config.timestep, 0.0001, 0.1, "timestep")
    validate_range(config.friction, 0.0, 2.0, "friction")
    validate_int_positive(config.substeps, "substeps")

    if config.timestep > 0.05:
        raise ValueError(f"timestep 过大 ({config.timestep}s)，仿真可能不稳定")


# ══════════════════════════════════════════════════════════
# 边界值裁剪
# ══════════════════════════════════════════════════════════

def clamp_float(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """安全裁剪浮点数"""
    return float(np.clip(value, lo, hi))


def clamp_int(value: int, lo: int, hi: int) -> int:
    """安全裁剪整数"""
    return int(np.clip(value, lo, hi))


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法 (除零返回 default)"""
    if b == 0.0 or not np.isfinite(b):
        return default
    result = a / b
    if not np.isfinite(result):
        return default
    return float(result)


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    """安全求均值 (空序列返回 default)"""
    if not values:
        return default
    return float(np.mean(values))


# ══════════════════════════════════════════════════════════
# 类型别名 (方便跨模块使用)
# ══════════════════════════════════════════════════════════

# 机械体相关
BodyList = List  # List[MechanicalBody]
BodyDict = Dict[str, Any]

# 配置相关
ConfigDict = Dict[str, Any]
ParamDict = Dict[str, Union[float, int, str, bool]]

# 评估相关
FitnessDict = Dict[str, float]
ResultDict = Dict[int, Dict[str, Any]]

# 仿真相关
Observation = np.ndarray  # shape: (obs_dim,)
Action = np.ndarray       # shape: (act_dim,)


__all__ = [
    # Basic
    "validate_positive", "validate_non_negative",
    "validate_range", "validate_probability",
    "validate_int_positive", "validate_int_range",
    # Containers
    "validate_not_empty", "validate_all_positive",
    "validate_same_length", "validate_dict_keys",
    # Types
    "check_type", "check_instance_any",
    # NumPy
    "validate_ndarray", "validate_finite",
    # Config
    "validate_evolution_config", "validate_rl_config",
    "validate_sim_config",
    # Clamp
    "clamp_float", "clamp_int",
    "safe_divide", "safe_mean",
    # Aliases
    "BodyList", "BodyDict", "ConfigDict", "ParamDict",
    "FitnessDict", "ResultDict", "Observation", "Action",
    # Flag
    "VALIDATE_INPUTS",
]
