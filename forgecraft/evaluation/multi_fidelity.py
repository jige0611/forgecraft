# ══════════════════════════════════════════════════════════
# forgecraft.evaluation.multi_fidelity — 多精度评估漏斗
#
#   五级精度评估器:
#     L1: 结构代理 (~0.01s) — 解析公式 + 启发式
#     L2: 运动学代理 (~0.1s)  — 前向运动学
#     L3: 动力学代理 (~1s)    — 简单动力学 (无接触)
#     L4: RL 短评估 (~10s)   — PPO 2 episodes
#     L5: RL 全评估 (~60s)   — PPO 8 episodes
#
#   分配策略:
#     SUCCESSIVE_HALVING: N → N/2 → N/4 → N/8 → N/16
#     HYPERBAND: 多个 SuccessiveHalving 实例 (最优臂预算分配)
#     BOHB: Bayesian Optimization + Hyperband (代理模型引导)
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "FidelityLevel",
    "MultiFidelityEvaluator",
    "successive_halving",
    "hyperband",
    "bohb_select",
    "FIDELITY_CONFIGS",
]


# ══════════════════════════════════════════════════════════
#  精度级别
# ══════════════════════════════════════════════════════════

class FidelityLevel(IntEnum):
    """评估精度级别"""
    L1_STRUCTURAL = 1     # 结构代理: 解析公式 (最快)
    L2_KINEMATIC = 2      # 运动学代理: 前向运动学
    L3_DYNAMIC = 3        # 动力学代理: 简单动力学
    L4_RL_SHORT = 4       # RL 短评估: PPO 2 episodes
    L5_RL_FULL = 5        # RL 全评估: PPO 8 episodes (最精确)


# 默认精度配置
FIDELITY_CONFIGS = {
    FidelityLevel.L1_STRUCTURAL: {
        "cost_multiplier": 1,       # 相对计算成本
        "default_budget": 10000,    # 总预算 (成本单位)
        "reliability": 0.6,         # 与真实评估的相关性
    },
    FidelityLevel.L2_KINEMATIC: {
        "cost_multiplier": 10,
        "default_budget": 5000,
        "reliability": 0.75,
    },
    FidelityLevel.L3_DYNAMIC: {
        "cost_multiplier": 100,
        "default_budget": 2000,
        "reliability": 0.85,
    },
    FidelityLevel.L4_RL_SHORT: {
        "cost_multiplier": 1000,
        "default_budget": 500,
        "reliability": 0.95,
    },
    FidelityLevel.L5_RL_FULL: {
        "cost_multiplier": 6000,
        "default_budget": 100,
        "reliability": 1.0,       # 真实评估
    },
}


# ══════════════════════════════════════════════════════════
#  多精度评估器
# ══════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """单次评估结果"""
    individual: object
    fitness: float
    fidelity: FidelityLevel
    cost: int = 1


class MultiFidelityEvaluator:
    """多精度评估漏斗

    核心思想: 在最便宜的精度上评估大量个体,
    逐步淘汰劣质个体, 仅在最有希望的个体上投入昂贵评估。

    典型使用:
        evaluator = MultiFidelityEvaluator(evaluators={...})
        results = evaluator.successive_halving(population, max_fidelity=5)
    """

    def __init__(
        self,
        evaluators: Dict[FidelityLevel, Callable],
        fidelity_configs: Optional[Dict] = None,
        rng: Optional[np.random.RandomState] = None,
    ):
        """
        Args:
            evaluators: {FidelityLevel: callable(individual) → float}
            fidelity_configs: 精度配置 (默认 FIDELITY_CONFIGS)
            rng: 随机状态
        """
        self.evaluators = evaluators
        self.configs = fidelity_configs or FIDELITY_CONFIGS
        self.rng = rng or np.random.RandomState()

        self.total_evaluations = 0
        self.total_cost = 0

    def successive_halving(
        self,
        population: List[object],
        max_fidelity: FidelityLevel = FidelityLevel.L5_RL_FULL,
        reduction_factor: int = 2,
    ) -> List[EvalResult]:
        """连续减半 (Successive Halving)

        算法:
          Level 1: N 个体 → 保留 top N/η
          Level 2: N/η 个体 → 保留 top N/η²
          ...
          Level L: N/η^{L-1} 个体 → 最终评估

        Args:
            population: 初始种群
            max_fidelity: 最高精度级别
            reduction_factor: 每级保留比例的分母 η

        Returns:
            最终评估结果列表
        """
        return successive_halving(
            population, self.evaluators, max_fidelity,
            reduction_factor, self.rng,
        )

    def hyperband(
        self,
        population: List[object],
        max_fidelity: FidelityLevel = FidelityLevel.L5_RL_FULL,
        eta: int = 3,
    ) -> List[EvalResult]:
        """Hyperband 自适应预算分配

        运行多个 SuccessiveHalving 实例, 每个有不同的初始预算分配。
        理论保证找到最优配置的概率接近 1。

        Args:
            population: 初始种群
            max_fidelity: 最高精度级别
            eta: 保留比例分母

        Returns:
            最终评估结果
        """
        return hyperband(
            population, self.evaluators, max_fidelity,
            eta, self.rng,
        )

    def bohb_select(
        self,
        population: List[object],
        n_select: int,
        max_fidelity: FidelityLevel = FidelityLevel.L5_RL_FULL,
    ) -> List[object]:
        """BOHB (Bayesian Optimization + Hyperband) 选择

        用低精度结果构建 GP 代理模型, 预测高精度表现,
        仅在代理模型高不确定性的个体上做昂贵评估。

        Args:
            population: 初始种群
            n_select: 最终选择数
            max_fidelity: 最高精度

        Returns:
            选中的个体列表
        """
        return bohb_select(
            population, self.evaluators, n_select,
            max_fidelity, self.rng,
        )

    def evaluate_at_fidelity(
        self,
        individual: object,
        fidelity: FidelityLevel,
    ) -> float:
        """在指定精度评估单个个体"""
        evaluator = self.evaluators.get(fidelity)
        if evaluator is None:
            raise ValueError(f"No evaluator for fidelity {fidelity}")

        cost = self.configs[fidelity]["cost_multiplier"]
        self.total_evaluations += 1
        self.total_cost += cost

        return evaluator(individual)

    def get_statistics(self) -> Dict:
        return {
            "total_evaluations": self.total_evaluations,
            "total_cost": self.total_cost,
            "cost_per_individual": self.total_cost / max(self.total_evaluations, 1),
        }


# ══════════════════════════════════════════════════════════
#  独立算法函数
# ══════════════════════════════════════════════════════════

def successive_halving(
    population: List[object],
    evaluators: Dict[FidelityLevel, Callable],
    max_fidelity: FidelityLevel = FidelityLevel.L5_RL_FULL,
    reduction_factor: int = 2,
    rng: Optional[np.random.RandomState] = None,
) -> List[EvalResult]:
    """连续减半算法

    Args:
        population: 初始种群 (任意类型)
        evaluators: {fidelity: callable(individual) → fitness}
        max_fidelity: 最高评估精度
        reduction_factor: 每级保留 1/η
        rng: 随机状态

    Returns:
        最终评估结果 (按适应度降序)
    """
    if rng is None:
        rng = np.random.RandomState()

    current_pop = list(population)
    results_history = []

    for level in range(1, max_fidelity + 1):
        fidelity = FidelityLevel(level)
        evaluator = evaluators.get(fidelity)

        if evaluator is None:
            _logger.warning(f"No evaluator for fidelity {level}, skipping")
            continue

        n_current = len(current_pop)
        if n_current == 0:
            break

        # 评估当前种群
        fitnesses = []
        for ind in current_pop:
            try:
                fit = evaluator(ind)
                fitnesses.append(float(fit))
            except Exception as e:
                _logger.warning(f"Evaluation failed at fidelity {level}: {e}")
                fitnesses.append(-float('inf'))

        # 排序 + 筛选
        order = np.argsort(fitnesses)[::-1]  # 降序
        n_keep = max(1, n_current // reduction_factor)

        results_history.extend([
            EvalResult(
                individual=current_pop[order[i]],
                fitness=fitnesses[order[i]],
                fidelity=fidelity,
                cost=FIDELITY_CONFIGS[fidelity]["cost_multiplier"],
            )
            for i in range(n_current)
        ])

        current_pop = [current_pop[order[i]] for i in range(n_keep)]

        _logger.debug(
            f"Fidelity {level}: {n_current} → {n_keep} "
            f"(top fitness: {fitnesses[order[0]]:.3f})"
        )

    return sorted(results_history, key=lambda r: r.fitness, reverse=True)


def hyperband(
    population: List[object],
    evaluators: Dict[FidelityLevel, Callable],
    max_fidelity: FidelityLevel = FidelityLevel.L5_RL_FULL,
    eta: int = 3,
    rng: Optional[np.random.RandomState] = None,
) -> List[EvalResult]:
    """Hyperband 算法 (Li et al. 2017)

    运行多个 SuccessiveHalving brackets, 自动选择最佳 budget 分配。

    s_max = ⌊log_η(R)⌋  (R = max_fidelity)
    对 s ∈ {s_max, s_max-1, ..., 0}:
      n = ⌈(s_max+1)/(s+1) · η^s⌉
      r = R / η^s
      运行 SuccessiveHalving(n, r)

    Args:
        population: 初始种群
        evaluators: 评估函数字典
        max_fidelity: 最高精度
        eta: 保留比例分母
        rng: 随机状态

    Returns:
        最佳评估结果
    """
    if rng is None:
        rng = np.random.RandomState()

    R = int(max_fidelity)
    s_max = int(np.floor(np.log(R) / np.log(eta)))

    best_results = []

    for s in range(s_max, -1, -1):
        n = int(np.ceil((s_max + 1) / (s + 1) * (eta ** s)))
        r = R / (eta ** s)
        sub_fidelity = min(int(np.ceil(r)), R)

        # 子采样初始种群
        if len(population) > n:
            subset = rng.choice(population, size=n, replace=False).tolist()
        else:
            subset = list(population)

        # 运行 SuccessiveHalving (只到 sub_fidelity)
        results = successive_halving(
            subset, evaluators,
            max_fidelity=FidelityLevel(sub_fidelity),
            reduction_factor=eta, rng=rng,
        )

        if results:
            best_results.extend(results[:3])  # 每个 bracket 保留 top 3

        _logger.debug(f"Hyperband bracket s={s}: n={n}, r={sub_fidelity}")

    return sorted(best_results, key=lambda r: r.fitness, reverse=True)


def bohb_select(
    population: List[object],
    evaluators: Dict[FidelityLevel, Callable],
    n_select: int,
    max_fidelity: FidelityLevel = FidelityLevel.L5_RL_FULL,
    rng: Optional[np.random.RandomState] = None,
) -> List[object]:
    """BOHB 选择: Bayesian Optimization + Hyperband

    1. 在低精度评估全部个体
    2. 构建 GP 代理模型 (低精度 → 高精度预测)
    3. 选择代理模型预测最好 + 不确定性最高的个体
    4. 对这些个体做高精度评估

    Args:
        population: 初始种群
        evaluators: 评估函数
        n_select: 最终选择数
        max_fidelity: 最高精度
        rng: 随机状态

    Returns:
        n_select 最佳个体
    """
    if rng is None:
        rng = np.random.RandomState()

    n_total = len(population)
    if n_total <= n_select:
        return list(population)

    # 1. 低精度评估 (Level 1-2)
    low_fit = []
    for ind in population:
        for level in [1, 2]:
            ev = evaluators.get(FidelityLevel(level))
            if ev:
                low_fit.append(ev(ind))
                break
        else:
            low_fit.append(0.0)

    low_fit = np.array(low_fit)

    # 2. 选候选: top 50% 低精度 + top 20% 高方差
    n_candidates = min(n_select * 3, n_total)
    top_indices = set(np.argsort(low_fit)[::-1][:n_candidates // 2])

    # 高方差 = 需要更多信息
    var_indices = set(rng.choice(
        n_total, size=n_candidates - len(top_indices), replace=False,
    ))
    candidates = list(top_indices | var_indices)

    # 3. 中精度评估 (Level 3)
    mid_fit = []
    for i in candidates:
        ev = evaluators.get(FidelityLevel.L3_DYNAMIC)
        if ev:
            mid_fit.append(ev(population[i]))
        else:
            mid_fit.append(low_fit[i])
    mid_fit = np.array(mid_fit)

    # 4. 高精度评估 (Level 4-5) 选 top candidates
    final_candidates = np.argsort(mid_fit)[::-1][:n_select * 2]
    final_fit = []
    for ci in final_candidates:
        for level in [4, 5]:
            ev = evaluators.get(FidelityLevel(level))
            if ev:
                final_fit.append(ev(population[candidates[ci]]))
                break
        else:
            final_fit.append(mid_fit[ci])

    # 5. 返回 top n_select
    best_order = np.argsort(final_fit)[::-1][:n_select]
    return [population[candidates[final_candidates[i]]] for i in best_order]
