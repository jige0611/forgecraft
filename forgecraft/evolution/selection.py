"""
选择算子模块 (Selection Operators)

单目标:
  - tournament_select:      随机抽 k 个, 选适应度最高的
  - select_elites:          按 fitness 降序取前 N

多目标 (NSGA-II):
  - non_dominated_sort:     Pareto 前沿分层 O(M·N²)
  - crowding_distance:      同前沿拥挤距离 (多样性)
  - pareto_tournament:      按 Pareto rank + crowding 选择
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from forgecraft.core.morphology import MechanicalBody
import logging


# ══════════════════════════════════════════════════════════
# 选择算子 (Selection Operators)
#
# 单目标:
#   - tournament_select:      随机抽 k 个, 选适应度最高的 (选压=1/k)
#   - select_elites:          按 fitness 降序取前 N
#
# 多目标 (NSGA-II, Deb et al. 2002):
#   - _dominates:             Pareto 支配关系判定
#   - non_dominated_sort:     O(M·N²) 非支配排序 → 前沿分层
#   - crowding_distance:     同一前沿内的拥挤距离 (多样性保持)
#   - pareto_elites:          从 Pareto 前沿选精英, 不足时退化为适应度排序
#   - pareto_tournament_select: 按 Pareto rank 排序做 tournament
#
# 支配关系: a dominates b 当且仅当:
#   ∀k: a[k] >= b[k] (maximize) 或 a[k] <= b[k] (minimize)
#   且 ∃k: a[k] > b[k] (strict)
# ══════════════════════════════════════════════════════════

__all__ = [
    "tournament_select",
    "non_dominated_sort",
    "crowding_distance",
    "pareto_elites",
    "pareto_tournament_select",
    "select_elites",
    "compute_population_stats",
    "compute_pareto_stats",
]

_logger = logging.getLogger(__name__)



def tournament_select(
    population: List[MechanicalBody],
    num_select: int,
    tournament_size: int = 3,
    rng: Optional[np.random.RandomState] = None,
) -> List[MechanicalBody]:
    if rng is None:
        rng = np.random.RandomState()
    if len(population) <= num_select:
        return list(population)

    selected = []
    indices = list(range(len(population)))
    for _ in range(num_select):
        candidates = rng.choice(indices, size=min(tournament_size, len(indices)), replace=False)
        winner_idx = max(candidates, key=lambda i: population[i].fitness)
        selected.append(population[winner_idx])
    return selected


def _dominates(
    obj_a: np.ndarray,
    obj_b: np.ndarray,
    directions: List[str],
) -> bool:
    better = True
    at_least_one_strict = False
    for k, d in enumerate(directions):
        if d == "maximize":
            if obj_a[k] < obj_b[k]:
                better = False
            if obj_a[k] > obj_b[k]:
                at_least_one_strict = True
        else:
            if obj_a[k] > obj_b[k]:
                better = False
            if obj_a[k] < obj_b[k]:
                at_least_one_strict = True
    return better and at_least_one_strict


def non_dominated_sort(
    objective_matrix: np.ndarray,
    directions: List[str],
) -> List[List[int]]:
    n = objective_matrix.shape[0]
    domination_counts = np.zeros(n, dtype=int)
    dominated_by = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(objective_matrix[j], objective_matrix[i], directions):
                domination_counts[i] += 1
                dominated_by[j].append(i)

    fronts = []
    current_front = [i for i in range(n) if domination_counts[i] == 0]

    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_by[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def crowding_distance(
    objective_matrix: np.ndarray,
    front_indices: List[int],
    directions: List[str],
) -> np.ndarray:
    m = len(front_indices)
    if m <= 2:
        return np.full(m, float("inf"))

    distances = np.zeros(m)
    n_obj = objective_matrix.shape[1]

    for k in range(n_obj):
        values = objective_matrix[front_indices, k]
        sorted_order = np.argsort(values)
        if directions[k] == "minimize":
            sorted_order = sorted_order[::-1]
        sorted_values = values[sorted_order]

        val_range = sorted_values[-1] - sorted_values[0] if m > 1 else 1.0
        if abs(val_range) < 1e-12:
            continue

        distances[sorted_order[0]] = float("inf")
        distances[sorted_order[-1]] = float("inf")

        for i in range(1, m - 1):
            distances[sorted_order[i]] += (
                sorted_values[i + 1] - sorted_values[i - 1]
            ) / val_range

    return distances


def pareto_elites(
    population: List[MechanicalBody],
    n_elites: int,
    objective_keys: List[str],
    directions: List[str],
) -> List[MechanicalBody]:
    if n_elites <= 0 or not population:
        return []

    n = len(population)
    objective_matrix = np.array([
        [population[i].fitness_components.get(k, 0.0) for k in objective_keys]
        for i in range(n)
    ])

    fronts = non_dominated_sort(objective_matrix, directions)

    selected_indices = []
    for front in fronts:
        if len(selected_indices) + len(front) <= n_elites:
            selected_indices.extend(front)
        else:
            remaining = n_elites - len(selected_indices)
            cd = crowding_distance(objective_matrix, front, directions)
            sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
            selected_indices.extend(sorted_front[:remaining])
            break

    return [population[i] for i in selected_indices]


def pareto_tournament_select(
    population: List[MechanicalBody],
    num_select: int,
    objective_keys: List[str],
    directions: List[str],
    tournament_size: int = 3,
    rng: Optional[np.random.RandomState] = None,
) -> List[MechanicalBody]:
    if rng is None:
        rng = np.random.RandomState()
    if len(population) <= num_select:
        return list(population)

    n = len(population)
    objective_matrix = np.array([
        [population[i].fitness_components.get(k, 0.0) for k in objective_keys]
        for i in range(n)
    ])
    fronts = non_dominated_sort(objective_matrix, directions)

    rank = np.zeros(n, dtype=int)
    for fi, front in enumerate(fronts):
        for idx in front:
            rank[idx] = fi

    cd_all = np.zeros(n)
    for front in fronts:
        cd = crowding_distance(objective_matrix, front, directions)
        for i, idx in enumerate(front):
            cd_all[idx] = cd[i]

    indices = list(range(n))
    selected = []
    for _ in range(num_select):
        candidates = rng.choice(indices, size=min(tournament_size, len(indices)), replace=False)
        winner_idx = min(candidates, key=lambda i: (rank[i], -cd_all[i]))
        selected.append(population[winner_idx])

    return selected


def compute_pareto_stats(
    population: List[MechanicalBody],
    objective_keys: List[str],
    directions: List[str],
) -> Dict:
    if not population:
        return {}

    n = len(population)
    objective_matrix = np.array([
        [population[i].fitness_components.get(k, 0.0) for k in objective_keys]
        for i in range(n)
    ])
    fronts = non_dominated_sort(objective_matrix, directions)

    pareto_front_indices = fronts[0] if fronts else []
    pareto_front_fitnesses = [population[i].fitness for i in pareto_front_indices]

    fitnesses = [b.fitness for b in population]
    return {
        "best_fitness": max(fitnesses),
        "mean_fitness": np.mean(fitnesses),
        "median_fitness": np.median(fitnesses),
        "min_fitness": min(fitnesses),
        "std_fitness": np.std(fitnesses),
        "population_size": len(population),
        "n_fronts": len(fronts),
        "front_sizes": [len(f) for f in fronts],
        "pareto_front_size": len(pareto_front_indices),
        "pareto_best_fitness": max(pareto_front_fitnesses) if pareto_front_fitnesses else 0.0,
        "pareto_mean_fitness": np.mean(pareto_front_fitnesses) if pareto_front_fitnesses else 0.0,
    }


def select_elites(
    population: List[MechanicalBody],
    n_elites: int,
    objective_keys: Optional[List[str]] = None,
    directions: Optional[List[str]] = None,
) -> List[MechanicalBody]:
    if objective_keys and directions and len(objective_keys) > 1:
        return pareto_elites(population, n_elites, objective_keys, directions)
    sorted_pop = sorted(population, key=lambda b: b.fitness, reverse=True)
    return sorted_pop[:n_elites]


def compute_population_stats(
    population: List[MechanicalBody],
    objective_keys: Optional[List[str]] = None,
    directions: Optional[List[str]] = None,
) -> Dict:
    if objective_keys and directions and len(objective_keys) > 1:
        stats = compute_pareto_stats(population, objective_keys, directions)
        return stats

    if not population:
        return {}

    fitnesses = [b.fitness for b in population]
    return {
        "best_fitness": max(fitnesses),
        "mean_fitness": np.mean(fitnesses),
        "median_fitness": np.median(fitnesses),
        "min_fitness": min(fitnesses),
        "std_fitness": np.std(fitnesses),
        "population_size": len(population),
    }
