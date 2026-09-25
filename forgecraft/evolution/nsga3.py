# ══════════════════════════════════════════════════════════
# forgecraft.evolution.nsga3 — NSGA-III 选择器
#
#   Deb & Jain (2014): "An Evolutionary Many-Objective Optimization
#   Algorithm Using Reference-Point-Based Nondominated Sorting Approach"
#
#   NSGA-II 用拥挤距离在 2-3 目标下有效, 但 5+ 目标时几乎所有解
#   都非支配 → 拥挤距离退化为随机选择。
#   NSGA-III 用预生成参考点替代拥挤距离, 保证高维多样性。
#
#   算法:
#     1. 非支配排序 (同 NSGA-II)
#     2. 自适应归一化 (理想点 + ASF 极值点 + 超平面截距)
#     3. 关联: 每个解关联到最近参考点
#     4. 小生境保存: 从小生境计数最少的参考点开始填充
# ══════════════════════════════════════════════════════════

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from forgecraft.core.morphology import MechanicalBody
from forgecraft.evolution.selection import (
    non_dominated_sort, crowding_distance, _dominates,
)
import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "NSGA3Selector",
    "generate_reference_points",
    "normalize_objectives",
    "associate_to_reference_points",
    "niching_select",
    "nsga3_select",
    "nsga3_pareto_elites",
]


# ══════════════════════════════════════════════════════════
#  参考点生成 (Das & Dennis)
# ══════════════════════════════════════════════════════════

def generate_reference_points(n_objectives: int, n_divisions: int) -> np.ndarray:
    """Das & Dennis 系统性参考点生成

    在单位单纯形上生成参考点:
      Σ r_i = 1,  r_i ∈ {0, 1/H, 2/H, ..., 1}

    点数: C(n_objectives + n_divisions - 1, n_divisions)

    Args:
        n_objectives: 目标数 M
        n_divisions: 每维度划分数 H

    Returns:
        (N_ref, M) 参考点 (归一化超平面上)

    Examples:
        >>> pts = generate_reference_points(3, 4)   # C(6,4)=15 点
        >>> pts = generate_reference_points(5, 6)   # C(10,6)=210 点
    """
    def _recursive_gen(obj_idx, left, current):
        if obj_idx == n_objectives - 1:
            current.append(left / n_divisions)
            points.append(current[:])
            current.pop()
        else:
            for i in range(left + 1):
                current.append(i / n_divisions)
                _recursive_gen(obj_idx + 1, left - i, current)
                current.pop()

    points = []
    _recursive_gen(0, n_divisions, [])
    return np.array(points)  # (N_ref, M)


# ══════════════════════════════════════════════════════════
#  自适应归一化
# ══════════════════════════════════════════════════════════

def normalize_objectives(
    F: np.ndarray,
    directions: List[str],
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NSGA-III 自适应归一化

    步骤:
      1. 理想点 z_min = 逐目标最小值
      2. 极值点: 用 ASF 在每根轴上找极值
      3. 超平面截距: M 个极值点确定的超平面截距
      4. 归一化: f'_i = (f_i - z_min_i) / a_i

    Args:
        F: (N, M) 目标矩阵
        directions: 每目标方向 ("maximize"/"minimize")
        epsilon: 防止除零

    Returns:
        (F_normalized, ideal_point, intercepts)
    """
    N, M = F.shape
    F_sign = F.copy()
    # 统一为 minimize 方向
    for k in range(M):
        if directions[k] == "maximize":
            F_sign[:, k] = -F_sign[:, k]

    # 理想点
    ideal_point = np.min(F_sign, axis=0)

    # 平移
    F_shifted = F_sign - ideal_point

    # ASF 权重 (扫描每个轴方向)
    extreme_points = np.zeros((M, M))
    for k in range(M):
        w = np.full(M, epsilon)
        w[k] = 1.0
        # ASF = max_i (f_i / w_i) for each solution
        asf_vals = np.max(F_shifted / (w + epsilon), axis=1)
        best_idx = np.argmin(asf_vals)
        extreme_points[k] = F_shifted[best_idx]

    # 超平面截距
    try:
        # 解线性方程组: extreme_points @ a_inv = 1
        intercepts = np.linalg.solve(extreme_points, np.ones(M))
        # 如果解出负截距, 回退到每目标最大值
        if np.any(intercepts < epsilon):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts = np.max(F_shifted, axis=0)

    # 归一化
    a = np.maximum(intercepts, epsilon)
    F_norm = F_shifted / a

    return F_norm, ideal_point, intercepts


# ══════════════════════════════════════════════════════════
#  关联到参考点
# ══════════════════════════════════════════════════════════

def associate_to_reference_points(
    F_norm: np.ndarray,
    reference_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """将每个解关联到最近参考点

    关联标准: 到参考方向线的垂直距离 (非欧氏距离)

    距离 d⊥(s, r) = ‖s - (s·r/‖r‖²)·r‖

    Args:
        F_norm: (N, M) 归一化目标值
        reference_points: (R, M) 参考点

    Returns:
        (association, distances): 各 (N,) - 最近参考点索引 + 垂直距离
    """
    N = F_norm.shape[0]
    R = reference_points.shape[0]

    association = np.zeros(N, dtype=int)
    distances = np.zeros(N)

    # 预计算参考点范数
    ref_norm_sq = np.sum(reference_points ** 2, axis=1)  # (R,)

    for i in range(N):
        s = F_norm[i]
        # 到每个参考方向的投影距离
        for r_idx in range(R):
            r = reference_points[r_idx]
            rns = ref_norm_sq[r_idx]
            if rns < 1e-15:
                d = np.linalg.norm(s)
            else:
                proj = np.dot(s, r) / rns
                proj = np.clip(proj, 0, None)  # 只考虑正向投影
                d = np.linalg.norm(s - proj * r)
            if r_idx == 0 or d < distances[i]:
                distances[i] = d
                association[i] = r_idx

    return association, distances


# ══════════════════════════════════════════════════════════
#  小生境保存
# ══════════════════════════════════════════════════════════

def niching_select(
    front_indices: List[int],
    n_select: int,
    association: np.ndarray,
    niche_counts: np.ndarray,
    distances: np.ndarray,
    rng: Optional[np.random.RandomState] = None,
) -> List[int]:
    """NSGA-III 小生境保存

    从临界前沿 F_l 中选择 n_select 个个体,
    优先选小生境计数最少的参考点关联的个体。

    Args:
        front_indices: 临界前沿中的个体索引
        n_select: 还需选择的个体数
        association: 所有个体的参考点关联
        niche_counts: 每个参考点的已选计数
        distances: 关联垂直距离
        rng: 随机状态

    Returns:
        选中个体索引列表
    """
    if rng is None:
        rng = np.random.RandomState()

    selected = []
    # 该前沿中个体的参考点关联
    front_assoc = association[front_indices]
    front_dist = distances[front_indices]

    remaining = set(range(len(front_indices)))

    for _ in range(n_select):
        if not remaining:
            break

        rem_list = list(remaining)
        # 找小生境计数最少的参考点
        min_count = niche_counts[front_assoc[rem_list]].min()
        candidate_refs = set()
        for j in rem_list:
            if niche_counts[front_assoc[j]] == min_count:
                candidate_refs.add(front_assoc[j])

        # 在候选参考点中找最近个体
        candidates = [j for j in rem_list if front_assoc[j] in candidate_refs]
        if candidates:
            best_j = candidates[np.argmin(front_dist[candidates])]
        else:
            # 随机选
            best_j = rng.choice(rem_list)

        selected.append(front_indices[best_j])
        remaining.remove(best_j)
        niche_counts[front_assoc[best_j]] += 1

    return selected


# ══════════════════════════════════════════════════════════
#  NSGA-III 选择器类
# ══════════════════════════════════════════════════════════

class NSGA3Selector:
    """NSGA-III 选择器

    用途: 在 3+ 目标优化中维持 Pareto 前沿多样性。
    不适于单目标或双目标优化 (此时用 NSGA-II 拥挤距离即可)。

    使用:
        selector = NSGA3Selector(n_objectives=5, n_divisions=6)
        next_pop = selector.select(parents + offspring, n_select=100, directions=[...])
    """

    def __init__(
        self,
        n_objectives: int,
        n_divisions: int = 6,
        random_seed: Optional[int] = None,
    ):
        """初始化 NSGA-III 选择器

        Args:
            n_objectives: 目标数 M
            n_divisions: 每维度划分数 H
            random_seed: 随机种子
        """
        self.n_objectives = n_objectives
        self.n_divisions = n_divisions
        self.rng = np.random.RandomState(random_seed)

        # 生成参考点
        self.reference_points = generate_reference_points(n_objectives, n_divisions)
        self.n_ref = len(self.reference_points)

        _logger.info(
            f"NSGA-III: M={n_objectives}, H={n_divisions}, "
            f"N_ref={self.n_ref}"
        )

    def select(
        self,
        population: List[MechanicalBody],
        n_select: int,
        objective_keys: List[str],
        directions: List[str],
        return_indices: bool = False,
    ) -> List[MechanicalBody]:
        """NSGA-III 环境选择

        Args:
            population: 父代 + 子代 (2N)
            n_select: 选择数量 N
            objective_keys: 适应度组件键
            directions: 每目标方向
            return_indices: 返回索引而非对象

        Returns:
            选中的 N 个个体
        """
        return nsga3_select(
            population, n_select, objective_keys, directions,
            self.reference_points, self.rng, return_indices,
        )

    def select_pareto_elites(
        self,
        population: List[MechanicalBody],
        n_elites: int,
        objective_keys: List[str],
        directions: List[str],
    ) -> List[MechanicalBody]:
        """从种群中选 NSGA-III 精英"""
        return nsga3_pareto_elites(
            population, n_elites, objective_keys, directions,
            self.reference_points, self.rng,
        )

    def get_reference_point_count(self) -> int:
        return self.n_ref

    def set_divisions(self, n_divisions: int):
        """动态调整参考点密度"""
        self.n_divisions = n_divisions
        self.reference_points = generate_reference_points(
            self.n_objectives, n_divisions,
        )
        self.n_ref = len(self.reference_points)


# ══════════════════════════════════════════════════════════
#  独立函数 (不依赖类)
# ══════════════════════════════════════════════════════════

def nsga3_select(
    population: List[MechanicalBody],
    n_select: int,
    objective_keys: List[str],
    directions: List[str],
    reference_points: Optional[np.ndarray] = None,
    rng: Optional[np.random.RandomState] = None,
    return_indices: bool = False,
) -> List[MechanicalBody]:
    """NSGA-III 环境选择

    Args:
        population: 合并种群 (P ∪ Q)
        n_select: 保留数量
        objective_keys: 目标键
        directions: 每目标方向
        reference_points: 参考点 (None=自动生成)
        rng: 随机状态
        return_indices: 返回索引而非对象

    Returns:
        选中的个体列表
    """
    if rng is None:
        rng = np.random.RandomState()

    n_total = len(population)
    n_obj = len(objective_keys)

    if n_total <= n_select:
        return list(range(n_total)) if return_indices else list(population)

    # 生成参考点
    if reference_points is None:
        n_div = max(4, int(math.pow(1000, 1.0 / (n_obj - 1))) if n_obj > 1 else 12)
        reference_points = generate_reference_points(n_obj, n_div)

    # 目标矩阵
    F = np.array([
        [population[i].fitness_components.get(k, 0.0) for k in objective_keys]
        for i in range(n_total)
    ])

    # 1. 非支配排序
    fronts = non_dominated_sort(F, directions)
    fronts_flat = [idx for front in fronts for idx in front]

    # 2. 找到临界前沿 F_l
    selected = []
    l = 0
    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected.extend(front)
            l += 1
        else:
            break

    if len(selected) >= n_select:
        if return_indices:
            return selected[:n_select]
        return [population[i] for i in selected[:n_select]]

    n_remaining = n_select - len(selected)

    # 3. 归一化
    F_norm, _, _ = normalize_objectives(F, directions)

    # 4. 关联: 仅对已选中的个体 + 临界前沿
    n_selected = len(selected)
    F_combined = F_norm[np.array(selected + fronts[l])] if l < len(fronts) else F_norm[selected]

    if l < len(fronts):
        # 为已选个体计算小生境计数
        association_all, distances_all = associate_to_reference_points(
            F_norm, reference_points,
        )
        niche_counts = np.zeros(len(reference_points), dtype=int)
        for idx in selected:
            niche_counts[association_all[idx]] += 1

        # 从临界前沿选择
        new_selected = niching_select(
            fronts[l], n_remaining,
            association_all, niche_counts, distances_all, rng,
        )
        selected.extend(new_selected)
    else:
        # 不足: 随机补
        remaining_indices = [i for i in range(n_total) if i not in selected]
        extras = list(rng.choice(remaining_indices, size=n_remaining, replace=False))
        selected.extend(extras)

    if return_indices:
        return selected[:n_select]
    return [population[i] for i in selected[:n_select]]


def nsga3_pareto_elites(
    population: List[MechanicalBody],
    n_elites: int,
    objective_keys: List[str],
    directions: List[str],
    reference_points: Optional[np.ndarray] = None,
    rng: Optional[np.random.RandomState] = None,
) -> List[MechanicalBody]:
    """NSGA-III 精英选择 (便捷接口)

    等价于 nsga3_select, 但从整个种群中选精英。
    """
    return nsga3_select(
        population, n_elites, objective_keys, directions,
        reference_points, rng,
    )
