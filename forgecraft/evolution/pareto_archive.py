# ══════════════════════════════════════════════════════════
# forgecraft.evolution.pareto_archive — Pareto 存档
#
#   跨代 Pareto 前沿管理器:
#     - 非支配集维护
#     - 超体积计算 (WPF 递推)
#     - 膝点检测 (边际效用递减)
#     - 参考点偏好查询
#     - 前沿多样性指标 (间距/展度/IGD+)
#
#   与 MAP-Elites 存档的区别:
#     MAP-Elites: 按行为描述符分区 (质量多样性)
#     ParetoArchive: 按目标空间支配关系 (多目标最优性)
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "ParetoArchive",
    "compute_hypervolume",
    "detect_knee_point",
    "reference_point_query",
    "spacing_metric",
    "spread_metric",
]


# ══════════════════════════════════════════════════════════
#  ParetoArchive 类
# ══════════════════════════════════════════════════════════

class ParetoArchive:
    """跨代 Pareto 前沿管理器

    维护非支配解集, 提供:
      - 插入 (自动去除非支配)
      - 超体积轨迹 (追踪优化进程)
      - 膝点检测 (单解推荐)
      - 偏好查询 (参考点最接近的解)
    """

    def __init__(
        self,
        n_objectives: int,
        directions: Optional[List[str]] = None,
        reference_point: Optional[np.ndarray] = None,
        capacity: int = 10000,
    ):
        """
        Args:
            n_objectives: 目标数
            directions: 每个目标的优化方向 ("maximize"/"minimize")
            reference_point: 超体积参考点 (最差点)
            capacity: 最大存档容量
        """
        self.n_objectives = n_objectives
        self.directions = directions or ["maximize"] * n_objectives
        self.capacity = capacity

        # 默认参考点 (如果未指定)
        if reference_point is None:
            # 使用安全默认值: 各目标为 0 (假定额外信息会更新)
            self.reference_point = np.zeros(n_objectives)
        else:
            self.reference_point = np.array(reference_point)

        # 内部存储
        self._solutions: List[Any] = []
        self._objectives: np.ndarray = np.empty((0, n_objectives))

        # 统计
        self.total_added = 0
        self.total_rejected = 0
        self.hypervolume_history: List[float] = []

    def _sign_objectives(self, F: np.ndarray) -> np.ndarray:
        """统一为 maximize 方向 (超体积要求)"""
        F_signed = F.copy()
        for k in range(self.n_objectives):
            if self.directions[k] == "minimize":
                F_signed[:, k] = self.reference_point[k] - F_signed[:, k]
            else:
                F_signed[:, k] = F_signed[:, k] - self.reference_point[k]
        return F_signed

    def add(self, solution: Any, objectives: np.ndarray) -> bool:
        """添加个体到存档

        Args:
            solution: 个体对象 (任意类型)
            objectives: (n_objectives,) 目标值

        Returns:
            是否添加成功
        """
        obj = np.asarray(objectives, dtype=np.float64).ravel()

        # 检查是否被现有解支配
        if len(self._solutions) > 0:
            if self._is_dominated(obj, self._objectives):
                self.total_rejected += 1
                return False

        # 移除被新解支配的旧解
        if len(self._solutions) > 0:
            keep_mask = ~self._is_dominated_by_others(self._objectives, obj)
            if not np.all(keep_mask):
                removed_count = int(np.sum(~keep_mask))
                self._solutions = [s for i, s in enumerate(self._solutions) if keep_mask[i]]
                self._objectives = self._objectives[keep_mask]

        # 添加新解
        self._solutions.append(solution)
        if len(self._objectives) == 0:
            self._objectives = obj.reshape(1, -1)
        else:
            self._objectives = np.vstack([self._objectives, obj.reshape(1, -1)])

        self.total_added += 1

        # 容量管理: 移除拥挤的解
        if len(self._solutions) > self.capacity:
            self._prune()

        return True

    def add_batch(self, solutions: List[Any], objectives: np.ndarray) -> int:
        """批量添加

        Args:
            solutions: 个体列表
            objectives: (N, n_objectives) 目标矩阵

        Returns:
            添加的个体数
        """
        added = 0
        for i in range(len(solutions)):
            if self.add(solutions[i], objectives[i]):
                added += 1
        return added

    def _is_dominated(self, obj: np.ndarray, archive: np.ndarray) -> bool:
        """检查 obj 是否被 archive 中任一个体支配"""
        for k in range(self.n_objectives):
            if self.directions[k] == "maximize":
                if np.any(obj[k] < archive[:, k]):
                    pass
            else:
                if np.any(obj[k] > archive[:, k]):
                    pass

        # 逐行检查
        for i in range(len(archive)):
            dominated = True
            at_least_one_better = False
            for k in range(self.n_objectives):
                if self.directions[k] == "maximize":
                    if obj[k] > archive[i, k]:
                        dominated = False
                        break
                    if obj[k] < archive[i, k]:
                        at_least_one_better = True
                else:
                    if obj[k] < archive[i, k]:
                        dominated = False
                        break
                    if obj[k] > archive[i, k]:
                        at_least_one_better = True
            if dominated and at_least_one_better:
                return True
        return False

    def _is_dominated_by_others(self, archive: np.ndarray, obj: np.ndarray) -> np.ndarray:
        """返回被 obj 支配的 archive 个体掩码"""
        n = len(archive)
        dominated_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            dominates = True
            at_least_one_better = False
            for k in range(self.n_objectives):
                if self.directions[k] == "maximize":
                    if obj[k] < archive[i, k]:
                        dominates = False
                        break
                    if obj[k] > archive[i, k]:
                        at_least_one_better = True
                else:
                    if obj[k] > archive[i, k]:
                        dominates = False
                        break
                    if obj[k] < archive[i, k]:
                        at_least_one_better = True
            dominated_mask[i] = dominates and at_least_one_better
        return dominated_mask

    def _prune(self):
        """拥挤修剪: 移除最拥挤的解"""
        cd = crowding_distance(self._objectives)
        # 移除去重后的最拥挤个体 (保留边界)
        if len(cd) > 2:
            worst = np.argmin(cd[1:-1]) + 1  # 跳过边界
            self._solutions.pop(worst)
            self._objectives = np.delete(self._objectives, worst, axis=0)

    def compute_hypervolume(self) -> float:
        """计算当前存档的超体积

        使用 Walking Fish Group (WFG) 递推算法
        复杂度 O(N^{d/2}), 但 d ≤ 5 时可用
        """
        if len(self._solutions) == 0:
            return 0.0
        F_signed = self._sign_objectives(self._objectives)
        hv = compute_hypervolume(F_signed, self.reference_point)
        self.hypervolume_history.append(hv)
        return hv

    def detect_knee_point(self) -> Optional[int]:
        """检测 Pareto 前沿膝点

        方法: 最大边际效用递减
        在归一化前沿上, 对每对相邻解计算角度变化,
        角度变化最大的解为膝点。

        Returns:
            膝点在存档中的索引, 或 None
        """
        if len(self._solutions) < 3:
            return None if len(self._solutions) == 0 else 0

        F = self._objectives
        n = len(F)

        # 归一化
        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        f_range = f_max - f_min
        f_range[f_range < 1e-10] = 1.0
        F_norm = (F - f_min) / f_range

        # 按第一个目标排序
        order = np.argsort(F_norm[:, 0])
        if self.directions[0] == "minimize":
            order = order[::-1]

        # 计算相邻点间角度
        angles = []
        for i in range(1, n - 1):
            v1 = F_norm[order[i]] - F_norm[order[i - 1]]
            v2 = F_norm[order[i + 1]] - F_norm[order[i]]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_theta = np.dot(v1, v2) / (n1 * n2)
                cos_theta = np.clip(cos_theta, -1, 1)
                angles.append(float(180 - np.arccos(cos_theta) * 180 / np.pi))
            else:
                angles.append(0.0)

        if angles:
            return int(order[np.argmax(angles) + 1])
        return None

    def reference_point_query(
        self, aspiration: np.ndarray, n_results: int = 5,
    ) -> List[int]:
        """基于参考点的偏好查询

        返回最接近 aspiration_levels 的 Pareto 解。

        Args:
            aspiration: (n_objectives,) 期望目标值
            n_results: 返回解数量

        Returns:
            存档索引列表
        """
        if len(self._solutions) == 0:
            return []

        F = self._objectives
        aspiration = np.asarray(aspiration).ravel()

        # L2 距离 (归一化)
        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        f_range = f_max - f_min
        f_range[f_range < 1e-10] = 1.0
        F_norm = (F - f_min) / f_range
        aspiration_norm = (aspiration - f_min) / f_range

        distances = np.linalg.norm(F_norm - aspiration_norm, axis=1)
        top_indices = np.argsort(distances)[:n_results]

        return [int(i) for i in top_indices]

    def get_solutions(self) -> List[Any]:
        return list(self._solutions)

    def get_objectives(self) -> np.ndarray:
        return self._objectives.copy()

    def get_knee_solution(self) -> Optional[Any]:
        idx = self.detect_knee_point()
        return self._solutions[idx] if idx is not None and idx < len(self._solutions) else None

    def size(self) -> int:
        return len(self._solutions)

    def diversity_metrics(self) -> Dict[str, float]:
        """计算前沿多样性指标"""
        if len(self._solutions) < 2:
            return {"spacing": 0.0, "spread": 0.0}

        F = self._objectives
        spacing = spacing_metric(F)
        spread = spread_metric(F)

        return {
            "spacing": float(spacing),
            "spread": float(spread),
            "hypervolume": float(self.compute_hypervolume()) if len(self._solutions) > 0 else 0.0,
        }

    def get_statistics(self) -> Dict[str, Any]:
        knee_idx = self.detect_knee_point()
        return {
            "archive_size": len(self._solutions),
            "total_added": self.total_added,
            "total_rejected": self.total_rejected,
            "hypervolume": self.hypervolume_history[-1] if self.hypervolume_history else None,
            "knee_point_index": knee_idx,
            "objectives_range": {
                f"obj_{k}": (float(np.min(self._objectives[:, k])), float(np.max(self._objectives[:, k])))
                for k in range(min(self.n_objectives, 5))
            } if len(self._solutions) > 0 else {},
        }


# ══════════════════════════════════════════════════════════
#  独立函数
# ══════════════════════════════════════════════════════════

def compute_hypervolume(F: np.ndarray, reference_point: np.ndarray) -> float:
    """计算超体积 (递推 WFG 算法)

    所有目标必须统一为 maximize 方向, 且已平移使 reference_point 为原点。

    Args:
        F: (N, M) 目标值 (maximize方向, 已减 ref)
        reference_point: 参考点 (应为 0 向量如果已平移)

    Returns:
        超体积值
    """
    N, M = F.shape
    if N == 0:
        return 0.0
    if N == 1:
        hv = np.prod(np.maximum(F[0], 0))
        return float(hv)

    # 递归: 按第一目标排序
    order = np.argsort(F[:, 0])[::-1]  # 降序
    F_sorted = F[order]

    hv = 0.0
    prev = 0.0
    for i in range(N):
        current = max(F_sorted[i, 0], 0.0)
        slice_width = current - prev
        if slice_width > 0 and M > 1:
            # 超体积贡献: width × HV_{M-1}(剩余点)
            if i > 0:
                restricted = F_sorted[:i, 1:]
                restricted = np.maximum(restricted, 0)
                sub_hv = compute_hypervolume(restricted, reference_point[1:])
                hv += slice_width * sub_hv
        prev = current

    # 最后一个矩形
    if M > 1:
        restricted = F_sorted[:, 1:]
        restricted = np.maximum(restricted, 0)
        sub_hv = compute_hypervolume(restricted, reference_point[1:])
        hv += max(F_sorted[-1, 0], 0.0) * sub_hv
    else:
        hv += max(F_sorted[0, 0], 0.0)

    return float(hv)


def detect_knee_point(objectives: np.ndarray) -> int:
    """独立膝点检测函数 (返回索引)

    Args:
        objectives: (N, M) 目标矩阵

    Returns:
        膝点索引
    """
    n, m = objectives.shape
    if n < 3:
        return 0 if n > 0 else -1

    # 归一化
    f_min = np.min(objectives, axis=0)
    f_max = np.max(objectives, axis=0)
    f_range = f_max - f_min
    f_range[f_range < 1e-10] = 1.0
    F_norm = (objectives - f_min) / f_range

    # 按第一个目标排序
    order = np.argsort(F_norm[:, 0])

    angles = []
    for i in range(1, n - 1):
        v1 = F_norm[order[i]] - F_norm[order[i - 1]]
        v2 = F_norm[order[i + 1]] - F_norm[order[i]]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-10 and n2 > 1e-10:
            cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            angles.append(float(180 - np.arccos(cos_theta) * 180 / np.pi))
        else:
            angles.append(0.0)

    if angles:
        return int(order[np.argmax(angles) + 1])
    return 0


def reference_point_query(
    objectives: np.ndarray, aspiration: np.ndarray, n_results: int = 5,
) -> List[int]:
    """独立参考点查询"""
    f_min = np.min(objectives, axis=0)
    f_max = np.max(objectives, axis=0)
    f_range = f_max - f_min
    f_range[f_range < 1e-10] = 1.0
    F_norm = (objectives - f_min) / f_range
    aspiration_norm = (aspiration - f_min) / f_range
    distances = np.linalg.norm(F_norm - aspiration_norm, axis=1)
    return [int(i) for i in np.argsort(distances)[:n_results]]


def spacing_metric(F: np.ndarray) -> float:
    """间距指标 (Schott 1995)

    衡量非支配解分布的均匀性:
      S = sqrt(1/(N-1) Σ (d̄ - d_i)²)
    其中 d_i = min_{j≠i} ‖F_i - F_j‖

    S 越小越均匀。

    Args:
        F: (N, M) 目标矩阵

    Returns:
        间距值
    """
    N = F.shape[0]
    if N < 2:
        return 0.0

    distances = np.zeros(N)
    for i in range(N):
        min_dist = float('inf')
        for j in range(N):
            if i != j:
                d = np.linalg.norm(F[i] - F[j])
                min_dist = min(min_dist, d)
        distances[i] = min_dist

    d_mean = np.mean(distances)
    spacing = np.sqrt(np.sum((distances - d_mean) ** 2) / (N - 1))
    return float(spacing)


def spread_metric(F: np.ndarray) -> float:
    """展度指标 (Deb et al. 2002)

    衡量非支配解覆盖的范围:
      Δ = (d_f + d_l + Σ|d_i - d̄|) / (d_f + d_l + (N-1)d̄)
    其中 d_f, d_l 为边界到极端的距离, d_i 为相邻解间距

    Δ ∈ [0, 1], 越小越好 (0 = 完美覆盖)

    Args:
        F: (N, M) 目标矩阵

    Returns:
        展度值 [0, 1]
    """
    N, M = F.shape
    if N < 3:
        return 0.0

    # 按第一个目标排序
    order = np.argsort(F[:, 0])
    F_sorted = F[order]

    # 相邻解间距
    distances = np.zeros(N - 1)
    for i in range(N - 1):
        distances[i] = np.linalg.norm(F_sorted[i + 1] - F_sorted[i])

    d_mean = np.mean(distances)

    # 边界距离
    extremes = np.array([
        np.min(F, axis=0),
        np.max(F, axis=0),
    ])

    df = np.linalg.norm(F_sorted[0] - extremes[0])
    dl = np.linalg.norm(F_sorted[-1] - extremes[1])

    denominator = df + dl + (N - 1) * d_mean
    if denominator < 1e-15:
        return 0.0

    spread = (df + dl + np.sum(np.abs(distances - d_mean))) / denominator
    return float(np.clip(spread, 0, 1))


def crowding_distance(points: np.ndarray) -> np.ndarray:
    """计算拥挤距离 (内部修剪用)

    Args:
        points: (N, D) 点集

    Returns:
        (N,) 拥挤距离
    """
    N, D = points.shape
    if N <= 2:
        return np.full(N, float('inf'))

    distances = np.zeros(N)
    for d in range(D):
        order = np.argsort(points[:, d])
        sorted_vals = points[order, d]
        val_range = sorted_vals[-1] - sorted_vals[0]
        if val_range < 1e-12:
            continue

        distances[order[0]] = float('inf')
        distances[order[-1]] = float('inf')

        for i in range(1, N - 1):
            distances[order[i]] += (sorted_vals[i + 1] - sorted_vals[i - 1]) / val_range

    return distances
