# ══════════════════════════════════════════════════════════
# forgecraft.evolution.cma_me — CMA-ME 协方差自适应 + MAP-Elites 发射器
#
#   CMA-ES (Hansen, 2006): 多元正态分布自适应, 进化路径控制步长+协方差
#   CMA-ME: CMA-ES + MAP-Elites → 分布偏向"可改进存档"的搜索方向
#
#   组件:
#     - CMAES: (μ/μ_w, λ)-CMA-ES 核心
#     - CMAMEmitter: MAP-Elites 的 CMA 引导发射器
#     - OptimizerEmitter: map_elites.py 预留的 EmitterType.OPTIMIZER 实现
# ══════════════════════════════════════════════════════════

from typing import Dict, List, Optional, Tuple

import numpy as np

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "CMAES",
    "CMAMEmitter",
    "OptimizerEmitter",
    "cmame_sample",
]


# ══════════════════════════════════════════════════════════
#  CMA-ES 核心 (μ/μ_w, λ)-CMA-ES
# ══════════════════════════════════════════════════════════

class CMAES:
    """(μ/μ_w, λ)-CMA-ES (Hansen 2006)

    均值 m ∈ R^n, 协方差 C ∈ R^{n×n}, 步长 σ > 0
    采样: x_k = m + σ · B · D · z_k,  z_k ~ N(0, I)
    其中 C = B·D²·B^T (特征分解)

    更新:
      - 均值: m ← m + c_m · σ · Σ w_i · z_{i:λ}
      - 步长: σ ← σ · exp(c_σ/d_σ · (‖p_σ‖/E‖N(0,I)‖ - 1))
      - 协方差: C ← (1 - c_1 - c_μ)·C + c_1·p_c·p_cᵀ + c_μ·Σ w_i·y_i·y_iᵀ
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma0: float = 0.3,
        popsize: Optional[int] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        random_seed: Optional[int] = None,
    ):
        """初始化 CMA-ES

        Args:
            mean: 初始均值向量 (n,)
            sigma0: 初始步长
            popsize: 每代采样数 λ (None=4+⌊3log(n)⌋)
            bounds: (lower, upper) 箱约束 (可选)
            random_seed: 随机种子
        """
        self.rng = np.random.RandomState(random_seed)
        self.dim = len(mean)
        self.mean = mean.copy()

        # 种群参数
        if popsize is None:
            self.popsize = 4 + int(3 * np.log(self.dim))
        else:
            self.popsize = max(popsize, 4)
        self.popsize = min(self.popsize, 2000)  # 上限

        self.mu = self.popsize // 2  # 父代数量

        # 权重 (log递减)
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights /= np.sum(weights)
        self.weights = weights

        # 有效方差选择质量
        self.mu_eff = 1.0 / np.sum(weights ** 2)

        # 步长
        self.sigma = sigma0

        # 进化路径
        self.pc = np.zeros(self.dim)   # 协方差进化路径
        self.ps = np.zeros(self.dim)   # 步长进化路径

        # 协方差矩阵
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)

        # 学习率 (从 Hansen 2006 的默认公式)
        self.cc = (4.0 + self.mu_eff / self.dim) / (self.dim + 4.0 + 2.0 * self.mu_eff / self.dim)
        self.cs = (self.mu_eff + 2.0) / (self.dim + self.mu_eff + 5.0)
        self.c1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((self.dim + 2.0) ** 2 + self.mu_eff),
        )
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (self.dim + 1.0)) - 1.0) + self.cs

        # 统一乘子
        self.chiN = np.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim ** 2))

        # 箱约束
        self.bounds = bounds  # (low, high)

        # 统计
        self.generation = 0
        self.history_best = []
        self.eigen_update_counter = 0

    def ask(self, n_samples: Optional[int] = None) -> np.ndarray:
        """从分布采样

        Args:
            n_samples: 采样数 (None=默认 popsize)

        Returns:
            (n_samples, dim) 样本矩阵
        """
        if n_samples is None:
            n_samples = self.popsize

        # 采样 z ~ N(0, I)
        z = self.rng.randn(n_samples, self.dim)

        # x = mean + sigma * B * D * z
        samples = self.mean + self.sigma * (self.B @ (self.D[:, None] * z.T)).T

        # 箱约束: 简单反射
        if self.bounds is not None:
            low, high = self.bounds
            for i in range(self.dim):
                if np.isfinite(low[i]) or np.isfinite(high[i]):
                    mask_low = samples[:, i] < low[i]
                    mask_high = samples[:, i] > high[i]
                    samples[mask_low, i] = 2.0 * low[i] - samples[mask_low, i]
                    samples[mask_high, i] = 2.0 * high[i] - samples[mask_high, i]
                    samples[:, i] = np.clip(samples[:, i], low[i], high[i])

        return samples

    def tell(self, solutions: np.ndarray, fitnesses: np.ndarray):
        """用评估结果更新分布

        假定已经按适应度从优到劣排列 (降序 / maximize)。

        Args:
            solutions: (λ, n) 样本
            fitnesses: (λ,) 适应度 (越大越好)
        """
        n = len(fitnesses)
        mu = self.mu

        # 排序 (降序 = 最大化方向)
        order = np.argsort(fitnesses)[::-1][:mu]
        best_solutions = solutions[order]
        best_fitness = fitnesses[order[0]]

        # 计算 z 值
        z_best = np.linalg.solve(
            self.B @ np.diag(self.D),
            (best_solutions - self.mean).T,
        ).T

        # 加权重组
        z_w = np.sum(z_best * self.weights[:, None], axis=0)

        # 均值更新
        self.mean += self.sigma * (self.B @ (self.D * z_w))

        # 步长更新
        C_inv_sqrt = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mu_eff) * z_w
        ps_norm = np.linalg.norm(self.ps)
        self.sigma *= np.exp(
            (self.cs / self.damps) * (ps_norm / self.chiN - 1.0)
        )
        self.sigma = max(self.sigma, 1e-12)

        # 协方差进化路径
        hsig = int(ps_norm / np.sqrt(1.0 - (1.0 - self.cs) ** (2 * (self.generation + 1))) / self.chiN < 1.4 + 2.0 / (self.dim + 1.0))
        self.pc = (1.0 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mu_eff) * z_w

        # 协方差更新 (rank-μ + rank-1)
        y = z_best  # (μ, n)
        dy = (self.B @ (self.D[:, None] * y.T)).T
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(mu):
            rank_mu += self.weights[i] * np.outer(dy[i], dy[i])

        self.C = (
            (1.0 - self.c1 - self.cmu) * self.C
            + self.c1 * np.outer(self.pc, self.pc)
            + self.cmu * rank_mu
        )

        # 确保对称+正定
        self.C = 0.5 * (self.C + self.C.T)

        # 特征分解 (每N代, 或条件数过高时)
        self.eigen_update_counter += 1
        if self.eigen_update_counter >= max(1, self.dim // 10):
            self._update_eigen()
            self.eigen_update_counter = 0

        self.generation += 1
        self.history_best.append(float(best_fitness))

        # 修正极小特征值
        diag_C = np.diag(self.C)
        if np.any(diag_C < 1e-20):
            self.C += np.eye(self.dim) * 1e-10

    def _update_eigen(self):
        """特征分解 C = B·D²·B^T"""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.C)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            self.D = np.sqrt(eigenvalues)
            self.B = eigenvectors
        except np.linalg.LinAlgError:
            _logger.warning("CMA-ES: 特征分解失败, 重置协方差")
            self.C = np.eye(self.dim)
            self.B = np.eye(self.dim)
            self.D = np.ones(self.dim)

    def restart(self, mean: Optional[np.ndarray] = None, sigma_factor: float = 2.0):
        """IPOP-CMA-ES 重启

        增大种群, 重新初始化协方差为对角。

        Args:
            mean: 新均值 (None=当前均值)
            sigma_factor: 步长倍数
        """
        if mean is not None:
            self.mean = mean.copy()

        self.popsize = min(self.popsize * 2, 2000)
        self.mu = self.popsize // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights /= np.sum(weights)
        self.weights = weights
        self.mu_eff = 1.0 / np.sum(weights ** 2)

        self.sigma = self.sigma0 * sigma_factor if hasattr(self, 'sigma0') else self.sigma * sigma_factor
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.generation = 0

    def best(self) -> np.ndarray:
        """当前最佳估计 (均值)"""
        return self.mean.copy()

    def get_diagnostics(self) -> dict:
        """返回诊断信息"""
        cond = 0.0
        if self.D[0] > 0 and self.D[-1] > 0:
            cond = (self.D[-1] / self.D[0]) ** 2
        return {
            "generation": self.generation,
            "sigma": float(self.sigma),
            "condition_number": float(cond),
            "mean_norm": float(np.linalg.norm(self.mean)),
            "axis_ratio": float(np.max(self.D) / np.min(self.D)) if np.min(self.D) > 1e-15 else float('inf'),
            "best_history": int(len(self.history_best)),
            "last_best": float(self.history_best[-1]) if self.history_best else float('nan'),
        }


# ══════════════════════════════════════════════════════════
#  CMA-ME 发射器 (MAP-Elites 集成)
# ══════════════════════════════════════════════════════════

class CMAMEmitter:
    """CMA-ME 发射器: CMA-ES + MAP-Elites 存档

    采样策略:
      - 70% CMA 分布采样 (偏向改进已有精英)
      - 30% 空单元格中心的高斯噪声 (探索新区域)

    更新策略 (选择性反馈):
      仅用真正改进了存档的解来更新 CMA 分布。
      这是 CMA-ME 的核心创新: 不盲目追踪种群最优,
      而是追踪"哪些解对存档有贡献"。
    """

    def __init__(
        self,
        archive,  # CVTArchive / GridArchive
        n_dim: int,
        popsize: int = 20,
        sigma0: float = 0.3,
        empty_niche_prob: float = 0.3,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            archive: MAP-Elites 存档
            n_dim: 参数维度
            popsize: CMA 每代采样数
            sigma0: 初始步长
            empty_niche_prob: 空单元格采样概率
            random_seed: 随机种子
        """
        self.archive = archive
        self.n_dim = n_dim
        self.empty_niche_prob = empty_niche_prob
        self.rng = np.random.RandomState(random_seed)

        # 初始化 CMA
        self.cma = CMAES(
            mean=np.zeros(n_dim),
            sigma0=sigma0,
            popsize=popsize,
            random_seed=random_seed,
        )

        # 统计
        self.generation = 0
        self.improvements_tracked: List[bool] = []

    def ask(self, n_samples: Optional[int] = None) -> Tuple[np.ndarray, List[Optional[int]]]:
        """采样新解

        Args:
            n_samples: 采样数

        Returns:
            (solutions, target_cells): 参数矩阵 + 每解的目标单元格 (None=CMA采样)
        """
        if n_samples is None:
            n_samples = self.cma.popsize

        n_empty = int(n_samples * self.empty_niche_prob)
        n_cma = n_samples - n_empty

        solutions = []
        target_cells = []

        # CMA 指导的采样
        if n_cma > 0:
            cma_samples = self.cma.ask(n_cma)
            solutions.append(cma_samples)
            target_cells.extend([None] * n_cma)

        # 空单元格引导采样
        if n_empty > 0:
            empty_samples = self._sample_empty_niches(n_empty)
            empty_vals, empty_cells = empty_samples
            solutions.append(empty_vals)
            target_cells.extend(empty_cells)

        if len(solutions) > 1:
            all_solutions = np.vstack(solutions)
        else:
            all_solutions = solutions[0]

        return all_solutions, target_cells

    def tell(
        self,
        solutions: np.ndarray,
        fitnesses: np.ndarray,
        improvements: List[bool],  # 是否改进了存档
    ):
        """用存档改进结果更新 CMA

        Args:
            solutions: (n_samples, n_dim) 参数
            fitnesses: (n_samples,) 适应度
            improvements: (n_samples,) 是否改进了存档单元格
        """
        # 选择性反馈: 仅用改进了存档的解
        improved_mask = np.array(improvements)
        if not np.any(improved_mask):
            self.generation += 1
            return

        improved_solutions = solutions[improved_mask]
        improved_fitnesses = fitnesses[improved_mask]

        # 用改进了存档的解更新 CMA
        self.cma.tell(improved_solutions, improved_fitnesses)

        self.generation += 1
        self.improvements_tracked.append(np.mean(improved_mask))

    def _sample_empty_niches(self, n: int) -> Tuple[np.ndarray, List[Optional[int]]]:
        """从空单元格/低适应度单元格采样

        对 CVT 存档: 随机选空质心 → 质心位置加高斯噪声
        对 Grid 存档: 随机选空格子 → 格子中心加高斯噪声
        """
        samples = np.zeros((n, self.n_dim))
        cells = []

        # 获取已占用单元格索引
        if hasattr(self.archive, 'cells'):
            occupied = set(self.archive.cells.keys())
            n_total = self.archive.num_centroids
        elif hasattr(self.archive, 'grid'):
            occupied = set(self.archive.grid.keys())
            n_total = np.prod(self.archive.grid_dims)
        else:
            return self.rng.randn(n, self.n_dim) * 0.1, [None] * n

        empty_cells = [i for i in range(n_total) if i not in occupied]
        if not empty_cells:
            # 全部填满: 在已填单元格中心加噪声
            empty_cells = list(occupied)[:max(n, 10)]

        for i in range(n):
            cell_idx = self.rng.choice(empty_cells)
            cells.append(int(cell_idx))

            # 获取单元格中心位置 (行为空间 → 参数空间需匹配维度)
            if hasattr(self.archive, 'centroids'):
                centroid = self.archive.centroids[cell_idx % len(self.archive.centroids)]
                # 若行为空间维度 ≠ 参数维度, 使用零中心
                if len(centroid) == self.n_dim:
                    center = centroid
                else:
                    center = np.zeros(self.n_dim)
            else:
                center = np.zeros(self.n_dim)

            # 添加自适应噪声
            noise_scale = 0.1 * self.cma.sigma
            noise = self.rng.randn(self.n_dim) * noise_scale

            # 将归一化中心映射回参数空间 (简化: 直接使用)
            samples[i] = center + noise

        return samples, cells

    def get_diagnostics(self) -> dict:
        diag = self.cma.get_diagnostics()
        diag["emitter_generation"] = self.generation
        if self.improvements_tracked:
            diag["improvement_rate"] = float(np.mean(self.improvements_tracked[-20:]))
        return diag


# ══════════════════════════════════════════════════════════
#  OptimizerEmitter (map_elites.py EmitterType.OPTIMIZER)
# ══════════════════════════════════════════════════════════

class OptimizerEmitter:
    """优化器发射器 — 实现 map_elites.py 预留的 EmitterType.OPTIMIZER

    可用底层:
      - "CMA-ME": 协方差矩阵自适应 (推荐)
      - "CMA-ES": 纯 CMA-ES (无 MAP-Elites 反馈)
      - "simple": 随机采样 + 高斯扰动

    用于:
      from forgecraft.evolution.map_elites import MAPElitesEngine
      from forgecraft.evolution.cma_me import OptimizerEmitter

      emitter = OptimizerEmitter(archive, method="CMA-ME")
      solutions = emitter.ask(32)
      emitter.tell(solutions, fitnesses, improvements)
    """

    def __init__(
        self,
        archive,
        n_dim: int,
        method: str = "CMA-ME",
        **kwargs,
    ):
        """初始化优化器发射器

        Args:
            archive: MAP-Elites 存档 (CVTArchive / GridArchive)
            n_dim: 参数空间维度
            method: 优化方法 ("CMA-ME", "CMA-ES", "simple")
            **kwargs: 传递给底层优化器
        """
        self.archive = archive
        self.n_dim = n_dim
        self.method = method
        self.rng = kwargs.get('rng', np.random.RandomState())

        if method == "CMA-ME":
            self.optimizer = CMAMEmitter(
                archive, n_dim,
                popsize=kwargs.get('popsize', 20),
                sigma0=kwargs.get('sigma0', 0.3),
                empty_niche_prob=kwargs.get('empty_niche_prob', 0.3),
                random_seed=kwargs.get('random_seed'),
            )
        elif method == "CMA-ES":
            self.optimizer = CMAES(
                mean=np.zeros(n_dim),
                sigma0=kwargs.get('sigma0', 0.3),
                popsize=kwargs.get('popsize', 20),
                random_seed=kwargs.get('random_seed'),
            )
        else:
            self.optimizer = None  # "simple" mode

    def ask(self, n_samples: int) -> np.ndarray:
        """采样 n_samples 个解"""
        if self.method == "CMA-ME":
            solutions, _ = self.optimizer.ask(n_samples)
            return solutions
        elif self.method == "CMA-ES":
            return self.optimizer.ask(n_samples)
        else:
            # Simple: 随机 + 精英扰动
            elites = self.archive.get_all_elites()
            if elites and len(elites) > 0:
                # 从精英周围采样
                elite_params = self._elite_params(elites)
                base = self.rng.choice(len(elite_params), size=n_samples)
                noise = self.rng.randn(n_samples, self.n_dim) * 0.1
                return elite_params[base] + noise
            else:
                return self.rng.randn(n_samples, self.n_dim) * 0.5

    def tell(
        self,
        solutions: np.ndarray,
        fitnesses: np.ndarray,
        improvements: List[bool],
    ):
        """反馈优化器"""
        if self.method == "CMA-ME":
            self.optimizer.tell(solutions, fitnesses, improvements)
        elif self.method == "CMA-ES":
            self.optimizer.tell(solutions, fitnesses)

    def _elite_params(self, elites: List) -> np.ndarray:
        """从精英条目提取参数向量 (简化实现: 用适应度作为代理)"""
        # 实际使用时, 这可替换为从 body 中提取连续参数
        n = len(elites)
        params = np.zeros((n, self.n_dim))
        for i, e in enumerate(elites):
            # 用 body 属性填充参数 (简化)
            if hasattr(e.body, 'fitness'):
                params[i, 0] = e.body.fitness
        return params

    def restart(self):
        """重启优化器"""
        if self.method in ("CMA-ME", "CMA-ES"):
            self.optimizer.restart()

    def get_diagnostics(self) -> dict:
        if self.method == "CMA-ME":
            return self.optimizer.get_diagnostics()
        elif self.method == "CMA-ES":
            return self.optimizer.get_diagnostics()
        return {"method": self.method}


# ══════════════════════════════════════════════════════════
#  便捷函数
# ══════════════════════════════════════════════════════════

def cmame_sample(
    archive,
    n_dim: int,
    n_samples: int,
    method: str = "CMA-ME",
    emitter: Optional[CMAMEmitter] = None,
) -> Tuple[np.ndarray, Optional[CMAMEmitter]]:
    """便捷: 从 CMA-ME 发射器采样

    Args:
        archive: MAP-Elites 存档
        n_dim: 参数维度
        n_samples: 采样数
        method: 方法
        emitter: 已有发射器 (None=创建新)

    Returns:
        (solutions, emitter)
    """
    if emitter is None:
        emitter = CMAMEmitter(archive, n_dim)
    solutions, _ = emitter.ask(n_samples)
    return solutions, emitter
