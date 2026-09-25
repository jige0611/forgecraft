"""
MAP-Elites 质量多样性进化模块 (深度整合版)

实现:
  - 多维行为特征空间 (速度/能耗/稳定性/对称性/质量)
  - CVT (Centroidal Voronoi Tessellation) 高维空间自适应划分
  - 多发射器策略 (improvement / random / optimizer)
  - 精英注入NSGA-II主流
  - QD-score / 覆盖率追踪
  - 存档持久化

核心算法参考:
  - Mouret & Clune (2015) "Illuminating search spaces by mapping elites"
    → 提出 MAP-Elites 算法: 按行为特征将精英分散存储在高维网格中
  - pyribs (https://github.com/icaros-usc/pyribs)
    → 参考实现, 未直接依赖
  - Vassiliades et al. (2018) "Using Centroidal Voronoi Tessellations to
    Scale Up the Multidimensional Archive of Phenotypic Elites Algorithm"
    → CVT 网格划分: 用 K-means 聚类中心代替均匀网格

算法流程:
  1. 计算每个形态的行为特征 (BC) → 归一化到 [0,1]
  2. 在 CVT 网格中定位 cell index
  3. 若 cell 为空, 或新形态适应度更高 → 替换
  4. 每 N 代从 archive 中注入 elite 到 NSGA-II 种群
  5. 追踪 QD-score = sum(max_fitness_per_cell)
"""

import numpy as np
import pickle
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import math

from forgecraft.core.morphology import MechanicalBody
import logging

__all__ = [
    "MAPElitesConfig",
    "MAPElitesEngine",
    "CVTArchive",
    "GridArchive",
    "EliteEntry",
    "BaseEmitter",
    "ImprovementEmitter",
    "RandomEmitter",
    "AdaptiveEmitter",
    "OptimizerEmitter",
    "EmitterType",
    "compute_behavior_characteristics",
    "compute_behavior_characteristics_with_objectives",
    "create_map_elites_engine",
    "inject_elites",
]

_logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# 数据类型
# ══════════════════════════════════════════════════════════

@dataclass
class MAPElitesConfig:
    """MAP-Elites 配置"""
    # 行为特征维度
    bc_names: Tuple[str, ...] = (
        "speed", "energy_efficiency", "stability",
        "symmetry", "body_mass"
    )
    # 每个维度的范围
    bc_ranges: Tuple[Tuple[float, float], ...] = (
        (-2.0, 10.0),    # speed
        (-10.0, 2.0),    # energy_efficiency (log scale)
        (0.0, 1.0),      # stability
        (0.0, 1.0),      # symmetry
        (0.1, 20.0),     # body_mass (kg)
    )
    # 存档大小 (每维分格数, 或 CVT 质心数)
    archive_size: int = 1000
    # 使用 CVT (高维推荐) 还是 Grid (2-3维)
    use_cvt: bool = True
    
    # 发射器
    num_emitters: int = 5
    emitter_batch_size: int = 32
    
    # 每代评估次数
    evaluations_per_generation: int = 50
    
    # 精英注入主流频率
    inject_every_n_gens: int = 5
    inject_count: int = 3
    
    # QD-score 追踪
    track_qd_score: bool = True
    
    # 存档持久化
    save_archive: bool = True
    archive_path: str = "map_elites_archive.pkl"


class EmitterType(Enum):
    """发射器类型"""
    IMPROVEMENT = "improvement"    # 从存档采样 + 变异
    RANDOM = "random"              # 完全随机
    OPTIMIZER = "optimizer"        # 基于梯度的优化 (CMA-ES/梯度)


@dataclass
class EliteEntry:
    """存档中的精英条目"""
    body: MechanicalBody
    fitness: float
    behavior: np.ndarray
    generation: int
    objective_values: Dict[str, float] = field(default_factory=dict)


class CVTArchive:
    """
    CVT (Centroidal Voronoi Tessellation) 存档
    
    使用 k-means 质心将高维行为空间划分为区域,
    每个区域只保留最好的个体。
    """
    
    def __init__(
        self,
        num_centroids: int,
        bc_dim: int,
        bc_ranges: List[Tuple[float, float]],
        random_seed: int = 42,
    ):
        self.num_centroids = num_centroids
        self.bc_dim = bc_dim
        self.bc_ranges = np.array(bc_ranges)
        self.rng = np.random.RandomState(random_seed)
        
        # 初始化质心 (在归一化空间 [0,1]^d 中均匀采样)
        self.centroids = self.rng.uniform(0, 1, (num_centroids, bc_dim))
        # 每个质心对应的最佳个体
        self.cells: Dict[int, EliteEntry] = {}
        # 质心计数器 (用于在线更新)
        self.centroid_counts = np.zeros(num_centroids, dtype=np.int64)
        self.centroid_sums = np.zeros((num_centroids, bc_dim))
        
        # 统计
        self.total_evaluations = 0
        self.total_added = 0
        self.total_replaced = 0
        
    def _normalize_bc(self, bc: np.ndarray) -> np.ndarray:
        """将行为特征归一化到 [0, 1]"""
        normalized = np.zeros(self.bc_dim)
        for i in range(self.bc_dim):
            low, high = self.bc_ranges[i]
            if high > low:
                normalized[i] = (bc[i] - low) / (high - low)
            else:
                normalized[i] = 0.5
        return np.clip(normalized, 0, 1)
    
    def _find_nearest_centroid(self, norm_bc: np.ndarray) -> int:
        """找最近质心索引"""
        distances = np.linalg.norm(self.centroids - norm_bc, axis=1)
        return int(np.argmin(distances))
    
    def _update_centroids(self):
        """在线更新质心位置 (moving average)"""
        for idx in range(self.num_centroids):
            if self.centroid_counts[idx] > 0:
                self.centroids[idx] = (
                    self.centroid_sums[idx] / self.centroid_counts[idx]
                )
    
    def add(self, body: MechanicalBody, fitness: float, bc: np.ndarray,
            generation: int = 0, objective_values: Dict[str, float] = None):
        """添加个体到存档"""
        self.total_evaluations += 1
        
        norm_bc = self._normalize_bc(bc)
        cell_idx = self._find_nearest_centroid(norm_bc)
        
        # 更新质心 (在线 k-means)
        self.centroid_counts[cell_idx] += 1
        lr = 1.0 / self.centroid_counts[cell_idx]
        self.centroid_sums[cell_idx] += norm_bc
        self.centroids[cell_idx] += lr * (norm_bc - self.centroids[cell_idx])
        
        entry = EliteEntry(
            body=body.clone(),
            fitness=fitness,
            behavior=bc.copy(),
            generation=generation,
            objective_values=objective_values or {},
        )
        
        if cell_idx not in self.cells or fitness > self.cells[cell_idx].fitness:
            if cell_idx in self.cells:
                self.total_replaced += 1
            self.cells[cell_idx] = entry
            self.total_added += 1
            return True
        return False
    
    def get_elite(self, cell_idx: int) -> Optional[EliteEntry]:
        return self.cells.get(cell_idx)
    
    def get_all_elites(self) -> List[EliteEntry]:
        return list(self.cells.values())
    
    def get_all_bodies(self) -> List[MechanicalBody]:
        return [e.body for e in self.cells.values()]
    
    def coverage(self) -> float:
        return len(self.cells) / self.num_centroids
    
    def random_cell(self) -> Optional[int]:
        if not self.cells:
            return None
        return int(self.rng.choice(list(self.cells.keys())))
    
    def random_elite(self) -> Optional[EliteEntry]:
        cell = self.random_cell()
        return self.cells[cell] if cell is not None else None
    
    def sample_elites(self, n: int) -> List[EliteEntry]:
        """采样 n 个精英 (有放回)"""
        elites = self.get_all_elites()
        if not elites:
            return []
        indices = self.rng.choice(len(elites), size=min(n, len(elites)), replace=True)
        return [elites[i] for i in indices]
    
    def best_elite(self) -> Optional[EliteEntry]:
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda e: e.fitness)
    
    def best_fitness(self) -> float:
        best = self.best_elite()
        return best.fitness if best else -float("inf")
    
    def compute_qd_score(self) -> float:
        """计算 QD-score: 所有单元格适应度之和"""
        return sum(e.fitness for e in self.cells.values())
    
    def get_stats(self) -> Dict[str, Any]:
        best = self.best_elite()
        return {
            "num_elites": len(self.cells),
            "coverage": self.coverage(),
            "qd_score": self.compute_qd_score(),
            "best_fitness": best.fitness if best else -float("inf"),
            "total_evaluations": self.total_evaluations,
            "total_added": self.total_added,
            "total_replaced": self.total_replaced,
            "behavior_range": self._compute_behavior_range(),
        }
    
    def _compute_behavior_range(self) -> Dict[str, Tuple[float, float]]:
        """计算存档中行为特征的实际范围"""
        if not self.cells:
            return {}
        bcs = np.array([e.behavior for e in self.cells.values()])
        ranges = {}
        for i in range(min(self.bc_dim, bcs.shape[1])):
            ranges[f"dim_{i}"] = (float(bcs[:, i].min()), float(bcs[:, i].max()))
        return ranges


class GridArchive:
    """
    网格存档 (低维行为空间, 2-3维)
    """
    
    def __init__(
        self,
        grid_dims: Tuple[int, ...],
        bc_ranges: List[Tuple[float, float]],
    ):
        self.grid_dims = grid_dims
        self.bc_ranges = np.array(bc_ranges)
        self.grid: Dict[Tuple[int, ...], EliteEntry] = {}
        self.total_evaluations = 0
        self.total_added = 0
        self.total_replaced = 0
    
    def _to_cell(self, bc: np.ndarray) -> Tuple[int, ...]:
        """行为特征映射到网格坐标"""
        cell = []
        for i, dim in enumerate(self.grid_dims):
            low, high = self.bc_ranges[i]
            if high > low:
                idx = int((bc[i] - low) / (high - low) * dim)
                idx = max(0, min(dim - 1, idx))
            else:
                idx = 0
            cell.append(idx)
        return tuple(cell)
    
    def add(self, body: MechanicalBody, fitness: float, bc: np.ndarray,
            generation: int = 0, objective_values: Dict[str, float] = None):
        self.total_evaluations += 1
        
        cell = self._to_cell(bc)
        entry = EliteEntry(
            body=body.clone(), fitness=fitness, behavior=bc.copy(),
            generation=generation,
            objective_values=objective_values or {},
        )
        
        if cell not in self.grid or fitness > self.grid[cell].fitness:
            if cell in self.grid:
                self.total_replaced += 1
            self.grid[cell] = entry
            self.total_added += 1
            return True
        return False
    
    def coverage(self) -> float:
        total_cells = 1
        for d in self.grid_dims:
            total_cells *= d
        return len(self.grid) / total_cells
    
    def random_elite(self) -> Optional[EliteEntry]:
        if not self.grid:
            return None
        cell = list(self.grid.keys())[np.random.randint(len(self.grid))]
        return self.grid[cell]
    
    def get_all_elites(self) -> List[EliteEntry]:
        return list(self.grid.values())
    
    def get_all_bodies(self) -> List[MechanicalBody]:
        return [e.body for e in self.grid.values()]
    
    def best_elite(self) -> Optional[EliteEntry]:
        if not self.grid:
            return None
        return max(self.grid.values(), key=lambda e: e.fitness)
    
    def best_fitness(self) -> float:
        best = self.best_elite()
        return best.fitness if best else -float("inf")
    
    def compute_qd_score(self) -> float:
        return sum(e.fitness for e in self.grid.values())
    
    def sample_elites(self, n: int) -> List[EliteEntry]:
        elites = self.get_all_elites()
        if not elites:
            return []
        indices = np.random.choice(len(elites), size=min(n, len(elites)), replace=True)
        return [elites[i] for i in indices]
    
    def get_stats(self) -> Dict[str, Any]:
        best = self.best_elite()
        return {
            "num_elites": len(self.grid),
            "coverage": self.coverage(),
            "qd_score": self.compute_qd_score(),
            "best_fitness": best.fitness if best else -float("inf"),
            "total_evaluations": self.total_evaluations,
            "total_added": self.total_added,
            "total_replaced": self.total_replaced,
        }


# ══════════════════════════════════════════════════════════
# 发射器
# ══════════════════════════════════════════════════════════

class BaseEmitter:
    """发射器基类"""
    
    def __init__(self, batch_size: int, rng: np.random.RandomState):
        self.batch_size = batch_size
        self.rng = rng
    
    def ask(self, archive) -> List[MechanicalBody]:
        raise NotImplementedError
    
    def tell(self, archive, bodies: List[MechanicalBody],
             fitnesses: List[float], bcs: List[np.ndarray],
             generation: int = 0):
        for body, fitness, bc in zip(bodies, fitnesses, bcs):
            archive.add(body, fitness, bc, generation)


class ImprovementEmitter(BaseEmitter):
    """
    改进发射器: 从存档采样精英, 变异后评估
    
    参考 pyribs ImprovementEmitter:
    - 选择父代: 均匀随机 + 适应度加权
    - 添加高斯噪声变异
    """
    
    def __init__(self, batch_size: int, rng: np.random.RandomState,
                 mutation_strength: float = 0.1):
        super().__init__(batch_size, rng)
        self.mutation_strength = mutation_strength
    
    def ask(self, archive) -> List[MechanicalBody]:
        elites = archive.get_all_elites()
        if not elites:
            return []
        
        offspring = []
        for _ in range(self.batch_size):
            # 均匀随机选择父代 (也可以按适应度加权)
            parent = self.rng.choice(elites)
            child = parent.body.clone()
            
            # 对参数添加随机扰动
            self._mutate_body(child)
            offspring.append(child)
        
        return offspring
    
    def _mutate_body(self, body: MechanicalBody):
        """对机械体的零件参数添加随机扰动"""
        for part_id in body.graph.nodes:
            part = body.get_part(part_id)
            for key in list(part.params.keys()):
                if isinstance(part.params[key], (int, float)):
                    noise = self.rng.normal(0, self.mutation_strength)
                    part.params[key] += noise * abs(part.params[key])
        
        # 对位置添加扰动
        for part_id in body.graph.nodes:
            part = body.get_part(part_id)
            if part_id != body.root_id:
                noise = self.rng.normal(0, self.mutation_strength * 0.05, 3)
                part.position += noise


class RandomEmitter(BaseEmitter):
    """
    随机发射器: 完全随机生成新形态
    
    维持探索性, 防止过早收敛
    """
    
    def __init__(self, batch_size: int, rng: np.random.RandomState,
                 generator_fn: Callable[[], MechanicalBody]):
        super().__init__(batch_size, rng)
        self.generator_fn = generator_fn
    
    def ask(self, archive) -> List[MechanicalBody]:
        return [self.generator_fn() for _ in range(self.batch_size)]


class AdaptiveEmitter(BaseEmitter):
    """
    自适应发射器: 使用已有进化算子
    
    集成自适应变异控制器和交叉操作
    """
    
    def __init__(self, batch_size: int, rng: np.random.RandomState,
                 adaptive_controller=None, crossover_rate: float = 0.3):
        super().__init__(batch_size, rng)
        self.adaptive_controller = adaptive_controller
        self.crossover_rate = crossover_rate
    
    def ask(self, archive) -> List[MechanicalBody]:
        elites = archive.get_all_elites()
        if len(elites) < 2:
            return []
        
        offspring = []
        for _ in range(self.batch_size):
            # 有概率进行交叉
            if self.rng.random() < self.crossover_rate and len(elites) >= 2:
                p1, p2 = self.rng.choice(elites, 2, replace=False)
                child = self._crossover(p1.body, p2.body)
            else:
                parent = self.rng.choice(elites)
                child = parent.body.clone()
            
            self._mutate_body(child)
            offspring.append(child)
        
        return offspring
    
    def _crossover(self, body1: MechanicalBody, body2: MechanicalBody) -> MechanicalBody:
        """简化交叉: 交换子树"""
        child = body1.clone()
        
        # 随机选择 body2 中的一个零件, 复制它的参数到 child 对应位置
        parts2 = body2.parts()
        parts1 = child.parts()
        
        if parts2 and parts1:
            for key in parts1[0].params:
                if key in parts2[0].params and self.rng.random() < 0.5:
                    parts1[0].params[key] = parts2[0].params[key]
        
        return child
    
    def _mutate_body(self, body: MechanicalBody):
        """轻量变异"""
        for part_id in body.graph.nodes:
            part = body.get_part(part_id)
            for key in list(part.params.keys()):
                if isinstance(part.params[key], (int, float)):
                    if self.rng.random() < 0.3:
                        noise = self.rng.normal(0, 0.05)
                        part.params[key] *= (1 + noise)


# ══════════════════════════════════════════════════════════
#  OptimizerEmitter — 基于梯度的优化发射器
# ══════════════════════════════════════════════════════════

class OptimizerEmitter(BaseEmitter):
    """优化器发射器 — CMA-ES / CMA-ME 引导的采样

    实现 EmitterType.OPTIMIZER 语义:
      使用协方差矩阵自适应 (CMA-ES) 或 CMA-ME 分布
      来生成下一代候选解。

    当与 CVTArchive 配合使用时, CMA-ME 模式会
    偏向"可能改进存档"的区域采样。
    """

    def __init__(self, batch_size: int, rng: np.random.RandomState,
                 n_params: int = 20, method: str = "CMA-ME",
                 sigma0: float = 0.3):
        super().__init__(batch_size, rng)
        self.n_params = n_params
        self.method = method
        self.sigma0 = sigma0
        self._delegate = None
        self._generation = 0

    def _init_delegate(self, archive):
        """延迟初始化 CMA-ME 委托 (需要 archive)"""
        if self._delegate is not None:
            return
        try:
            from forgecraft.evolution.cma_me import OptimizerEmitter as CmaOptimizer
            self._delegate = CmaOptimizer(
                archive, self.n_params, method=self.method,
                sigma0=self.sigma0, rng=self.rng,
            )
        except ImportError:
            _logger.warning("CMA-ME not available, falling back to random")
            self.method = "simple"

    def ask(self, archive) -> List[MechanicalBody]:
        if self.method == "simple":
            return self._simple_ask(archive)

        self._init_delegate(archive)
        if self._delegate is None:
            return self._simple_ask(archive)

        try:
            if self.method == "CMA-ME":
                solutions, _ = self._delegate.ask(self.batch_size)
            else:
                solutions = self._delegate.ask(self.batch_size)

            # 将参数向量转换回 MechanicalBody
            # (简化: 克隆精英并修改参数)
            bodies = []
            elites = archive.get_all_elites() if hasattr(archive, 'get_all_elites') else []
            for i in range(min(len(solutions), self.batch_size)):
                if elites and i < len(elites):
                    body = elites[i].body.clone()
                else:
                    body = self._random_body()
                self._apply_params(body, solutions[i])
                bodies.append(body)
            return bodies
        except Exception as e:
            _logger.warning(f"CMA sampling failed: {e}")
            return self._simple_ask(archive)

    def tell(self, solutions: List[MechanicalBody], fitnesses: List[float],
             improvements: Optional[List[bool]] = None):
        """反馈优化器"""
        if self._delegate is not None and self.method in ("CMA-ME", "CMA-ES"):
            try:
                param_matrix = np.array([
                    self._extract_params(b) for b in solutions
                ])
                imp = improvements or [True] * len(solutions)
                self._delegate.tell(param_matrix, np.array(fitnesses), imp)
            except Exception as e:
                _logger.warning(f"CMA tell failed: {e}")

        self._generation += 1

    def _simple_ask(self, archive) -> List[MechanicalBody]:
        """回退: 简单随机采样"""
        elites = archive.get_all_elites()
        if not elites:
            return [self._random_body() for _ in range(self.batch_size)]

        bodies = []
        for _ in range(self.batch_size):
            elite = self.rng.choice(elites)
            body = elite.body.clone()
            # 参数扰动
            for part_id in body.graph.nodes:
                part = body.get_part(part_id)
                for key in list(part.params.keys()):
                    if isinstance(part.params[key], (int, float)):
                        part.params[key] += self.rng.normal(0, self.sigma0 * abs(part.params[key]) + 1e-6)
            bodies.append(body)
        return bodies

    def _random_body(self) -> MechanicalBody:
        """创建随机体 (占位)"""
        from forgecraft.core.morphology import MechanicalBody
        return MechanicalBody(name="random")

    def _extract_params(self, body: MechanicalBody) -> np.ndarray:
        """从 MechanicalBody 提取参数向量"""
        params = np.zeros(self.n_params)
        idx = 0
        for part_id in body.graph.nodes:
            part = body.get_part(part_id)
            for key in sorted(part.params.keys()):
                if idx < self.n_params:
                    val = part.params[key]
                    params[idx] = float(val) if isinstance(val, (int, float)) else 0.0
                    idx += 1
        return params

    def _apply_params(self, body: MechanicalBody, params: np.ndarray):
        """将参数向量应用到 MechanicalBody"""
        idx = 0
        for part_id in body.graph.nodes:
            part = body.get_part(part_id)
            for key in sorted(part.params.keys()):
                if idx < len(params) and isinstance(part.params[key], (int, float)):
                    part.params[key] = float(params[idx])
                    idx += 1


# ══════════════════════════════════════════════════════════
# 主类: MAP-Elites 引擎
# ══════════════════════════════════════════════════════════

class MAPElitesEngine:
    """
    MAP-Elites 进化引擎 (深度整合版)
    
    用法:
        engine = MAPElitesEngine(config)
        engine.initialize(generator_fn)
        
        for gen in range(max_gens):
            engine.step(evaluator_fn)
            stats = engine.get_stats()
            
            # 周期性注入主流
            if gen % inject_every == 0:
                elites = engine.get_injection_elites(n)
    """
    
    def __init__(self, config: MAPElitesConfig = None, seed: int = 42):
        self.config = config or MAPElitesConfig()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        bc_dim = len(self.config.bc_names)
        
        # 创建存档
        if self.config.use_cvt:
            self.archive = CVTArchive(
                num_centroids=self.config.archive_size,
                bc_dim=bc_dim,
                bc_ranges=list(self.config.bc_ranges),
                random_seed=seed,
            )
        else:
            # Grid: 每维约 archive_size^(1/d) 格
            grid_dim = int(self.config.archive_size ** (1.0 / bc_dim))
            grid_dim = max(2, grid_dim)
            self.archive = GridArchive(
                grid_dims=tuple([grid_dim] * bc_dim),
                bc_ranges=list(self.config.bc_ranges),
            )
        
        # 发射器列表
        self.emitters: List[BaseEmitter] = []
        self.generation = 0
        self.history: List[Dict] = []
        
    def initialize(self, generator_fn: Callable[[], MechanicalBody],
                   body_generator: Any = None):
        """
        初始化引擎
        
        Args:
            generator_fn: 随机形态生成函数
            body_generator: BodyGenerator 实例 (用于 AdaptiveEmitter)
        """
        # 改进发射器 (主要)
        for i in range(max(1, self.config.num_emitters - 2)):
            self.emitters.append(
                ImprovementEmitter(
                    batch_size=self.config.emitter_batch_size,
                    rng=np.random.RandomState(self.seed + i),
                    mutation_strength=0.08 + 0.02 * i,
                )
            )
        
        # 随机发射器 (维持探索)
        self.emitters.append(
            RandomEmitter(
                batch_size=self.config.emitter_batch_size,
                rng=np.random.RandomState(self.seed + 100),
                generator_fn=generator_fn,
            )
        )
        
        # 自适应发射器 (使用进化算子)
        self.emitters.append(
            AdaptiveEmitter(
                batch_size=self.config.emitter_batch_size,
                rng=np.random.RandomState(self.seed + 200),
                adaptive_controller=None,
                crossover_rate=0.3,
            )
        )
    
    def step(self, evaluator_fn: Callable,
             batch_size: int = None) -> Dict[str, Any]:
        """
        执行一步进化
        
        Args:
            evaluator_fn: 评估函数 (body) -> (fitness, behavior_descriptor, results)
            batch_size: 本步评估个体数
        
        Returns:
            本步统计信息
        """
        batch_size = batch_size or self.config.evaluations_per_generation
        bodies_per_emitter = max(1, batch_size // len(self.emitters))
        
        total_evaluated = 0
        
        for emitter in self.emitters:
            # 修改 batch_size
            emitter.batch_size = bodies_per_emitter
            
            # 从存档生成后代
            offspring = emitter.ask(self.archive)
            if not offspring:
                continue
            
            # 评估
            fitnesses = []
            bcs = []
            valid_bodies = []
            
            for body in offspring:
                try:
                    fitness, bc, _ = evaluator_fn(body)
                    fitnesses.append(fitness)
                    bcs.append(bc)
                    valid_bodies.append(body)
                    total_evaluated += 1
                except Exception:
                    continue
            
            # 更新存档
            if valid_bodies:
                emitter.tell(self.archive, valid_bodies, fitnesses, bcs,
                            self.generation)
        
        self.generation += 1
        
        stats = self.archive.get_stats()
        stats["generation"] = self.generation
        stats["total_evaluated_this_step"] = total_evaluated
        
        self.history.append(stats)
        return stats
    
    def get_injection_elites(self, n: int = 3) -> List[MechanicalBody]:
        """
        获取用于注入主流的精英
        
        选择策略: 从不同区域采样, 确保多样性
        """
        elites = self.archive.get_all_elites()
        if len(elites) <= n:
            return [e.body for e in elites]
        
        # 按适应度排序, 取 top-k
        sorted_elites = sorted(elites, key=lambda e: e.fitness, reverse=True)
        
        # 均匀采样 + 最佳个体
        selected = [sorted_elites[0]]  # 最佳
        if n > 1:
            # 剩余从不同适应度区间均匀取
            step = (len(sorted_elites) - 1) / (n - 1)
            for i in range(1, n):
                idx = int(i * step) + 1
                idx = min(idx, len(sorted_elites) - 1)
                selected.append(sorted_elites[idx])
        
        return [e.body.clone() for e in selected]
    
    def get_stats(self) -> Dict[str, Any]:
        return self.archive.get_stats()
    
    def get_qd_score_history(self) -> List[float]:
        return [h.get("qd_score", 0) for h in self.history]
    
    def get_coverage_history(self) -> List[float]:
        return [h.get("coverage", 0) for h in self.history]
    
    def save(self, path: str = None):
        path = path or self.config.archive_path
        data = {
            "archive": self.archive,
            "generation": self.generation,
            "history": self.history,
            "config": self.config,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> "MAPElitesEngine":
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        engine = cls(config=data["config"])
        engine.archive = data["archive"]
        engine.generation = data["generation"]
        engine.history = data["history"]
        return engine
    
    def print_summary(self):
        stats = self.get_stats()
        print("=" * 60)
        print("MAP-Elites 存档摘要")
        print("=" * 60)
        print(f"  代数: {self.generation}")
        print(f"  精英数: {stats['num_elites']}")
        print(f"  覆盖率: {stats['coverage']:.1%}")
        print(f"  QD-score: {stats['qd_score']:.2f}")
        print(f"  最佳适应度: {stats['best_fitness']:.4f}")
        print(f"  总评估次数: {stats['total_evaluations']}")
        print(f"  添加/替换: {stats['total_added']}/{stats['total_replaced']}")
        print("=" * 60)


# ══════════════════════════════════════════════════════════
# 行为特征计算
# ══════════════════════════════════════════════════════════

def compute_behavior_characteristics(
    body: MechanicalBody,
    episode_info: dict,
) -> np.ndarray:
    """
    从评估结果提取多维行为特征。
    
    Returns:
        行为特征向量 [speed, energy_efficiency, stability, symmetry, body_mass]
    """
    # 速度
    speed = episode_info.get("speed", 0.0)
    if isinstance(speed, (list, np.ndarray)):
        speed = float(np.mean(speed))
    
    # 能量效率 (位移 / 能耗, log scale)
    displacement = episode_info.get("displacement", 0.0)
    energy = episode_info.get("energy", 10.0)
    if isinstance(energy, (list, np.ndarray)):
        energy = float(np.sum(np.abs(energy)))
    efficiency = math.log(max(abs(displacement / max(abs(energy), 1e-6)), 1e-10))
    
    # 稳定性 (基于质心加速度方差)
    stability = episode_info.get("stability", 0.5)
    if isinstance(stability, (list, np.ndarray)):
        stability = float(1.0 / (1.0 + np.var(stability)))
    
    # 对称性 (左右零件数量差异)
    symmetry = _compute_symmetry(body)
    
    # 体重
    body_mass = _estimate_body_mass(body)
    
    return np.array([speed, efficiency, stability, symmetry, body_mass],
                    dtype=np.float32)


def _compute_symmetry(body: MechanicalBody) -> float:
    """计算机械体左右对称性"""
    left_count = 0
    right_count = 0
    
    for part in body.parts():
        if part.position[1] < -0.01:
            left_count += 1
        elif part.position[1] > 0.01:
            right_count += 1
    
    total = left_count + right_count
    if total == 0:
        return 1.0
    
    return 1.0 - abs(left_count - right_count) / total


def _estimate_body_mass(body: MechanicalBody) -> float:
    """估算机械体总质量"""
    total_mass = 0.0
    for part in body.parts():
        length = part.params.get("length", 0.1)
        width = part.params.get("width", part.params.get("thickness", 0.02))
        height = part.params.get("height", part.params.get("thickness", 0.02))
        volume = length * width * height
        density = 1000.0  # 默认密度
        total_mass += volume * density
    return max(total_mass, 0.01)


def compute_behavior_characteristics_with_objectives(
    body: MechanicalBody,
    fitness_components: Dict[str, float],
    episode_info: dict,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    结合适应度分量和行为特征
    
    Returns:
        (行为特征向量, 目标值字典)
    """
    bc = compute_behavior_characteristics(body, episode_info)
    
    objectives = {
        "speed": bc[0],
        "energy_efficiency": bc[1],
        "stability": bc[2],
        "symmetry": bc[3],
        "body_mass": bc[4],
    }
    # 合并传入的适应度分量
    objectives.update(fitness_components)
    
    return bc, objectives


# ══════════════════════════════════════════════════════════
# 与主流进化循环的集成接口
# ══════════════════════════════════════════════════════════

def create_map_elites_engine(
    config: MAPElitesConfig = None,
    seed: int = 42,
) -> MAPElitesEngine:
    """创建 MAP-Elites 引擎实例"""
    return MAPElitesEngine(config=config, seed=seed)


def inject_elites(
    population: List[MechanicalBody],
    engine: MAPElitesEngine,
    n: int = 3,
    replace_worst: bool = True,
) -> List[MechanicalBody]:
    """
    将 MAP-Elites 精英注入主流种群
    
    Args:
        population: 主流种群
        engine: MAP-Elites 引擎
        n: 注入数量
        replace_worst: 是否替换最差个体
    
    Returns:
        注入后的种群
    """
    elites = engine.get_injection_elites(n)
    if not elites:
        return population
    
    actual_n = len(elites)
    
    if replace_worst and len(population) > actual_n:
        # 按适应度排序, 移除最差的 n 个
        sorted_pop = sorted(population, key=lambda b: b.fitness)
        population = sorted_pop[actual_n:] + elites
    else:
        population = population + elites
    
    return population
