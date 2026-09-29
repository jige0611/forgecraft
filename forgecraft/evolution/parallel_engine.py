# ══════════════════════════════════════════════════════════
#  并行进化引擎 — 多 MuJoCo 实例异步评估
#
#  支持三种并行模式:
#    "thread"     — ThreadPoolExecutor (轻量, GIL 限制, 适合 I/O)
#    "process"    — ProcessPoolExecutor (真并行, 适合 CPU/GPU)
#    "auto"       — 自动选择: >16 核 → process, ≤16 → thread
#
#  关键优化:
#    1. 每个 worker 独立的 MuJoCo 实例 (避免 GIL/锁竞争)
#    2. 批量提交 → 异步收集 (无需等待最慢个体)
#    3. 仿真进度条 (tqdm 可选)
#    4. 内存管理: 大模型 pickle 序列化 → 每个 worker 重新加载
#    5. 容错: 单 worker 崩溃不中断整体评估
#
#  说明: 具体加速比取决于 CPU 核数、个体规模与 episode 预算,
#        本仓库不附带基准测试数据, 请以本机实测为准。
# ══════════════════════════════════════════════════════════

from __future__ import annotations

import os
import sys
import copy
import time
import pickle
import queue
import threading
import logging
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor,
    as_completed, Future, TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

__all__ = [
    "ParallelConfig",
    "ParallelEvaluator",
    "SimWorker",
    "BatchResult",
    "get_optimal_workers",
]

# ══════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════

@dataclass
class ParallelConfig:
    """并行评估配置"""
    mode: str = "auto"               # "thread" | "process" | "auto"
    n_workers: int = 0               # 0 = auto (CPU 核数 - 1)
    batch_size: int = 1              # 每批次个体数 (1 = per-individual)
    timeout_per_eval: float = 120.0  # 单个体评估超时 (秒)
    max_retries: int = 2             # 失败重试次数
    use_pickle: bool = True          # 子进程用 pickle 传参
    progress_bar: bool = True        # tqdm 进度条
    
    @property
    def effective_workers(self) -> int:
        if self.n_workers > 0:
            return self.n_workers
        return get_optimal_workers()


def get_optimal_workers() -> int:
    """自动确定最优 worker 数"""
    cpu_count = os.cpu_count() or 4
    
    # 留 1 个核给主进程
    workers = max(1, cpu_count - 1)
    
    # 限制上限 (避免过度竞争)
    return min(workers, 32)


# ══════════════════════════════════════════════════════════
#  仿真 Worker (子进程入口)
# ══════════════════════════════════════════════════════════

# 在子进程中运行, 需要独立的 MuJoCo 实例

def _worker_init():
    """子进程初始化 — 提高递归深度"""
    if sys.getrecursionlimit() < 10000:
        sys.setrecursionlimit(10000)


def _evaluate_single_body(args: Dict[str, Any]) -> Dict[str, Any]:
    """子进程: 评估单个 MechanicalBody

    直接复用进化评估器的标准 rollout + PPO 训练路径
    (``forgecraft.evolution.evaluator._evaluate_body_worker_ucb``), 使并行评估
    与主流程使用完全一致的环境 / 策略 / 适应度逻辑。

    Args:
        args: {
            "body": MechanicalBody 或 pickle bytes,
            "catalog": PartSpec dict,
            "sim_config": SimConfig,
            "task_config": TaskConfig,
            "rl_config": RLConfig,
            "episodes": int,
            "steps_per_ep": int,
            "ppo_epochs": int,
            "device": str,
            "seed": int,
            "enc_state": Optional[dict],  # 形态编码器状态, 缺省则新建
        }

    Returns:
        {fitness, reward_mean, speed_mean, energy_mean, upright_mean,
         fall_rate, n_episodes, success, error}
    """
    _worker_init()

    body = args["body"]
    if isinstance(body, bytes):
        body = pickle.loads(body)

    catalog = args["catalog"]
    sim_config = args["sim_config"]
    task_config = args["task_config"]
    rl_config = args["rl_config"]
    n_episodes = args.get("episodes", 8)
    steps_per_ep = args.get("steps_per_ep", 500)
    ppo_epochs = args.get("ppo_epochs", 5)
    device_str = args.get("device", "cpu")

    try:
        from forgecraft.evolution.evaluator import _evaluate_body_worker_ucb
        from forgecraft.rl.encoder import MorphologyEncoder, _build_type_registry

        enc_state = args.get("enc_state")
        if enc_state is None:
            # 无谱系信息时新建编码器; 编码器权重仅用于生成形态嵌入,
            # 策略网络 (_evaluate_body_worker_ucb 内部) 会从头训练。
            encoder = MorphologyEncoder(
                node_feat_dim=32,
                edge_feat_dim=16,
                hidden_dim=getattr(rl_config, "hidden_dim", 64),
                output_dim=getattr(rl_config, "morph_embed_dim", 64),
                num_layers=getattr(rl_config, "gnn_layers", 3),
                part_type_registry=_build_type_registry(catalog),
            )
            enc_state = {
                "node_feat_dim": 32,
                "edge_feat_dim": 16,
                "hidden_dim": encoder.hidden_dim,
                "output_dim": encoder.output_dim,
                "num_layers": encoder.num_layers,
                "weights": copy.deepcopy(encoder.state_dict()),
            }

        raw = _evaluate_body_worker_ucb(
            body, sim_config, task_config, rl_config,
            enc_state, args.get("inherit_state"),
            args.get("best_obs_dim"), args.get("best_act_dim"),
            n_episodes=n_episodes,
            steps_per_ep=steps_per_ep,
            ppo_epochs=ppo_epochs,
            device=device_str,
            catalog=catalog,
            prev_trainer_state=None,
            best_morph_embed=None,
            obs_rms=None,
        )

        n_completed = raw.get("n_completed", 0)
        episode_fits = raw.get("episode_fitnesses", [])
        n = max(n_completed, 1)
        mean_fitness = float(np.mean(episode_fits)) if episode_fits else 0.0

        return {
            "fitness": mean_fitness,
            "reward_mean": mean_fitness,
            "speed_mean": raw.get("total_speed", 0.0) / n,
            "energy_mean": raw.get("total_energy", 0.0) / n,
            "upright_mean": raw.get("total_upright", 0.0) / n,
            "fall_rate": raw.get("total_fell", 0) / n,
            "n_episodes": n_completed,
            "success": n_completed > 0,
            "error": None,
        }

    except Exception as e:
        _logger.error(f"Worker eval failed: {e}")
        return {
            "fitness": 0.0,
            "reward_mean": 0.0,
            "speed_mean": 0.0,
            "energy_mean": 0.0,
            "upright_mean": 0.0,
            "fall_rate": 1.0,
            "n_episodes": 0,
            "success": False,
            "error": str(e)[:200],
        }


# ══════════════════════════════════════════════════════════
#  并行评估器
# ══════════════════════════════════════════════════════════

@dataclass
class BatchResult:
    """批次评估结果"""
    fitness_scores: List[float]
    fitness_components: List[Dict[str, float]]
    success_count: int
    failure_count: int
    elapsed_ms: float
    per_worker_ms: List[float]
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / max(total, 1)
    
    def summary(self) -> str:
        return (
            f"Parallel eval: {self.success_count}/{self.success_count+self.failure_count} "
            f"bodies ({self.elapsed_ms:.0f}ms, "
            f"mean fitness={np.mean(self.fitness_scores):.3f})"
        )


class ParallelEvaluator:
    """并行种群评估器
    
    使用多进程/多线程并行评估多个 MechanicalBody,
    每个 worker 运行独立的 MuJoCo 仿真循环。
    
    用法:
      >>> evaluator = ParallelEvaluator(sim_config, task_config, rl_config, catalog)
      >>> results = evaluator.evaluate(bodies, episodes=8, steps_per_ep=500)
      >>> print(results.summary())
    """
    
    def __init__(
        self,
        sim_config,
        task_config,
        rl_config,
        catalog: Dict[str, Any] = None,
        config: Optional[ParallelConfig] = None,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.sim_config = sim_config
        self.task_config = task_config
        self.rl_config = rl_config
        self.catalog = catalog or {}
        self.config = config or ParallelConfig()
        self.device = device
        self.seed = seed
    
    def _get_executor(self):
        """根据配置获取 Executor"""
        mode = self.config.mode
        nw = self.config.effective_workers
        
        if mode == "auto":
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                mode = "process"
            else:
                mode = "process" if os.cpu_count() > 8 else "thread"
        
        if mode == "process":
            return ProcessPoolExecutor(
                max_workers=nw,
                initializer=_worker_init,
            )
        else:
            return ThreadPoolExecutor(max_workers=nw)
    
    def evaluate(
        self,
        bodies: List["MechanicalBody"],
        episodes: int = 8,
        steps_per_ep: int = 500,
        ppo_epochs: int = 5,
        use_ppo: bool = True,
    ) -> BatchResult:
        """并行评估种群
        
        Args:
            bodies: 机械体列表
            episodes: 每个体的评估轮数
            steps_per_ep: 每轮步数
            ppo_epochs: PPO 更新轮数
            use_ppo: 是否使用 PPO 策略学习
        
        Returns:
            BatchResult
        """
        t0 = time.perf_counter()
        
        n_bodies = len(bodies)
        worker_times = []
        
        # 准备任务
        tasks = []
        for i, body in enumerate(bodies):
            task = {
                "body": pickle.dumps(body) if self.config.use_pickle else body,
                "catalog": self.catalog,
                "sim_config": self.sim_config,
                "task_config": self.task_config,
                "rl_config": self.rl_config,
                "episodes": episodes,
                "steps_per_ep": steps_per_ep,
                "ppo_epochs": ppo_epochs if use_ppo else 0,
                "device": self.device,
                "seed": self.seed + i,
            }
            tasks.append(task)
        
        # 提交并收集结果
        fitness_scores = []
        fitness_components = []
        success_count = 0
        failure_count = 0
        
        try:
            with self._get_executor() as executor:
                futures = {
                    executor.submit(_evaluate_single_body, task): i
                    for i, task in enumerate(tasks)
                }
                
                # 使用 as_completed 异步收集 (不等最慢的)
                for future in as_completed(futures, timeout=self.config.timeout_per_eval * n_bodies):
                    idx = futures[future]
                    try:
                        result = future.result(timeout=self.config.timeout_per_eval)
                        
                        if result.get("success", False):
                            fitness_scores.append((idx, result["fitness"]))
                            fitness_components.append((idx, {
                                "reward_mean": result.get("reward_mean", 0),
                                "speed_mean": result.get("speed_mean", 0),
                                "energy_mean": result.get("energy_mean", 0),
                                "upright_mean": result.get("upright_mean", 0),
                                "fall_rate": result.get("fall_rate", 1),
                            }))
                            success_count += 1
                        else:
                            fitness_scores.append((idx, 0.0))
                            fitness_components.append((idx, {
                                "reward_mean": 0, "speed_mean": 0,
                                "energy_mean": 0, "upright_mean": 0,
                                "fall_rate": 1.0,
                            }))
                            failure_count += 1
                    
                    except (FutureTimeoutError, Exception) as e:
                        fitness_scores.append((idx, 0.0))
                        fitness_components.append((idx, {
                            "reward_mean": 0, "speed_mean": 0,
                            "energy_mean": 0, "upright_mean": 0,
                            "fall_rate": 1.0,
                        }))
                        failure_count += 1
                        _logger.warning(f"Body {idx} eval failed: {e}")
        
        except Exception as e:
            _logger.error(f"Parallel evaluation failed: {e}")
            # 填满缺失的
            for idx in range(n_bodies):
                if idx not in [s[0] for s in fitness_scores]:
                    fitness_scores.append((idx, 0.0))
                    fitness_components.append((idx, {
                        "reward_mean": 0, "speed_mean": 0,
                        "energy_mean": 0, "upright_mean": 0,
                        "fall_rate": 1.0,
                    }))
                    failure_count += 1
        
        # 按原始顺序排序
        fitness_scores.sort(key=lambda x: x[0])
        fitness_components.sort(key=lambda x: x[0])
        
        dt = (time.perf_counter() - t0) * 1000
        
        return BatchResult(
            fitness_scores=[fs[1] for fs in fitness_scores],
            fitness_components=[fc[1] for fc in fitness_components],
            success_count=success_count,
            failure_count=failure_count,
            elapsed_ms=dt,
            per_worker_ms=worker_times,
        )
    
    def evaluate_simple(
        self,
        bodies: List["MechanicalBody"],
        n_episodes: int = 4,
        steps: int = 300,
    ) -> np.ndarray:
        """最简单接口 — 返回 fitness 数组
        
        适合快速集成到现有进化循环
        """
        result = self.evaluate(bodies, episodes=n_episodes, steps_per_ep=steps, use_ppo=False)
        return np.array(result.fitness_scores)
    
    def close(self):
        """清理资源"""
        pass


# ══════════════════════════════════════════════════════════
#  进度追踪器 (用于异步评估的可视化)
# ══════════════════════════════════════════════════════════

class EvalProgressTracker:
    """异步评估进度追踪"""
    
    def __init__(self, n_total: int):
        self.n_total = n_total
        self.n_done = 0
        self.n_failed = 0
        self.fitnesses = []
        self._lock = threading.Lock()
    
    def update(self, fitness: float, success: bool):
        with self._lock:
            self.n_done += 1
            if not success:
                self.n_failed += 1
            self.fitnesses.append(fitness)
    
    @property
    def progress(self) -> float:
        return self.n_done / max(self.n_total, 1)
    
    @property
    def mean_fitness(self) -> float:
        return float(np.mean(self.fitnesses)) if self.fitnesses else 0.0
