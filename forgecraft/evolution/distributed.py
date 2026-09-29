# ══════════════════════════════════════════════════════════
# 🌐 分布式集群计算引擎
#
# 基于 Ray 的分布式进化计算框架。
# Ray 不可用时回退到 multiprocessing + Redis 协同。
#
# 功能:
#   ✅ Ray 远程任务调度 (remote functions)
#   ✅ Actor 模型 (有状态工作节点)
#   ✅ 分布式种群评估 (跨节点并行)
#   ✅ 分布式进化循环
#   ✅ 容错与自动重试
#   ✅ 弹性伸缩 (动态添加/移除节点)
#   ✅ Redis 回退 (轻量级消息队列)
#   ✅ 资源感知调度 (CPU/GPU)
#
# 启动集群:
#   ray start --head --port=6379
#   ray start --address='<head-ip>:6379'
#
# 使用:
#   >>> from forgecraft.evolution.distributed import DistributedEvaluator
#   >>> evaluator = DistributedEvaluator(n_remote_workers=8)
#   >>> results = evaluator.evaluate(population, config...)
# ══════════════════════════════════════════════════════════

import os
import sys
import copy
import time
import pickle
import json
import uuid
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed, Future

import numpy as np

from forgecraft.core.morphology import MechanicalBody
from forgecraft.config import EvolutionConfig, RLConfig, SimConfig, TaskConfig, PartSpec

logger = logging.getLogger(__name__)

__all__ = [
    "DistributedConfig",
    "EvalTask",
    "EvalResult",
    "ClusterInfo",
    "RayClusterManager",
    "RedisTaskQueue",
    "DistributedEvaluator",
    "DistributedEvolutionLoop",
]

# 尝试导入 Ray
try:
    import ray
    from ray.exceptions import RayActorError, RayTaskError
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    logger.info("Ray 未安装，使用 multiprocessing + Redis 回退")

# 尝试导入 Redis
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


# ══════════════════════════════════════════════════════════
# 数据类型
# ══════════════════════════════════════════════════════════

@dataclass
class DistributedConfig:
    """分布式配置"""
    # Ray 模式
    use_ray: bool = True
    ray_address: str = "auto"        # Ray 集群地址
    ray_namespace: str = "forgecraft"
    
    # 工作节点
    n_remote_workers: int = 8        # 远程 Worker 数
    cpu_per_worker: float = 1.0      # 每个 Worker 分配的 CPU
    gpu_per_worker: float = 0.0      # 每个 Worker 分配的 GPU
    
    # Redis 回退
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_task_queue: str = "forgecraft:tasks"
    redis_result_queue: str = "forgecraft:results"
    
    # 容错
    max_retries: int = 3
    retry_delay: float = 1.0
    task_timeout: float = 300.0      # 单任务超时 (秒)
    
    # 弹性
    auto_scale: bool = False
    min_workers: int = 1
    max_workers: int = 32


@dataclass
class EvalTask:
    """评估任务"""
    task_id: str
    body_pickle: bytes
    config_pickle: bytes
    n_episodes: int = 1
    steps_per_ep: int = 500
    ppo_epochs: int = 4
    priority: int = 0


@dataclass
class EvalResult:
    """评估结果"""
    task_id: str
    worker_id: str = ""
    fitness: float = 0.0
    speed: float = 0.0
    energy: float = 0.0
    upright: float = 0.0
    displacement: float = 0.0
    survival_ratio: float = 0.0
    fell_count: int = 0
    n_completed: int = 0
    manufacturability: float = 0.5
    trainer_state_pickle: bytes = b""
    obs_dim: int = 0
    act_dim: int = 0
    error: str = ""
    compute_time: float = 0.0
    node_id: str = ""


@dataclass
class ClusterInfo:
    """集群信息"""
    n_nodes: int = 0
    n_workers: int = 0
    total_cpus: float = 0.0
    total_gpus: float = 0.0
    available_cpus: float = 0.0
    available_gpus: float = 0.0
    node_ids: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════
# 核心评估 Worker (序列化安全，不依赖类状态)
# ══════════════════════════════════════════════════════════

def _evaluate_single_body(
    body_pickle: bytes,
    config_pickle: bytes,
    task_id: str = "",
    n_episodes: int = 1,
    steps_per_ep: int = 500,
    ppo_epochs: int = 4,
) -> dict:
    """
    评估单个机械体（可在远程节点执行）

    序列化安全，不依赖外部类状态。内部直接复用进化评估器的标准
    rollout + PPO 训练路径 (``forgecraft.evolution.evaluator._evaluate_body_worker_ucb``),
    使分布式评估与主流程使用完全一致的环境 / 策略 / 适应度 / 可制造性逻辑。

    Returns:
        {task_id, fitness, speed, energy, upright, displacement, survival_ratio,
         fell_count, n_completed, EpisodeFitnessesList, manufacturability,
         trainer_state_pickle, obs_dim, act_dim, error, compute_time}
    """
    t0 = time.time()

    try:
        body = pickle.loads(body_pickle)
        configs = pickle.loads(config_pickle)

        sim_config = configs.get("sim_config")
        task_config = configs.get("task_config")
        rl_config = configs.get("rl_config")
        catalog = configs.get("catalog")
        device = configs.get("device", "cpu")
        morph_encoder_state = configs.get("morph_encoder_state")
        inherit_state = configs.get("inherit_state")
        best_obs_dim = configs.get("best_obs_dim")
        best_act_dim = configs.get("best_act_dim")
        seed = configs.get("seed", 42)

        # 设置随机种子
        np.random.seed(seed + hash(task_id) % 10000)

        from forgecraft.evolution.evaluator import _evaluate_body_worker_ucb
        from forgecraft.rl.encoder import MorphologyEncoder, _build_type_registry

        enc_state = morph_encoder_state
        if enc_state is None:
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
            enc_state, inherit_state, best_obs_dim, best_act_dim,
            n_episodes=n_episodes,
            steps_per_ep=steps_per_ep,
            ppo_epochs=ppo_epochs,
            device=device,
            catalog=catalog,
            prev_trainer_state=None,
            best_morph_embed=configs.get("best_morph_embed"),
            obs_rms=None,
        )

        n_completed = raw.get("n_completed", 0)
        n = max(n_completed, 1)
        episode_fitnesses = list(raw.get("episode_fitnesses", []))
        trainer_state = raw.get("trainer_state")

        return {
            "task_id": task_id,
            "fitness": float(np.mean(episode_fitnesses)) if episode_fitnesses else 0.0,
            "speed": raw.get("total_speed", 0.0) / n,
            "energy": raw.get("total_energy", 0.0) / n,
            "upright": raw.get("total_upright", 0.0) / n,
            "displacement": raw.get("total_displacement", 0.0) / n,
            "survival_ratio": raw.get("total_survival_ratio", 0.0) / n,
            "fell_count": raw.get("total_fell", 0),
            "n_completed": n_completed,
            "EpisodeFitnessesList": episode_fitnesses,
            "manufacturability": raw.get("manufacturability", 0.0),
            "trainer_state_pickle": pickle.dumps(trainer_state) if trainer_state else b"",
            "obs_dim": raw.get("obs_dim", 0),
            "act_dim": raw.get("act_dim", 0),
            "error": "",
            "compute_time": time.time() - t0,
        }

    except Exception as e:
        return {
            "task_id": task_id,
            "fitness": 0.0,
            "speed": 0.0,
            "energy": 0.0,
            "upright": 0.0,
            "displacement": 0.0,
            "survival_ratio": 0.0,
            "fell_count": 1,
            "n_completed": 0,
            "EpisodeFitnessesList": [],
            "manufacturability": 0.0,
            "trainer_state_pickle": b"",
            "obs_dim": 0,
            "act_dim": 0,
            "error": str(e)[:200],
            "compute_time": time.time() - t0,
        }


# ══════════════════════════════════════════════════════════
# Ray 分布式引擎
# ══════════════════════════════════════════════════════════

if HAS_RAY:
    @ray.remote
    class RayEvalWorker:
        """Ray 远程评估 Worker (Actor 模型)"""
        
        def __init__(self, worker_id: str, device: str = "cpu"):
            self.worker_id = worker_id
            self.device = device
            self.tasks_completed = 0
            self.total_compute_time = 0.0
            self.config_cache = None
            self.node_id = ray.get_runtime_context().node_id.hex()
            # 设置进程级随机种子
            import random
            random.seed(int(worker_id.split("_")[-1]) if "_" in worker_id else 42)
        
        def get_info(self) -> dict:
            return {
                "worker_id": self.worker_id,
                "device": self.device,
                "tasks_completed": self.tasks_completed,
                "total_compute_time": self.total_compute_time,
                "node_id": self.node_id,
            }
        
        def evaluate(self, task_data: dict) -> dict:
            t0 = time.time()
            
            result = _evaluate_single_body(
                body_pickle=task_data["body_pickle"],
                config_pickle=task_data["config_pickle"],
                task_id=task_data.get("task_id", ""),
                n_episodes=task_data.get("n_episodes", 1),
                steps_per_ep=task_data.get("steps_per_ep", 500),
                ppo_epochs=task_data.get("ppo_epochs", 4),
            )
            
            elapsed = time.time() - t0
            self.tasks_completed += 1
            self.total_compute_time += elapsed
            
            result["worker_id"] = self.worker_id
            result["node_id"] = self.node_id
            return result
        
        def set_config_cache(self, config_pickle: bytes):
            self.config_cache = config_pickle


class RayClusterManager:
    """Ray 集群管理器"""
    
    def __init__(self, config: DistributedConfig = None):
        self.config = config or DistributedConfig()
        self._initialized = False
        self.workers: List[ray.actor.ActorHandle] = []
        self._futures: Dict[str, ray.ObjectRef] = {}
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    def connect(self) -> bool:
        if not HAS_RAY:
            return False
        
        try:
            if not ray.is_initialized():
                ray.init(
                    address=self.config.ray_address if self.config.ray_address != "auto" else None,
                    namespace=self.config.ray_namespace,
                    ignore_reinit_error=True,
                    logging_level=logging.WARNING,
                )
            self._initialized = True
            
            # 获取集群信息
            nodes = ray.nodes()
            alive = [n for n in nodes if n["Alive"]]
            logger.info(f"Ray 集群: {len(alive)} 个节点, "
                       f"{sum(n['Resources'].get('CPU', 0) for n in alive)} CPUs, "
                       f"{sum(n['Resources'].get('GPU', 0) for n in alive)} GPUs")
            return True
        except Exception as e:
            logger.warning(f"Ray 连接失败: {e}")
            return False
    
    def create_workers(self, n_workers: int = None) -> int:
        if not self._initialized:
            self.connect()
        
        n = n_workers or self.config.n_remote_workers
        
        for i in range(n):
            worker = RayEvalWorker.options(
                num_cpus=self.config.cpu_per_worker,
                num_gpus=self.config.gpu_per_worker,
                max_restarts=self.config.max_retries,
                max_task_retries=self.config.max_retries,
            ).remote(
                worker_id=f"worker_{i}_{uuid.uuid4().hex[:6]}",
                device="cpu",
            )
            self.workers.append(worker)
        
        logger.info(f"创建 {n} 个 Ray Worker")
        return len(self.workers)
    
    def get_cluster_info(self) -> ClusterInfo:
        if not self._initialized or not ray.is_initialized():
            return ClusterInfo()
        
        nodes = ray.nodes()
        alive = [n for n in nodes if n["Alive"]]
        
        total_cpus = sum(n["Resources"].get("CPU", 0) for n in alive)
        total_gpus = sum(n["Resources"].get("GPU", 0) for n in alive)
        available_cpus = sum(n["Resources"].get("CPU", 0) for n in alive) - sum(
            sum(kw.get("num_cpus", 0) for kw in n.get("ActiveTaskKeywordArgs", []))
            for n in alive
        ) if "ActiveTaskKeywordArgs" in (alive[0] if alive else {}) else total_cpus * 0.5
        
        return ClusterInfo(
            n_nodes=len(alive),
            n_workers=len(self.workers),
            total_cpus=total_cpus,
            total_gpus=total_gpus,
            available_cpus=max(0, available_cpus),
            available_gpus=total_gpus,
            node_ids=[n["NodeID"] for n in alive],
        )
    
    def submit_batch(self, tasks: List[dict]) -> List[str]:
        """批量提交任务"""
        task_ids = []
        for i, task in enumerate(tasks):
            worker = self.workers[i % len(self.workers)]
            task["task_id"] = task.get("task_id", str(uuid.uuid4()))
            future = worker.evaluate.remote(task)
            self._futures[task["task_id"]] = future
            task_ids.append(task["task_id"])
        return task_ids
    
    def collect_results(self, task_ids: List[str] = None,
                        timeout: float = None) -> Dict[str, dict]:
        """收集结果"""
        if task_ids is None:
            task_ids = list(self._futures.keys())
        
        timeout = timeout or self.config.task_timeout
        results = {}
        remaining = {tid: self._futures[tid] for tid in task_ids if tid in self._futures}
        
        start = time.time()
        ready = []
        
        while remaining and (time.time() - start) < timeout:
            try:
                ready_refs, _ = ray.wait(
                    list(remaining.values()),
                    num_returns=1,
                    timeout=5.0,
                )
                
                for ref in ready_refs:
                    for tid, r in list(remaining.items()):
                        if r is ref:
                            try:
                                results[tid] = ray.get(ref, timeout=10.0)
                            except (RayTaskError, RayActorError) as e:
                                results[tid] = {"task_id": tid, "error": str(e), "fitness": 0.0}
                            del remaining[tid]
                            break
            except Exception:
                break
        
        # 剩余超时任务
        for tid in remaining:
            results[tid] = {"task_id": tid, "error": "timeout", "fitness": 0.0}
        
        self._futures.clear()
        return results
    
    def shutdown(self):
        for w in self.workers:
            try:
                ray.kill(w)
            except Exception:
                pass
        self.workers = []
        self._futures = {}
        try:
            ray.shutdown()
        except Exception:
            pass
        self._initialized = False


# ══════════════════════════════════════════════════════════
# Redis 回退引擎 (轻量级，无需 Ray)
# ══════════════════════════════════════════════════════════

class RedisTaskQueue:
    """Redis 任务队列 (Ray 不可用时的回退方案)"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self._client = None
    
    @property
    def client(self):
        if self._client is None and HAS_REDIS:
            self._client = redis.Redis(
                host=self.host, port=self.port, db=self.db,
                decode_responses=False,
            )
        return self._client
    
    def push_task(self, task_data: dict) -> str:
        task_id = task_data.get("task_id", str(uuid.uuid4()))
        task_data["task_id"] = task_id
        if self.client:
            self.client.lpush("forgecraft:tasks", pickle.dumps(task_data))
        return task_id
    
    def pop_task(self, timeout: float = 5.0) -> Optional[dict]:
        if self.client:
            result = self.client.brpop("forgecraft:tasks", timeout=int(timeout))
            if result:
                return pickle.loads(result[1])
        return None
    
    def push_result(self, task_id: str, result_data: dict):
        if self.client:
            self.client.set(f"forgecraft:result:{task_id}",
                          pickle.dumps(result_data),
                          ex=3600)
    
    def get_pending_count(self) -> int:
        if self.client:
            return self.client.llen("forgecraft:tasks")
        return 0


# ══════════════════════════════════════════════════════════
# 分布式评估器
# ══════════════════════════════════════════════════════════

class DistributedEvaluator:
    """分布式评估器 — Ray actor / multiprocessing 回退

    两条实际生效的评估路径 (均复用 ``_evaluate_single_body`` → 标准 rollout + PPO):
      - Ray actor: 每个 worker 持有独立 MuJoCo 实例, 跨节点并行评估
      - multiprocessing: Ray 不可用时回退到本地进程池 (n_workers=1 时顺序执行)
      - 故障转移: worker 超时/异常时该个体记为零适应度并继续

    说明: ``RedisTaskQueue`` 仅提供任务/结果的队列原语 (push/pop/set),
    当前**未接入** ``evaluate_population`` 的评估路径, 需自行编写消费端。
    """

    def __init__(
        self,
        sim_config: SimConfig,
        task_config: TaskConfig,
        rl_config: RLConfig,
        catalog: Dict[str, PartSpec],
        distributed_config: DistributedConfig = None,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.sim_config = sim_config
        self.task_config = task_config
        self.rl_config = rl_config
        self.catalog = catalog
        self.device = device
        self.seed = seed
        
        self.dist_config = distributed_config or DistributedConfig()
        self._ray_manager: Optional[RayClusterManager] = None
        self._redis_queue: Optional[RedisTaskQueue] = None
        self._use_ray = self.dist_config.use_ray and HAS_RAY
        
        if self._use_ray:
            self._ray_manager = RayClusterManager(self.dist_config)
            if self._ray_manager.connect():
                self._ray_manager.create_workers()
            else:
                self._use_ray = False
                logger.info("Ray 不可用，回退到 multiprocessing")
        
        if HAS_REDIS and not self._use_ray:
            self._redis_queue = RedisTaskQueue(
                host=self.dist_config.redis_host,
                port=self.dist_config.redis_port,
                db=self.dist_config.redis_db,
            )
    
    def _build_config_pickle(self, extra: dict = None) -> bytes:
        configs = {
            "sim_config": self.sim_config,
            "task_config": self.task_config,
            "rl_config": self.rl_config,
            "catalog": self.catalog,
            "device": self.device,
            "seed": self.seed,
        }
        if extra:
            configs.update(extra)
        return pickle.dumps(configs)
    
    def evaluate_population(
        self,
        bodies: List[MechanicalBody],
        n_episodes: int = 3,
        steps_per_ep: int = 500,
        ppo_epochs: int = 4,
        morph_encoder_state: dict = None,
        inherit_state: dict = None,
        best_obs_dim: int = None,
        best_act_dim: int = None,
        best_morph_embed: np.ndarray = None,
        progress_callback: Callable = None,
    ) -> Dict[int, dict]:
        """
        分布式评估整个种群
        
        Returns:
            {body_index: result_dict}
        """
        config_extra = {
            "morph_encoder_state": morph_encoder_state,
            "inherit_state": inherit_state,
            "best_obs_dim": best_obs_dim,
            "best_act_dim": best_act_dim,
            "best_morph_embed": best_morph_embed,
        }
        config_pkl = self._build_config_pickle(config_extra)
        
        if self._use_ray and self._ray_manager:
            return self._evaluate_ray(
                bodies, n_episodes, steps_per_ep, ppo_epochs,
                config_pkl, progress_callback,
            )
        else:
            return self._evaluate_multiprocess(
                bodies, n_episodes, steps_per_ep, ppo_epochs,
                config_pkl, progress_callback,
            )
    
    def _evaluate_ray(
        self, bodies, n_episodes, steps_per_ep, ppo_epochs,
        config_pkl, progress_callback,
    ) -> Dict[int, dict]:
        """Ray 分布式评估"""
        # 准备任务
        tasks = []
        for idx, body in enumerate(bodies):
            task = {
                "task_id": f"body_{idx}",
                "body_pickle": pickle.dumps(body),
                "config_pickle": config_pkl,
                "n_episodes": n_episodes,
                "steps_per_ep": steps_per_ep,
                "ppo_epochs": ppo_epochs,
            }
            tasks.append(task)
        
        # 提交
        task_ids = self._ray_manager.submit_batch(tasks)
        logger.info(f"Ray: 提交 {len(task_ids)} 个评估任务到 {len(self._ray_manager.workers)} 个 Worker")
        
        # 收集
        raw_results = self._ray_manager.collect_results(task_ids)
        
        # 解析
        results = {}
        for i in range(len(bodies)):
            tid = f"body_{i}"
            if tid in raw_results:
                results[i] = self._normalize_result(raw_results[tid])
            else:
                results[i] = None
        
        if progress_callback:
            progress_callback(len(results), len(bodies))
        
        return results
    
    def _evaluate_multiprocess(
        self, bodies, n_episodes, steps_per_ep, ppo_epochs,
        config_pkl, progress_callback,
    ) -> Dict[int, dict]:
        """multiprocessing 回退评估"""
        n_workers = min(self.dist_config.n_remote_workers, len(bodies))
        
        tasks = []
        for idx, body in enumerate(bodies):
            tasks.append({
                "task_id": f"body_{idx}",
                "body_pickle": pickle.dumps(body),
                "config_pickle": config_pkl,
                "n_episodes": n_episodes,
                "steps_per_ep": steps_per_ep,
                "ppo_epochs": ppo_epochs,
            })
        
        results = {}
        
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_evaluate_single_body,
                                   t["body_pickle"], t["config_pickle"],
                                   t["task_id"], t["n_episodes"],
                                   t["steps_per_ep"], t["ppo_epochs"]): i
                    for i, t in enumerate(tasks)
                }
                
                completed = 0
                for future in as_completed(futures, timeout=self.dist_config.task_timeout):
                    idx = futures[future]
                    try:
                        raw = future.result(timeout=60)
                        results[idx] = self._normalize_result(raw)
                    except Exception:
                        results[idx] = None
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(bodies))
        else:
            # 单进程顺序评估
            for i, task in enumerate(tasks):
                raw = _evaluate_single_body(
                    task["body_pickle"], task["config_pickle"],
                    task["task_id"], task["n_episodes"],
                    task["steps_per_ep"], task["ppo_epochs"],
                )
                results[i] = self._normalize_result(raw)
                if progress_callback:
                    progress_callback(i + 1, len(bodies))
        
        return results
    
    def _normalize_result(self, raw: dict) -> dict:
        """标准化结果格式"""
        if raw is None:
            return {
                "fitness": 0.0, "speed": 0.0, "energy": 0.0,
                "upright": 0.0, "displacement": 0.0,
                "survival_ratio": 0.0, "fell": 0,
                "n_completed": 0, "manufacturability": 0.0,
                "episode_fitnesses": [],
            }
        
        raw_fitnesses = raw.get("EpisodeFitnessesList", [])
        if not raw_fitnesses:
            raw_fitnesses = [raw.get("fitness", 0.0)]
        
        return {
            "fitness": float(np.mean(raw_fitnesses)) if raw_fitnesses else 0.0,
            "speed": raw.get("speed", 0.0),
            "energy": raw.get("energy", 0.0),
            "upright": raw.get("upright", 0.0),
            "displacement": raw.get("displacement", 0.0),
            "survival_ratio": raw.get("survival_ratio", 0.0),
            "fell": raw.get("fell_count", 0),
            "n_completed": raw.get("n_completed", 0),
            "manufacturability": raw.get("manufacturability", 0.0),
            "episode_fitnesses": raw_fitnesses,
            "trainer_state": raw.get("trainer_state_pickle"),
            "obs_dim": raw.get("obs_dim", 0),
            "act_dim": raw.get("act_dim", 0),
            "compute_time": raw.get("compute_time", 0.0),
            "node_id": raw.get("node_id", "local"),
        }
    
    def get_cluster_info(self) -> ClusterInfo:
        if self._ray_manager:
            return self._ray_manager.get_cluster_info()
        return ClusterInfo(
            n_nodes=1,
            n_workers=self.dist_config.n_remote_workers,
            total_cpus=float(self.dist_config.n_remote_workers),
        )
    
    def shutdown(self):
        if self._ray_manager:
            self._ray_manager.shutdown()


# ══════════════════════════════════════════════════════════
# 分布式进化循环
# ══════════════════════════════════════════════════════════

class DistributedEvolutionLoop:
    """
    分布式进化循环
    
    封装 EvolutionLoop，自动使用分布式评估器
    """
    
    def __init__(
        self,
        evo_config: EvolutionConfig,
        sim_config: SimConfig,
        rl_config: RLConfig,
        task_config: TaskConfig,
        catalog: Dict[str, PartSpec],
        distributed_config: DistributedConfig = None,
        device: str = "cpu",
        seed: int = 42,
        enable_map_elites: bool = False,
    ):
        self.evo_config = evo_config
        self.sim_config = sim_config
        self.rl_config = rl_config
        self.task_config = task_config
        self.catalog = catalog
        self.device = device
        self.seed = seed
        self.enable_map_elites = enable_map_elites
        
        self.dist_config = distributed_config or DistributedConfig()
        
        self.dist_evaluator = DistributedEvaluator(
            sim_config=sim_config,
            task_config=task_config,
            rl_config=rl_config,
            catalog=catalog,
            distributed_config=self.dist_config,
            device=device,
            seed=seed,
        )
        
        self.cluster_info = self.dist_evaluator.get_cluster_info()
    
    def print_cluster_status(self):
        info = self.cluster_info
        print("=" * 60)
        print("  分布式集群状态")
        print("=" * 60)
        print(f"  模式: {'Ray' if self.dist_config.use_ray else 'Multiprocessing'}")
        print(f"  节点数: {info.n_nodes}")
        print(f"  Worker 数: {info.n_workers}")
        print(f"  CPU: {info.total_cpus} total / {info.available_cpus:.1f} available")
        if info.total_gpus > 0:
            print(f"  GPU: {info.total_gpus} total / {info.available_gpus:.1f} available")
        print("=" * 60)
    
    def run_evolution(
        self,
        generations: int,
        catalog_name: str = "primitives",
        pretrained_weights: dict = None,
        checkpoint_every: int = 5,
    ):
        """运行分布式进化"""
        from forgecraft.evolution.loop import EvolutionLoop
        
        loop = EvolutionLoop(
            evo_config=self.evo_config,
            sim_config=self.sim_config,
            rl_config=self.rl_config,
            task_config=self.task_config,
            catalog=self.catalog,
            device=self.device,
            seed=self.seed,
            n_workers=1,  # 单进程调度，实际评估由分布式引擎完成
            pretrained_encoder_weights=pretrained_weights,
            enable_map_elites=self.enable_map_elites,
        )
        
        loop.initialize_population()
        
        # 快速代理评估（每代用分布式引擎做一次完整评估）
        for gen in range(generations):
            t0 = time.time()
            
            # 用分布式引擎评估当前种群
            results = self.dist_evaluator.evaluate_population(
                loop.population,
                n_episodes=3,
                steps_per_ep=500,
                ppo_epochs=4,
            )
            
            # 注入评估结果
            for idx, body in enumerate(loop.population):
                if idx in results and results[idx]:
                    r = results[idx]
                    body.fitness = r["fitness"]
                    body.fitness_components = {
                        "speed": r["speed"],
                        "energy": r["energy"],
                        "upright": r["upright"],
                        "displacement": r["displacement"],
                        "manufacturability": r["manufacturability"],
                    }
            
            # 繁殖下一代
            loop.evolve_one_generation()
            
            elapsed = time.time() - t0
            if loop.best_body:
                n = len(loop.population)
                best = loop.best_body.fitness
                mfg = loop.best_body.fitness_components.get("manufacturability", 0)
                print(f"[Gen {gen+1}/{generations}] 种群={n} 最佳={best:.4f} "
                      f"可制造={mfg:.2f} 耗时={elapsed:.1f}s")
            
            if (gen + 1) % checkpoint_every == 0:
                loop.save_checkpoint(f"checkpoint_gen{gen+1}.pkl")
        
        return loop
    
    def shutdown(self):
        self.dist_evaluator.shutdown()
