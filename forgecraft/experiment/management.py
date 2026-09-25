# ══════════════════════════════════════════════════════════
# 🧪 实验管理系统
#
# 功能:
#   ✅ 实验追踪 - 自动记录所有配置/参数/种子
#   ✅ 指标记录 - 每代适应度/多样性/种群统计
#   ✅ 断点快照 - 自动保存最佳形态 + 训练状态
#   ✅ 实验对比 - 多实验横向对比
#   ✅ 可复现性 - 完整记录环境信息 (git hash, 系统, 时间戳)
#   ✅ 导出报告 - JSON/CSV/YAML 格式
#   ✅ 实验恢复 - 从任意断点恢复
#
# 目录结构:
#   experiments/
#     <experiment_name>/
#       run_<timestamp>/
#         config.json       ← 完整配置快照
#         metrics.csv       ← 每代指标
#         checkpoints/      ← 断点文件
#         best_body.pkl     ← 最佳形态
#         report.json       ← 最终报告
#         metadata.json     ← 环境信息
#     experiments_index.json
#
# 用法:
#   python -m forgecraft.main --task walk --experiment-name my_run
#   python -m forgecraft.main --list-experiments
#   python -m forgecraft.main --compare exp1,exp2
# ══════════════════════════════════════════════════════════

import csv
import json
import logging
import os
import pickle
import platform
import shutil
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "collect_run_metadata",
    "ExperimentRecord",
    "ExperimentSummary",
    "ExperimentTracker",
    "ExperimentManager",
    "TrackedEvolutionLoop",
]


# ══════════════════════════════════════════════════════════
# 环境元数据收集
# ══════════════════════════════════════════════════════════

def _get_git_hash() -> str:
    """获取当前 git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    return "unknown"


def _get_git_branch() -> str:
    """获取当前 git branch"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_git_dirty() -> bool:
    """检查工作区是否有未提交的修改"""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True, timeout=5,
        )
        if result.returncode != 0:
            return True
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True, timeout=5,
        )
        return result.returncode != 0
    except Exception:
        return False


def collect_run_metadata(seed: int = 0, extra: Dict = None) -> Dict[str, Any]:
    """
    收集当前运行的完整环境信息
    
    Returns:
        包含系统/环境/版本等元数据的字典
    """
    import torch
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "timestamp_unix": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
        "git_hash": _get_git_hash(),
        "git_branch": _get_git_branch(),
        "git_dirty": _get_git_dirty(),
        "seed": seed,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    # NumPy 版本
    try:
        metadata["numpy_version"] = np.__version__
    except Exception:
        metadata["numpy_version"] = "unknown"
    
    # PyTorch 版本
    try:
        metadata["torch_version"] = torch.__version__
    except Exception:
        metadata["torch_version"] = "unknown"
    
    # Optuna 版本 (如果安装)
    try:
        import optuna
        metadata["optuna_version"] = optuna.__version__
    except ImportError:
        pass
    
    # Ray 版本 (如果安装)
    try:
        import ray
        metadata["ray_version"] = ray.__version__
    except ImportError:
        pass
    
    # 额外信息
    if extra:
        metadata["extra"] = extra
    
    return metadata


# ══════════════════════════════════════════════════════════
# 数据类
# ══════════════════════════════════════════════════════════

@dataclass
class ExperimentRecord:
    """单实验记录"""
    experiment_name: str
    run_id: str
    run_dir: str
    timestamp: str
    status: str = "running"  # running, completed, interrupted, error
    configs: Dict = field(default_factory=dict)
    metrics_history: List[Dict] = field(default_factory=list)
    best_fitness: float = 0.0
    best_generation: int = 0
    best_fitness_components: Dict = field(default_factory=dict)
    n_generations_completed: int = 0
    total_duration_seconds: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class ExperimentSummary:
    """实验摘要 (用于列表展示)"""
    run_id: str
    experiment_name: str
    timestamp: str
    status: str
    n_generations: int
    best_fitness: float
    duration_seconds: float
    config_summary: str = ""


# ══════════════════════════════════════════════════════════
# 实验追踪器
# ══════════════════════════════════════════════════════════

class ExperimentTracker:
    """
    单次实验追踪器
    
    在进化循环中记录每一代的指标，自动保存配置和断点。
    
    用法:
        tracker = ExperimentTracker(
            experiment_name="walk_v2",
            experiments_dir="experiments",
            seed=42,
        )
        tracker.start(evo_config, rl_config, sim_config, task_config)
        
        for gen in range(n_generations):
            # ... 进化一代 ...
            tracker.record_generation(
                generation=gen,
                best_fitness=0.85,
                mean_fitness=0.42,
                population_size=64,
                diversity=0.3,
                n_parts_list=[3, 5, 4, ...],
                extra={"map_elites_coverage": 0.25},
            )
        
        tracker.finish(best_body=best_body, status="completed")
    """
    
    def __init__(
        self,
        experiment_name: str = "default",
        experiments_dir: str = "experiments",
        seed: int = 42,
        auto_save_interval: int = 5,
    ):
        self.experiment_name = experiment_name
        self.experiments_dir = Path(experiments_dir)
        self.seed = seed
        self.auto_save_interval = auto_save_interval
        
        # 运行时状态
        self.run_id: str = ""
        self.run_dir: Path = None
        self._start_time: float = 0.0
        self._active: bool = False
        self._metrics_buffer: List[Dict] = []
        self._best_fitness: float = -float("inf")
        self._best_generation: int = 0
        self._best_body_bytes: bytes = b""
        self._metadata: Dict = {}
        self._config_snapshot: Dict = {}
    
    @property
    def is_active(self) -> bool:
        return self._active
    
    def start(
        self,
        evo_config = None,
        rl_config = None,
        sim_config = None,
        task_config = None,
        catalog_info: Dict = None,
        extra_info: Dict = None,
    ) -> str:
        """
        开始实验
        
        Returns:
            run_id: 本次运行的唯一标识
        """
        # 生成 run_id
        timestamp = datetime.now()
        self.run_id = timestamp.strftime("%Y%m%d_%H%M%S_") + f"{os.getpid()}"
        
        # 创建目录
        self.run_dir = self.experiments_dir / self.experiment_name / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        
        # 收集元数据
        self._metadata = collect_run_metadata(
            seed=self.seed,
            extra=extra_info,
        )
        
        # 保存配置快照
        self._config_snapshot = self._snapshot_configs(
            evo_config, rl_config, sim_config, task_config, catalog_info
        )
        
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self._config_snapshot, f, indent=2, ensure_ascii=False, default=str)
        
        with open(self.run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False, default=str)
        
        # 初始化 CSV
        self._init_metrics_csv()
        
        self._start_time = time.time()
        self._active = True
        
        return self.run_id
    
    def record_generation(
        self,
        generation: int,
        best_fitness: float = 0.0,
        mean_fitness: float = 0.0,
        median_fitness: float = 0.0,
        std_fitness: float = 0.0,
        population_size: int = 0,
        diversity: float = 0.0,
        n_parts_list: List[int] = None,
        n_species: int = 0,
        best_n_parts: int = 0,
        best_n_joints: int = 0,
        elapsed_seconds: float = 0.0,
        qud_score: float = 0.0,
        computation_cost: int = 0,
        extra: Dict = None,
    ) -> Dict:
        """
        记录一代的指标
        
        Returns:
            记录的指标字典
        """
        if not self._active:
            return {}
        
        if best_fitness > self._best_fitness:
            self._best_fitness = best_fitness
            self._best_generation = generation
        
        metrics = {
            "generation": generation,
            "best_fitness": float(best_fitness),
            "mean_fitness": float(mean_fitness),
            "median_fitness": float(median_fitness),
            "std_fitness": float(std_fitness),
            "population_size": population_size,
            "diversity": float(diversity),
            "n_species": n_species,
            "best_n_parts": best_n_parts,
            "best_n_joints": best_n_joints,
            "elapsed_cumulative": time.time() - self._start_time,
            "elapsed_generation": float(elapsed_seconds),
            "qud_score": float(qud_score),
            "computation_cost": computation_cost,
        }
        
        if n_parts_list:
            metrics["mean_n_parts"] = float(np.mean(n_parts_list)) if n_parts_list else 0.0
            metrics["max_n_parts"] = int(max(n_parts_list)) if n_parts_list else 0
            metrics["min_n_parts"] = int(min(n_parts_list)) if n_parts_list else 0
        
        if extra:
            for k, v in extra.items():
                if k not in metrics:
                    metrics[k] = v
        
        self._metrics_buffer.append(metrics)
        
        # 定时刷新到 CSV
        if len(self._metrics_buffer) >= self.auto_save_interval:
            self._flush_metrics()
        
        return metrics
    
    def save_checkpoint(self, state_dict: Dict, generation: int):
        """保存断点"""
        if not self._active:
            return
        
        path = self.run_dir / "checkpoints" / f"gen_{generation}.pkl"
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)
        
        # 保留最近 10 个断点
        checkpoints = sorted(
            (self.run_dir / "checkpoints").glob("gen_*.pkl"),
            key=lambda p: int(p.stem.split("_")[1])
        )
        for old in checkpoints[:-10]:
            old.unlink()
    
    def save_best_body(self, body, fitness: float, generation: int):
        """保存最佳形态"""
        if not self._active:
            return
        
        self._best_body_bytes = pickle.dumps(body)
        path = self.run_dir / "best_body.pkl"
        with open(path, "wb") as f:
            pickle.dump(body, f)
    
    def finish(
        self,
        best_body=None,
        status: str = "completed",
        error_message: str = "",
    ):
        """
        结束实验
        
        Args:
            best_body: 最佳形态 (MechanicalBody)
            status: completed, interrupted, error
        """
        if not self._active:
            return
        
        total_time = time.time() - self._start_time
        
        # 刷新剩余指标
        self._flush_metrics()
        
        # 保存最佳形态
        if best_body is not None:
            self.save_best_body(best_body, self._best_fitness, self._best_generation)
        
        # 生成报告
        report = self._generate_report(status, total_time, error_message)
        with open(self.run_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 更新实验索引
        self._update_index(report)
        
        self._active = False
    
    def get_record(self) -> ExperimentRecord:
        """获取实验记录"""
        return ExperimentRecord(
            experiment_name=self.experiment_name,
            run_id=self.run_id,
            run_dir=str(self.run_dir) if self.run_dir else "",
            timestamp=datetime.now().isoformat(),
            status="running" if self._active else "completed",
            configs=self._config_snapshot,
            metrics_history=list(self._metrics_buffer),
            best_fitness=self._best_fitness,
            best_generation=self._best_generation,
            metadata=self._metadata,
            n_generations_completed=len(self._metrics_buffer),
            total_duration_seconds=time.time() - self._start_time if self._active else 0.0,
        )
    
    # ── 内部方法 ──
    
    def _snapshot_configs(self, evo_config, rl_config, sim_config, task_config, catalog_info) -> Dict:
        """序列化配置为 JSON 兼容格式"""
        snapshot = {}
        
        if evo_config is not None:
            snapshot["evolution"] = {
                "population_size": getattr(evo_config, "population_size", None),
                "elite_count": getattr(evo_config, "elite_count", None),
                "mutation_rate": getattr(evo_config, "mutation_rate", None),
                "crossover_rate": getattr(evo_config, "crossover_rate", None),
                "topo_mutation_prob": getattr(evo_config, "topo_mutation_prob", None),
                "param_mutation_prob": getattr(evo_config, "param_mutation_prob", None),
                "param_mutation_scale": getattr(evo_config, "param_mutation_scale", None),
                "selection_pressure": getattr(evo_config, "selection_pressure", None),
            }
        
        if rl_config is not None:
            snapshot["rl"] = {
                "actor_lr": getattr(rl_config, "actor_lr", None),
                "critic_lr": getattr(rl_config, "critic_lr", None),
                "gamma": getattr(rl_config, "gamma", None),
                "lam": getattr(rl_config, "lam", None),
                "clip_ratio": getattr(rl_config, "clip_ratio", None),
                "entropy_coef": getattr(rl_config, "entropy_coef", None),
                "ppo_epochs": getattr(rl_config, "ppo_epochs", None),
                "hidden_dim": getattr(rl_config, "hidden_dim", None),
                "gnn_hidden": getattr(rl_config, "gnn_hidden", None),
                "morph_embed_dim": getattr(rl_config, "morph_embed_dim", None),
                "gnn_layers": getattr(rl_config, "gnn_layers", None),
            }
        
        if sim_config is not None:
            snapshot["simulation"] = {
                "timestep": getattr(sim_config, "timestep", None),
                "friction": getattr(sim_config, "friction", None),
                "substeps": getattr(sim_config, "substeps", None),
                "gravity": getattr(sim_config, "gravity", None),
            }
        
        if task_config is not None:
            snapshot["task"] = {
                "name": getattr(task_config, "name", None),
                "domain": getattr(task_config, "domain", None),
                "max_steps": getattr(task_config, "max_episode_steps", None),
            }
        
        if catalog_info:
            snapshot["catalog"] = catalog_info
        
        return snapshot
    
    def _init_metrics_csv(self):
        """初始化 CSV 文件头"""
        path = self.run_dir / "metrics.csv"
        fields = [
            "generation", "best_fitness", "mean_fitness", "median_fitness",
            "std_fitness", "population_size", "diversity", "n_species",
            "best_n_parts", "best_n_joints", "mean_n_parts",
            "max_n_parts", "min_n_parts",
            "elapsed_cumulative", "elapsed_generation",
            "qud_score", "computation_cost",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
    
    def _flush_metrics(self):
        """将缓冲的指标写入 CSV"""
        if not self._metrics_buffer:
            return
        
        path = self.run_dir / "metrics.csv"
        # 动态获取 CSV 的现有字段，以支持 extra 字段
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
        
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            for m in self._metrics_buffer:
                try:
                    writer.writerow(m)
                except ValueError:
                    # 如果字段不匹配，跳过 extra 字段
                    row = {k: v for k, v in m.items() if k in fields}
                    writer.writerow(row)
        
        self._metrics_buffer.clear()
    
    def _generate_report(self, status: str, total_time: float, error: str) -> Dict:
        """生成最终报告"""
        n_gens = len(self._metrics_buffer) if not self._metrics_buffer else max(
            (m.get("generation", 0) for m in self._metrics_buffer), default=0
        ) + 1
        
        fitnesses = [m.get("best_fitness", 0) for m in self._metrics_buffer] if self._metrics_buffer else []
        diversities = [m.get("diversity", 0) for m in self._metrics_buffer] if self._metrics_buffer else []
        
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "start_time": self._metadata.get("timestamp", ""),
            "end_time": datetime.now().isoformat(),
            "status": status,
            "error": error,
            "total_duration_seconds": round(total_time, 2),
            "total_duration_human": f"{total_time/60:.1f}min" if total_time < 3600 else f"{total_time/3600:.1f}h",
            "n_generations": n_gens,
            "n_metrics_records": len(self._metrics_buffer),
            "best_fitness": round(float(self._best_fitness), 6),
            "best_generation": self._best_generation,
            "final_fitness": round(float(fitnesses[-1]), 6) if fitnesses else 0.0,
            "fitness_improvement": round(float(fitnesses[-1] - fitnesses[0]), 6) if len(fitnesses) >= 2 else 0.0,
            "fitness_variance_final": round(float(np.var(fitnesses[-10:])), 6) if len(fitnesses) >= 10 else 0.0,
            "mean_diversity": round(float(np.mean(diversities)), 4) if diversities else 0.0,
            "diversity_final": round(float(diversities[-1]), 4) if diversities else 0.0,
            "hostname": self._metadata.get("hostname", ""),
            "git_hash": self._metadata.get("git_hash", "unknown"),
            "git_branch": self._metadata.get("git_branch", "unknown"),
            "seed": self.seed,
            "platform": self._metadata.get("platform", ""),
            "gpu": self._metadata.get("gpu_name", "N/A"),
            "config_summary": f"pop={self._config_snapshot.get('evolution',{}).get('population_size','?')} "
                            f"mut={self._config_snapshot.get('evolution',{}).get('mutation_rate','?')}",
        }
    
    def _update_index(self, report: Dict) -> None:
        """更新实验索引文件（原子写入 + 绝对路径）"""
        import tempfile
        
        index_path = self.experiments_dir / "experiments_index.json"
        
        # 读取现有索引（加文件锁防止并发竞争）
        index: list = []
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
            except (json.JSONDecodeError, Exception):
                index = []
        
        # 避免重复
        index = [e for e in index if e.get("run_id") != self.run_id]
        
        # 使用 resolve() 确保绝对路径（防止 working dir 变化）
        run_dir_abs = str(self.run_dir.resolve())
        
        index.append({
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "timestamp": report.get("start_time", ""),
            "status": report.get("status", ""),
            "best_fitness": report.get("best_fitness", 0),
            "n_generations": report.get("n_generations", 0),
            "duration_seconds": report.get("total_duration_seconds", 0),
            "config_summary": report.get("config_summary", ""),
            "run_dir": run_dir_abs,
        })
        
        # 保留最近 500 条
        index = sorted(
            index,
            key=lambda x: x.get("timestamp", ""),
            reverse=True,
        )[:500]
        
        # 原子写入: tmp → fsync → rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".json", dir=str(self.experiments_dir.resolve())
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(index_path.resolve()))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        # 更新特定实验的索引（也使用绝对路径 + 原子写入）
        exp_index_path = self.experiments_dir / self.experiment_name / "runs_index.json"
        exp_dir = exp_index_path.parent
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        exp_runs = [e for e in index if e.get("experiment_name") == self.experiment_name]
        tmp_fd2, tmp_path2 = tempfile.mkstemp(
            suffix=".json", dir=str(exp_dir.resolve())
        )
        try:
            with os.fdopen(tmp_fd2, "w", encoding="utf-8") as f:
                json.dump(exp_runs, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path2, str(exp_index_path.resolve()))
        finally:
            if os.path.exists(tmp_path2):
                os.unlink(tmp_path2)


# ══════════════════════════════════════════════════════════
# 实验管理器
# ══════════════════════════════════════════════════════════

class ExperimentManager:
    """
    多实验管理器
    
    功能: 列表/加载/对比/删除实验
    """
    
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
    
    def list_experiments(self, n_recent: int = 20) -> List[ExperimentSummary]:
        """列出最近的实验"""
        index_path = self.experiments_dir / "experiments_index.json"
        if not index_path.exists():
            return []
        
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        
        summaries = []
        for entry in index[:n_recent]:
            summaries.append(ExperimentSummary(
                run_id=entry.get("run_id", ""),
                experiment_name=entry.get("experiment_name", ""),
                timestamp=entry.get("timestamp", ""),
                status=entry.get("status", "unknown"),
                n_generations=entry.get("n_generations", 0),
                best_fitness=entry.get("best_fitness", 0.0),
                duration_seconds=entry.get("duration_seconds", 0.0),
                config_summary=entry.get("config_summary", ""),
            ))
        return summaries
    
    def load_experiment(self, run_id: str) -> Optional[ExperimentRecord]:
        """加载实验记录"""
        # 在索引中查找
        index_path = self.experiments_dir / "experiments_index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            
            for entry in index:
                if entry.get("run_id") == run_id:
                    return self._load_from_dir(entry.get("run_dir", ""))
        
        # 遍历目录查找
        for exp_dir in self.experiments_dir.iterdir():
            if not (exp_dir / "runs_index.json").exists():
                continue
            with open(exp_dir / "runs_index.json", "r", encoding="utf-8") as f:
                runs = json.load(f)
            for run in runs:
                if run.get("run_id") == run_id:
                    return self._load_from_dir(run.get("run_dir", ""))
        
        return None
    
    def compare_experiments(self, run_ids: List[str]) -> Dict[str, Any]:
        """对比多个实验（带重试）"""
        records = {}
        for rid in run_ids:
            rec = self.load_experiment(rid)
            if rec:
                records[rid] = rec
        
        if len(records) < 2:
            # 重试一次（处理文件系统延迟）
            import time
            time.sleep(0.3)
            for rid in run_ids:
                if rid not in records:
                    rec = self.load_experiment(rid)
                    if rec:
                        records[rid] = rec
        
        if len(records) < 2:
            return {"error": "需要至少 2 个有效实验", "records": records}
        
        # 提取对比数据
        fitness_curves = {}
        diversity_curves = {}
        stats = {}
        
        for rid, rec in records.items():
            label = f"{rec.experiment_name}/{rid[:12]}"
            
            # 适应度曲线
            if rec.metrics_history:
                fitness_curves[label] = [
                    m.get("best_fitness", 0.0) for m in rec.metrics_history
                ]
                diversity_curves[label] = [
                    m.get("diversity", 0.0) for m in rec.metrics_history
                ]
            
            # 汇总统计
            stats[label] = {
                "best_fitness": rec.best_fitness,
                "best_generation": rec.best_generation,
                "n_generations": rec.n_generations_completed,
                "duration_minutes": round(rec.total_duration_seconds / 60, 1),
                "status": rec.status,
                "seed": rec.metadata.get("seed", "N/A"),
                "git_hash": rec.metadata.get("git_hash", "unknown"),
                "config": rec.configs.get("evolution", {}),
            }
        
        # 找到最佳实验
        best_label = max(stats.keys(), key=lambda k: stats[k]["best_fitness"])
        
        return {
            "n_experiments": len(records),
            "fitness_curves": fitness_curves,
            "diversity_curves": diversity_curves,
            "stats": stats,
            "best_experiment": best_label,
            "best_fitness": stats[best_label]["best_fitness"],
        }
    
    def get_experiment_names(self) -> List[str]:
        """获取所有实验名称"""
        if not self.experiments_dir.exists():
            return []
        return [
            d.name for d in self.experiments_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("_")
        ]
    
    def remove_experiment(self, run_id: str) -> bool:
        """删除实验记录"""
        rec = self.load_experiment(run_id)
        if rec and rec.run_dir and os.path.exists(rec.run_dir):
            shutil.rmtree(rec.run_dir)
            
            # 更新索引
            index_path = self.experiments_dir / "experiments_index.json"
            if index_path.exists():
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                index = [e for e in index if e.get("run_id") != run_id]
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(index, f, indent=2, ensure_ascii=False)
            return True
        return False
    
    def print_experiments_table(self, n_recent: int = 20):
        """打印实验列表"""
        experiments = self.list_experiments(n_recent)
        
        if not experiments:
            print("(暂无实验记录)")
            return
        
        print("=" * 90)
        print(f"  实验记录 (最近 {len(experiments)} 条)")
        print("=" * 90)
        print(f"  {'ID':<20} {'状态':<6} {'世代':<5} {'最佳适应度':<12} {'耗时':<10} {'名称'}")
        print("-" * 90)
        
        for exp in experiments:
            rid_short = exp.run_id[:18] if len(exp.run_id) > 18 else exp.run_id
            status_icon = {"completed": "✔", "running": "▶", "interrupted": "⏸", "error": "✘"}.get(exp.status, "?")
            
            mins = exp.duration_seconds / 60
            if mins < 1:
                dur_str = f"{exp.duration_seconds:.0f}s"
            elif mins < 60:
                dur_str = f"{mins:.1f}m"
            else:
                dur_str = f"{mins/60:.1f}h"
            
            print(f"  {rid_short:<20} {status_icon:<6} {exp.n_generations:<5} "
                  f"{exp.best_fitness:<12.4f} {dur_str:<10} {exp.experiment_name}")
        
        print("=" * 90)
    
    def print_comparison(self, run_ids: List[str]):
        """打印实验对比"""
        result = self.compare_experiments(run_ids)
        
        if "error" in result:
            print(f"错误: {result['error']}")
            return
        
        print("=" * 70)
        print(f"  实验对比 ({result['n_experiments']} 个实验)")
        print("=" * 70)
        
        # 表头
        print(f"  {'实验':<25} {'最佳适应度':<14} {'最佳世代':<10} {'代':<6} {'耗时':<10} {'种'}")
        print("-" * 70)
        
        best_name = result["best_experiment"]
        for label, stat in result["stats"].items():
            marker = " ★" if label == best_name else "  "
            mins = stat["duration_minutes"]
            dur_str = f"{mins:.1f}min" if mins < 60 else f"{mins/60:.1f}h"
            
            print(f"{marker} {label:<23} {stat['best_fitness']:<14.4f} "
                  f"{stat['best_generation']:<10} {stat['n_generations']:<6} "
                  f"{dur_str:<10} {stat['seed']}")
        
        print("-" * 70)
        print(f"  ★ 最佳: {best_name} (适应度={result['best_fitness']:.4f})")
        print("=" * 70)
    
    def _load_from_dir(self, run_dir: str) -> Optional[ExperimentRecord]:
        """从目录加载实验记录"""
        run_path = Path(run_dir)
        if not run_path.exists():
            return None
        
        # 解析 run_id
        run_id = run_path.name.replace("run_", "")
        
        # 实验名称 = 上级目录名
        experiment_name = run_path.parent.name if run_path.parent != self.experiments_dir else "unknown"
        
        # 加载配置
        configs = {}
        config_path = run_path / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                configs = json.load(f)
        
        # 加载元数据
        metadata = {}
        meta_path = run_path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        
        # 加载报告
        report = {}
        report_path = run_path / "report.json"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        
        # 加载指标
        metrics = []
        metrics_path = run_path / "metrics.csv"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 自动转换数值字段
                    for key in row:
                        try:
                            row[key] = float(row[key])
                            if row[key] == int(row[key]):
                                row[key] = int(row[key])
                        except (ValueError, TypeError):
                            pass
                    metrics.append(row)
        
        return ExperimentRecord(
            experiment_name=experiment_name,
            run_id=run_id,
            run_dir=str(run_path),
            timestamp=metadata.get("timestamp", ""),
            status=report.get("status", "unknown"),
            configs=configs,
            metrics_history=metrics,
            best_fitness=report.get("best_fitness", 0.0),
            best_generation=report.get("best_generation", 0),
            n_generations_completed=report.get("n_generations", 0),
            total_duration_seconds=report.get("total_duration_seconds", 0.0),
            metadata=metadata,
        )


# ══════════════════════════════════════════════════════════
# 进化循环包装器 (简化集成)
# ══════════════════════════════════════════════════════════

class TrackedEvolutionLoop:
    """
    带实验追踪的进化循环
    
    用法:
        tracked = TrackedEvolutionLoop(loop, experiment_name="walk_v2")
        tracked.run(n_generations=100)
    """
    
    def __init__(
        self,
        loop,
        experiment_name: str = "default",
        experiments_dir: str = "experiments",
    ):
        self.loop = loop
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            experiments_dir=experiments_dir,
            seed=getattr(loop, "seed", 42),
        )
    
    def run(self, n_generations: int, save_best: bool = True,
            on_generation_complete: callable = None):
        """运行带追踪的进化"""
        loop = self.loop
        
        # 提取 catalog 信息
        catalog_info = {}
        if hasattr(loop, "catalog") and loop.catalog:
            catalog_info = {"n_parts": len(loop.catalog)}
        
        # 开始追踪
        self.tracker.start(
            evo_config=getattr(loop, "evo_config", None),
            rl_config=getattr(loop, "rl_config", None),
            sim_config=getattr(loop, "sim_config", None),
            task_config=getattr(loop, "task_config", None),
            catalog_info=catalog_info,
            extra_info={"population_init_size": len(loop.population) if hasattr(loop, "population") else 0},
        )
        
        try:
            # 如果还没初始化种群
            if not hasattr(loop, "population") or not loop.population:
                loop.initialize_population()
            
            # 设置循环状态
            loop._is_final_gen = False
            total_gens = getattr(loop.evo_config, "generations", n_generations)
            
            # 应用课程学习
            if hasattr(loop, "_apply_curriculum"):
                loop._apply_curriculum()
            
            for gen in range(n_generations):
                t0 = time.time()
                
                # 设置进化状态 (与 run() 保持一致)
                loop.generation = gen
                gens_remaining = n_generations - gen
                loop._is_final_gen = (gens_remaining <= 2)
                
                # 应用课程
                if gen > 0 and hasattr(loop, "_apply_curriculum"):
                    loop._apply_curriculum()
                
                # 进化一代
                loop.evolve_one_generation()
                
                # 收集指标
                gen_time = time.time() - t0
                
                body = loop.best_body
                best_fit = body.fitness if body else 0.0
                n_parts = body.num_parts() if body else 0
                n_joints = body.num_joints() if body else 0
                
                # 种群统计
                population = getattr(loop, "population", [])
                fitnesses = [b.fitness for b in population] if population else [0.0]
                mean_fit = float(np.mean(fitnesses)) if fitnesses else 0.0
                median_fit = float(np.median(fitnesses)) if fitnesses else 0.0
                std_fit = float(np.std(fitnesses)) if fitnesses else 0.0
                
                n_parts_list = [b.num_parts() for b in population] if population else []
                
                # 从 history 获取额外信息
                stats = loop.history[-1] if loop.history else {}
                qud_score = stats.get("qd_score", stats.get("archive_coverage", 0.0))
                diversity = stats.get("diversity", stats.get("species_count", float(std_fit)))
                n_species = stats.get("n_species", stats.get("species_count", 0))
                
                if isinstance(diversity, (list, tuple)):
                    diversity = float(np.mean(diversity)) if diversity else 0.0
                
                self.tracker.record_generation(
                    generation=gen,
                    best_fitness=best_fit,
                    mean_fitness=mean_fit,
                    median_fitness=median_fit,
                    std_fitness=std_fit,
                    population_size=len(population),
                    diversity=float(diversity),
                    n_parts_list=n_parts_list,
                    n_species=int(n_species),
                    best_n_parts=n_parts,
                    best_n_joints=n_joints,
                    elapsed_seconds=gen_time,
                    qud_score=float(qud_score) if qud_score else 0.0,
                    extra={
                        "mode": stats.get("mode", "standard"),
                        "n_elites": stats.get("n_elites", 0),
                    },
                )
                
                # 定期保存断点
                if (gen + 1) % 5 == 0:
                    try:
                        checkpoint = {
                            "generation": gen + 1,
                            "population": pickle.dumps(population),
                            "history": loop.history,
                        }
                        self.tracker.save_checkpoint(checkpoint, gen + 1)
                    except Exception:
                        pass

                # 实时仪表盘回调
                if on_generation_complete:
                    try:
                        on_generation_complete(loop)
                    except Exception:
                        pass
            
            # 保存最佳形态
            if save_best and loop.best_body:
                self.tracker.save_best_body(
                    loop.best_body,
                    loop.best_body.fitness,
                    getattr(loop, "best_generation", 0),
                )
            
            self.tracker.finish(best_body=loop.best_body, status="completed")
            
        except KeyboardInterrupt:
            self.tracker.finish(
                best_body=getattr(loop, "best_body", None),
                status="interrupted",
            )
            raise
        except Exception as e:
            self.tracker.finish(
                best_body=getattr(loop, "best_body", None),
                status="error",
                error_message=str(e),
            )
            raise
