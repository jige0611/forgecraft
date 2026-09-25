# ═══════════════════════════════════════════════════════════════
#  ForgeCraft 断点续跑 / 错误恢复系统
#  ═══════════════════════════════════════════════════════════════
#
#  功能：
#  - 自动保存检查点（每N代）
#  - 从检查点恢复进化
#  - 错误捕获和自动重试
#  - 实验状态持久化
#
#  使用：
#    from checkpoint import CheckpointManager
#    ckpt = CheckpointManager("v11_results")
#    ckpt.save(generation, population, best_individual)
#    state = ckpt.load_latest()
# ═══════════════════════════════════════════════════════════════

import json
import pickle
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger("ForgeCraft.Checkpoint")


@dataclass
class CheckpointState:
    """检查点状态数据"""
    generation: int = 0
    population: List[Dict] = None  # 序列化的种群
    best_individual: Dict = None   # 最佳个体
    best_fitness: float = 0.0
    history: List[Dict] = None     # 进化历史
    config: Dict = None             # 运行配置
    timestamp: str = ""
    
    def __post_init__(self):
        if self.population is None:
            self.population = []
        if self.history is None:
            self.history = []
        if self.timestamp == "":
            self.timestamp = datetime.now().isoformat()


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, results_dir: str = "v11_results", 
                 checkpoint_interval: int = 10,
                 max_checkpoints: int = 5):
        self.results_dir = Path(results_dir)
        self.checkpoint_dir = self.results_dir / "checkpoints"
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        
        # 确保目录存在
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_state = CheckpointState()
    
    @property
    def has_checkpoint(self) -> bool:
        """是否存在可恢复的检查点"""
        return len(list(self.checkpoint_dir.glob("ckpt_*.json"))) > 0
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """获取最新检查点文件"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("ckpt_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return checkpoints[0] if checkpoints else None
    
    def save(self, generation: int, population: list, 
              best_individual=None, best_fitness: float = 0.0,
              history: list = None, config: dict = None):
        """
        保存检查点
        
        Args:
            generation: 当前代数
            population: 当前种群（MechanicalBody列表）
            best_individual: 最佳个体
            best_fitness: 最佳适应度
            history: 进化历史记录
            config: 配置信息
        """
        try:
            # 序列化种群（简化：只保存关键属性）
            pop_data = []
            for body in population:
                body_data = {
                    "name": getattr(body, 'name', ''),
                    "fitness": getattr(body, 'fitness', 0),
                    "num_parts": body.num_parts() if hasattr(body, 'num_parts') else 0,
                }
                pop_data.append(body_data)
            
            # 序列化最佳个体
            best_data = None
            if best_individual:
                best_data = {
                    "name": getattr(best_individual, 'name', ''),
                    "fitness": best_fitness,
                    "type": getattr(best_individual, 'robot_type', 'unknown'),
                }
            
            state = CheckpointState(
                generation=generation,
                population=pop_data,
                best_individual=best_data,
                best_fitness=best_fitness,
                history=history or [],
                config=config or {},
                timestamp=datetime.now().isoformat()
            )
            
            # 保存JSON
            ckpt_path = self.checkpoint_dir / f"ckpt_gen{generation:04d}.json"
            with open(ckpt_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(state), f, indent=2, default=str)
            
            # 同时保存pickle版本（完整对象）
            pickle_path = self.checkpoint_dir / f"ckpt_gen{generation:04d}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump({
                    'generation': generation,
                    'population': population,
                    'best': best_individual,
                    'best_fitness': best_fitness,
                    'history': history,
                }, f)
            
            # 清理旧检查点
            self._cleanup_old_checkpoints()
            
            logger.info(f"Checkpoint saved: gen {generation} -> {ckpt_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load(self, checkpoint_path: str = None) -> Optional[CheckpointState]:
        """
        加载检查点
        
        Args:
            checkpoint_path: 指定路径，或使用最新的
            
        Returns:
            CheckpointState 或 None
        """
        try:
            if checkpoint_path:
                path = Path(checkpoint_path)
            else:
                path = self.get_latest_checkpoint()
            
            if not path or not path.exists():
                logger.warning("No checkpoint found")
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            state = CheckpointState(**data)
            logger.info(f"Loaded checkpoint: {path.name}, gen {state.generation}")
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def load_pickle(self, checkpoint_path: str = None) -> Optional[dict]:
        """加载pickle格式的完整检查点"""
        try:
            if checkpoint_path:
                path = Path(checkpoint_path).with_suffix('.pkl')
            else:
                json_ckpt = self.get_latest_checkpoint()
                if json_ckpt:
                    path = json_ckpt.with_suffix('.pkl')
                else:
                    return None
            
            if not path.exists():
                return None
            
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded pickle checkpoint: {path.name}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load pickle checkpoint: {e}")
            return None
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点，保留最近的N个"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("ckpt_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for old_ckpt in checkpoints[self.max_checkpoints:]:
            # 删除JSON和Pickle
            old_ckpt.unlink(missing_ok=True)
            old_ckpt.with_suffix('.pkl').unlink(missing_ok=True)
            logger.debug(f"Removed old checkpoint: {old_ckpt.name}")
    
    def get_resume_info(self) -> Dict[str, Any]:
        """获取恢复信息摘要"""
        latest = self.get_latest_checkpoint()
        
        if not latest:
            return {
                "can_resume": False,
                "message": "No checkpoints found"
            }
        
        state = self.load(latest)
        
        return {
            "can_resume": True,
            "checkpoint_file": str(latest),
            "generation": state.generation if state else 0,
            "best_fitness": state.best_fitness if state else 0,
            "timestamp": state.timestamp if state else "",
            "population_size": len(state.population) if state and state.population else 0,
        }
    
    def delete_all(self):
        """删除所有检查点"""
        for f in self.checkpoint_dir.glob("ckpt_*"):
            f.unlink()
        logger.info("All checkpoints deleted")


class ErrorRecovery:
    """错误恢复管理器"""
    
    MAX_RETRIES = 3
    RETRY_DELAYS = [1, 5, 15]  # 秒
    
    def __init__(self):
        self.error_log: List[Dict] = []
        self.retry_counts: Dict[str, int] = {}
    
    @staticmethod
    def retry_on_error(func, *args, max_retries=3, **kwargs):
        """
        带重试的函数执行
        
        Args:
            func: 要执行的函数
            max_retries: 最大重试次数
        """
        import time
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    delay = ErrorREcovery.RETRY_DELAYS[min(attempt, len(ErrorREcovery.RETRY_DELAYS)-1)]
                    logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        
        raise last_exception
    
    def log_error(self, context: str, error: Exception, severity: str = "error"):
        """记录错误"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity
        }
        self.error_log.append(entry)
        
        if severity == "error":
            logger.error(f"[{context}] {type(error).__name__}: {error}")
        elif severity == "warning":
            logger.warning(f"[{context}] {error}")
    
    def save_error_log(self, output_path: str):
        """保存错误日志到文件"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.error_log, f, indent=2, default=str)


def create_experiment_state(results_dir: str = "v11_results"):
    """
    创建实验状态跟踪器
    
    用于跟踪长时间运行的实验状态
    """
    
    class ExperimentState:
        def __init__(self, results_dir):
            self.results_dir = Path(results_dir)
            self.state_file = self.results_dir / ".experiment_state.json"
            self.state = {
                "status": "idle",  # idle, running, paused, completed, failed
                "started_at": None,
                "completed_at": None,
                "current_generation": 0,
                "total_generations": 0,
                "best_fitness": 0.0,
                "last_update": None,
                "pid": None,
            }
            self._load()
        
        def _load(self):
            if self.state_file.exists():
                with open(self.state_file) as f:
                    saved = json.load(f)
                    self.state.update(saved)
        
        def save(self):
            self.state["last_update"] = datetime.now().isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        
        def start(self, total_gens: int):
            self.state.update({
                "status": "running",
                "started_at": datetime.now().isoformat(),
                "total_generations": total_gens,
                "current_generation": 0,
            })
            self.save()
        
        def update(self, generation: int, best_fitness: float):
            self.state["current_generation"] = generation
            self.state["best_fitness"] = best_fitness
            self.save()
        
        def complete(self):
            self.state["status"] = "completed"
            self.state["completed_at"] = datetime.now().isoformat()
            self.save()
        
        def fail(self, error: str):
            self.state["status"] = "failed"
            self.state["error"] = error
            self.save()
        
        def is_running(self) -> bool:
            return self.state["status"] == "running"
        
        def get_progress(self) -> float:
            if self.state["total_generations"] == 0:
                return 0.0
            return self.state["current_generation"] / self.state["total_generations"]
    
    return ExperimentState(results_dir)


if __name__ == "__main__":
    # 测试检查点系统
    print("="*50)
    print("  Checkpoint System Test")
    print("="*50)
    
    ckpt = CheckpointManager("test_checkpoint_test", checkpoint_interval=2)
    
    # 模拟保存
    print("\nSaving test checkpoints...")
    for gen in [10, 20, 30]:
        ckpt.save(
            generation=gen,
            population=[],
            best_fitness=0.1 * gen,
            history=[]
        )
    
    # 测试加载
    info = ckpt.get_resume_info()
    print(f"\nResume Info:")
    print(f"  Can resume: {info['can_resume']}")
    print(f"  Generation: {info['generation']}")
    print(f"  Best Fitness: {info['best_fitness']}")
    
    # 清理
    import shutil
    if Path("test_checkpoint_test").exists():
        shutil.rmtree("test_checkpoint_test")
        print("\nCleaned up test files.")
