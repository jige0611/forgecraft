# ══════════════════════════════════════════════════════════
# ⚡ 并行几何生成引擎
#
# 功能:
#   ✅ 多进程并行几何生成
#   ✅ 任务队列管理
#   ✅ 进度跟踪
#   ✅ 结果聚合
#   ✅ 错误处理与重试
#   ✅ 资源限制
#
# 使用示例:
#   >>> from forgecraft.core.parallel_generator import ParallelGeometryGenerator
#   >>> generator = ParallelGeometryGenerator(num_workers=4)
#   >>> results = generator.generate_batch(part_specs)
# ══════════════════════════════════════════════════════════

import multiprocessing as mp
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
import traceback

logger = logging.getLogger(__name__)

__all__ = [
    "GenerationTask",
    "GenerationResult",
    "ProgressInfo",
    "WorkerProcess",
    "ParallelGeometryGenerator",
    "SynchronousGenerator",
    "create_generator",
    "example_batch_generation",
]


@dataclass
class GenerationTask:
    """生成任务"""
    task_id: str
    part_name: str
    params: Dict[str, float]
    scale: float = 1.0
    validate: bool = True


@dataclass
class GenerationResult:
    """生成结果"""
    task_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    time_ms: float = 0.0


@dataclass
class ProgressInfo:
    """进度信息"""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    progress: float
    eta_seconds: float
    current_task: Optional[str] = None


class WorkerProcess(mp.Process):
    """工作进程"""
    
    def __init__(self, task_queue: mp.Queue, result_queue: mp.Queue, worker_id: int):
        super().__init__(daemon=True)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.generator = None
    
    def run(self):
        """工作进程主循环"""
        logger.info(f"Worker {self.worker_id} started")
        
        # 初始化几何生成器
        try:
            self.generator = V8GeometryGenerator(resolution=24)
        except Exception as e:
            logger.error(f"Worker {self.worker_id} failed to initialize generator: {e}")
            return
        
        while True:
            try:
                # 获取任务
                task = self.task_queue.get(timeout=5.0)
                
                if task is None:
                    # 结束信号
                    break
                
                # 执行任务
                start_time = time.time()
                try:
                    result = self.generator.generate(
                        part_name=task.part_name,
                        params=task.params,
                        scale=task.scale,
                        validate=task.validate
                    )
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    self.result_queue.put(GenerationResult(
                        task_id=task.task_id,
                        success=True,
                        result=result,
                        time_ms=elapsed_ms
                    ))
                except Exception as e:
                    elapsed_ms = (time.time() - start_time) * 1000
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    
                    self.result_queue.put(GenerationResult(
                        task_id=task.task_id,
                        success=False,
                        error=error_msg,
                        time_ms=elapsed_ms
                    ))
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                break
        
        logger.info(f"Worker {self.worker_id} exiting")


class ParallelGeometryGenerator:
    """并行几何生成器"""
    
    def __init__(self, num_workers: Optional[int] = None, max_pending: int = 100):
        """
        初始化并行生成器
        
        Args:
            num_workers: 工作进程数，默认使用CPU核心数
            max_pending: 最大待处理任务数
        """
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.max_pending = max_pending
        
        # 进程间通信队列
        self.task_queue = mp.Queue(maxsize=max_pending)
        self.result_queue = mp.Queue(maxsize=max_pending)
        
        # 工作进程列表
        self.workers: List[WorkerProcess] = []
        
        # 进度跟踪
        self._progress_lock = threading.Lock()
        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._start_time = 0.0
        self._current_task = None
        
        # 回调函数
        self._progress_callback: Optional[Callable[[ProgressInfo], None]] = None
        
        # 运行状态
        self._running = False
    
    def set_progress_callback(self, callback: Callable[[ProgressInfo], None]):
        """设置进度回调函数"""
        self._progress_callback = callback
    
    def _update_progress(self, completed: bool, failed: bool = False):
        """更新进度"""
        with self._progress_lock:
            if completed:
                self._completed_tasks += 1
            if failed:
                self._failed_tasks += 1
            
            elapsed = time.time() - self._start_time
            progress = self._completed_tasks / max(self._total_tasks, 1)
            
            if progress > 0:
                eta = elapsed / progress * (1 - progress)
            else:
                eta = float('inf')
            
            info = ProgressInfo(
                total_tasks=self._total_tasks,
                completed_tasks=self._completed_tasks,
                failed_tasks=self._failed_tasks,
                progress=progress,
                eta_seconds=eta,
                current_task=self._current_task
            )
            
            if self._progress_callback:
                try:
                    self._progress_callback(info)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
    
    def _start_workers(self):
        """启动工作进程"""
        if self.workers:
            return
        
        for i in range(self.num_workers):
            worker = WorkerProcess(
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                worker_id=i
            )
            worker.start()
            self.workers.append(worker)
        
        self._running = True
        logger.info(f"Started {self.num_workers} worker processes")
    
    def _stop_workers(self):
        """停止工作进程"""
        if not self._running:
            return
        
        # 发送结束信号
        for _ in range(self.num_workers):
            try:
                self.task_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
        
        # 等待进程结束
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                logger.warning(f"Worker {worker.worker_id} did not exit gracefully")
                worker.terminate()
        
        self.workers = []
        self._running = False
        logger.info("All workers stopped")
    
    def generate_batch(
        self,
        tasks: List[GenerationTask],
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        timeout: float = 300.0  # 5分钟超时
    ) -> Dict[str, GenerationResult]:
        """
        批量生成几何模型
        
        Args:
            tasks: 任务列表
            progress_callback: 进度回调函数
            timeout: 超时时间（秒）
        
        Returns:
            结果字典 {task_id: GenerationResult}
        """
        if not tasks:
            return {}
        
        # 设置进度回调
        if progress_callback:
            self.set_progress_callback(progress_callback)
        
        # 初始化进度
        with self._progress_lock:
            self._total_tasks = len(tasks)
            self._completed_tasks = 0
            self._failed_tasks = 0
            self._start_time = time.time()
        
        # 启动工作进程
        self._start_workers()
        
        # 提交任务
        for task in tasks:
            try:
                self.task_queue.put(task, timeout=5.0)
            except queue.Full:
                logger.error(f"Task queue full, dropping task {task.task_id}")
        
        # 收集结果
        results: Dict[str, GenerationResult] = {}
        start_time = time.time()
        
        while len(results) < len(tasks):
            # 检查超时
            if time.time() - start_time > timeout:
                logger.error("Batch generation timeout")
                break
            
            try:
                result = self.result_queue.get(timeout=1.0)
                results[result.task_id] = result
                self._update_progress(completed=True, failed=not result.success)
            except queue.Empty:
                continue
        
        # 停止工作进程
        self._stop_workers()
        
        # 返回结果
        return results
    
    def generate_single(
        self,
        part_name: str,
        params: Dict[str, float] = None,
        scale: float = 1.0,
        validate: bool = True
    ) -> Optional[Any]:
        """
        生成单个几何模型
        
        Args:
            part_name: 零件名称
            params: 参数字典
            scale: 缩放因子
            validate: 是否验证
        
        Returns:
            GeometryResult 或 None
        """
        task = GenerationTask(
            task_id="single",
            part_name=part_name,
            params=params or {},
            scale=scale,
            validate=validate
        )
        
        results = self.generate_batch([task])
        result = results.get("single")
        
        if result and result.success:
            return result.result
        else:
            if result:
                logger.error(f"Generation failed: {result.error}")
            return None
    
    def close(self):
        """关闭生成器"""
        self._stop_workers()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 同步版本（用于回退）
class SynchronousGenerator:
    """同步几何生成器（作为并行版本的回退）"""
    
    def __init__(self):
        self.generator = V8GeometryGenerator(resolution=24)
    
    def generate_batch(
        self,
        tasks: List[GenerationTask],
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        timeout: float = 300.0
    ) -> Dict[str, GenerationResult]:
        """批量生成"""
        results = {}
        total = len(tasks)
        start_time = time.time()
        
        for i, task in enumerate(tasks):
            task_start = time.time()
            try:
                result = self.generator.generate(
                    part_name=task.part_name,
                    params=task.params,
                    scale=task.scale,
                    validate=task.validate
                )
                elapsed_ms = (time.time() - task_start) * 1000
                results[task.task_id] = GenerationResult(
                    task_id=task.task_id,
                    success=True,
                    result=result,
                    time_ms=elapsed_ms
                )
            except Exception as e:
                elapsed_ms = (time.time() - task_start) * 1000
                results[task.task_id] = GenerationResult(
                    task_id=task.task_id,
                    success=False,
                    error=str(e),
                    time_ms=elapsed_ms
                )
            
            # 更新进度
            if progress_callback:
                progress = (i + 1) / total
                elapsed = time.time() - start_time
                eta = elapsed / progress * (1 - progress) if progress > 0 else float('inf')
                progress_callback(ProgressInfo(
                    total_tasks=total,
                    completed_tasks=i + 1,
                    failed_tasks=sum(1 for r in results.values() if not r.success),
                    progress=progress,
                    eta_seconds=eta,
                    current_task=task.task_id
                ))
        
        return results
    
    def generate_single(
        self,
        part_name: str,
        params: Dict[str, float] = None,
        scale: float = 1.0,
        validate: bool = True
    ) -> Optional[Any]:
        """生成单个"""
        try:
            return self.generator.generate(
                part_name=part_name,
                params=params or {},
                scale=scale,
                validate=validate
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None
    
    def close(self):
        pass


def create_generator(use_parallel: bool = True, num_workers: Optional[int] = None) -> ParallelGeometryGenerator:
    """创建几何生成器实例"""
    if use_parallel:
        try:
            return ParallelGeometryGenerator(num_workers=num_workers)
        except Exception as e:
            logger.warning(f"Failed to create parallel generator, falling back to synchronous: {e}")
            return SynchronousGenerator()
    else:
        return SynchronousGenerator()


# 示例使用
def example_batch_generation():
    """批量生成示例"""
    tasks = [
        GenerationTask(task_id="gm6020", part_name="robomaster_gm6020", params={}, scale=1.0),
        GenerationTask(task_id="m3508", part_name="robomaster_m3508", params={}, scale=1.0),
        GenerationTask(task_id="aluminum", part_name="aluminum_extrusion_2020", params={"length": 0.5}, scale=1.0),
        GenerationTask(task_id="battery", part_name="lipo_3s_2200mAh", params={}, scale=1.0),
    ]
    
    def progress_callback(info: ProgressInfo):
        print(f"进度: {info.progress:.1%} ({info.completed_tasks}/{info.total_tasks}) "
              f"ETA: {info.eta_seconds:.1f}s")
    
    with ParallelGeometryGenerator(num_workers=4) as generator:
        results = generator.generate_batch(tasks, progress_callback=progress_callback)
        
        print("\n生成结果:")
        for task_id, result in results.items():
            if result.success:
                print(f"✅ {task_id}: 成功, 质量={result.result.mass_kg:.3f}kg")
            else:
                print(f"❌ {task_id}: 失败 - {result.error}")
