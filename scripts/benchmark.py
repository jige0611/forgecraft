# ═══════════════════════════════════════════════════════════════
#  ForgeCraft 性能基准测试套件
#  ═══════════════════════════════════════════════════════════════
#
#  功能：
#  - 模板生成基准（吞吐量、延迟）
#  - 模型构建基准（编译时间、内存）
#  - 评估基准（仿真速度、GPU利用率）
#  - 进化基准（收敛速度、多样性保持）
#  - 历史记录和趋势对比
#
#  使用：
#    from benchmark import BenchmarkSuite
#    suite = BenchmarkSuite()
#    results = suite.run_all()
#    suite.print_report()
# ═══════════════════════════════════════════════════════════════

import time
import json
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger("ForgeCraft.Benchmark")


@dataclass 
class BenchmarkResult:
    """单个基准测试结果"""
    name: str
    operation: str
    iterations: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    std_dev: float = 0.0
    throughput: float = 0.0  # ops/sec
    memory_mb: float = 0.0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """完整基准报告"""
    timestamp: str = ""
    system_info: Dict = field(default_factory=dict)
    results: List[BenchmarkResult] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
        }


class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self):
        self._start = None
        self._laps: List[float] = []
    
    def start(self):
        self._start = time.perf_counter()
        return self
    
    def lap(self) -> float:
        if self._start is None:
            self.start()
        now = time.perf_counter()
        elapsed = now - self._start
        self._laps.append(elapsed)
        self._start = now
        return elapsed
    
    def stop(self) -> float:
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start
    
    @property
    def laps(self) -> List[float]:
        return self._laps.copy()


class BenchmarkSuite:
    """综合基准测试套件"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.results: List[BenchmarkResult] = []
        self.report = BenchmarkReport()
        self.catalog = None
        self._warmup_done = False
    
    def _get_system_info(self) -> Dict:
        """获取系统信息"""
        info = {}
        try:
            import platform
            info["os"] = platform.system()
            info["python"] = platform.python_version()
            info["machine"] = platform.machine()
        except:
            pass
        
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        except:
            pass
        
        try:
            import mujoco
            info["mujoco_version"] = mujoco.__version__
        except:
            pass
        
        try:
            import numpy as np
            info["numpy_version"] = np.__version__
        except:
            pass
        
        try:
            import psutil
            process = psutil.Process()
            info["cpu_count"] = psutil.cpu_count()
            info["memory_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
        except:
            pass
        
        return info
    
    def run_all(self, quick_mode: bool = False) -> BenchmarkReport:
        """运行所有基准测试"""
        print("\n" + "="*60)
        print("  ForgeCraft Performance Benchmark Suite")
        print("="*60 + "\n")
        
        self.report.timestamp = datetime.now().isoformat()
        self.report.system_info = self._get_system_info()
        
        # 预热
        print("[*] Warming up...")
        self._warmup()
        
        # 运行各项测试
        benchmarks = [
            ("Template Generation", self.bench_template_generation),
            ("Model Building", self.bench_model_building),
            ("Model Compilation (MuJoCo)", self.bench_model_compilation),
            ("Direct Drive Evaluation", self.bench_evaluation),
            ("Full Pipeline", self.bench_full_pipeline),
        ]
        
        if not quick_mode:
            benchmarks.extend([
                ("Memory Usage Profile", self.bench_memory_profile),
                ("Scalability Test", self.bench_scalability),
            ])
        
        for name, bench_fn in benchmarks:
            try:
                print(f"\n[Running] {name}...")
                result = bench_fn(quick=quick_mode)
                if result:
                    self.results.append(result)
                    self._print_result(result)
            except Exception as e:
                print(f"  [ERROR] {name} failed: {e}")
                logger.error(f"Benchmark {name} failed: {e}")
        
        # 汇总报告
        self.report.results = self.results
        self.report.summary = self._generate_summary()
        
        # 保存结果
        self.save_report()
        
        print("\n" + "="*60)
        print(f"  Benchmark Complete! {len(self.results)} tests passed")
        print(f"  Results saved to: {self.output_dir}")
        print("="*60 + "\n")
        
        return self.report
    
    def _warmup(self):
        """预热运行"""
        from locomotion_templates import LocomotionTemplateGenerator
        gen = LocomotionTemplateGenerator(seed=999)
        pop = gen.generate_population(size=2)
        self._warmup_done = True
    
    def _print_result(self, result: BenchmarkResult):
        """打印单个结果"""
        print(f"  {result.name}:")
        print(f"    Operations:     {result.iterations}")
        print(f"    Avg Time:       {result.avg_time*1000:.2f} ms")
        print(f"    Throughput:     {result.throughput:.1f} ops/s")
        if result.memory_mb > 0:
            print(f"    Memory:         {result.memory_mb:.1f} MB")
        if result.std_dev > 0:
            print(f"    Std Dev:        {result.std_dev*1000:.2f} ms")
    
    def bench_template_generation(self, quick: bool = False) -> BenchmarkResult:
        """模板生成基准"""
        from locomotion_templates import LocomotionTemplateGenerator
        
        n_iter = 5 if quick else 20
        n_per_iter = 10
        
        timer = PerformanceTimer()
        times = []
        total_generated = 0
        
        for i in range(n_iter):
            timer.start()
            gen = LocomotionTemplateGenerator(seed=i * 100)
            pop = gen.generate_population(size=n_per_iter)
            elapsed = timer.stop()
            times.append(elapsed)
            total_generated += len(pop)
        
        return BenchmarkResult(
            name="Template Generation",
            operation="generate_population()",
            iterations=total_generated,
            total_time=sum(times),
            avg_time=sum(times)/len(times)/n_per_iter,
            min_time=min(times)/n_per_iter,
            max_time=max(times)/n_per_iter,
            std_dev=statistics.stdev([t/n_per_iter for t in times]) if len(times) > 1 else 0,
            throughput=total_generated / sum(times),
        )
    
    def bench_model_building(self, quick: bool = False) -> BenchmarkResult:
        """模型构建基准"""
        from locomotion_templates import LocomotionTemplateGenerator
        from forgecraft.simulation.builder import build_mjcf_model
        
        if not self.catalog:
            from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
            self.catalog = load_catalog()
        
        gen = LocomotionTemplateGenerator(seed=42)
        population = gen.generate_population(size=10 if not quick else 5)
        
        timer = PerformanceTimer()
        times = []
        
        for body in population:
            timer.start()
            build_mjcf_model(body, self.catalog)
            elapsed = timer.stop()
            times.append(elapsed)
        
        valid_times = [t for t in times if t < 10]  # 过滤异常值
        
        return BenchmarkResult(
            name="Model Building",
            operation="build_mjcf_model()",
            iterations=len(valid_times),
            total_time=sum(valid_times),
            avg_time=statistics.mean(valid_times) if valid_times else 0,
            min_time=min(valid_times) if valid_times else 0,
            max_time=max(valid_times) if valid_times else 0,
            std_dev=statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
            throughput=len(valid_times) / sum(valid_times) if sum(valid_times) > 0 else 0,
        )
    
    def bench_model_compilation(self, quick: bool = False) -> BenchmarkResult:
        """MuJoCo模型编译基准"""
        import mujoco
        from locomotion_templates import LocomotionTemplateGenerator
        from forgecraft.simulation.builder import build_mjcf_model
        
        if not self.catalog:
            from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
            self.catalog = load_catalog()
        
        gen = LocomotionTemplateGenerator(seed=42)
        population = gen.generate_population(size=5 if not quick else 3)
        
        timer = PerformanceTimer()
        times = []
        
        for body in population[:5]:
            result = build_mjcf_model(body, self.catalog)
            xml_string = result[0] if isinstance(result, tuple) else result
            
            timer.start()
            model = mujoco.MjModel.from_xml_string(xml_string)
            data = mujoco.MjData(model)
            elapsed = timer.stop()
            times.append(elapsed)
        
        return BenchmarkResult(
            name="Model Compilation (MuJoCo)",
            operation="MjModel.from_xml_string()",
            iterations=len(times),
            total_time=sum(times),
            avg_time=statistics.mean(times) if times else 0,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            std_dev=statistics.stdev(times) if len(times) > 1 else 0,
            throughput=len(times) / sum(times) if sum(times) > 0 else 0,
            extra_metrics={
                "avg_bodies": statistics.mean([1 for _ in times]),
            }
        )
    
    def bench_evaluation(self, quick: bool = False) -> BenchmarkResult:
        """直接驱动评估基准"""
        from locomotion_templates import LocomotionTemplateGenerator
        from direct_drive_evaluator import DirectDriveEvaluator, DirectDriveConfig
        
        if not self.catalog:
            from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
            self.catalog = load_catalog()
        
        config = DirectDriveConfig(sim_steps=300 if quick else 1000, n_frequencies=1, episodes_per_freq=1)
        evaluator = DirectDriveEvaluator(config)
        
        gen = LocomotionTemplateGenerator(seed=42)
        population = gen.generate_population(size=5 if not quick else 3)
        
        timer = PerformanceTimer()
        times = []
        
        for body in population:
            timer.start()
            evaluator.evaluate_body(body, self.catalog, n_episodes=1)
            elapsed = timer.stop()
            times.append(elapsed)
        
        return BenchmarkResult(
            name="Direct Drive Evaluation",
            operation="evaluate_body()",
            iterations=len(times),
            total_time=sum(times),
            avg_time=statistics.mean(times) if times else 0,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            std_dev=statistics.stdev(times) if len(times) > 1 else 0,
            throughput=len(times) / sum(times) if sum(times) > 0 else 0,
            extra_metrics={
                "sim_steps": config.sim_steps,
            }
        )
    
    def bench_full_pipeline(self, quick: bool = False) -> BenchmarkResult:
        """完整流程基准：生成→构建→评估"""
        from locomotion_templates import LocomotionTemplateGenerator
        from forgecraft.simulation.builder import build_mjcf_model
        from direct_drive_evaluator import DirectDriveEvaluator, DirectDriveConfig
        import mujoco
        
        if not self.catalog:
            from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
            self.catalog = load_catalog()
        
        config = DirectDriveConfig(sim_steps=200 if quick else 500, n_frequencies=1, episodes_per_freq=1)
        evaluator = DirectDriveEvaluator(config)
        
        gen = LocomotionTemplateGenerator(seed=42)
        population = gen.generate_population(size=3 if not quick else 2)
        
        timer = PerformanceTimer()
        times = []
        
        for body in population:
            timer.start()
            
            # 完整流程
            build_mjcf_model(body, self.catalog)
            evaluator.evaluate_body(body, self.catalog, n_episodes=1)
            
            elapsed = timer.stop()
            times.append(elapsed)
        
        return BenchmarkResult(
            name="Full Pipeline (Gen→Build→Eval)",
            operation="end-to-end",
            iterations=len(times),
            total_time=sum(times),
            avg_time=statistics.mean(times) if times else 0,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            std_dev=statistics.stdev(times) if len(times) > 1 else 0,
            throughput=len(times) / sum(times) if sum(times) > 0 else 0,
        )
    
    def bench_memory_profile(self, quick: bool = False) -> BenchmarkResult:
        """内存使用分析"""
        import psutil
        from locomotion_templates import LocomotionTemplateGenerator
        from direct_drive_evaluator import DirectDriveEvaluator, DirectDriveConfig
        
        process = psutil.Process()
        
        if not self.catalog:
            from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
            self.catalog = load_catalog()
        
        baseline_mem = process.memory_info().rss / 1024 / 1024
        
        # 生成大量机器人
        gen = LocomotionTemplateGenerator(seed=42)
        pop = gen.generate_population(size=50 if not quick else 20)
        
        after_gen_mem = process.memory_info().rss / 1024 / 1024
        
        # 评估
        config = DirectDriveConfig(sim_steps=200, n_frequencies=1, episodes_per_freq=1)
        evaluator = DirectDriveEvaluator(config)
        for body in pop[:10]:
            evaluator.evaluate_body(body, self.catalog, n_episodes=1)
        
        after_eval_mem = process.memory_info().rss / 1024 / 1024
        
        return BenchmarkResult(
            name="Memory Usage Profile",
            operation="memory_tracking",
            iterations=50,
            memory_mb=after_eval_mem,
            extra_metrics={
                "baseline_mb": round(baseline_mem, 1),
                "after_generation_mb": round(after_gen_mem, 1),
                "after_evaluation_mb": round(after_eval_mem, 1),
                "gen_delta_mb": round(after_gen_mem - baseline_mem, 1),
                "eval_delta_mb": round(after_eval_mem - after_gen_mem, 1),
                "per_robot_kb": round((after_gen_mem - baseline_mem) * 1024 / 50, 1),
            }
        )
    
    def bench_scalability(self, quick: bool = False) -> BenchmarkResult:
        """可扩展性测试（不同种群大小）"""
        from locomotion_templates import LocomotionTemplateGenerator
        from direct_drive_evaluator import DirectDriveEvaluator, DirectDriveConfig
        
        if not self.catalog:
            from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
            self.catalog = load_catalog()
        
        sizes = [10, 20, 30, 40, 50]
        config = DirectDriveConfig(sim_steps=200, n_frequencies=1, episodes_per_freq=1)
        evaluator = DirectDriveEvaluator(config)
        
        scalability_data = {}
        
        for size in sizes:
            gen = LocomotionTemplateGenerator(seed=42)
            pop = gen.generate_population(size=size)
            
            start = time.perf_counter()
            for body in pop:
                evaluator.evaluate_body(body, self.catalog, n_episodes=1)
            elapsed = time.perf_counter() - start
            
            scalability_data[f"pop_{size}"] = {
                "size": size,
                "total_time_s": round(elapsed, 2),
                "per_robot_ms": round(elapsed / size * 1000, 1),
            }
        
        return BenchmarkResult(
            name="Scalability Test",
            operation="population_scaling",
            iterations=sum(sizes),
            total_time=sum(d["total_time_s"] for d in scalability_data.values()),
            extra_metrics=scalability_data,
        )
    
    def _generate_summary(self) -> Dict:
        """生成摘要统计"""
        if not self.results:
            return {"status": "no_results"}
        
        throughputs = [r.throughput for r in self.results if r.throughput > 0]
        avg_times = [r.avg_time * 1000 for r in self.results if r.avg_time > 0]
        
        summary = {
            "total_tests": len(self.results),
            "avg_throughput": round(statistics.mean(throughputs), 1) if throughputs else 0,
            "avg_latency_ms": round(statistics.mean(avg_times), 2) if avg_times else 0,
            "bottleneck": "",
            "recommendation": "",
        }
        
        # 找瓶颈
        if avg_times:
            slowest_idx = avg_times.index(max(avg_times))
            if slowest_idx < len(self.results):
                summary["bottleneck"] = f"{self.results[slowest_idx].name}: {max(avg_times):.1f}ms"
        
        # 建议
        if any(r.avg_time > 1.0 for r in self.results):
            summary["recommendation"] = "Consider GPU acceleration or Numba optimization"
        elif all(r.avg_time < 0.01 for r in self.results):
            summary["recommendation"] = "Excellent performance!"
        else:
            summary["recommendation"] = "Performance within acceptable range"
        
        return summary
    
    def save_report(self) -> Path:
        """保存报告到文件"""
        report_path = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report.to_dict(), f, indent=2, default=str)
        
        # 同时保存最新报告
        latest_path = self.output_dir / "latest_benchmark.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(self.report.to_dict(), f, indent=2, default=str)
        
        return report_path


def run_quick_benchmark():
    """快速基准测试入口"""
    suite = BenchmarkSuite()
    return suite.run_all(quick_mode=True)


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    suite = BenchmarkSuite()
    report = suite.run_all(quick_mode=quick)
