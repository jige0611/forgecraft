#!/usr/bin/env python3
"""ForgeCraft 性能基准测试

测量各模块的运行时性能：
  - 形态生成速度
  - GNN 编码吞吐量
  - PPO 更新速度
  - 物理仿真 FPS
  - 完整一代进化流程耗时

用法:
    python -m forgecraft.testing.benchmark          # 标准基准
    python -m forgecraft.testing.benchmark --quick  # 快速模式 (减少迭代)
    python -m forgecraft.testing.benchmark --gpu    # GPU 模式
"""

import argparse
import time
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
__all__ = [
    "run_benchmark",
    "benchmark_generation",
    "benchmark_simulation",
    "benchmark_encoding",
    "benchmark_evolution_step",
]




# ── 内部依赖 ────────────────────────────────────────────────
PRJ = Path(__file__).resolve().parents[1]
if str(PRJ) not in sys.path:
    sys.path.insert(0, str(PRJ))


def run_benchmarks(device: str = "cpu", quick: bool = False) -> Dict:
    """运行所有性能基准"""
    results = {
        "system": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": device,
            "platform": platform.platform(),
        },
        "benchmarks": {},
    }

    n_generations = 2 if quick else 5
    pop_size = 8 if quick else 20
    bodies: list = []
    part_specs: dict = {}
    task_config = None

    # ── 0. 初始化进化循环 (用来生成 bodies + 获取配置) ──
    try:
        from forgecraft.core.loader import load_catalog, load_task
        from forgecraft.evolution.loop import EvolutionLoop
        from forgecraft.config import EvolutionConfig

        catalog = load_catalog(name="default")
        task = load_task(name="speed")
        part_specs = catalog.to_part_specs()
        task_config = task.to_task_config()

        evo = EvolutionConfig(
            population_size=pop_size,
            generations=n_generations,
            mutation_rate=0.3,
            elite_count=max(2, pop_size // 5),
        )
        loop = EvolutionLoop(
            evo_config=evo,
            task_config=task_config,
            catalog=part_specs,
            device=device,
            n_workers=2,
        )
    except Exception as e:
        results["benchmarks"]["init"] = {"error": str(e)}
        print(f"  初始化:    FAILED - {e}")
        loop = None

    # ── 1. 形态生成 ──────────────────────────────────────
    try:
        if loop is None:
            raise RuntimeError("EvolutionLoop 未初始化")
        from forgecraft.core.generator import BodyGenerator

        generator = BodyGenerator(part_specs, seed=42)

        t0 = time.perf_counter()
        bodies = [generator.generate_random_body() for _ in range(pop_size)]
        gen_time = time.perf_counter() - t0

        results["benchmarks"]["morphology_generation"] = {
            "population_size": pop_size,
            "total_seconds": round(gen_time, 4),
            "bodies_per_second": round(pop_size / gen_time, 1),
            "avg_n_parts": round(np.mean([b.num_parts() for b in bodies]), 1),
        }
        print(f"  形态生成:  {pop_size} 体 / {gen_time:.2f}s = {pop_size/gen_time:.0f} 体/s")
    except Exception as e:
        results["benchmarks"]["morphology_generation"] = {"error": str(e)}
        print(f"  形态生成:  FAILED - {e}")
        bodies = []

    # ── 2. GNN 编码 ──────────────────────────────────────
    try:
        from forgecraft.rl.encoder import MorphologyEncoder
        from forgecraft.rl.batch_encoder import BatchMorphologyEncoder

        if not bodies:
            raise RuntimeError("无形态数据")

        encoder = MorphologyEncoder(output_dim=64)
        batch_enc = BatchMorphologyEncoder(encoder, device=device)

        # 预热
        _ = batch_enc.encode_batch(bodies[:4])

        t0 = time.perf_counter()
        n_runs = 20 if quick else 50
        for _ in range(n_runs):
            _ = batch_enc.encode_batch(bodies)
        enc_time = time.perf_counter() - t0

        results["benchmarks"]["gnn_encoding"] = {
            "batch_size": pop_size,
            "runs": n_runs,
            "total_seconds": round(enc_time, 4),
            "avg_ms_per_batch": round(enc_time / n_runs * 1000, 2),
            "bodies_per_second": round(n_runs * pop_size / enc_time, 0),
        }
        print(f"  GNN 编码:   {n_runs}x{pop_size} 体 = {enc_time:.2f}s ({enc_time/n_runs*1000:.1f} ms/batch)")
    except Exception as e:
        results["benchmarks"]["gnn_encoding"] = {"error": str(e)}
        print(f"  GNN 编码:  FAILED - {e}")

    # ── 3. PPO 更新 ──────────────────────────────────────
    try:
        from forgecraft.config import RLConfig
        from forgecraft.rl.ppo import PPOBuffer, PPOTrainer

        config = RLConfig(
            hidden_dim=128, batch_size=64, ppo_epochs=8,
            clip_ratio=0.2, value_loss_coef=0.5, entropy_coef=0.01,
            actor_lr=3e-4, critic_lr=1e-3,
        )
        buffer = PPOBuffer(
            obs_dim=24, act_dim=6, morph_dim=64,
            max_size=200,
        )

        # 填充随机数据
        for _ in range(200):
            buffer.store(
                obs=np.random.randn(24).astype(np.float32),
                morph=np.random.randn(64).astype(np.float32),
                act=np.random.randn(6).astype(np.float32),
                rew=float(np.random.randn()),
                val=float(np.random.randn()),
                logp=float(np.random.randn()),
                done=False,
            )
        buffer.compute_gae(last_val=0.0, gamma=0.99, lam=0.95)

        trainer = PPOTrainer(
            obs_dim=24, act_dim=6, morph_dim=64,
            config=config, device=device, compile_nets=False,
        )

        # 预热
        _ = trainer.update(buffer)
        buffer.clear()
        for _ in range(200):
            buffer.store(
                obs=np.random.randn(24).astype(np.float32),
                morph=np.random.randn(64).astype(np.float32),
                act=np.random.randn(6).astype(np.float32),
                rew=float(np.random.randn()),
                val=float(np.random.randn()),
                logp=float(np.random.randn()),
                done=False,
            )
        buffer.compute_gae(last_val=0.0, gamma=0.99, lam=0.95)

        t0 = time.perf_counter()
        n_updates = 3 if quick else 10
        for _ in range(n_updates):
            info = trainer.update(buffer)
        ppo_time = time.perf_counter() - t0

        results["benchmarks"]["ppo_update"] = {
            "updates": n_updates,
            "total_seconds": round(ppo_time, 4),
            "avg_ms_per_update": round(ppo_time / n_updates * 1000, 2),
            "actor_loss": round(info.get("actor_loss", 0), 4),
            "critic_loss": round(info.get("critic_loss", 0), 4),
        }
        print(f"  PPO 更新:   {n_updates} 次 = {ppo_time:.2f}s ({ppo_time/n_updates*1000:.1f} ms/update)")
    except Exception as e:
        results["benchmarks"]["ppo_update"] = {"error": str(e)}
        print(f"  PPO 更新:   FAILED - {e}")

    # ── 4. MuJoCo 仿真 ────────────────────────────────────
    try:
        from forgecraft.simulation.builder import build_mjcf_model
        from forgecraft.config import SimConfig

        if not bodies:
            raise RuntimeError("无形态数据")

        body = bodies[0]
        sim_config = SimConfig(timestep=0.005, max_steps=300)

        # 预热
        xml_str, _, _, _ = build_mjcf_model(body, part_specs)

        import mujoco
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        target_iters = 100 if quick else 500

        t0 = time.perf_counter()
        steps = 0
        for _ in range(target_iters // sim_config.max_steps + 1):
            xml_str, _, _, _ = build_mjcf_model(body, part_specs)
            m = mujoco.MjModel.from_xml_string(xml_str)
            d = mujoco.MjData(m)
            for _ in range(sim_config.max_steps):
                mujoco.mj_step(m, d)
                steps += 1
        sim_time = time.perf_counter() - t0

        results["benchmarks"]["physics_simulation"] = {
            "backend": "MuJoCo",
            "total_seconds": round(sim_time, 4),
            "total_steps": steps,
            "avg_ms_per_step": round(sim_time / steps * 1000, 4),
            "effective_fps": round(steps / sim_time, 0),
        }
        print(f"  物理仿真:   {steps} 步 / {sim_time:.2f}s = {steps/sim_time:.0f} FPS")
    except Exception as e:
        results["benchmarks"]["physics_simulation"] = {"error": str(e)}
        print(f"  物理仿真:   FAILED - {e}")

    # ── 5. 完整进化流程 ────────────────────────────────────
    try:
        if loop is None:
            raise RuntimeError("EvolutionLoop 未初始化")

        # 单进程模式避免 OOM
        loop.n_workers = 1
        loop.initialize_population()

        t0 = time.perf_counter()
        loop.run(n_generations=n_generations, safe=False)
        evo_time = time.perf_counter() - t0

        results["benchmarks"]["full_evolution"] = {
            "population_size": pop_size,
            "generations": n_generations,
            "total_seconds": round(evo_time, 2),
            "seconds_per_generation": round(evo_time / n_generations, 2),
            "final_best_fitness": round(float(loop.best_body.fitness), 4) if loop.best_body else 0.0,
        }
        print(f"  完整进化:   {n_generations} 代 / {evo_time:.1f}s = {evo_time/n_generations:.1f}s/代")
    except Exception as e:
        results["benchmarks"]["full_evolution"] = {"error": str(e)}
        print(f"  完整进化:   FAILED - {e}")

    # ── 6. 增量编码 vs 全量编码 ──────────────────────────
    try:
        from forgecraft.rl.encoder import MorphologyEncoder
        from forgecraft.rl.incremental_encoder import IncrementalMorphologyEncoder

        if not bodies or len(bodies) < 2:
            raise RuntimeError("需要至少 2 个形态")

        encoder = MorphologyEncoder(output_dim=64)
        inc_enc = IncrementalMorphologyEncoder(encoder, device=device)

        body_a = bodies[0]
        body_b = bodies[1]

        # 全量编码预热
        _ = inc_enc.encode_with_diff(body_a, None)
        _ = inc_enc.encode_with_diff(body_a, body_b)

        t0 = time.perf_counter()
        n_runs = 50 if quick else 200
        for _ in range(n_runs):
            _ = inc_enc.encode_with_diff(body_a, None)  # 全量
        full_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = inc_enc.encode_with_diff(body_a, body_b)  # 增量
        inc_time = time.perf_counter() - t0

        speedup = full_time / inc_time if inc_time > 0 else 0
        results["benchmarks"]["incremental_encoding"] = {
            "runs": n_runs,
            "full_ms_per_call": round(full_time / n_runs * 1000, 3),
            "incremental_ms_per_call": round(inc_time / n_runs * 1000, 3),
            "speedup": round(speedup, 2),
            "cache_hit_rate": inc_enc.stats.get("hit_rate", 0) if hasattr(inc_enc, "stats") else "N/A",
        }
        print(f"  增量编码:   全量 {full_time/n_runs*1000:.2f}ms vs 增量 {inc_time/n_runs*1000:.2f}ms ({speedup:.1f}x)")
    except Exception as e:
        results["benchmarks"]["incremental_encoding"] = {"error": str(e)}
        print(f"  增量编码:   FAILED - {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="ForgeCraft 性能基准")
    parser.add_argument("--quick", action="store_true", help="快速模式")
    parser.add_argument("--gpu", action="store_true", help="GPU 模式")
    parser.add_argument("--json", type=str, default=None, help="导出结果到 JSON")
    args = parser.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    mode_str = "GPU" if device == "cuda" else "CPU"
    q_str = "快速" if args.quick else "标准"

    print(f"\n{'='*60}")
    print(f"  ForgeCraft 性能基准 — {mode_str} {q_str} 模式")
    print(f"{'='*60}\n")

    results = run_benchmarks(device=device, quick=args.quick)

    print(f"\n{'='*60}")
    print(f"  基准完成")
    print(f"{'='*60}\n")

    # 汇总
    ok = sum(1 for v in results["benchmarks"].values() if "error" not in v)
    total = len(results["benchmarks"])
    print(f"  通过: {ok}/{total}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"  结果已写入: {args.json}")


if __name__ == "__main__":
    main()
