"""
ForgeCraft 程序入口

命令行模式:
  python -m forgecraft run          — 完整进化流程
  python -m forgecraft list-specs   — 列出可用零件箱/任务
  python -m forgecraft dashboard    — 启动 Web 看板
  python -m forgecraft validate     — 验证环境配置
"""

import argparse
import json
import os

__all__ = ["run"]

import torch

from forgecraft.config import EvolutionConfig, RLConfig
from forgecraft.core.loader import load_catalog, load_task, list_catalogs, list_tasks
from forgecraft.evolution.loop import EvolutionLoop


def _launch_dashboard(args) -> None:
    """启动 Web 仪表盘（从已有进化数据生成 HTML 并启动 HTTP 服务）"""
    from forgecraft.dashboard import generate_dashboard_data, render_dashboard_html, serve, WEB_DIR
    export_dir = args.export_dir if args.export else "."
    design_dir = "design_output"
    hist_file = os.path.join(export_dir, "evolution_history.json")
    body_file = os.path.join(export_dir, "best_body.json")
    if not os.path.exists(hist_file):
        hist_file = os.path.join(design_dir, "evolution_history.json")
        body_file = os.path.join(design_dir, "best_body.json")
    print(f"  进化数据: {hist_file}")
    print(f"  最佳形态: {body_file}")
    data = generate_dashboard_data(hist_file, body_file, export_dir if os.path.isdir(export_dir) else design_dir)
    html = render_dashboard_html(data)
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    (WEB_DIR / "dashboard.html").write_text(html, encoding="utf-8")
    print(f"  仪表盘: http://localhost:8080")
    serve(port=8080)


def main():
    parser = argparse.ArgumentParser(
        description="ForgeCraft - 基于强化学习的机械形态进化系统"
    )
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--elites", type=int, default=3)
    parser.add_argument("--config", type=str, default=None,
                        help="从 YAML 配置文件加载实验参数 (覆盖命令行参数)")
    parser.add_argument("--workers", type=int, default=4, help="并行进程数 (默认: 4)")
    parser.add_argument("--cuda", action="store_true", help="启用 CUDA GPU 加速")
    parser.add_argument("--catalog", type=str, default="primitives",
                        help=f"零件箱名称或路径")
    parser.add_argument("--task", type=str, default="speed",
                        help=f"任务名称或路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--export", action="store_true", help="进化完成后自动导出制造文件 (STL+STEP+URDF+BOM)")
    parser.add_argument("--export-dir", type=str, default="design_output", help="制造文件输出目录")
    parser.add_argument("--quality", type=str, default="high", choices=["low","medium","high","ultra"],
                        help="网格质量: low/medium/high/ultra (默认high)")
    parser.add_argument("--dashboard", action="store_true", help="进化完成后启动 Web 可视化仪表盘")
    parser.add_argument("--live-dashboard", action="store_true", help="进化过程中启动实时 WebSocket 看板")
    parser.add_argument("--dashboard-port", type=int, default=8080, help="仪表盘端口 (默认8080)")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--turbo", action="store_true", help="极速模式: torch.compile + 渐进式评估")
    parser.add_argument("--resume", type=str, default=None, help="从断点文件恢复进化")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="每N代自动保存断点 (默认5)")
    parser.add_argument("--curriculum", action="store_true", help="启用课程学习 (简单→困难渐进训练)")
    parser.add_argument("--pretrain", action="store_true", help="进化前预测练形态编码器 (SimCLR对比学习)")
    parser.add_argument("--pretrain-epochs", type=int, default=200, help="预训练轮数 (默认200)")
    parser.add_argument("--pretrain-bodies", type=int, default=256, help="预训练生成的形态数 (默认256)")
    parser.add_argument("--map-elites", action="store_true", help="启用 MAP-Elites 质量多样性进化 (提升形态多样性)")
    parser.add_argument("--hpo", action="store_true", help="启用超参数自动优化 (Optuna/Bayesian)")
    parser.add_argument("--hpo-trials", type=int, default=30, help="HPO 试验次数 (默认30)")
    parser.add_argument("--hpo-generations", type=int, default=8, help="每试验进化代数 (默认8)")
    parser.add_argument("--distributed", action="store_true", help="启用分布式计算 (Ray 集群)")
    parser.add_argument("--ray-address", type=str, default="auto", help="Ray 集群地址")
    parser.add_argument("--dist-workers", type=int, default=8, help="分布式 Worker 数 (默认8)")
    parser.add_argument("--experiment-name", type=str, default=None, help="实验名称 (启用实验追踪)")
    parser.add_argument("--experiments-dir", type=str, default="experiments", help="实验存储目录")
    parser.add_argument("--list-experiments", action="store_true", help="列出所有实验记录")
    parser.add_argument("--compare", type=str, default=None, help="对比实验 (逗号分隔 run_id)")
    parser.add_argument("--list-catalogs", action="store_true", help="列出所有可用零件箱")
    parser.add_argument("--list-tasks", action="store_true", help="列出所有可用任务")

    args = parser.parse_args()

    # ════════════════════════════════════════════════════════
    # 实验管理命令 (不需要初始化整个进化循环)
    # ════════════════════════════════════════════════════════
    if args.list_experiments:
        from forgecraft.experiment.management import ExperimentManager
        mgr = ExperimentManager(experiments_dir=args.experiments_dir)
        mgr.print_experiments_table()
        return

    if args.compare:
        from forgecraft.experiment.management import ExperimentManager
        run_ids = [rid.strip() for rid in args.compare.split(",")]
        mgr = ExperimentManager(experiments_dir=args.experiments_dir)
        mgr.print_comparison(run_ids)
        return

    if args.list_catalogs:
        print("可用零件箱:")
        for name in list_catalogs():
            cat = load_catalog(name=name)
            print(f"\n{cat.describe()}")
        return

    if args.list_tasks:
        print("可用任务:")
        for name in list_tasks():
            task = load_task(name=name)
            print(f"\n  {name}: {task.description.split(chr(10))[0][:100]}")
        return

    # ── 仪表盘独立模式 (不需要进化数据) ──────────────
    if args.dashboard and args.generations == 10 and args.population == 16:
        # 未指定显式进化参数 → 尝试从已有数据启动看板
        export_dir = args.export_dir if args.export else "."
        hist_file = os.path.join(export_dir, "evolution_history.json")
        if os.path.exists(hist_file) or os.path.exists("design_output/evolution_history.json"):
            print("从已有进化数据启动仪表盘...")
            _launch_dashboard(args)
            return

    # ── YAML 配置热加载 ──────────────────────────────
    if args.config and os.path.exists(args.config):
        from forgecraft.experiment_config import ExperimentConfig
        yaml_cfg = ExperimentConfig(args.config)
        # YAML 参数覆盖命令行空值
        if args.generations == 10:  # 命令行默认值, 用 YAML 覆盖
            args.generations = yaml_cfg.generations
        if args.population == 16:
            args.population = yaml_cfg.population
        if args.elites == 3:
            args.elites = yaml_cfg.evolution.elite_count
        print(f"[Config] 从 {args.config} 加载实验参数")
        print(f"  {yaml_cfg.name}: {args.generations}代 × {args.population}个体")

    if args.cuda and torch.cuda.is_available():
        device = "cuda"
        torch.set_float32_matmul_precision('high')
    elif args.cuda:
        print("警告: CUDA 不可用，退回 CPU 模式")
        device = "cpu"
    else:
        device = "cpu"

    catalog = load_catalog(name=args.catalog)
    task_spec = load_task(name=args.task)

    sim_config = task_spec.to_sim_config()
    task_config = task_spec.to_task_config()

    if args.quick:
        task_config.max_episode_steps = 300

    evo_config = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        elite_count=args.elites,
    )

    rl_config = RLConfig(
        hidden_dim=64,
        gnn_hidden=32,
        morph_embed_dim=32,
    )

    print("=" * 60)
    print("  ForgeCraft v0.4.0")
    print("  基于强化学习的机械形态共进化系统")
    print("=" * 60)
    print(f"  设备: GPU ({torch.cuda.get_device_name(0)})" if device == "cuda" else "  设备: CPU")
    print(f"  零件箱: {catalog.name} ({len(catalog.parts)} 种零件)")
    print(f"  任务: {task_spec.name}")
    print(f"  并行进程: {args.workers} | 种群: {args.population} | 世代: {args.generations}")
    print(f"  模式: {'快速测试' if args.quick else '标准'}")
    print("=" * 60)

    if args.resume and os.path.exists(args.resume):
        loop = EvolutionLoop.load_checkpoint(args.resume, device=device)
        loop.n_workers = max(1, args.workers // 2)  # MuJoCo并行限制，防止OOM
        print(f"从断点恢复: {args.resume}")
        print(f"世代: {loop.generation}/{args.generations}, 种群: {len(loop.population)}, 最佳: {loop.best_body.fitness if loop.best_body else 0:.4f}")
    else:
        if args.resume:
            print(f"警告: 断点文件不存在 ({args.resume})，将从头开始")

        pretrained_weights = None
        if args.pretrain:
            print("\n" + "=" * 50)
            print("  形态编码器预训练 (SimCLR 对比学习)")
            print("=" * 50)
            from forgecraft.rl.pretrain import pretrain_encoder
            pretrain_result = pretrain_encoder(
                catalog.to_part_specs(),
                n_bodies=args.pretrain_bodies,
                epochs=args.pretrain_epochs,
                device=device,
                seed=args.seed,
                output_dir=args.export_dir if args.export else None,
                hidden_dim=rl_config.gnn_hidden,
                output_dim=rl_config.morph_embed_dim,
                num_layers=rl_config.gnn_layers,
            )
            pretrained_weights = pretrain_result["encoder_state"]
            print(f"  预训练完成: loss={pretrain_result['final_loss']:.4f}")

        # ════════════════════════════════════════════════════
        # HPO 超参数自动优化
        # ════════════════════════════════════════════════════
        if args.hpo:
            print("\n" + "=" * 50)
            print("  超参数自动优化 (HPO)")
            print("=" * 50)
            print(f"  试验数: {args.hpo_trials}")
            print(f"  每试验代数: {args.hpo_generations}")
            print("=" * 50)
            
            from forgecraft.evolution.hyperparam_opt import (
                HyperParamOptimizer, HyperParameterSpace,
                create_hpo_objective, print_optimization_report,
            )
            
            objective = create_hpo_objective(
                catalog=catalog.to_part_specs(),
                task_config=task_config,
                sim_config=sim_config,
                generations=args.hpo_generations,
                workers=args.workers,
                device=device,
                seed=args.seed,
            )
            
            optimizer = HyperParamOptimizer(seed=args.seed)
            result = optimizer.optimize(
                objective_fn=objective,
                n_trials=args.hpo_trials,
                early_stopping_patience=10,
            )
            
            print_optimization_report(result)
            
            # 用最优参数更新配置
            best = result.best_params
            if best:
                space = HyperParameterSpace()
                evo_config = space.to_evo_config(best)
                rl_config = space.to_rl_config(best)
                sim_config = space.to_sim_config(best)
                
                print("\n已应用最优超参数, 开始最终进化...")

        loop = EvolutionLoop(
            evo_config=evo_config,
            sim_config=sim_config,
            rl_config=rl_config,
            task_config=task_config,
            catalog=catalog.to_part_specs(),
            device=device,
            seed=args.seed,
            n_workers=args.workers,
            pretrained_encoder_weights=pretrained_weights,
            enable_map_elites=args.map_elites,
        )

    if args.curriculum:
        total = args.generations
        mid_point = total // 3
        if args.task in ("climb",):
            stages = [(0, "efficiency"), (mid_point, "speed"), (mid_point * 2, args.task)]
        else:
            stages = [(0, "efficiency"), (mid_point, args.task)]
        loop.setup_curriculum(stages)

    # ════════════════════════════════════════════════════════
    # 分布式计算模式
    # ════════════════════════════════════════════════════════
    if args.distributed:
        print("\n" + "=" * 50)
        print("  分布式集群计算模式")
        print("=" * 50)
        
        from forgecraft.evolution.distributed import (
            DistributedConfig, DistributedEvolutionLoop,
            HAS_RAY, HAS_REDIS,
        )
        
        print(f"  Ray 可用: {'是' if HAS_RAY else '否 (回退 multiprocessing)'}")
        print(f"  Redis 可用: {'是' if HAS_REDIS else '否'}")
        
        dist_config = DistributedConfig(
            use_ray=HAS_RAY,
            ray_address=args.ray_address,
            n_remote_workers=args.dist_workers,
            task_timeout=600.0,
        )
        
        dist_loop = DistributedEvolutionLoop(
            evo_config=evo_config,
            sim_config=sim_config,
            rl_config=rl_config,
            task_config=task_config,
            catalog=catalog.to_part_specs(),
            distributed_config=dist_config,
            device=device,
            seed=args.seed,
            enable_map_elites=args.map_elites,
        )
        
        dist_loop.print_cluster_status()
        
        try:
            loop = dist_loop.run_evolution(
                generations=args.generations,
                catalog_name=args.catalog,
                pretrained_weights=pretrained_weights,
                checkpoint_every=args.checkpoint_every,
            )
        except KeyboardInterrupt:
            print("\n\n用户中断。")
        finally:
            dist_loop.shutdown()
    else:
        # ── 实时仪表盘 ──────────────────────────────────
        _dashboard_stop = None
        if args.live_dashboard:
            import threading
            from forgecraft.dashboard_live import LiveDashboard, push_stats, push_log, set_config
            live_dashboard = LiveDashboard(port=args.dashboard_port)
            t = threading.Thread(target=live_dashboard.serve, daemon=True)
            t.start()
            import time; time.sleep(1.5)

            set_config(total_generations=args.generations,
                       task_name=args.task,
                       catalog_name=args.catalog)
            push_log(f"进化启动: {args.catalog} x {args.task}, "
                     f"{args.population}个体 x {args.generations}代")

            # 后台轮询线程：每 3 秒直接从 loop 读取状态写入共享字典
            _stop_polling = threading.Event()

            def _poll_loop_state():
                while not _stop_polling.is_set():
                    try:
                        hist = loop.history
                        if hist:
                            last = hist[-1]
                            best = loop.best_body
                            best_fit = getattr(best, 'fitness', 0) or 0
                            parts = best.num_parts() if best and hasattr(best, 'num_parts') else 0
                            push_stats(
                                generation=loop.generation,
                                best_fitness=best_fit,
                                mean_fitness=last.get('mean_fitness', 0) or 0,
                                population_size=last.get('population_size', 0) or 0,
                                best_parts=parts,
                            )
                            push_log(f"Gen {loop.generation}: best={best_fit:.4f} "
                                     f"avg={last.get('mean_fitness', 0):.4f} parts={parts}")
                    except Exception:
                        pass
                    _stop_polling.wait(3.0)

            poll_thread = threading.Thread(target=_poll_loop_state, daemon=True)
            poll_thread.start()
            _dashboard_stop = _stop_polling

        # 实验追踪
        if args.experiment_name:
            from forgecraft.experiment.management import TrackedEvolutionLoop
            tracked = TrackedEvolutionLoop(
                loop,
                experiment_name=args.experiment_name,
                experiments_dir=args.experiments_dir,
            )
            try:
                tracked.run(n_generations=args.generations, save_best=True)
            except KeyboardInterrupt:
                print("\n\n用户中断。")
        else:
            try:
                loop.run(n_generations=args.generations)
            except KeyboardInterrupt:
                print("\n\n用户中断。")

    if loop.best_body:
        print("\n" + "-" * 40)
        print("  最佳形态制造性分析")
        print("-" * 40)
        from forgecraft.evaluation.metrics import compute_manufacturability_detail
        mfg_detail = compute_manufacturability_detail(loop.best_body)
        print(f"  总分: {mfg_detail['overall']:.3f}")
        for check_name, check_data in mfg_detail.items():
            if check_name == "overall":
                continue
            score = check_data["score"]
            detail = check_data["detail"]
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            label_map = {"bed_size": "打印床尺寸", "wall_thickness": "壁厚",
                         "assembly": "装配复杂度", "torque_budget": "力矩预算",
                         "cantilever": "悬臂约束"}
            label = label_map.get(check_name, check_name)
            print(f"  [{bar}] {label:8s} {score:.3f}")
            if check_name == "bed_size" and detail.get("violations", 0) > 0:
                blim = detail.get("bed_limit", [0.22] * 3)
                print(f"        {detail['violations']}个零件超出打印床 "
                      f"(限{blim[0]:.2f}×{blim[1]:.2f}×{blim[2]:.2f})")
                for v in detail.get("largest_violations", [])[:2]:
                    s = v.get("size", [0, 0, 0])
                    print(f"        {v.get('part_type','?')}: {s[0]:.3f}×{s[1]:.3f}×{s[2]:.3f}m")
            if check_name == "wall_thickness":
                violations = detail.get("violations", [])
                if violations:
                    for v in violations[:3]:
                        print(f"        太薄: {v[1]}.{v[2]}={v[3]:.4f}m (最小{detail.get('min_wall', 0.0008):.4f}m)")
            if check_name == "torque_budget":
                ratio = detail.get("ratio", 1.0)
                if ratio < detail.get("min_ratio", 0.3):
                    print(f"        力矩/质量比={ratio:.3f} (需≥{detail.get('min_ratio', 0.3)})")

        if args.export or args.output:
            export_dir = args.export_dir if args.export else args.output
            os.makedirs(export_dir, exist_ok=True)
            from forgecraft.manufacturing.pipeline import ManufacturingPipeline, score_manufacturability
            pipeline = ManufacturingPipeline(quality=args.quality)
            bom = pipeline.generate_bom(loop.best_body, os.path.join(export_dir, "bom.json"))
            mfg_score, _ = score_manufacturability(loop.best_body)
            mfg_detail.update({
                "manufacturability_score": mfg_score,
                "export_dir": export_dir,
            })
            mfg_path = os.path.join(export_dir, "manufacturability.json")
            with open(mfg_path, "w", encoding="utf-8") as f:
                json.dump(mfg_detail, f, indent=2, ensure_ascii=False, default=str)
            print(f"  BOM + 可制造性报告 → {export_dir}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        _save_results(loop, args.output)

    if args.dashboard:
        print("\n" + "=" * 50)
        print("  启动 Web 仪表盘...")
        print("=" * 50)
        _launch_dashboard(args)
        return  # serve() blocks until Ctrl+C

    if loop.best_body:
        print("\n最佳形态可直接渲染：")
        from forgecraft.simulation.builder import build_mjcf_model
        xml_str, _, _, _ = build_mjcf_model(loop.best_body, catalog.to_part_specs())
        xml_path = "best_body.xml" if not args.output else os.path.join(args.output, "best_body.xml")
        with open(xml_path, "w") as f:
            f.write(xml_str)
        print(f"  MuJoCo 模型 → {xml_path}")

    if args.export and loop.best_body:
        print("\n" + "=" * 50)
        print("  制造导出中...")
        print("=" * 50)
        from forgecraft.manufacturing.pipeline import ManufacturingPipeline

        _save_results(loop, args.export_dir)
        
        pipeline = ManufacturingPipeline(quality=args.quality)
        report = pipeline.export_all(loop.best_body, args.export_dir)
        
        print(f"  输出目录: {args.export_dir}")
        print(f"  可制造性: {report.manufacturability_score:.1%}")
        print(f"  总成本: ${report.total_cost_usd:.2f}")
        print(f"  打印耗时: {report.print_time_hours:.1f}h")
        print(f"  导出文件: {len(report.exported_files)} 个")
        for f in report.exported_files:
            print(f"    - {os.path.basename(f)}")
        if report.warnings:
            print(f"  警告: {len(report.warnings)} 个")
        if report.suggestions:
            print(f"  建议: {len(report.suggestions)} 个")


def _save_results(loop: EvolutionLoop, output_dir: str):
    history_path = os.path.join(output_dir, "evolution_history.json")
    with open(history_path, "w") as f:
        json.dump(loop.history, f, indent=2, default=str)

    if loop.best_body:
        best_path = os.path.join(output_dir, "best_body.json")
        body_data = {
            "name": loop.best_body.name,
            "fitness": loop.best_body.fitness,
            "fitness_components": loop.best_body.fitness_components,
            "num_parts": loop.best_body.num_parts(),
            "num_joints": loop.best_body.num_joints(),
            "parts": [p.to_dict() for p in loop.best_body.parts()],
            "joints": [j.to_dict() for j in loop.best_body.joints()],
        }
        with open(best_path, "w") as f:
            json.dump(body_data, f, indent=2, default=str)

    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
