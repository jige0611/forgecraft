# ═══════════════════════════════════════════════════════════════
#  ForgeCraft CLI - 标准化命令行入口
#  ═══════════════════════════════════════════════════════════════
#
#  用法：
#    python forgecraft.py evolve          # 运行进化实验
#    python forgecraft.py view            # 查看机器人3D仿真
#    python forgecraft.py evaluate        # 评估单个机器人
#    python forgecraft.py export          # 导出模型(STL/URDF)
#    python forgecraft.py report          # 生成实验报告
#    python forgecraft.py benchmark       # 运行性能基准
#    python forgecraft.py test            # 运行测试套件
#    python forgecraft.py info            # 显示系统信息
#
#  示例：
#    python forgecraft.py evolve --gens 50 --pop 30
#    python forgecraft.py view --template diff --headless
#    python forgecraft.py export --format stl --output robot.stl
# ═══════════════════════════════════════════════════════════════

import sys
import os
import argparse
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def cmd_evolve(args):
    """运行进化实验"""
    from v11_ultimate_engine import V11Config, V11EvolutionEngine
    
    config = V11Config(
        total_generations=args.generations,
        population_size=args.population,
        n_evaluations=args.evaluations,
        elite_size=args.elite,
        output_dir=args.output,
    )
    
    engine = V11EvolutionEngine(config)
    result = engine.run()
    
    if result:
        print(f"\n{'='*60}")
        print(f"  Evolution Complete!")
        print(f"{'='*60}")
        print(f"  Best Fitness:   {result.fitness:.4f}")
        print(f"  Displacement:   {result.displacement*100:.2f} cm")
        print(f"  Speed:          {result.speed:.3f} m/s")
        print(f"  Survival:       {result.survival*100:.1f}%")
        print(f"  Type:           {result.robot_type}")
        print(f"{'='*60}")


def cmd_view(args):
    """查看机器人3D仿真"""
    from viewer import ViewerConfig, RealtimeViewer, LocomotionTemplateGenerator
    from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
    
    # 生成或加载机器人
    template_gen = LocomotionTemplateGenerator(seed=args.seed)
    
    if args.template:
        template_map = {
            "diff": "differential_wheeled",
            "four": "quad_wheeled",
            "bipedal": "bipedal",
            "quad": "quadruped",
            "crawler": "crawler",
        }
        for k in template_gen.template_weights:
            template_gen.template_weights[k] = 0.0
        ttype = template_map.get(args.template, args.template)
        if ttype in template_gen.template_weights:
            template_gen.template_weights[ttype] = 1.0
    
    population = template_gen.generate_population(size=1)
    body = population[0]
    catalog = load_catalog()
    
    config = ViewerConfig(
        auto_drive=not args.no_drive,
        drive_speed=args.speed,
        show_hud=not args.no_hud,
    )
    
    viewer = RealtimeViewer(body, config, catalog)
    
    if viewer.model is None:
        print("ERROR: Failed to create model!")
        sys.exit(1)
    
    if args.headless or args.duration:
        duration = args.duration or 10.0
        results = viewer.run_headless(duration=duration, save_images=True)
        print(f"\nResults:")
        print(f"  Duration: {results['duration']}s sim, {results['elapsed_real']:.1f}s real")
        print(f"  Displacement: {results['final_displacement']*100:.2f}cm")
        print(f"  Avg Speed: {results['avg_speed']:.3f}m/s")
        print(f"  Output: {results['output_dir']}")
    else:
        viewer.run_interactive()


def cmd_evaluate(args):
    """评估单个机器人"""
    from locomotion_templates import LocomotionTemplateGenerator
    from direct_drive_evaluator import DirectDriveEvaluator, DirectDriveConfig
    from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
    
    template_gen = LocomotionTemplateGenerator(seed=args.seed)
    
    if args.template:
        for k in template_gen.template_weights:
            template_gen.template_weights[k] = 0.0
        template_map = {
            "diff": "differential_wheeled", "four": "quad_wheeled",
            "bipedal": "bipedal", "quad": "quadruped", "crawler": "crawler",
        }
        ttype = template_map.get(args.template, args.template)
        if ttype in template_gen.template_weights:
            template_gen.template_weights[ttype] = 1.0
    
    population = template_gen.generate_population(size=args.count)
    catalog = load_catalog()
    
    eval_config = DirectDriveConfig(
        sim_steps=args.steps,
        n_frequencies=args.frequencies,
        episodes_per_freq=args.episodes,
    )
    
    evaluator = DirectDriveEvaluator(eval_config)
    
    print(f"\nEvaluating {len(population)} robots...")
    print(f"{'Robot':>6} {'Type':>12} {'Fitness':>8} {'Disp(cm)':>9} {'Speed':>8} {'Survival':>8}")
    print("-" * 60)
    
    results = []
    for i, body in enumerate(population):
        result = evaluator.evaluate(body, catalog)
        if result:
            results.append(result)
            print(f"{i+1:>6} {result.get('robot_type','?'):>12} "
                  f"{result.get('fitness',0):>8.4f} "
                  f"{result.get('displacement',0)*100:>9.2f} "
                  f"{result.get('avg_speed',0):>8.3f} "
                  f"{result.get('survival_ratio',0)*100:>7.1f}%")
    
    if results:
        best = max(results, key=lambda x: x.get('fitness', 0))
        print(f"\nBest: fitness={best['fitness']:.4f}, disp={best['displacement']*100:.2f}cm")


def cmd_export(args):
    """导出模型"""
    from locomotion_templates import LocomotionTemplateGenerator
    from v8_3d_model_builder import V83DModelBuilder
    from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
    from forgecraft.simulation.builder import build_mjcf_model
    import json
    
    template_gen = LocomotionTemplateGenerator(seed=args.seed)
    population = template_gen.generate_population(size=1)
    body = population[0]
    catalog = load_catalog()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fmt = args.format.lower()
    
    if fmt == "stl":
        builder = V83DModelBuilder(catalog)
        meshes = builder.build_body_meshes(body)
        
        for name, mesh in meshes.items():
            mesh_path = output_dir / f"{name}.stl"
            mesh.export(str(mesh_path))
            print(f"Exported: {mesh_path}")
        
        # 组装体
        assembly = builder.create_assembly(meshes)
        assembly_path = output_dir / "assembly.stl"
        assembly.export(str(assembly_path))
        print(f"Assembly: {assembly_path}")
    
    elif fmt == "urdf":
        xml_string, joint_map, motor_map, _ = build_mjcf_model(body, catalog)
        urdf_path = output_dir / "robot.urdf"
        with open(urdf_path, 'w') as f:
            f.write(xml_string)
        print(f"URDF: {urdf_path}")
    
    elif fmt == "json":
        body_data = {
            "name": body.name,
            "parts": [
                {"id": p.part_id, "type": p.part_type, "pos": p.position.tolist(), 
                 "params": p.params}
                for p in body.parts.values()
            ],
            "joints": [
                {"parent": j.parent_id, "child": j.child_id, "type": j.joint_type,
                 "axis": j.axis.tolist(), "params": j.params}
                for j in body.joints
            ],
        }
        json_path = output_dir / "body.json"
        with open(json_path, 'w') as f:
            json.dump(body_data, f, indent=2)
        print(f"JSON: {json_path}")
    
    else:
        print(f"Unknown format: {fmt}. Supported: stl, urdf, json")


def cmd_report(args):
    """生成实验报告"""
    from pathlib import Path
    import json
    from datetime import datetime
    
    results_dir = Path("v11_results")
    
    if not results_dir.exists():
        print("No experiment results found. Run 'forgecraft evolve' first.")
        return
    
    # 加载数据
    final_results = results_dir / "v11_final_results.json"
    if not final_results.exists():
        print("No final results found.")
        return
    
    with open(final_results) as f:
        data = json.load(f)
    
    best = data.get("best_result", {})
    history = data.get("history", [])
    
    # 生成HTML报告
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ForgeCraft Experiment Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metric {{ display: inline-block; margin: 15px; padding: 20px; background: #f9f9f9; border-radius: 8px; min-width: 150px; text-align: center; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .footer {{ margin-top: 40px; text-align: center; color: #999; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ForgeCraft Evolution Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Best Individual</h2>
        <div class="metric">
            <div class="metric-value">{best.get('fitness', 0):.4f}</div>
            <div class="metric-label">Fitness</div>
        </div>
        <div class="metric">
            <div class="metric-value">{best.get('displacement', 0)*100:.2f}</div>
            <div class="metric-label">Displacement (cm)</div>
        </div>
        <div class="metric">
            <div class="metric-value">{best.get('speed', 0):.3f}</div>
            <div class="metric-label">Speed (m/s)</div>
        </div>
        <div class="metric">
            <div class="metric-value">{best.get('survival', 0)*100:.0f}%</div>
            <div class="metric-label">Survival Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{best.get('robot_type', '?')}</div>
            <div class="metric-label">Type</div>
        </div>
        <div class="metric">
            <div class="metric-value">{best.get('n_parts', 0)}</div>
            <div class="metric-label">Parts</div>
        </div>
        <div class="metric">
            <div class="metric-value">{best.get('n_motors', 0)}</div>
            <div class="metric-label">Motors</div>
        </div>
        
        <h2>Evolution History (Last 20 Generations)</h2>
        <table>
            <tr><th>Gen</th><th>Avg Fitness</th><th>Best Fitness</th><th>Avg Disp (cm)</th><th>Valid</th></tr>
"""
    
    for gen in history[-20:]:
        html += f"""<tr>
            <td>{gen.get('generation', 0)}</td>
            <td>{gen.get('avg_fitness', 0):.4f}</td>
            <td>{gen.get('best_fitness', 0):.4f}</td>
            <td>{gen.get('avg_displacement', 0)*100:.3f}</td>
            <td>{gen.get('valid_count', 0)}</td>
        </tr>\n"""
    
    html += f"""</table>
        
        <h2>Configuration</h2>
        <ul>
            <li>Total Generations: {len(history)}</li>
            <li>Population Size: {history[0].get('valid_count', 30) if history else 30}</li>
            <li>Evaluation: Direct Drive (MuJoCo)</li>
            <li>Templates: Differential, Quadruped, Bipedal, Crawler</li>
        </ul>
        
        <div class="footer">
            ForgeCraft Robot Evolution System | Powered by MuJoCo + Numba + CasADi
        </div>
    </div>
</body>
</html>"""
    
    output_path = Path(args.output) if args.output else Path("v11_results/report.html")
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Report generated: {output_path}")


def cmd_benchmark(args):
    """运行性能基准"""
    import time
    import numpy as np
    from locomotion_templates import LocomotionTemplateGenerator
    from direct_drive_evaluator import DirectDriveEvaluator, DirectDriveConfig
    from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
    
    print("\n" + "="*60)
    print("  ForgeCraft Performance Benchmark")
    print("="*60 + "\n")
    
    # 测试1：模板生成速度
    print("[1/4] Template Generation...")
    start = time.time()
    template_gen = LocomotionTemplateGenerator(seed=42)
    pop = template_gen.generate_population(size=30)
    gen_time = time.time() - start
    print(f"      Generated 30 robots in {gen_time:.2f}s ({gen_time/30*1000:.1f}ms/robot)")
    
    # 测试2：模型构建速度
    print("[2/4] Model Building...")
    from forgecraft.simulation.builder import build_mjcf_model
    catalog = load_catalog()
    
    build_times = []
    for body in pop[:10]:
        s = time.time()
        try:
            build_mjcf_model(body, catalog)
            build_times.append(time.time() - s)
        except:
            pass
    
    avg_build = np.mean(build_times) if build_times else 0
    print(f"      Avg build time: {avg_build*1000:.1f}ms")
    
    # 测试3：评估速度
    print("[3/4] Evaluation Speed...")
    eval_config = DirectDriveConfig(sim_steps=500, n_frequencies=1, episodes_per_freq=1)
    evaluator = DirectDriveEvaluator(eval_config)
    
    eval_times = []
    for body in pop[:5]:
        s = time.time()
        evaluator.evaluate(body, catalog)
        eval_times.append(time.time() - s)
    
    avg_eval = np.mean(eval_times) if eval_times else 0
    print(f"      Avg evaluation: {avg_eval:.2f}s")
    
    # 测试4：内存使用
    print("[4/4] Memory Usage...")
    import psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"      Current memory: {mem_mb:.1f} MB")
    
    # 汇总
    print("\n" + "-"*60)
    print("  Summary:")
    print(f"    Template Generation: {30/gen_time:.1f} robots/s")
    print(f"    Model Build:         {1/avg_build:.0f} models/s" if avg_build > 0 else "    Model Build: N/A")
    print(f"    Evaluation:          {1/avg_eval:.2f} evals/s" if avg_eval > 0 else "    Evaluation: N/A")
    print(f"    Memory:              {mem_mb:.1f} MB")
    print("="*60 + "\n")


def cmd_test(args):
    """运行测试套件"""
    import subprocess
    
    test_dir = Path(__file__).parent / "tests"
    
    if not test_dir.exists():
        print("No tests directory found.")
        return
    
    tests = list(test_dir.glob("test_*.py"))
    
    if not tests:
        print("No test files found.")
        return
    
    print(f"Running {len(tests)} test files...\n")
    
    passed = 0
    failed = 0
    
    for test_file in sorted(tests):
        print(f"  {test_file.name}...", end=" ")
        result = subprocess.run([sys.executable, str(test_file)], 
                              capture_output=True, timeout=60)
        if result.returncode == 0:
            print("PASSED")
            passed += 1
        else:
            print("FAILED")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)} tests")


def cmd_info(args):
    """显示系统信息"""
    import platform
    import mujoco
    import torch
    import numpy as np
    
    print("\n" + "="*60)
    print("  ForgeCraft System Information")
    print("="*60 + "\n")
    
    print(f"  OS:          {platform.system()} {platform.release()}")
    print(f"  Python:      {platform.python_version()}")
    print(f"  Platform:    {platform.machine()}")
    print()
    print(f"  MuJoCo:      {mujoco.__version__}")
    print(f"  NumPy:       {np.__version__}")
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  CUDA:        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:        {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # 检查实验结果
    results_dir = Path("v11_results")
    if results_dir.exists():
        final = results_dir / "v11_final_results.json"
        if final.exists():
            import json
            with open(final) as f:
                data = json.load(f)
            best = data.get("best_result", {})
            print(f"  Last Experiment:")
            print(f"    Best Fitness:     {best.get('fitness', 'N/A')}")
            print(f"    Best Type:        {best.get('robot_type', 'N/A')}")
            print(f"    Generations:      {len(data.get('history', []))}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        prog="forgecraft",
        description="ForgeCraft Robot Evolution System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # evolve 命令
    p_evolve = subparsers.add_parser("evolve", help="Run evolution experiment")
    p_evolve.add_argument("--gens", "-g", type=int, default=100, help="Number of generations")
    p_evolve.add_argument("--pop", "-p", type=int, default=30, help="Population size")
    p_evolve.add_argument("--evals", "-e", type=int, default=4, help="Evaluations per individual")
    p_evolve.add_argument("--elite", type=int, default=3, help="Elite size")
    p_evolve.add_argument("--output", "-o", default="v11_results", help="Output directory")
    
    # view 命令
    p_view = subparsers.add_parser("view", help="View robot in 3D simulation")
    p_view.add_argument("--template", "-t", choices=["diff", "four", "bipedal", "quad", "crawler"])
    p_view.add_argument("--headless", action="store_true", help="Headless mode (screenshots)")
    p_view.add_argument("--duration", "-d", type=float, help="Simulation duration (seconds)")
    p_view.add_argument("--no-drive", action="store_true", help="Disable auto driving")
    p_view.add_argument("--speed", "-s", type=float, default=0.8, help="Drive speed")
    p_view.add_argument("--no-hud", action="store_true", help="Disable HUD")
    p_view.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # evaluate 命令
    p_eval = subparsers.add_parser("evaluate", help="Evaluate robot performance")
    p_eval.add_argument("--template", "-t", choices=["diff", "four", "bipedal", "quad", "crawler"])
    p_eval.add_argument("--count", "-n", type=int, default=5, help="Number of robots to evaluate")
    p_eval.add_argument("--steps", type=int, default=1500, help="Simulation steps")
    p_eval.add_argument("--frequencies", "-f", type=int, default=3, help="Test frequencies")
    p_eval.add_argument("--episodes", type=int, default=2, help="Episodes per frequency")
    p_eval.add_argument("--seed", type=int, default=42)
    
    # export 命令
    p_export = subparsers.add_parser("export", help="Export robot model")
    p_export.add_argument("--format", "-f", choices=["stl", "urdf", "json"], default="stl", help="Output format")
    p_export.add_argument("--output", "-o", default="exported_models", help="Output directory")
    p_export.add_argument("--seed", type=int, default=42)
    
    # report 命令
    p_report = subparsers.add_parser("report", help="Generate experiment report")
    p_report.add_argument("--output", "-o", help="Output file path")
    
    # benchmark 命令
    subparsers.add_parser("benchmark", help="Run performance benchmark")
    
    # test 命令
    subparsers.add_parser("test", help="Run test suite")
    
    # info 命令
    subparsers.add_parser("info", help="Show system information")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # 配置日志
    log_level = getattr(logging, os.environ.get("FORGECRAFT_LOG", "INFO").upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 路由到命令处理函数
    commands = {
        "evolve": cmd_evolve,
        "view": cmd_view,
        "evaluate": cmd_evaluate,
        "export": cmd_export,
        "report": cmd_report,
        "benchmark": cmd_benchmark,
        "test": cmd_test,
        "info": cmd_info,
    }
    
    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
