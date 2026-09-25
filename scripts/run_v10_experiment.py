"""
v10 实验: 创新零件集成测试

目标:
- 首次在完整进化流程中使用Phase 1创新零件
- 验证弹簧、中空管、半球脚垫在实际进化中的表现
- 对比v7基线性能，评估提升幅度

实验配置:
- 零件箱: parameterized_parts_v2 (9种零件: 6基础 + 3创新)
- 任务: speed_density (单位体积最快速度)
- 种群: 20 (适度规模，便于观察)
- 世代: 30 (快速验证)

运行方式:
    python run_v10_experiment.py
    
或使用main.py:
    python -m forgecraft.main --catalog parameterized_parts_v2 --task speed_density --generations 30 --population 20
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="v10: 创新零件集成实验")
    
    # 基础参数
    parser.add_argument("--generations", type=int, default=30, help="进化世代数 (默认: 30)")
    parser.add_argument("--population", type=int, default=20, help="种群大小 (默认: 20)")
    parser.add_argument("--elites", type=int, default=4, help="精英个体数 (默认: 4)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 硬件配置
    parser.add_argument("--cuda", action="store_true", help="启用GPU加速")
    parser.add_argument("--workers", type=int, default=4, help="并行进程数 (默认: 4)")
    
    # V2特定参数
    parser.add_argument("--catalog", type=str, default="parameterized_parts_v2", 
                        help="零件箱 (默认: parameterized_parts_v2)")
    parser.add_argument("--task", type=str, default="speed_density",
                        help="任务 (默认: speed_density)")
    
    # 输出选项
    parser.add_argument("--output-dir", type=str, default="design_output_v10",
                        help="输出目录 (默认: design_output_v10)")
    parser.add_argument("--export", action="store_true", 
                        help="完成后自动导出STL/URDF/ONNX")
    parser.add_argument("--dashboard", action="store_true",
                        help="完成后启动可视化仪表盘")
    
    # 调试/快速模式
    parser.add_argument("--quick", action="store_true", 
                        help="快速测试模式 (减少episode步数)")
    parser.add_argument("--dry-run", action="store_true",
                        help="干运行: 仅初始化不执行进化 (用于调试)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 v10 实验: Phase 1 创新零件集成测试")
    print("=" * 70)
    
    print("\n📋 实验配置:")
    print(f"   📦 零件箱: {args.catalog} (包含spring_element, hollow_tube, hemisphere_foot)")
    print(f"   🎯 任务: {args.task}")
    print(f"   👥 种群: {args.population}")
    print(f"   🔄 世代: {args.generations}")
    print(f"   👑 精英: {args.elites}")
    print(f"   💻 设备: {'CUDA GPU' if args.cuda else 'CPU'}")
    print(f"   📁 输出: {args.output_dir}/")
    
    if args.quick:
        print("   ⚡ 模式: 快速测试")
    if args.dry_run:
        print("   🔍 模式: 干运行 (无实际进化)")
    
    # 导入forgecraft主模块
    try:
        from forgecraft.config import EvolutionConfig, RLConfig
        from forgecraft.core.loader import load_catalog, load_task
        from forgecraft.evolution.loop import EvolutionLoop
        
        print("\n✅ 模块导入成功")
        
    except ImportError as e:
        print(f"\n❌ 模块导入失败: {e}")
        print("   请确保已安装所有依赖: pip install -r requirements.txt")
        return False
    
    # 加载catalog和task
    print(f"\n📂 加载零件箱: {args.catalog}...")
    try:
        catalog = load_catalog(name=args.catalog)
        print(f"   ✅ 成功! 共{len(catalog.parts)}种零件可用")
        
        # 显示创新零件
        innovative_parts = ['spring_element', 'hollow_tube', 'hemisphere_foot']
        available_innovative = [p for p in innovative_parts if p in catalog.parts]
        
        if available_innovative:
            print(f"\n   🆕 创新零件已加载:")
            for part_name in available_innovative:
                spec = catalog.parts[part_name]
                print(f"      • {part_name}: {spec.geometry_type} (质量: {spec.mass:.3f}kg)")
        else:
            print(f"   ⚠️ 未找到创新零件，将仅使用基础零件")
            
    except Exception as e:
        print(f"   ❌ 加载零件箱失败: {e}")
        return False
    
    print(f"\n🎯 加载任务: {args.task}...")
    try:
        task = load_task(name=args.task)
        print(f"   ✅ 成功! {task.description.split(chr(10))[0][:100]}")
    except Exception as e:
        print(f"   ❌ 加载任务失败: {e}")
        return False
    
    if args.dry_run:
        print("\n🔍 干运行模式 - 跳过实际进化")
        print("\n✅ 配置验证完成! 可以移除 --dry-run 参数运行完整实验")
        return True
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置进化参数
    evo_config = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        elite_count=args.elites,
    )
    
    # 配置RL参数
    rl_config = RLConfig()
    if args.quick:
        rl_config.max_episode_steps = 300
    
    # 转换为SimConfig和TaskConfig
    sim_config = task.to_sim_config()
    task_config = task.to_task_config()
    
    # 启动进化循环
    print(f"\n{'='*70}")
    print(f"🧬 开始进化...")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        loop = EvolutionLoop(
            catalog=catalog.to_part_specs(),
            evo_config=evo_config,
            sim_config=sim_config,
            rl_config=rl_config,
            task_config=task_config,
            seed=args.seed,
            device="cuda" if args.cuda else "cpu",
            n_workers=args.workers,
        )
        
        best_individual = loop.run()
        
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        print(f"\n{'='*70}")
        print(f"✨ 进化完成!")
        print(f"{'='*70}")
        print(f"⏱️  总耗时: {hours}小时{minutes}分钟 ({elapsed:.0f}秒)")
        print(f"🏆 最佳适应度: {best_individual.fitness:.4f}")
        print(f"📊 零件数: {len(best_individual.body.parts)}")
        
        # 统计创新零件使用情况
        innovative_usage = {}
        for part in best_individual.body.parts:
            if part.part_type in innovative_parts:
                innovative_usage[part.part_type] = innovative_usage.get(part.part_type, 0) + 1
        
        if innovative_usage:
            print(f"\n🆕 创新零件使用统计:")
            for part_type, count in innovative_usage.items():
                print(f"   • {part_type}: {count}个")
        else:
            print(f"\n⚠️ 最佳个体未使用创新零件")
        
        # 自动导出
        if args.export:
            print(f"\n📦 导出制造文件到 {args.output_dir}/...")
            
            from forgecraft.manufacturing.advanced_mesh_builder import AdvancedMeshBuilder
            from forgecraft.manufacturing.advanced_mesh_builder_v2 import AdvancedMeshBuilderV2
            
            builder = AdvancedMeshBuilderV2(quality_level="high")
            
            export_path = os.path.join(args.output_dir, "manufacturing")
            os.makedirs(export_path, exist_ok=True)
            
            for i, part in enumerate(best_individual.body.parts):
                try:
                    if part.part_type == "spring_element":
                        mesh = builder.create_spring_element(
                            stiffness=part.params.get('stiffness', 500),
                            max_deformation=part.params.get('max_deformation', 0.02),
                        )
                    elif part.part_type == "hollow_tube":
                        mesh = builder.create_hollow_tube(
                            outer_diameter=part.params.get('outer_diameter', 0.03),
                            length=part.params.get('length', 0.15),
                        )
                    elif part.part_type == "hemisphere_foot":
                        mesh = builder.create_hemisphere_foot(
                            radius=part.params.get('radius', 0.015),
                        )
                    else:
                        continue
                    
                    stl_path = os.path.join(export_path, f"{part.part_type}_{i}.stl")
                    builder.export_part_to_stl(mesh, stl_path)
                    
                except Exception as e:
                    print(f"   ⚠️ 导出{part.part_type}失败: {e}")
            
            print(f"   ✅ 制造文件导出完成")
        
        # 启动仪表盘
        if args.dashboard:
            print(f"\n🖥️  启动Web仪表盘...")
            os.system(f"python -m forgecraft.dashboard --output-dir {args.output_dir}")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ 用户中断进化过程")
        return False
        
    except Exception as e:
        print(f"\n❌ 进化过程中发生错误:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
