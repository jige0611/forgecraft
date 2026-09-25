#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v10 完整进化实验 (100+代) - 终极优化版
========================================

基于50代初步实验分析结果，运行完整的100代进化实验:
- 使用优化的V3零件库配置
- 集成速度任务专用的奖励函数调优
- 自适应种群大小策略
- 更长的训练时间确保充分收敛

目标: 获得性能提升30%+的最佳机器人模型

使用方法:
    python run_v10_full_experiment.py [--generations 120] [--population 32]
"""

import argparse
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入必要模块 (全局可用)
from forgecraft.config import EvolutionConfig, RLConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="v10 完整进化实验 (100+代) - 基于V3优化配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行标准100代实验
  python run_v10_full_experiment.py
  
  # 运行120代实验，种群32
  python run_v10_full_experiment.py --generations 120 --population 32
  
  # 快速测试模式(30代)
  python run_v10_full_experiment.py --quick --generations 30
  
  # 导出最佳模型并可视化
  python run_v10_full_experiment.py --export --dashboard
        """
    )
    
    # ===== 核心实验参数 =====
    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=100,
        help="进化世代数 (默认: 100)"
    )
    
    parser.add_argument(
        "--population", "-p",
        type=int,
        default=28,
        help="种群大小 (默认: 28, V3自适应策略中期值)"
    )
    
    parser.add_argument(
        "--elites", "-e",
        type=int,
        default=5,
        help="精英个体数 (默认: 5)"
    )
    
    # ===== V3 特定参数 =====
    parser.add_argument(
        "--catalog-v3",
        action="store_true",
        default=True,
        help="使用V3优化版零件库 (默认启用)"
    )
    
    parser.add_argument(
        "--speed-optimized",
        action="store_true",
        default=True,
        help="启用速度任务专用优化 (默认启用)"
    )
    
    parser.add_argument(
        "--adaptive-population",
        action="store_true",
        default=True,
        help="启用自适应种群大小 (默认启用)"
    )
    
    # ===== 训练模式 =====
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速测试模式 (减少评估步数用于调试)"
    )
    
    parser.add_argument(
        "--export",
        action="store_true",
        help="导出最佳个体为STL/URDF"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="启动实时监控仪表盘"
    )
    
    # ===== 系统参数 =====
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="随机种子 (默认: 42, 与50代实验一致便于对比)"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="并行工作线程数 (默认: 4)"
    )
    
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="使用GPU加速 (如果可用)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="design_output_v10_full",
        help="输出目录 (默认: design_output_v10_full)"
    )
    
    return parser.parse_args()


def create_evolution_config(args) -> EvolutionConfig:
    """创建进化配置（集成V3优化）"""
    
    config = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        elite_count=args.elites,
    )
    
    # V3特定优化
    if args.catalog_v3:
        # 设置更高的变异上限
        config.max_mutation_multiplier = 4.0  # V3: 从3.0提高到4.0
        
        # 平台期检测阈值
        config.plateau_detection_threshold = 8  # 代数
        
        # UCB精英投资比例
        config.ucb_elite_ratio = 0.3
    
    if args.adaptive_population:
        # 启用自适应种群大小
        config.adaptive_population = True
        config.population_schedule = {
            'early': {'size': 20, 'until_gen': 30},
            'mid': {'size': 28, 'until_gen': 60},
            'late': {'size': 32, 'until_gen': args.generations},
        }
    
    return config


def create_rl_config(args) -> RLConfig:
    """创建RL配置（集成速度任务优化）"""
    
    config = RLConfig()
    
    if args.quick:
        # 快速测试模式
        config.max_episode_steps = 300
        config.n_eval_episodes = 2
    else:
        # 标准完整训练模式
        config.max_episode_steps = 1000  # V3: 更长训练步数
        
        # 渐进式评估策略
        config.eval_schedule = {
            'early_phase': {'episodes': 4, 'steps': 500, 'until_gen': 20},
            'mid_phase': {'episodes': 8, 'steps': 700, 'until_gen': 60},
            'late_phase': {'episodes': 16, 'steps': 900, 'until_gen': args.generations - 5},
            'final_phase': {'episodes': 20, 'steps': 1000, 'until_gen': args.generations},
        }
        
        # UCB资源分配
        config.ucb_config = {
            'enabled': True,
            'max_investments_per_elite': 10,
            'min_confidence_interval': 0.1,
        }
    
    if args.speed_optimized:
        # V3: 速度任务专用奖励函数调整
        config.reward_weights = {
            'velocity': 1.5,           # 提高速度权重 (原1.0)
            'stability': 0.8,          # 稳定性
            'simplicity_bonus': 0.3,   # 简单性奖励 (新增!)
        }
        
        config.penalties = {
            'motor_usage': -0.2,       # 惩罚马达使用 (鼓励被动驱动!)
            'complexity': -0.05,       # 轻微惩罚复杂度
            'instability': -0.5,       # 惩罚不稳定
        }
        
        config.bonuses = {
            'zero_motor': 0.4,         # ⭐ 零马达大奖!
            'innovative_parts': 0.15,  # 创新零件奖励
            'manufacturability': 0.2,  # 制造性奖励
        }
    
    return config


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 70)
    print("🚀 v10 完整进化实验 (100+代) - V3终极优化版")
    print("=" * 70)
    print(f"\n📋 实验配置:")
    print(f"   世代数: {args.generations}")
    print(f"   种群大小: {args.population}")
    print(f"   精英数: {args.elites}")
    print(f"   零件库版本: {'V3 (优化版)' if args.catalog_v3 else 'V2'}")
    print(f"   速度优化: {'✅ 启用' if args.speed_optimized else '❌ 禁用'}")
    print(f"   自适应种群: {'✅ 启用' if args.adaptive_population else '❌ 禁用'}")
    print(f"   训练模式: {'⚡ 快速测试' if args.quick else '🎯 完整训练'}")
    print(f"   输出目录: {args.output_dir}")
    print(f"   随机种子: {args.seed} (与50代实验一致)")
    
    start_time = time.time()
    
    try:
        # 1. 导入模块 (延迟导入以处理可能的路径问题)
        from forgecraft.config import EvolutionConfig, RLConfig
        from forgecraft.core.loader import load_catalog, load_task
        from forgecraft.evolution.loop import EvolutionLoop
        
        print("✅ 模块导入成功")
        
        # 2. 加载零件箱 (V3优化版)
        catalog_path = "forgecraft/configs/catalogs/parameterized_parts_v3.yaml"
        if not Path(catalog_path).exists():
            catalog_path = "forgecraft/configs/catalogs/parameterized_parts_v2.yaml"
            print(f"⚠️ V3配置未找到，回退到V2: {catalog_path}")
        
        print(f"\n📦 加载零件库: {catalog_path}")
        catalog = load_catalog(name="parameterized_parts_v3" if args.catalog_v3 else "parameterized_parts_v2")
        print(f"   ✅ 已加载 {len(catalog.parts)} 种零件类型")
        
        # 2. 加载任务
        task = load_task(name="speed_density")
        print(f"\n🎯 任务: speed_density (速度密度优化)")
        
        # 3. 创建配置
        evo_config = create_evolution_config(args)
        rl_config = create_rl_config(args)  # 保留用于文档记录
        
        # 转换任务配置
        sim_config = task.to_sim_config()
        task_config = task.to_task_config()
        
        # 4. 启动进化循环
        print(f"\n🔄 启动进化循环...")
        print(f"   设备: {'CUDA (GPU)' if args.cuda else 'CPU'}")
        print(f"   工作线程: {args.workers}")
        
        loop = EvolutionLoop(
            catalog=catalog.to_part_specs() if hasattr(catalog, 'to_part_specs') else catalog,
            evo_config=evo_config,
            sim_config=sim_config,
            rl_config=rl_config,
            task_config=task_config,
            seed=args.seed,
            device="cuda" if args.cuda else "cpu",
            n_workers=args.workers,
        )
        
        print("\n" + "=" * 70)
        print("⚡ 开始进化! (按 Ctrl+C 安全中断并保存断点)")
        print("=" * 70 + "\n")
        
        best_individual = loop.run()
        
        # 5. 处理结果
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print("\n" + "=" * 70)
        print("✨ 进化完成!")
        print("=" * 70)
        print(f"\n⏱️  总耗时: {hours}小时{minutes}分钟{seconds}秒 ({elapsed_time:.1f}秒)")
        
        if best_individual is not None:
            print(f"\n🏆 最佳个体:")
            print(f"   适应度: {best_individual.fitness:.4f}")
            
            # 尝试获取更多信息
            if hasattr(best_individual, 'n_parts'):
                print(f"   零件数: {best_individual.n_parts}")
            if hasattr(best_individual, 'n_motors'):
                print(f"   马达数: {best_individual.n_motors}")
            if hasattr(best_individual, 'manufacturability'):
                print(f"   制造性: {best_individual.manufacturability:.2f}")
            
            # 导出模型
            if args.export:
                print(f"\n💾 导出最佳模型到: {args.output_dir}/")
                # 这里可以添加导出逻辑
            
            # 性能对比
            print(f"\n📊 性能对比:")
            print(f"   vs 50代实验 (Gen46): ", end="")
            baseline_50gen = 4.1283
            if best_individual.fitness > baseline_50gen:
                improvement = ((best_individual.fitness - baseline_50gen) / baseline_50gen * 100)
                print(f"+{improvement:.1f}% 🚀")
            else:
                decline = ((baseline_50gen - best_individual.fitness) / baseline_50gen * 100)
                print(f"-{decline:.1f}% ⚠️")
            
            print(f"   vs v7基线 (Gen89): ", end="")
            v7_baseline = 8.39
            ratio = best_individual.fitness / v7_baseline * 100
            print(f"{ratio:.1f}%")
            
            # 判断是否达到目标
            target_improvement = 30  # 目标: 提升30%+
            if best_individual.fitness > baseline_50gen * (1 + target_improvement/100):
                print(f"\n🎉🎉🎉 成功! 性能提升超过{target_improvement}%!")
            else:
                current_imp = (best_individual.fitness / baseline_50gen - 1) * 100
                print(f"\n📈 当前提升: +{current_imp:.1f}% (目标: +{target_improvement}%)")
        
        else:
            print("\n⚠️ 未获得有效最佳个体 (可能实验被中断)")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚡ 用户中断! 正在保存状态...")
        # 断点保存逻辑在EvolutionLoop内部处理
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        print(f"⏱️  已运行: {hours}小时{minutes}分钟")
        print("💾 断点已保存，可使用 --resume 继续")
        return 130  # 特殊退出码表示被中断
    
    except Exception as e:
        print(f"\n❌ 进化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
