#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v10-V4 修正版进化实验 (100代) - 强制主动驱动
=============================================

⚠️ 重要修正: 基于V3实验的教训，重新设计奖励函数

核心修正:
1. 移除零马达奖励 (之前的严重错误!)
2. 惩罚零马达设计 (视为无效)
3. 强制要求至少2个主动驱动马达
4. 弹簧仅作为辅助零件，不能替代马达

目标: 获得真正可用的、自包含的、依靠马达驱动的机器人模型

使用方法:
    python run_v10_v4_corrected.py [--generations 100]
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from forgecraft.config import EvolutionConfig, RLConfig
    from forgecraft.evolution.loop import EvolutionLoop
    from forgecraft.core.loader import load_task, load_catalog
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="v10-V4 修正版进化实验 - 强制主动驱动",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--generations", "-g", type=int, default=100,
                        help="进化世代数 (默认: 100)")
    parser.add_argument("--population", "-p", type=int, default=28,
                        help="种群大小 (默认: 28)")
    parser.add_argument("--elites", "-e", type=int, default=5,
                        help="精英个体数 (默认: 5)")
    
    # V4特定参数
    parser.add_argument("--catalog-v4", action="store_true", default=True,
                        help="使用V4修正版零件库 (默认启用)")
    parser.add_argument("--min-motors", type=int, default=2,
                        help="最小马达数量要求 (默认: 2)")
    parser.add_argument("--enforce-motors", action="store_true", default=True,
                        help="强制执行马达数量要求 (默认启用)")
    
    # 训练模式
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式 (30代)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    
    return parser.parse_args()


def create_evolution_config(args) -> EvolutionConfig:
    """创建进化配置"""
    config = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        elite_count=args.elites,
    )
    
    if args.quick:
        config.generations = 30
        config.population_size = 16
    
    return config


def create_rl_config(args) -> RLConfig:
    """创建强化学习配置 (V4修正版!)"""
    config = RLConfig()
    
    # V4关键修正: 正确的奖励函数!
    if args.catalog_v4:
        config.reward_weights = {
            'velocity': 1.5,           # 保持速度权重
            'stability': 0.8,
            'simplicity_bonus': 0.2,   # 降低简单性奖励
        }
        
        # ⚠️ V4: 关键修正!
        config.penalties = {
            'zero_motor_penalty': -1.0,          # 严重惩罚零马达!!!
            'insufficient_motors': -0.5,         # 马达不足惩罚
            'external_dependency': -0.8,         # 外部依赖惩罚
        }
        
        config.bonuses = {
            'motor_usage_bonus': 0.3,            # ✅ 奖励使用马达
            'optimal_motor_count': 0.2,          # 马达数量适中时奖励
            'energy_efficiency': 0.25,           # 能效奖励
            'innovative_part_usage': 0.1,        # 创新零件小奖励
        }
        
        # V4: 设计验证规则
        config.validation_rules = {
            'min_motors_required': args.min_motors,
            'must_be_self_contained': True,
            'spring_role': 'auxiliary_only',     # 弹簧只能辅助!
        }
    
    return config


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 70)
    print("🤖 v10-V4 修正版进化实验")
    print("=" * 70)
    print(f"⚠️  关键修正: 强制要求主动驱动 (最少{args.min_motors}个马达)")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    try:
        # 1. 加载任务配置
        print("📋 加载任务配置...")
        task_path = project_root / "forgecraft" / "configs" / "tasks" / "speed.yaml"
        task = load_task(task_path)
        
        # 2. 加载V4零件库
        print("📦 加载V4修正版零件库...")
        catalog_path = project_root / "forgecraft" / "configs" / "catalogs" / "parameterized_parts_v4.yaml"
        
        if not catalog_path.exists():
            print(f"❌ V4配置文件不存在: {catalog_path}")
            return False
        
        catalog = load_catalog(catalog_path)
        print(f"   ✅ 成功加载V4零件库")
        print(f"   📌 核心规则:")
        print(f"      - 最少马达数: {args.min_motors}")
        print(f"      - 零马达惩罚: -1.0")
        print(f"      - 弹簧角色: 仅辅助")
        
        # 转换为零件规格字典 (EvolutionLoop需要字典格式)
        catalog_dict = catalog.to_part_specs()
        
        # 3. 创建配置
        evo_config = create_evolution_config(args)
        rl_config = create_rl_config(args)
        
        # 4. 初始化进化循环
        print("\n🔄 初始化进化循环...")
        loop = EvolutionLoop(
            evo_config=evo_config,
            sim_config=task.to_sim_config(),
            task_config=task.to_task_config(),
            rl_config=rl_config,
            catalog=catalog_dict,  # 使用字典格式
            seed=args.seed,
        )
        
        # 5. 运行进化
        print(f"\n🚀 开始进化训练 ({args.generations}代)...")
        print(f"   种群大小: {args.population}")
        print(f"   精英数量: {args.elites}")
        print(f"   马达要求: ≥{args.min_motors}个")
        print()
        
        best_individual = loop.run()
        
        # 6. 输出结果
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("✨ 进化完成!")
        print("=" * 70)
        
        if best_individual is not None:
            fitness = getattr(best_individual, 'fitness', None)
            n_parts = getattr(best_individual, 'n_parts', 0)
            n_motors = getattr(best_individual, 'n_motors', 0)
            
            print(f"\n🏆 最佳个体信息:")
            print(f"   适应度: {fitness}")
            print(f"   零件数: {n_parts}")
            print(f"   ⚡ 马达数: {n_motors}")
            
            # V4验证!
            if n_motors >= args.min_motors:
                print(f"\n✅✅✅ 验证通过! 这是一个真正的机器人!")
                print(f"   包含{n_motors}个主动驱动马达")
                print(f"   可以自主运行，不依赖外部能量")
            else:
                print(f"\n❌❌❌ 警告! 马达数({n_motors})不足要求({args.min_motors})!")
                print(f"   这可能意味着V4配置未正确应用")
                print(f"   或算法找到了规避方法")
            
        else:
            print("\n⚠️ 未获得有效最佳个体")
        
        print(f"\n⏱️  总耗时: {elapsed_time/3600:.1f}小时 ({elapsed_time:.1f}秒)")
        print(f"📅 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 实验出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
