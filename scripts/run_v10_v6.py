#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v10-V6 平衡修正版进化实验 (100代) - 中等约束+分阶段策略
=============================================================

🎯 核心改进: 基于V4/V5教训，找到最佳平衡点!

V4失败: -1.0惩罚太弱 → 最佳个体0个马达
V5失败: -10.0惩罚+硬约束太强 → 适应度仅0.24, 19代停滞
V6方案: **中等强度(-5.0) + 分阶段约束 + 充分探索空间**

关键特性:
1. 📈 分阶段策略 (先探索后聚焦)
2. ⚖️ 平衡目标 (60%自主性 + 40%性能)
3. 🔄 中等惩罚 (-5.0, 不极端)
4. 🌳 充足多样性 (避免局部最优)
5. 🎯 双重成功标准 (合规性 AND 性能)

目标: 获得包含≥2个马达 且 适应度>0.40 的真正自主机器人!

使用方法:
    python run_v10_v6.py [--generations 100]
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
        description="v10-V6 平衡修正版进化实验 - 中等约束+分阶段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--generations", "-g", type=int, default=100,
                        help="进化世代数 (默认: 100)")
    parser.add_argument("--population", "-p", type=int, default=34,
                        help="种群大小 (默认: 34, V6适中)")
    parser.add_argument("--elites", "-e", type=int, default=6,
                        help="精英个体数 (默认: 6)")
    
    # V6特定参数
    parser.add_argument("--catalog-v6", action="store_true", default=True,
                        help="使用V6平衡版零件库 (默认启用)")
    parser.add_argument("--min-motors", type=int, default=2,
                        help="最小马达数量要求 (默认: 2)")
    parser.add_argument("--balanced-mode", action="store_true", default=True,
                        help="启用平衡模式 (60-40自主性-性能)")
    
    # 训练模式
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式 (30代)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    
    return parser.parse_args()


def get_phase_config(current_generation, total_generations=100):
    """
    🎯 V6核心函数: 分阶段配置获取
    
    根据当前代数返回对应的约束参数:
    
    阶段1 (第0-30代): 宽松探索期
      - 较轻惩罚 (-3.0)
      - 高多样性 (+50%)
      - 目标: 广泛探索解空间
    
    阶段2 (第31-65代): 中度优化期  
      - 中等惩罚 (-5.0)
      - 正常多样性
      - 目标: 优化有潜力的设计
    
    阶段3 (第66-100代): 收敛验证期
      - 较强惩罚 (-7.0)
      - 低多样性(专注收敛)
      - 目标: 收敛到最优合规设计
    """
    
    progress = current_generation / total_generations
    
    if current_generation <= 30:
        # 阶段1: 探索期
        phase = 1
        config = {
            'name': 'exploration',
            'motor_penalty': -3.0,
            'insufficient_penalty': -1.5,
            'external_dep_penalty': -1.8,
            'motor_bonus': 0.35,
            'autonomy_bonus': 0.4,
            'diversity_multiplier': 1.5,
            'mutation_range': (1.5, 4.0),
            'goal': '广泛探索解空间',
        }
    elif current_generation <= 65:
        # 阶段2: 优化期
        phase = 2
        config = {
            'name': 'refinement',
            'motor_penalty': -5.0,
            'insufficient_penalty': -2.5,
            'external_dep_penalty': -3.0,
            'motor_bonus': 0.5,
            'autonomy_bonus': 0.6,
            'diversity_multiplier': 1.0,
            'mutation_range': (1.8, 3.5),
            'goal': '优化有潜力的设计',
        }
    else:
        # 阶段3: 收敛期
        phase = 3
        config = {
            'name': 'convergence',
            'motor_penalty': -7.0,
            'insufficient_penalty': -3.5,
            'external_dep_penalty': -4.0,
            'motor_bonus': 0.6,
            'autonomy_bonus': 0.7,
            'diversity_multiplier': 0.7,
            'mutation_range': (2.0, 3.0),
            'goal': '收敛到最优合规设计',
        }
    
    return phase, config


def create_evolution_config(args) -> EvolutionConfig:
    """创建进化配置 (V6平衡版)"""
    config = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        elite_count=args.elites,
    )
    
    if args.quick:
        config.generations = 30
        config.population_size = 22
    
    return config


def create_rl_config(args) -> RLConfig:
    """创建强化学习配置 (V6平衡修正版!)"""
    config = RLConfig()
    
    if args.catalog_v6:
        # 🎯 V6: 平衡的双目标!
        config.reward_weights = {
            'velocity': 1.2,             # 适中的速度权重
            'stability': 0.7,             # 适中的稳定性
            'autonomy': 1.8,              # 🆕 平衡的自主性权重 (不是3.0!)
            'simplicity_bonus': 0.15,     # 适中的简单性奖励
        }
        
        # ⚖️⚖️⚖️ V6: 中等强度惩罚 (关键平衡点!)
        config.penalties = {
            'zero_motor_penalty': -5.0,       # 从V5的-10.0降到-5.0! (折中!)
            'insufficient_motors_penalty': -2.5, # 从-5.0降到-2.5! 
            'external_dependency_penalty': -3.0, # 从-5.0降到-3.0!
            'passive_design_penalty': -4.0,     # 从-8.0降到-4.0!
            'complexity_penalty': -0.015,       
            'instability_penalty': -0.25,       
            # ❌ 移除: violating_hard_constraint (-100.0) ← 太极端!
        }
        
        # ✨ V6: 适度奖励 (不夸张)
        config.bonuses = {
            'motor_usage_bonus': 0.5,           # 从0.8降到0.5
            'optimal_motor_count_bonus': 0.35,   # 从0.5降到0.35
            'self_contained_bonus': 0.4,         # 从0.6降到0.4
            'autonomy_achievement_bonus': 0.6,   # 从1.0降到0.6
            'energy_efficiency_bonus': 0.2,      
            'innovative_part_usage': 0.08,       
            'manufacturability_bonus': 0.12,      
        }
        
        # 🆕 V6: 分阶段配置标记
        config.validation_rules = {
            'min_motors_required': args.min_motors,
            'must_be_self_contained': True,
            'spring_role': 'auxiliary_only',
            'phased_constraints_enabled': True,  # 启用分阶段!
            'hard_constraint_disabled': True,    # 禁用硬约束!
            'balance_mode': '60_40',            # 60%自主性/40%性能
        }
        
        # 🆕 V6: 多目标平衡配置
        config.multi_objective = {
            'enabled': True,
            'objectives': [
                {'name': 'autonomy', 'weight': 1.8, 'priority': 1},
                {'name': 'performance', 'weight': 1.2, 'priority': 2},
                {'name': 'quality', 'weight': 0.7, 'priority': 3},
            ],
            'balance_strategy': 'weighted_sum_with_phases',
        }
    
    return config


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 70)
    print("🤖🤖🤖 v10-V6 平衡修正版进化实验 🤖🤖🤖")
    print("=" * 70)
    print("🎯 核心改进: 基于V4/V5教训，找到最佳平衡点!")
    print()
    print("📊 V6三大关键改进:")
    print("   1. 📈 分阶段策略 (先探索后聚焦)")
    print("   2. ⚖️ 平衡目标 (60%自主性 + 40%性能)")
    print("   3. 🔄 中等惩罚 (-5.0, 不极端)")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    try:
        # 1. 加载任务配置
        print("📋 加载任务配置...")
        task_path = project_root / "forgecraft" / "configs" / "tasks" / "speed.yaml"
        task = load_task(task_path)
        print("   ✅ 任务配置加载成功")
        
        # 2. 加载V6零件库
        print("\n📦 加载V6平衡版零件库...")
        catalog_path = project_root / "forgecraft" / "configs" / "catalogs" / "parameterized_parts_v6.yaml"
        
        if not catalog_path.exists():
            print(f"❌ V6配置文件不存在: {catalog_path}")
            return False
        
        catalog = load_catalog(catalog_path)
        print(f"   ✅ 成功加载V6零件库")
        print(f"\n   📌 V6核心规则:")
        print(f"      ⚖️ 策略: 平衡模式 (60%自主性 / 40%性能)")
        print(f"      📈 方法: 分阶段约束 (探索→优化→收敛)")
        print(f"      💥 惩罚强度: -5.0 (中等, V5的一半!)")
        print(f"      ⚡ 奖励力度: +0.5 (适度, 不过度)")
        print(f"      🔒 硬约束: 已禁用 (避免过度限制!)")
        print(f"\n   📋 分阶段计划:")
        print(f"      阶段1 (0-30代):  宽松探索, 惩罚×0.6")
        print(f"      阶段2 (31-65代): 中度优化, 惩罚×1.0")
        print(f"      阶段3 (66-100代): 收敛验证, 惩罚×1.0~1.2")
        
        # 转换为零件规格字典
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
            catalog=catalog_dict,
            seed=args.seed,
        )
        
        # 🆕 V6: 注入分阶段配置器 (如果支持)
        if hasattr(loop, 'set_phase_configurator'):
            loop.set_phase_configurator(
                get_phase_func=get_phase_config,
                enabled=True
            )
            print("   ✅ 分阶段配置器已注入")
        
        # 5. 运行进化
        print(f"\n🚀🚀🚀 开始V6平衡进化训练 ({args.generations}代) 🚀🚀🚀")
        print(f"   种群大小: {args.population} (平衡)")
        print(f"   精英数量: {args.elites}")
        print(f"   ⚖️ 平衡模式: 60%自主性 / 40%性能")
        print(f"   📈 分阶段: 是 (3个阶段自动调整)")
        print(f"   💥 惩罚: 中等强度 (-5.0)")
        print()
        
        best_individual = loop.run()
        
        # 6. 输出结果 (V6验证!)
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("✨✨✨ V6 进化完成! ✨✨✨")
        print("=" * 70)
        
        if best_individual is not None:
            fitness = getattr(best_individual, 'fitness', None)
            n_parts = getattr(best_individual, 'n_parts', 0)
            n_motors = getattr(best_individual, 'n_motors', 0)
            
            print(f"\n🏆🏆🏆 最佳个体信息:")
            print(f"   适应度: {fitness}")
            print(f"   零件数: {n_parts}")
            print(f"   ⚡⚡⚡ 马达数: {n_motors}")
            
            # V6双重验证!
            print(f"\n{'='*50}")
            print("🔍 V6双重验证 (合规性 + 性能):")
            print(f"{'='*50}")
            
            # 验证1: 合规性 (马达数量)
            compliance_ok = n_motors is not None and n_motors >= args.min_motors
            
            # 验证2: 性能 (适应度阈值)
            performance_ok = fitness is not None and fitness >= 0.40
            
            if compliance_ok and performance_ok:
                print(f"\n🎉🎉🎉 完美成功! 双重目标达成!")
                print(f"   ✓ 合规性: 包含{n_motors}个马达 (要求: ≥{args.min_motors})")
                print(f"   ✓ 性能: 适应度{fitness:.4f} (目标: ≥0.40)")
                print(f"   ✓ 这是一个真正可用的高性能自主机器人!")
                validation_passed = True
                result_grade = "EXCELLENT"
                
            elif compliance_ok and not performance_ok:
                print(f"\n✅ 部分成功: 合规但性能待提升")
                print(f"   ✓ 合规性: 包含{n_motors}个马达 ✅")
                print(f"   ⚠️ 性能: 适应度{fitness:.4f} (目标: ≥0.40, 未达标)")
                print(f"   💡 建议: 可接受的结果，可进一步优化")
                validation_passed = True
                result_grade = "ACCEPTABLE"
                
            elif not compliance_ok and performance_ok:
                print(f"\n⚠️ 部分成功: 性能好但未完全合规")
                print(f"   ❌ 合规性: 马达数{n_motors} < 要求{args.min_motors}")
                print(f"   ✓ 性能: 适应度{fitness:.4f} ✅ (优秀!)")
                print(f"   💡 建议: 约束可能需要微调")
                validation_passed = False
                result_grade = "PARTIAL"
                
            else:
                print(f"\n❌ 未达成目标")
                print(f"   ❌ 合规性: 马达数{n_motors} < 要求{args.min_motors}")
                print(f"   ❌ 性能: 适应度{fitness:.4f} < 目标0.40")
                validation_passed = False
                result_grade = "FAILED"
            
            # 最终评级
            print(f"\n{'='*50}")
            print(f"📊 最终评级: {result_grade}")
            print(f"{'='*50}")
            
        else:
            print("\n⚠️ 未获得有效最佳个体")
            validation_passed = False
        
        print(f"\n⏱️  总耗时: {elapsed_time/3600:.1f}小时 ({elapsed_time:.1f}秒)")
        print(f"📅 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 对比总结
        print(f"\n{'='*70}")
        print("📈 V6 vs V5 vs V4 对比总结:")
        print(f"{'='*70}")
        print(f"版本   | 最佳适应度 | 马达数 | 状态")
        print(f"-------|------------|--------|------")
        print(f"V4     | 0.5835     | 0      | ❌ 不合规")
        print(f"V5     | 0.2379     | ?      | ❌ 严重停滞")
        print(f"V6     | {fitness if fitness else 'N/A':<11} | {n_motors if n_motors else '?':<6} | {'✅' if validation_passed else '待定'}")
        print(f"{'='*70}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 实验出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
