"""
V10-V7 实验启动脚本 - 集成工程级真实电机模型

核心改进:
  ✅ ModyPy耦合DC电机物理模型
  ✅ 真实参数空间 (Kv, R, L, J)
  ✅ 能耗建模与自主性评分
  ✅ 效率优化目标
  ✅ 分阶段约束策略 (继承V6)

运行命令:
  python run_v10_v7.py --generations 100 --min-motors 2
"""
import sys
import os
import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入forgecraft核心模块
from forgecraft.core.loader import load_catalog, load_task
from forgecraft.core.engineering_dc_motor import (
    EngineeringDCMotor,
    MotorSpecifications,
    ROBOT_MOTOR_DATABASE,
    create_motor_from_database
)
from forgecraft.core.motor_adapter import (
    MotorAdapter,
    MotorGenome,
    MotorPhenotype
)


def load_v7_config():
    """加载V7配置文件"""
    config_path = project_root / "forgecraft" / "configs" / "catalogs" / "parameterized_parts_v7.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"V7配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"✅ V7配置加载成功: {config_path}")
    print(f"   版本: {config.get('catalog_version', 'unknown')}")

    return config


def initialize_motor_adapter(config):
    """初始化电机适配器"""
    adapter = MotorAdapter(config)

    print("\n🔧 电机适配器初始化完成")
    print(f"   物理参数范围:")
    physics_cfg = config['parts']['engineering_dc_motor']['motor_physics']
    for param, cfg in physics_cfg.items():
        range_vals = cfg.get('range', [0, 1])
        try:
            r_min = float(range_vals[0])
            r_max = float(range_vals[1])
            print(f"     {param}: [{r_min:.6f}, {r_max:.6f}]")
        except (ValueError, TypeError):
            print(f"     {param}: {range_vals}")

    return adapter


def create_initial_population(adapter, population_size, min_motors):
    """创建初始种群（包含随机电机基因）"""
    population = []

    for i in range(population_size):
        num_motors = np.random.randint(min_motors, min_motors + 4)  # 2-5个电机
        motor_genomes = [
            adapter.generate_random_genome(motor_id=j)
            for j in range(num_motors)
        ]

        individual = {
            'id': i,
            'generation': 0,
            'motor_genomes': motor_genomes,
            'fitness': None,
            'evaluation_result': None
        }

        population.append(individual)

    print(f"\n🧬 初始种群创建完成: {population_size}个个体")
    print(f"   马达数量分布: {min_motors}-{min_motors+3}个/个体")

    return population


def evaluate_individual_v7(individual, adapter, generation, phase_config):
    """
    V7评估函数 - 集成真实电机模型
    
    Args:
        individual: 个体字典
        adapter: 电机适配器
        generation: 当前代数
        phase_config: 分阶段配置
        
    Returns:
        适应度分数
    """
    motor_genomes = individual['motor_genomes']
    num_motors = len(motor_genomes)

    # 基础电机性能评估
    eval_result = adapter.evaluate_robot_motors(
        motor_genomes,
        target_speed=100.0,
        target_torque_per_motor=0.005
    )

    fitness = eval_result['total_fitness']

    # 应用分阶段惩罚/奖励
    if not eval_result.get('is_valid', False):
        penalty = phase_config.get('motor_penalty', -5.0)
        fitness += penalty
        individual['penalty_reason'] = eval_result.get('penalty_reason', 'invalid')
    else:
        # 应用马达数量奖励
        if num_motors >= 2:
            bonus = phase_config.get('motor_bonus', 0.5)
            fitness += bonus * (num_motors / 2)

        # 应用效率奖励
        if eval_result['avg_efficiency'] > 50:
            efficiency_bonus = 0.3 * (eval_result['avg_efficiency'] / 100)
            fitness += efficiency_bonus

    # 更新个体数据
    individual['fitness'] = fitness
    individual['evaluation_result'] = eval_result
    individual['generation'] = generation

    return fitness


def get_phase_config_v7(current_generation, total_generations=100):
    """获取V7分阶段配置（增强版）"""
    if current_generation <= 30:
        return 1, {
            'name': 'exploration',
            'motor_penalty': -3.0,
            'insufficient_penalty': -1.5,
            'external_dep_penalty': -1.8,
            'motor_bonus': 0.35,
            'efficiency_bonus': 0.15,
            'autonomy_bonus': 0.3,
            'diversity_multiplier': 1.5,
            'mutation_range': (1.5, 4.0),
            'goal': '广泛探索解空间'
        }
    elif current_generation <= 65:
        return 2, {
            'name': 'refinement',
            'motor_penalty': -5.0,
            'insufficient_penalty': -2.5,
            'external_dep_penalty': -3.0,
            'motor_bonus': 0.5,
            'efficiency_bonus': 0.25,
            'autonomy_bonus': 0.5,
            'diversity_multiplier': 1.0,
            'mutation_range': (1.8, 3.5),
            'goal': '优化有潜力的设计'
        }
    else:
        return 3, {
            'name': 'convergence',
            'motor_penalty': -7.0,
            'insufficient_penalty': -3.5,
            'external_dep_penalty': -4.0,
            'motor_bonus': 0.6,
            'efficiency_bonus': 0.35,
            'autonomy_bonus': 0.6,
            'diversity_multiplier': 0.7,
            'mutation_range': (2.0, 3.0),
            'goal': '收敛到最优合规设计'
        }


def mutate_population_v7(population, adapter, mutation_strength):
    """V7变异操作（包含电机参数变异）"""
    mutated_population = []

    for individual in population:
        new_motor_genomes = []

        for genome in individual['motor_genomes']:
            # 变异电机参数
            mutated_genome = adapter.mutate_genome(genome, mutation_strength)
            new_motor_genomes.append(mutated_genome)

        mutated_individual = {
            'id': individual['id'],
            'generation': individual['generation'],
            'motor_genomes': new_motor_genomes,
            'fitness': None,
            'evaluation_result': None
        }

        mutated_population.append(mutated_individual)

    return mutated_population


def run_evolution_v7(args):
    """执行V7进化实验"""
    print("=" * 70)
    print("🚀 V10-V7 实验启动 (工程级真实电机集成)")
    print("=" * 70)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 参数设置:")
    print(f"   代数: {args.generations}")
    print(f"   最小马达数: {args.min_motors}")
    print(f"   平衡模式: {'启用' if args.balanced_mode else '禁用'}")
    print()

    # 加载配置和初始化组件
    try:
        config = load_v7_config()
        adapter = initialize_motor_adapter(config)

        # 创建初始种群
        pop_size = config.get('evolution_config', {}).get('population_size', 36)
        population = create_initial_population(
            adapter, pop_size, args.min_motors
        )

        # 进化主循环
        best_ever_fitness = -float('inf')
        best_ever_individual = None
        stagnation_count = 0

        for gen in range(args.generations):
            phase_id, phase_config = get_phase_config_v7(gen)

            print(f"\n{'='*60}")
            print(f"🔄 第 {gen+1}/{args.generations} 代 [阶段{phase_id}: {phase_config['name']}]")
            print(f"{'='*60}")

            # 评估当前种群
            gen_fitnesses = []
            for idx, individual in enumerate(population):
                fitness = evaluate_individual_v7(
                    individual, adapter, gen, phase_config
                )
                gen_fitnesses.append(fitness)

                # 输出进度（每10个个体）
                if (idx + 1) % 10 == 0 or idx == len(population) - 1:
                    avg_fit = np.mean(gen_fitnesses[:idx+1])
                    print(f"   已评估 {idx+1}/{len(population)} 个体 | "
                          f"当前平均适应度: {avg_fit:.4f}")

            # 统计本代最佳
            gen_best_idx = np.argmax(gen_fitnesses)
            gen_best_fitness = gen_fitnesses[gen_best_idx]
            gen_avg_fitness = np.mean(gen_fitnesses)

            print(f"\n📊 第{gen+1}代统计:")
            print(f"   最佳适应度: {gen_best_fitness:.4f}")
            print(f"   平均适应度: {gen_avg_fitness:.4f}")
            print(f"   中位数适应度: {np.median(gen_fitnesses):.4f}")

            # 更新全局最佳
            if gen_best_fitness > best_ever_fitness:
                best_ever_fitness = gen_best_fitness
                best_ever_individual = population[gen_best_idx].copy()
                stagnation_count = 0
                print(f"   🎉 新纪录! 适应度提升至 {best_ever_fitness:.4f}")
            else:
                stagnation_count += 1
                print(f"   ⚠️ 停滞 {stagnation_count} 代")

            # 自适应变异强度
            base_mutation = 0.2
            if stagnation_count > 5:
                adaptive_mutation = min(0.8, base_mutation + stagnation_count * 0.05)
                print(f"   [自适应变异] 停滞{stagnation_count}代 → "
                      f"变异强度×{adaptive_mutation/base_mutation:.1f}")
            else:
                adaptive_mutation = base_mutation

            # 变异生成下一代（最后一代不变异）
            if gen < args.generations - 1:
                population = mutate_population_v7(
                    population, adapter, adaptive_mutation
                )

            # 定期保存断点
            if (gen + 1) % 10 == 0:
                checkpoint_data = {
                    'generation': gen + 1,
                    'best_fitness': best_ever_fitness,
                    'best_individual': best_ever_individual,
                    'population_stats': {
                        'mean': gen_avg_fitness,
                        'std': np.std(gen_fitnesses),
                        'best': gen_best_fitness
                    },
                    'phase': phase_config['name']
                }

                checkpoint_file = project_root / "checkpoint_v7.pkl"
                import pickle
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)

                print(f"💾 断点已保存: {checkpoint_file}")

        # 实验结束总结
        print("\n" + "=" * 70)
        print("🎯 V10-V7 实验完成!")
        print("=" * 70)
        print(f"\n🏆 最终结果:")
        print(f"   最佳适应度: {best_ever_fitness:.4f}")

        if best_ever_individual and best_ever_individual.get('evaluation_result'):
            result = best_ever_individual['evaluation_result']
            print(f"   最佳个体特征:")
            print(f"     马达数量: {result['num_motors']}")
            print(f"     平均转速: {result['avg_speed']:.1f} RPM")
            print(f"     总扭矩: {result['total_torque']*1000:.2f} mNm")
            print(f"     平均效率: {result['avg_efficiency']:.1f}%")
            print(f"     自主性评分: {result['autonomy_score']:.2f}")
            print(f"     健康度: {result['health_ratio']*100:.0f}%")

            # 显示最佳个体的电机参数
            print(f"\n   最佳个体电机参数:")
            for i, genome in enumerate(best_ever_individual['motor_genomes']):
                phenotype = adapter.evaluate_motor_performance(genome)
                print(f"     马达{i}: Kv={genome.Kv:.4f}, R={genome.R:.2f}Ω, "
                      f"{phenotype.speed_rpm:.0f}RPM, "
                      f"{phenotype.efficiency_pct:.1f}%效率")

        print(f"\n⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return best_ever_individual

    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="V10-V7 进化实验 (工程级真实电机集成)"
    )

    parser.add_argument(
        "--generations", type=int, default=100,
        help="进化代数 (默认: 100)"
    )

    parser.add_argument(
        "--min-motors", type=int, default=2,
        help="最小马达数量要求 (默认: 2)"
    )

    parser.add_argument(
        "--balanced-mode", action="store_true", default=True,
        help="启用平衡模式 (55-30-15 自主性-性能-效率)"
    )

    parser.add_argument(
        "--output-dir", type=str, default="design_output_v10_v7",
        help="输出目录 (默认: design_output_v10_v7)"
    )

    args = parser.parse_args()

    # 执行进化实验
    best_individual = run_evolution_v7(args)

    if best_individual:
        print("\n✅ V7实验成功完成! 最佳个体已获得")
        return 0
    else:
        print("\n❌ V7实验失败")
        return 1


if __name__ == "__main__":
    exit(main())