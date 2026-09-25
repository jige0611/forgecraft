# ══════════════════════════════════════════════════════════
# 🚀 V10 V8 完整进化实验
#
# 特性:
#   ✅ 47种真实竞赛级零件 (RoboMaster/FRC/工业级)
#   ✅ 基于真实零件的基因编码
#   ✅ 功率预算与热约束验证
#   ✅ 多目标帕累托优化
#   ✅ 自适应变异 + UCB资源分配
#   ✅ 完整的MuJoCo物理仿真集成
#
# 运行方式:
#   python run_v10_v8.py --generations 100 --pop 48
#
# ══════════════════════════════════════════════════════════

import sys
import os
import argparse
import time
import json
from datetime import datetime

# 确保可以导入forgecraft模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forgecraft.core.catalog_v8 import create_v8_catalog, CompetitionPartSpec
from forgecraft.core.v8_evolution_engine import (
    V8EvolutionEngine,
    V8Genome,
    V8EvolutionResult
)


def setup_experiment(args):
    """设置实验环境"""
    print("\n" + "=" * 70)
    print("🔬 V10-V8 竞赛级机器人形态进化实验")
    print("=" * 70)
    
    # 加载V8零件目录
    print("\n📦 加载V8竞赛级零件库...")
    catalog = create_v8_catalog()
    
    # 显示目录统计
    motors = catalog.get_motors()
    batteries = catalog.get_batteries()
    sensors = catalog.get_by_category('sensor_inertial')
    transmissions = catalog.get_transmissions()
    
    print(f"   ✅ 目录加载成功")
    print(f"   📊 统计:")
    print(f"      电机: {len(motors)} 种")
    print(f"      电池: {len(batteries)} 种")
    print(f"      传感器: {len(sensors)} 种")
    print(f"      传动件: {len(transmissions)} 种")
    print(f"      总计: {len(catalog.catalog)} 种竞赛级零件")
    
    return catalog


def create_experiment_config(args) -> dict:
    """创建实验配置"""
    config = {
        'experiment_name': f'v10-v8-{args.target_type}-{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'target_robot_type': args.target_type,
        'population_size': args.pop_size,
        'generations': args.generations,
        'mutation_rate': 0.35,
        'crossover_rate': 0.6,
        'elite_ratio': 0.15,
        
        # 适应度权重
        'fitness_weights': {
            'autonomy': 2.5,
            'performance': 1.5,
            'manufacturability': 1.0
        },
        
        # 物理约束
        'constraints': {
            'min_motors': 4,
            'max_motors': 16,
            'min_runtime_hours': 0.25,
            'max_cost_usd': 5000,
            'collision_threshold_m': 0.02
        },
        
        # UCB参数
        'ucb_params': {
            'exploration_constant': 1.82,
            'base_evaluations': 3
        }
    }
    
    return config


def run_experiment(catalog, config, args):
    """运行主实验"""
    print(f"\n⚙️ 创建进化引擎...")
    engine = V8EvolutionEngine(
        catalog=catalog,
        population_size=config['population_size'],
        generations=config['generations'],
        target_robot_type=config['target_robot_type']
    )
    
    # 运行进化
    print(f"\n🚀 开始进化 ({config['generations']}代)...")
    start_time = time.time()
    
    result = engine.run_evolution(
        verbose=True,
        save_interval=args.save_interval,
        checkpoint_file=args.checkpoint_file
    )
    
    elapsed_time = time.time() - start_time
    
    # 输出最终结果
    print_final_results(engine, result, config, elapsed_time, catalog)
    
    # 保存结果
    save_experiment_results(engine, result, config, args)
    
    return engine, result


def print_final_results(engine, result, config, elapsed_time, catalog):
    """打印最终结果"""
    best = engine.best_ever_genome
    
    print("\n" + "═" * 70)
    print("🏆 最终结果报告")
    print("═" * 70)
    
    print(f"\n📈 进化性能:")
    print(f"   最佳适应度: {best.fitness:+.4f}")
    print(f"   出现代数: 第{best.generation + 1}代 / 共{config['generations']}代")
    print(f"   总耗时: {elapsed_time:.1f}秒 ({elapsed_time/60:.1f}分钟)")
    print(f"   平均每代耗时: {elapsed_time/config['generations']:.2f}秒")
    
    print(f"\n🤖 最佳机器人配置:")
    print(f"   类型: {config['target_robot_type']}")
    print(f"   马达数量: {len(best.motors)}")
    print(f"   马达型号分布:")
    
    motor_types = {}
    for m in best.motors:
        mtype = m['type']
        motor_types[mtype] = motor_types.get(mtype, 0) + 1
    
    for mtype, count in sorted(motor_types.items(), key=lambda x: -x[1]):
        spec = catalog.catalog.get(mtype)
        if spec:
            manufacturer = getattr(spec, 'manufacturer', 'Unknown')
            part_num = getattr(spec, 'part_number', '')
            torque = spec.get_performance('torque_rated', '?')
            speed = spec.get_performance('speed_no_load', '?')
            print(f"      ×{count} {mtype} ({manufacturer} {part_num})")
            print(f"         扭矩={torque}N·m, 转速={speed}RPM")
    
    if best.battery:
        btype = best.battery['type']
        bspec = catalog.catalog.get(btype)
        if bspec:
            voltage = bspec.electrical.get('voltage_nominal', '?')
            capacity = bspec.electrical.get('capacity_Ah', '?')
            energy = bspec.electrical.get('energy', '?')
            print(f"\n   🔋 能源系统: {btype}")
            print(f"      电压: {voltage}V | 容量: {capacity}Ah | 能量: {energy}Wh")
    
    if best.sensors:
        print(f"\n   📡 传感器: {[s['type'] for s in best.sensors]}")
    
    if best.transmissions:
        print(f"\n   ⚙️ 传动系统: {[t['type'] for t in best.transmissions]}")
    
    print(f"\n📊 适应度分项分析:")
    components = best.fitness_components
    for comp_name, comp_val in components.items():
        bar_len = int(abs(comp_val) * 20)
        bar_char = '█' if comp_val >= 0 else '░'
        bar = bar_char * max(0, bar_len)
        print(f"   {comp_name:15s}: {comp_val:+7.4f} |{bar:<20s}|")
    
    print(f"\n✅ 可行性检查:")
    if best.is_feasible:
        print(f"   状态: ✅ 可行设计 (可用于实际制造)")
    else:
        print(f"   状态: ⚠️ 存在约束违反")
        for violation in best.constraint_violations[:5]:
            print(f"      - {violation}")
    
    # 帕累托前沿信息
    if engine.pareto_archive:
        print(f"\n📐 帕累托前沿解 ({len(engine.pareto_archive)} 个):")
        for i, sol in enumerate(engine.pareto_archive[:5]):
            print(f"   解{i+1}: 适应度={sol.fitness:+.4f}, "
                  f"{len(sol.motors)}马达, "
                  f"{'可行' if sol.is_feasible else '不可行'}")


def save_experiment_results(engine, result, config, args):
    """保存实验结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存JSON报告
    report_file = f'v8_experiment_report_{timestamp}.json'
    report_data = {
        'experiment_config': config,
        'best_result': {
            'fitness': engine.best_ever_fitness,
            'generation': engine.best_ever_genome.generation,
            'is_feasible': engine.best_ever_genome.is_feasible,
            'motors': [
                {'type': m['type'], 
                 'position': m['position'],
                 'gear_ratio': m.get('gear_ratio', 1.0)}
                for m in engine.best_ever_genome.motors
            ],
            'battery': engine.best_ever_genome.battery,
            'sensors': [{'type': s['type']} for s in engine.best_ever_genome.sensors],
            'fitness_components': {
                k: float(v) if hasattr(v, '__float__') else v
                for k, v in engine.best_ever_genome.fitness_components.items()
            }
        },
        'evolution_history': [
            {
                'gen': r.generation + 1,
                'best': r.best_fitness,
                'avg': r.avg_fitness,
                'worst': r.worst_fitness,
                'stagnation': r.population_stats['stagnation'],
                'avg_motors': float(r.population_stats['avg_motors'])
            }
            for r in engine.history[-20:]  # 只保存最近20代
        ]
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存:")
    print(f"   📄 报告文件: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='V10-V8 竞赛级机器人形态进化实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_v10_v8.py --target-type humanoid --generations 100 --pop 48
  python run_v10_v8.py --target-type quadruped --generations 150 --pop 64
  
目标类型:
  humanoid     - 人形机器人 (双足行走)
  quadruped    - 四足机器人 (四足行走)
  wheeled      - 轮式机器人 (移动平台)
  drone        - 无人机 (飞行器)
  generic      - 通用型 (自由探索)
        """
    )
    
    parser.add_argument('--target-type', type=str, default='humanoid',
                       choices=['humanoid', 'quadruped', 'wheeled', 'drone', 'generic'],
                       help='目标机器人类型 (默认: humanoid)')
    
    parser.add_argument('--generations', type=int, default=100,
                       help='进化代数 (默认: 100)')
    
    parser.add_argument('--pop', '--population-size', type=int, dest='pop_size',
                       default=48, help='种群规模 (默认: 48)')
    
    parser.add_argument('--mutation-rate', type=float, default=0.35,
                       help='变异率 (默认: 0.35)')
    
    parser.add_argument('--save-interval', type=int, default=10,
                       help='检查点保存间隔(代) (默认: 10)')
    
    parser.add_argument('--checkpoint-file', type=str,
                       default='v8_evolution_checkpoint.pkl',
                       help='检查点文件名')
    
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子 (用于可重复实验)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"🔢 随机种子: {args.seed}")
    
    try:
        # 设置实验环境
        catalog = setup_experiment(args)
        
        # 创建配置
        config = create_experiment_config(args)
        
        # 运行实验
        engine, result = run_experiment(catalog, config, args)
        
        print("\n" + "=" * 70)
        print("✅ 实验完成!")
        print("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断实验")
        print("   可以使用 --checkpoint-file 加载最近检查点继续运行")
        return 130
        
    except Exception as e:
        print(f"\n❌ 实验出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import random
    exit(main())
