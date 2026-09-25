"""
Motor Adapter - 连接ModyPy电机模型与forgecraft进化系统
提供参数转换、性能评估、适应度计算等核心功能
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

# 导入工程级电机模型
import sys
import logging

_logger = logging.getLogger(__name__)

sys.path.insert(0, '.')
from forgecraft.core.engineering_dc_motor import (
    EngineeringDCMotor,
    MotorSpecifications,
    MotorPerformance,
    ROBOT_MOTOR_DATABASE,
    create_motor_from_database
)

__all__ = [
    "MotorGenome",
    "MotorPhenotype",
    "MotorAdapter",
    "create_motor_adapter_from_yaml",
]


@dataclass
class MotorGenome:
    """电机基因型（用于进化算法）"""
    motor_id: int
    Kv: float                    # 扭矩常数
    R: float                     # 电阻
    L: float                     # 电感
    J: float                     # 转动惯量
    V_nominal: float             # 工作电压
    position: Tuple[float, float, float]  # 安装位置 (x,y,z)
    orientation: Tuple[float, float, float]  # 安装方向 (rx,ry,rz)


@dataclass
class MotorPhenotype:
    """电机表现型（仿真结果）"""
    motor_id: int
    speed_rpm: float
    torque_nm: float
    current_a: float
    power_w: float
    efficiency_pct: float
    battery_drain_rate: float     # 电池消耗速率 (mAh/s)
    thermal_load: float           # 热负载 (W)
    is_healthy: bool              # 是否在安全工作范围


class MotorAdapter:
    """
    电机适配器 - 桥接ModyPy与进化系统
    
    核心功能:
    1. 基因型↔表现型映射
    2. 电机性能实时评估
    3. 多目标适应度计算
    4. 能耗与热管理验证
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化适配器
        
        Args:
            config: 从YAML加载的配置字典
        """
        self.config = config
        self.motor_configs = self._extract_motor_configs()
        self._motor_cache = {}  # 缓存已创建的电机实例

    def _extract_motor_configs(self) -> Dict[str, Any]:
        """提取电机相关配置"""
        parts_config = self.config.get('parts', {})
        motor_config = parts_config.get('engineering_dc_motor', {})

        return {
            'physics': motor_config.get('motor_physics', {}),
            'energy': motor_config.get('energy_system', {}),
            'weights': motor_config.get('performance_weights', {}),
            'validation': self.config.get('evolution_config', {}).get(
                'motor_validation', {}
            )
        }

    def genome_to_motor(self, genome: MotorGenome) -> EngineeringDCMotor:
        """
        将基因型转换为电机实例
        
        Args:
            genome: 电机基因
            
        Returns:
            EngineeringDCMotor实例
        """
        cache_key = f"{genome.motor_id}_{genome.Kv:.6f}_{genome.R:.4f}"

        if cache_key in self._motor_cache:
            return self._motor_cache[cache_key]

        specs = MotorSpecifications(
            Kv=genome.Kv,
            R=genome.R,
            L=genome.L,
            J=genome.J,
            V_nominal=genome.V_nominal,
            I_no_load=self.motor_configs['physics'].get(
                'no_load_current_A', 0.02
            ),
            T_stall=self.motor_configs['physics'].get(
                'stall_torque_Nm', 0.1
            ),
            speed_no_load=self.motor_configs['physics'].get(
                'no_load_speed_RPM', 7844
            )
        )

        motor = EngineeringDCMotor(specs)
        motor.initialize()

        self._motor_cache[cache_key] = motor
        return motor

    def evaluate_motor_performance(
        self,
        genome: MotorGenome,
        load_torque: float = 0.01,
        simulation_time: float = 0.1
    ) -> MotorPhenotype:
        """
        评估单个电机的性能
        
        Args:
            genome: 电机基因
            load_torque: 外部负载扭矩 (Nm)
            simulation_time: 仿真时长 (s)
            
        Returns:
            MotorPhenotype: 性能指标
        """
        motor = self.genome_to_motor(genome)

        try:
            perf = motor.simulate_step(
                load_torque=load_torque,
                duration=simulation_time
            )

            # 计算衍生指标
            battery_drain = perf.current_a * simulation_time / 3.6  # mAh
            thermal_load = (perf.current_a ** 2) * genome.R  # I²R损耗

            # 验证健康状态
            validation_cfg = self.motor_configs['validation']
            is_healthy = (
                perf.efficiency >= validation_cfg.get(
                    'min_motor_efficiency', 20.0
                ) if perf.efficiency > 0 else False and
                perf.current_a < self.motor_configs['energy'].get(
                    'max_current_draw_A', 50.0
                )
            )

            return MotorPhenotype(
                motor_id=genome.motor_id,
                speed_rpm=perf.speed_rpm,
                torque_nm=perf.torque_nm,
                current_a=perf.current_a,
                power_w=perf.power_out_w,
                efficiency_pct=perf.efficiency,
                battery_drain_rate=battery_drain / simulation_time if simulation_time > 0 else 0,
                thermal_load=thermal_load,
                is_healthy=is_healthy
            )

        except Exception as e:
            print(f"⚠️ 电机{genome.motor_id}评估失败: {e}")
            return MotorPhenotype(
                motor_id=genome.motor_id,
                speed_rpm=0, torque_nm=0, current_a=0,
                power_w=0, efficiency_pct=0,
                battery_drain_rate=0, thermal_load=0,
                is_healthy=False
            )

    def evaluate_robot_motors(
        self,
        motor_genomes: List[MotorGenome],
        target_speed: float = 100.0,
        target_torque_per_motor: float = 0.005
    ) -> Dict[str, Any]:
        """
        评估机器人整体电机系统
        
        Args:
            motor_genomes: 所有电机的基因列表
            target_speed: 目标转速 (RPM)
            target_torque_per_motor: 单个电机目标扭矩 (Nm)
            
        Returns:
            包含各项指标的字典
        """
        num_motors = len(motor_genomes)

        if num_motors < 2:
            return {
                'total_fitness': -10.0,  # 严重惩罚
                'num_motors': num_motors,
                'avg_speed': 0,
                'total_torque': 0,
                'avg_efficiency': 0,
                'total_power': 0,
                'autonomy_score': 0,
                'is_valid': False,
                'penalty_reason': 'insufficient_motors'
            }

        phenotypes = []
        total_torque = 0
        total_power = 0
        total_battery_drain = 0
        total_thermal = 0
        healthy_count = 0

        for genome in motor_genomes:
            phenotype = self.evaluate_motor_performance(
                genome,
                load_torque=target_torque_per_motor
            )
            phenotypes.append(phenotype)

            total_torque += phenotype.torque_nm
            total_power += phenotype.power_w

            # 安全提取battery_drain_rate
            drain = phenotype.battery_drain_rate
            if isinstance(drain, (int, float)):
                total_battery_drain += drain
            else:
                print(f"⚠️ 马达{genome.motor_id}电池消耗异常: {type(drain)}")

            total_thermal += phenotype.thermal_load

            if phenotype.is_healthy:
                healthy_count += 1

        # 计算聚合指标
        avg_speed = np.mean([p.speed_rpm for p in phenotypes])
        avg_efficiency = np.mean([p.efficiency_pct for p in phenotypes])

        # 自主性评分（基于电池容量）
        energy_cfg = self.motor_configs.get('energy', {})
        battery_capacity = energy_cfg.get('battery_capacity_mAh', 2200)

        # 确保是数值
        if isinstance(battery_capacity, dict):
            battery_capacity = 2200  # 默认值

        autonomy_hours = battery_capacity / (
            total_battery_drain * 3600 + 1e-6
        )
        autonomy_score = min(1.0, autonomy_hours / 2.0)  # 2小时为满分

        # 性能评分
        speed_score = min(1.0, avg_speed / target_speed)
        torque_score = min(1.0, total_torque / (num_motors * target_torque_per_motor))
        efficiency_score = avg_efficiency / 100.0

        # 综合适应度（多目标加权）
        weights = self.motor_configs['weights']
        fitness = (
            weights.get('speed_priority', 0.35) * speed_score +
            weights.get('torque_priority', 0.30) * torque_score +
            weights.get('efficiency_priority', 0.25) * efficiency_score +
            weights.get('autonomy_priority', 0.10) * autonomy_score
        )

        # 健康度惩罚
        health_ratio = healthy_count / num_motors
        if health_ratio < 1.0:
            fitness *= (0.5 + 0.5 * health_ratio)

        return {
            'total_fitness': fitness,
            'num_motors': num_motors,
            'avg_speed': avg_speed,
            'total_torque': total_torque,
            'avg_efficiency': avg_efficiency,
            'total_power': total_power,
            'autonomy_score': autonomy_score,
            'is_valid': True,
            'health_ratio': health_ratio,
            'phenotypes': phenotypes
        }

    def generate_random_genome(
        self,
        motor_id: int = 0
    ) -> MotorGenome:
        """
        生成随机电机基因（用于初始化种群）
        
        Args:
            motor_id: 电机ID
            
        Returns:
            随机生成的MotorGenome
        """
        physics_cfg = self.motor_configs['physics']

        # 从配置范围中采样
        def sample_param(param_name):
            cfg = physics_cfg.get(param_name, {})
            range_vals = cfg.get('range', [0, 1])

            # 确保是数值类型
            r_min = float(range_vals[0])
            r_max = float(range_vals[1])

            distribution = cfg.get('distribution', 'uniform')
            if distribution == 'log_uniform':
                if r_min <= 0 or r_max <= 0:
                    return np.random.uniform(r_min, r_max)
                log_min, log_max = np.log10(r_min), np.log10(r_max)
                return 10 ** np.random.uniform(log_min, log_max)
            else:
                return np.random.uniform(r_min, r_max)

        return MotorGenome(
            motor_id=motor_id,
            Kv=sample_param('motor_constant_Kv'),
            R=sample_param('resistance_R'),
            L=sample_param('inductance_L'),
            J=sample_param('moment_of_inertia_J'),
            V_nominal=np.random.choice(
                physics_cfg.get('nominal_voltage_V', {}).get(
                    'options', [12.0]
                )
            ),
            position=(
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.05, 0.15),
                np.random.uniform(0.0, 0.1)
            ),
            orientation=(
                np.random.uniform(-np.pi, np.pi),
                np.random.uniform(-np.pi, np.pi),
                np.random.uniform(-np.pi, np.pi)
            )
        )

    def mutate_genome(
        self,
        genome: MotorGenome,
        mutation_strength: float = 0.2
    ) -> MotorGenome:
        """
        变异电机基因
        
        Args:
            genome: 原始基因
            mutation_strength: 变异强度 (0-1)
            
        Returns:
            变异后的MotorGenome
        """
        physics_cfg = self.motor_configs['physics']

        def mutate_param(current_val, param_name):
            cfg = physics_cfg.get(param_name, {})
            range_vals = cfg.get('range', [0, 1])

            # 确保所有值都是数值
            r_min = float(range_vals[0])
            r_max = float(range_vals[1])
            current_val = float(current_val)

            max_delta = (r_max - r_min) * mutation_strength
            delta = np.random.uniform(-max_delta, max_delta)

            new_val = current_val + delta

            # 限制在有效范围内（使用已转换的数值）
            return np.clip(new_val, r_min, r_max)

        return MotorGenome(
            motor_id=genome.motor_id,
            Kv=mutate_param(genome.Kv, 'motor_constant_Kv'),
            R=mutate_param(genome.R, 'resistance_R'),
            L=mutate_param(genome.L, 'inductance_L'),
            J=mutate_param(genome.J, 'moment_of_inertia_J'),
            V_nominal=genome.V_nominal,  # 电压通常不变异
            position=genome.position,
            orientation=genome.orientation
        )

    def get_fitness_summary(
        self,
        evaluation_result: Dict[str, Any]
    ) -> str:
        """生成人类可读的适应度摘要"""
        if not evaluation_result.get('is_valid', False):
            return (
                f"❌ 无效设计: {evaluation_result.get('penalty_reason', '未知原因')}\n"
                f"   马达数量: {evaluation_result.get('num_motors', 0)}"
            )

        return (
            f"✅ 有效设计\n"
            f"   马达数量: {evaluation_result['num_motors']}\n"
            f"   平均转速: {evaluation_result['avg_speed']:.1f} RPM\n"
            f"   总扭矩: {evaluation_result['total_torque']*1000:.2f} mNm\n"
            f"   平均效率: {evaluation_result['avg_efficiency']:.1f}%\n"
            f"   总功率: {evaluation_result['total_power']:.3f} W\n"
            f"   自主性评分: {evaluation_result['autonomy_score']:.2f}\n"
            f"   健康度: {evaluation_result['health_ratio']*100:.0f}%\n"
            f"   🎯 综合适应度: {evaluation_result['total_fitness']:.4f}"
        )


def create_motor_adapter_from_yaml(yaml_path: str) -> MotorAdapter:
    """
    从YAML文件创建适配器
    
    Args:
        yaml_path: YAML配置文件路径
        
    Returns:
        MotorAdapter实例
    """
    import yaml

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return MotorAdapter(config)


if __name__ == "__main__":
    """快速测试适配器功能"""
    import yaml

    print("🔧 测试Motor Adapter")

    # 加载V7配置
    with open('forgecraft/configs/catalogs/parameterized_parts_v7.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    adapter = MotorAdapter(config)

    # 测试随机基因组生成
    print("\n--- 测试1: 随机基因组生成 ---")
    for i in range(3):
        genome = adapter.generate_random_genome(motor_id=i)
        print(f"马达{i}: Kv={genome.Kv:.4f}, R={genome.R:.2f}, V={genome.V_nominal}V")

    # 测试单电机评估
    print("\n--- 测试2: 单电机性能评估 ---")
    test_genome = adapter.generate_random_genome()
    phenotype = adapter.evaluate_motor_performance(test_genome)
    print(adapter.get_fitness_summary({
        'is_valid': True,
        'num_motors': 1,
        'avg_speed': phenotype.speed_rpm,
        'total_torque': phenotype.torque_nm,
        'avg_efficiency': phenotype.efficiency_pct,
        'total_power': phenotype.power_w,
        'autonomy_score': 0.8,
        'health_ratio': 1.0 if phenotype.is_healthy else 0.5,
        'total_fitness': 0.75
    }))

    # 测试多电机系统
    print("\n--- 测试3: 多电机系统评估 ---")
    robot_genomes = [adapter.generate_random_genome(i) for i in range(3)]
    result = adapter.evaluate_robot_motors(robot_genomes)
    print(adapter.get_fitness_summary(result))

    print("\n✅ Motor Adapter测试完成!")