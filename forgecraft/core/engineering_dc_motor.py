"""
EngineeringDCMotor - 基于ModyPy的工程级DC电机零件
为forgecraft进化系统提供真实的电机物理模型
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from modypy.blocks.elmech import DCMotor as ModyPyDCMotor
from modypy.model import System
from modypy.simulation import Simulator, SimulationResult
from modypy.blocks.sources import constant
import logging

__all__ = [
    "MotorSpecifications",
    "MotorPerformance",
    "EngineeringDCMotor",
    "create_motor_from_database",
    "ROBOT_MOTOR_DATABASE",
]

_logger = logging.getLogger(__name__)



@dataclass
class MotorSpecifications:
    """电机规格参数"""
    Kv: float = 0.02           # 扭矩常数 (Vs/rad)
    R: float = 1.0             # 电枢电阻 (Ohm)
    L: float = 0.001           # 电感 (H)
    J: float = 5e-5            # 转动惯量 (kg*m^2)
    V_nominal: float = 12.0    # 额定电压 (V)
    I_no_load: float = 0.02    # 空载电流 (A)
    T_stall: float = 0.1       # 堵转扭矩 (Nm)
    speed_no_load: float = 7844  # 空载转速 (RPM)


@dataclass
class MotorPerformance:
    """电机性能指标"""
    speed_rpm: float           # 当前转速 (RPM)
    torque_nm: float           # 输出扭矩 (Nm)
    current_a: float           # 电枢电流 (A)
    power_out_w: float         # 机械输出功率 (W)
    power_in_w: float          # 电输入功率 (W)
    efficiency: float          # 效率 (%)
    back_emf_v: float          # 反电动势 (V)


class EngineeringDCMotor:
    """
    工程级DC电机零件类
    
    基于ModyPy的真实物理模型，支持：
    - 完整的电气-机械耦合方程
    - 参数化配置（适配YAML零件库）
    - 实时仿真与性能计算
    - 与MuJoCo/forgecraft系统集成
    """

    def __init__(self, specs: MotorSpecifications):
        """
        初始化电机实例
        
        Args:
            specs: 电机规格参数
        """
        self.specs = specs
        self._system = None
        self._dcmotor = None
        self._simulator = None
        self._last_result = None

        # 当前状态
        self.current_speed_rpm = 0.0
        self.current_torque_nm = 0.0
        self.current_current_a = 0.0

    def initialize(self) -> bool:
        """初始化ModyPy仿真系统"""
        try:
            self._system = System()
            self._dcmotor = ModyPyDCMotor(
                self._system,
                motor_constant=self.specs.Kv,
                resistance=self.specs.R,
                inductance=max(self.specs.L, 1e-6),  # 避免除零
                moment_of_inertia=self.specs.J,
                initial_omega=0,
                initial_current=0
            )

            # 连接电压源
            voltage_signal = constant(self.specs.V_nominal)
            self._dcmotor.voltage.connect(voltage_signal)

            # 连接负载（初始为0）
            load_signal = constant(0.0)
            self._dcmotor.external_torque.connect(load_signal)

            return True
        except Exception as e:
            print(f"❌ 电机初始化失败: {e}")
            return False

    def simulate_step(
        self,
        load_torque: float = 0.0,
        duration: float = 0.01,
        voltage: Optional[float] = None
    ) -> MotorPerformance:
        """
        执行单步仿真
        
        Args:
            load_torque: 外部负载扭矩 (Nm)
            duration: 仿真时长 (s)
            voltage: 施加电压 (V)，None则使用额定电压
            
        Returns:
            MotorPerformance: 性能指标对象
        """
        if self._system is None:
            if not self.initialize():
                raise RuntimeError("电机系统未初始化")

        try:
            # 更新电压源（如果需要）
            if voltage is not None and voltage != self.specs.V_nominal:
                new_voltage = constant(voltage)
                self._dcmotor.voltage.connect(new_voltage)

            # 更新负载
            new_load = constant(load_torque)
            self._dcmotor.external_torque.connect(new_load)

            # 运行仿真
            self._simulator = Simulator(system=self._system, start_time=0)
            result = SimulationResult(self._system)

            for state in self._simulator.run_until(time_boundary=duration):
                result.append(state)

            self._last_result = result

            # 提取状态数据
            omega_idx = self._dcmotor.omega.state_index
            current_idx = self._dcmotor.current.state_index

            final_omega = result.state[omega_idx, -1]
            final_current = result.state[current_idx, -1]

            # 计算性能指标
            speed_rpm = (final_omega / (2 * np.pi)) * 60
            torque_nm = self.specs.Kv * final_current
            back_emf_v = self.specs.Kv * final_omega
            power_in_w = voltage or self.specs.V_nominal * final_current
            power_out_w = torque_nm * final_omega
            efficiency = (power_out_w / power_in_w * 100) if power_in_w > 0 else 0

            # 更新当前状态
            self.current_speed_rpm = speed_rpm
            self.current_torque_nm = torque_nm
            self.current_current_a = final_current

            return MotorPerformance(
                speed_rpm=speed_rpm,
                torque_nm=torque_nm,
                current_a=final_current,
                power_out_w=power_out_w,
                power_in_w=power_in_w,
                efficiency=efficiency,
                back_emf_v=back_emf_v
            )

        except Exception as e:
            print(f"⚠️ 仿真警告: {e}")
            return MotorPerformance(0, 0, 0, 0, 0, 0, 0)

    def get_characteristic_curve(
        self,
        voltage: float = 12.0,
        num_points: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        获取电机特性曲线（扭矩-转速曲线）
        
        Args:
            voltage: 工作电压 (V)
            num_points: 采样点数
            
        Returns:
            包含torque, speed, current, power, efficiency数组的字典
        """
        torques = np.linspace(0, self.specs.T_stall, num_points)
        speeds = []
        currents = []
        powers = []
        efficiencies = []

        for torque in torques:
            perf = self.simulate_step(load_torque=torque, voltage=voltage)
            speeds.append(perf.speed_rpm)
            currents.append(perf.current_a)
            powers.append(perf.power_out_w)
            efficiencies.append(perf.efficiency)

        return {
            'torque_nm': torques,
            'speed_rpm': np.array(speeds),
            'current_a': np.array(currents),
            'power_w': np.array(powers),
            'efficiency_pct': np.array(efficiencies)
        }

    def calculate_fitness_metrics(
        self,
        target_speed: float,
        target_torque: float,
        weight_speed: float = 0.6,
        weight_efficiency: float = 0.4
    ) -> float:
        """
        计算适应度指标（用于进化算法）
        
        Args:
            target_speed: 目标转速 (RPM)
            target_torque: 目标扭矩 (Nm)
            weight_speed: 速度权重
            weight_efficiency: 效率权重
            
        Returns:
            适应度分数 (0-1)
        """
        perf = self.simulate_step(load_torque=target_torque)

        # 速度得分（越接近目标越好）
        speed_score = 1.0 / (1.0 + abs(perf.speed_rpm - target_speed) / target_speed)

        # 效率得分
        efficiency_score = perf.efficiency / 100.0

        # 加权综合得分
        fitness = weight_speed * speed_score + weight_efficiency * efficiency_score

        return fitness

    @classmethod
    def from_yaml_config(cls, config: Dict[str, Any]) -> 'EngineeringDCMotor':
        """
        从YAML配置创建电机实例
        
        Args:
            config: YAML解析后的字典
            
        Returns:
            EngineeringDCMotor实例
        """
        specs = MotorSpecifications(
            Kv=config.get('motor_constant', 0.02),
            R=config.get('resistance', 1.0),
            L=config.get('inductance', 0.001),
            J=config.get('moment_of_inertia', 5e-5),
            V_nominal=config.get('nominal_voltage', 12.0),
            I_no_load=config.get('no_load_current', 0.02),
            T_stall=config.get('stall_torque', 0.1),
            speed_no_load=config.get('no_load_speed', 7844)
        )
        return cls(specs)

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典格式（用于序列化）"""
        return {
            'type': 'engineering_dc_motor',
            'specs': {
                'Kv': self.specs.Kv,
                'R': self.specs.R,
                'L': self.specs.L,
                'J': self.specs.J,
                'V_nominal': self.specs.V_nominal,
                'I_no_load': self.specs.I_no_load,
                'T_stall': self.specs.T_stall,
                'speed_no_load': self.specs.speed_no_load
            },
            'current_state': {
                'speed_rpm': self.current_speed_rpm,
                'torque_nm': self.current_torque_nm,
                'current_a': self.current_current_a
            }
        }

    def __repr__(self) -> str:
        return (
            f"EngineeringDCMotor("
            f"Kv={self.specs.Kv:.4f} Vs/rad, "
            f"R={self.specs.R:.2f} Ω, "
            f"V={self.specs.V_nominal}V)"
        )


# 预定义的机器人电机数据库
ROBOT_MOTOR_DATABASE = {
    'n20_micro': MotorSpecifications(
        Kv=0.02, R=1.0, L=0.001, J=5e-5,
        V_nominal=12.0, I_no_load=0.02,
        T_stall=0.1, speed_no_load=7844
    ),
    'n20_precise': MotorSpecifications(
        Kv=0.014, R=25.0, L=0.0001, J=5e-6,
        V_nominal=12.0, I_no_load=0.02,
        T_stall=0.006, speed_no_load=7844
    ),
    'rs380_standard': MotorSpecifications(
        Kv=0.05, R=0.3, L=0.002, J=1e-4,
        V_nominal=7.2, I_no_load=0.05,
        T_stall=0.15, speed_no_load=15000
    ),
    'high_torque_servo': MotorSpecifications(
        Kv=0.08, R=0.15, L=0.003, J=2e-4,
        V_nominal=6.0, I_no_load=0.1,
        T_stall=0.25, speed_no_load=6000
    )
}


def create_motor_from_database(motor_type: str) -> EngineeringDCMotor:
    """
    从预定义数据库创建电机
    
    Args:
        motor_type: 电机类型名称
        
    Returns:
        EngineeringDCMotor实例
    """
    if motor_type not in ROBOT_MOTOR_DATABASE:
        raise ValueError(f"未知电机类型: {motor_type}. 可选: {list(ROBOT_MOTOR_DATABASE.keys())}")

    return EngineeringDCMotor(ROBOT_MOTOR_DATABASE[motor_type])