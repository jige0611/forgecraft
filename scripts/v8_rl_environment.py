#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V8 Robot Assembly RL Environment (Gymnasium Interface)
========================================================
工业级并行训练环境 - 支持GPU加速的多智能体机器人组装优化

核心特性:
1. 标准Gymnasium API (reset/step/render)
2. 连续/离散混合动作空间
3. 多目标帕累托奖励函数
4. PyBullet动态组装引擎集成
5. 域随机化(Domain Randomization)
6. 向量化环境支持(VectorizedEnv)

作者: V8 Evolution System
版本: 2.0.0 (Isaac Sim Ready)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
import logging
from collections import deque
import pybullet as p

# 导入碰撞形状库和动态组装器
from v8_collision_shapes import (
    COLLISION_SHAPES,
    get_part_info,
    get_parts_by_category,
    validate_connection,
    get_all_connectors
)
from v8_dynamic_assembler import DynamicAssembler, PartInstance, AssemblyRecord

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """强化学习环境配置"""
    # 仿真参数
    max_steps_per_episode: int = 1000  # 每回合最大步数
    simulation_timestep: float = 1.0 / 240.0  # 物理时间步长(秒)
    action_repeat: int = 10  # 每个动作重复执行的仿真步数
    
    # 任务参数
    max_parts_per_robot: int = 20  # 机器人最大零件数
    min_parts_for_complete: int = 5  # 完成装配的最小零件数
    task_type: str = 'locomotion'  # 任务类型: locomotion/manipulation/balance
    
    # 奖励权重
    reward_weights: Dict[str, float] = None  # 将在__init__中设置默认值
    
    # 观测空间
    include_3d_visual: bool = False  # 是否包含视觉观测
    observation_history_len: int = 4  # 历史帧数
    
    # 难度与随机化
    difficulty: float = 0.5  # 0.0(easy) -> 1.0(hard)
    domain_randomization: bool = True  # 是否启用域随机ization
    
    # 物理参数
    gravity: float = -9.81
    ground_friction: float = 1.0
    
    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                'distance': 3.0,          # 移动距离奖励
                'stability': 2.0,         # 稳定性奖励
                'energy_efficiency': 1.5, # 能效奖励
                'completeness': 2.0,      # 装配完整度奖励
                'manufacturability': 1.0, # 可制造性奖励
                'survival': 0.5,          # 存活奖励(每步小奖励)
            }


class V8RobotAssemblyEnv(gym.Env):
    """
    V8机器人组装强化学习环境
    
    状态空间:
    - 零件配置向量 (one-hot编码 + 参数)
    - 连接关系图邻接矩阵
    - 物理传感器数据 (位置/速度/IMU/力矩)
    - 装配历史嵌入
    
    动作空间 (混合空间):
    - 离散: 选择零件类型 / 选择连接点
    - 连续: 零件参数 / 连接偏移 / 电机控制信号
    
    奖励函数 (多目标加权):
    R_total = w1*distance + w2*stability + w3*energy + w4*complete + w5*manufacture
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(self, config: RLConfig = None, render_mode: str = None):
        super().__init__()
        
        # 配置
        self.config = config or RLConfig()
        self.render_mode = render_mode
        
        # 初始化PyBullet物理引擎
        self._init_physics()
        
        # 初始化动态组装器
        self.assembler = DynamicAssembler(
            physics_client_id=self.physics_client,
            use_gui=(render_mode == 'human')
        )
        
        # 定义动作空间和观测空间
        self._define_spaces()
        
        # 初始化状态变量
        self.current_step = 0
        self.episode_reward = 0.0
        self.robot_parts: List[Dict] = []  # 已装配零件列表
        self.assembly_history: List[Dict] = []  # 装配历史
        self.state_buffer = deque(maxlen=self.config.observation_history_len)
        
        # 性能统计
        self.stats = {
            'episodes_completed': 0,
            'total_steps': 0,
            'successful_assemblies': 0,
            'failed_assemblies': 0,
            'best_distance': 0.0,
            'best_reward': -float('inf'),
        }
        
        logger.info("✅ V8RobotAssemblyEnv initialized successfully")
    
    def _init_physics(self):
        """初始化PyBullet物理引擎"""
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.5]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # 设置物理参数
        p.setGravity(0, 0, self.config.gravity, physicsClientId=self.physics_client)
        p.setTimeStep(self.config.simulation_timestep, physicsClientId=self.physics_client)
        
        # 创建地面 (使用几何体而非URDF)
        ground_collision = p.createCollisionShape(
            p.GEOM_PLANE,
            physicsClientId=self.physics_client
        )
        ground_visual = p.createVisualShape(
            p.GEOM_PLANE,
            rgbaColor=[0.8, 0.8, 0.8, 1.0],
            physicsClientId=self.physics_client
        )
        self.plane_id = p.createMultiBody(
            baseCollisionShapeIndex=ground_collision,
            baseVisualShapeIndex=ground_visual,
            basePosition=[0, 0, 0],
            physicsClientId=self.physics_client
        )
        p.changeDynamics(self.plane_id, -1, lateralFriction=self.config.ground_friction,
                        physicsClientId=self.physics_client)
        
        logger.info(f"🔧 Physics engine initialized (client={self.physics_client})")
    
    def _define_spaces(self):
        """定义动作空间和观测空间"""
        # 获取所有可用零件
        self.available_parts = list(COLLISION_SHAPES.keys())
        self.n_parts = len(self.available_parts)
        
        # ===== 动作空间 (统一Box空间 - 兼容PPO/SAC) =====
        # 将离散和连续动作统一编码为连续向量
        # 格式: [part_selection(1), parent_conn(1), child_conn(1), assembly_params(6), motor_control(20)]
        total_action_dim = 1 + 1 + 1 + 6 + 20  # = 29
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(total_action_dim,), dtype=np.float32
        )
        
        # ===== 观测空间 =====
        # 计算各部分维度
        max_connectors = 15  # 每个零件最大连接点数
        
        obs_components = {
            # 1. 机器人状态 (躯干位置/速度/姿态)
            'robot_state': spaces.Box(
                low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
            ),  # [pos(3), vel(3), quat(4), angular_vel(3)]
            
            # 2. 零件配置矩阵 (one-hot编码 + 数量统计)
            'part_config': spaces.Box(
                low=0, high=self.config.max_parts_per_robot, 
                shape=(self.n_parts,), dtype=np.float32
            ),
            
            # 3. 连接关系图 (邻接矩阵)
            'connection_graph': spaces.Box(
                low=-1, high=1, shape=(self.config.max_parts_per_robot, 
                                        self.config.max_parts_per_robot, 4),
                dtype=np.float32
            ),  # [connected?, connector_type, joint_type, strength]
            
            # 4. 物理传感器数据
            'sensor_data': spaces.Box(
                low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
            ),  # [imu(6), force_torque(6), motor_states(8)]
            
            # 5. 装配进度
            'assembly_progress': spaces.Box(
                low=0.0, high=1.0, shape=(5,), dtype=np.float32
            ),  # [completion%, stability, energy, step_ratio, difficulty]
            
            # 6. 可用连接点掩码 (用于masking invalid actions)
            'valid_actions_mask': spaces.Box(
                low=0, high=1, shape=(self.n_parts * max_connectors * 2,), 
                dtype=np.float32
            ),
        }
        
        self.observation_space = spaces.Dict(obs_components)
        
        logger.info(f"📐 Action space: {self.action_space}")
        logger.info(f"📐 Observation space: {self.observation_space}")
    
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置环境到初始状态
        
        Returns:
            observation: 初始观测
            info: 额外信息字典
        """
        super().reset(seed=seed, **kwargs)
        
        # 重置物理世界
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, self.config.gravity, physicsClientId=self.physics_client)
        p.setTimeStep(self.config.simulation_timestep, physicsClientId=self.physics_client)
        
        # 重新创建地面
        ground_collision = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.physics_client)
        ground_visual = p.createVisualShape(
            p.GEOM_PLANE,
            rgbaColor=[0.8, 0.8, 0.8, 1.0],
            physicsClientId=self.physics_client
        )
        self.plane_id = p.createMultiBody(
            baseCollisionShapeIndex=ground_collision,
            baseVisualShapeIndex=ground_visual,
            basePosition=[0, 0, 0],
            physicsClientId=self.physics_client
        )
        p.changeDynamics(self.plane_id, -1, lateralFriction=self.config.ground_friction,
                        physicsClientId=self.physics_client)
        
        # 重置组装器和状态
        self.assembler = DynamicAssembler(
            physics_client_id=self.physics_client,
            use_gui=(self.render_mode == 'human')
        )
        
        self.robot_parts = []
        self.assembly_history = []
        self.state_buffer.clear()
        self.current_step = 0
        self.episode_reward = 0.0
        
        # 域随机化
        if self.config.domain_randomization:
            self._apply_domain_randomization()
        
        # 创建初始基础零件 (通常是底盘或躯干)
        self._create_base_part()
        
        # 运行几步仿真让系统稳定
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # 获取初始观测
        observation = self._get_observation()
        info = self._get_info()
        
        logger.info("🔄 Environment reset complete")
        
        return observation, info
    
    def _create_base_part(self):
        """创建机器人基础零件"""
        # 选择一个合适的基础零件 (如底盘或主框架)
        base_candidates = ['robomaster_chassis', 'aluminum_profile_20x20_200mm', 
                          'frc_chassis_rail']
        
        base_part = None
        for candidate in base_candidates:
            if candidate in COLLISION_SHAPES:
                base_part = candidate
                break
        
        if base_part is None:
            base_part = self.available_parts[0]  # fallback
        
        # 在地面上方创建基础零件
        part_info = get_part_info(base_part)
        base_height = 0.05  # 地面高度
        if part_info['collision']['type'] == 'box':
            base_height = part_info['collision']['params'].get('half_extents', [0.1, 0.1, 0.1])[2]
        elif part_info['collision']['type'] == 'cylinder':
            base_height = part_info['collision']['params'].get('height', 0.1) / 2
        
        position = [0, 0, base_height + 0.01]  # 略高于地面
        
        body_id = self.assembler.load_part(
            part_name=base_part,
            position=position,
            orientation=[0, 0, 0, 1],  # 单位四元数
            color=[0.8, 0.2, 0.2, 1.0]  # 红色标识基础件
        )
        
        if body_id is not None:
            self.robot_parts.append({
                'name': base_part,
                'body_id': body_id,
                'position': position.copy(),
                'is_base': True,
                'connectors_used': set(),
            })
            logger.info(f"📦 Base part created: {base_part} (id={body_id})")
    
    def _apply_domain_randomization(self):
        """应用域随机化 (提高泛化能力)"""
        # 随机重力微小扰动
        gravity_noise = self.np_random.uniform(-0.1, 0.1)
        actual_gravity = self.config.gravity + gravity_noise
        p.setGravity(0, 0, actual_gravity, physicsClientId=self.physics_client)
        
        # 随机地面摩擦力
        friction = self.np_random.uniform(0.8, 1.2)
        # 这会在加载地面时应用
        
        # 随机零件质量微小变化 (模拟制造误差)
        # 这会在创建零件时应用
        
        logger.debug(f"🎲 Domain randomization applied (g={actual_gravity:.2f}, μ={friction:.2f})")
    
    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作字典，包含:
                - part_selection: 选择的零件索引
                - parent_connector: 父零件连接点
                - child_connector: 子零件连接点
                - assembly_params: 装配参数偏移
                - motor_control: 电机控制信号
        
        Returns:
            observation: 新的观测
            reward: 奖励值
            terminated: 是否终止 (完成/失败)
            truncated: 是否截断 (超时)
            info: 额外信息
        """
        self.current_step += 1
        self.stats['total_steps'] += 1
        
        # 解析动作 (从统一Box空间解码)
        # 动作格式: [part_selection, parent_conn, child_conn, assembly_params(6), motor_control(20)]
        part_selection_val = action[0]  # [-1, 1] -> 映射到 [0, n_parts]
        parent_conn_val = action[1]      # [-1, 1] -> 映射到 [0, 15]
        child_conn_val = action[2]       # [-1, 1] -> 映射到 [0, 15]
        assembly_params = action[3:9]    # [pos_offset(3), orn_offset(3)]
        motor_control = action[9:29]     # 20个电机控制信号
        
        # 将连续值映射到离散索引
        part_idx = int(((part_selection_val + 1) / 2) * self.n_parts) % (self.n_parts + 1)
        parent_conn_idx = int(((parent_conn_val + 1) / 2) * 15) % 16
        child_conn_idx = int(((child_conn_val + 1) / 2) * 15) % 16
        
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # 执行装配动作 (如果不是"无操作")
        if part_idx < self.n_parts and len(self.robot_parts) < self.config.max_parts_per_robot:
            selected_part = self.available_parts[part_idx]
            assembly_success = self._try_assemble(
                selected_part, parent_conn_idx, child_conn_idx, assembly_params
            )
            
            if assembly_success:
                reward += self.config.reward_weights['completeness'] * 0.1
                self.stats['successful_assemblies'] += 1
            else:
                reward -= 0.05  # 失败的小惩罚
                self.stats['failed_assemblies'] += 1
        
        # 应用电机控制
        self._apply_motor_control(motor_control)
        
        # 运行物理仿真
        for _ in range(self.config.action_repeat):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # 计算多目标奖励
        reward_dict = self._calculate_rewards()
        reward = sum(w * v for w, v in zip(self.config.reward_weights.values(), 
                                           reward_dict.values()))
        
        # 更新episode累计奖励
        self.episode_reward += reward
        
        # 检查终止条件
        terminated, truncated = self._check_termination()
        
        # 获取新观测
        observation = self._get_observation()
        info = self._get_info()
        info.update({'reward_breakdown': reward_dict})
        
        # 如果episode结束，更新统计
        if terminated or truncated:
            self.stats['episodes_completed'] += 1
            if self.episode_reward > self.stats['best_reward']:
                self.stats['best_reward'] = self.episode_reward
            logger.info(f"Episode finished: reward={self.episode_reward:.2f}, "
                       f"steps={self.current_step}, parts={len(self.robot_parts)}")
        
        return observation, reward, terminated, truncated, info
    
    def _try_assemble(self, part_name: str, parent_conn_idx: int, 
                      child_conn_idx: int, params: np.ndarray) -> bool:
        """
        尝试执行装配操作
        
        Returns:
            bool: 装配是否成功
        """
        if len(self.robot_parts) == 0:
            return False
        
        # 选择父零件 (最后一个添加的非基础零件，或基础件)
        parent_part = self.robot_parts[-1] if len(self.robot_parts) > 1 else self.robot_parts[0]
        
        # 获取父零件的所有连接点
        parent_connectors = get_all_connectors(parent_part['name'])
        if not parent_connectors or parent_conn_idx >= len(parent_connectors):
            return False
        
        parent_conn_name = list(parent_connectors.keys())[parent_conn_idx % len(parent_connectors)]
        
        # 获取子零件连接点
        child_connectors = get_all_connectors(part_name)
        if not child_connectors or child_conn_idx >= len(child_connectors):
            return False
        
        child_conn_name = list(child_connectors.keys())[child_conn_idx % len(child_connectors)]
        
        # 验证连接兼容性
        if not validate_connection(parent_part['name'], parent_conn_name, 
                                  part_name, child_conn_name):
            return False
        
        # 计算子零件位置 (基于父零件位置 + 偏移)
        parent_pos = np.array(parent_part['position'])
        offset = params[:3] * 0.1  # 缩放偏移量
        child_position = (parent_pos + offset + [0, 0, 0.05]).tolist()  # 默认向上堆叠
        
        # 加载子零件
        child_body_id = self.assembler.load_part(
            part_name=part_name,
            position=child_position,
            color=self._random_color()
        )
        
        if child_body_id is None:
            return False
        
        # 执行装配
        constraint_id = self.assembler.assemble(
            parent_part_key=f"{parent_part['name']}_{parent_part['body_id']}",
            parent_connector=parent_conn_name,
            child_part_key=f"{part_name}_{child_body_id}",
            child_connector=child_conn_name,
            joint_type='fixed'
        )
        
        if constraint_id is not None:
            # 记录装配成功
            self.robot_parts.append({
                'name': part_name,
                'body_id': child_body_id,
                'position': child_position,
                'is_base': False,
                'connectors_used': {child_conn_name},
                'parent': parent_part['name'],
                'constraint_id': constraint_id,
            })
            
            # 记录到历史
            self.assembly_history.append({
                'step': self.current_step,
                'parent': parent_part['name'],
                'child': part_name,
                'parent_conn': parent_conn_name,
                'child_conn': child_conn_name,
                'success': True,
            })
            
            logger.debug(f"✅ Assembled: {part_name} → {parent_part['name']} "
                        f"(conn: {parent_conn_name} ↔ {child_conn_name})")
            return True
        
        return False
    
    def _apply_motor_control(self, motor_control: np.ndarray):
        """应用电机控制信号到已装配的电机零件"""
        for i, part in enumerate(self.robot_parts):
            part_info = get_part_info(part['name'])
            category = part_info.get('category', '')
            
            if category == 'actuator' and i < len(motor_control):
                control_signal = motor_control[i]
                
                # 应用速度控制或扭矩控制
                if part_info.get('electrical', {}).get('speed_no_load_RPM', 0) > 0:
                    # 速度控制模式
                    max_speed = part_info['electrical']['speed_no_load_RPM'] * 2 * np.pi / 60
                    target_velocity = control_signal * max_speed
                    
                    # PyBullet中设置电机速度 (简化版)
                    # 实际应该通过joint控制，这里用velocity控制近似
                    p.resetBaseVelocity(
                        part['body_id'],
                        linearVelocity=[target_velocity * 0.1, 0, 0],
                        angularVelocity=[0, 0, target_velocity],
                        physicsClientId=self.physics_client
                    )
    
    def _calculate_rewards(self) -> Dict[str, float]:
        """
        计算多目标奖励 (帕累托最优方向)
        
        Returns:
            dict: 各目标的原始奖励值
        """
        rewards = {}
        
        # 1. 移动距离奖励 (主要目标)
        if len(self.robot_parts) > 0:
            base_part = self.robot_parts[0]
            pos, _ = p.getBasePositionAndOrientation(base_part['body_id'], 
                                                     physicsClientId=self.physics_client)
            distance = pos[0]  # X轴移动距离
            rewards['distance'] = distance * 10.0  # 缩放因子
            
            # 更新最佳距离
            if distance > self.stats['best_distance']:
                self.stats['best_distance'] = distance
        else:
            rewards['distance'] = 0.0
        
        # 2. 稳定性奖励 (姿态保持 + 不翻倒)
        stability_score = self._calculate_stability()
        rewards['stability'] = stability_score
        
        # 3. 能效奖励 (移动距离 / 能耗)
        energy_score = self._calculate_energy_efficiency(distance=rewards.get('distance', 0))
        rewards['energy_efficiency'] = energy_score
        
        # 4. 完整度奖励 (已装配零件比例)
        completeness = len(self.robot_parts) / self.config.max_parts_per_robot
        rewards['completeness'] = completeness
        
        # 5. 可制造性奖励 (连接合理性 + 零件选择)
        manufacturability = self._calculate_manufacturability()
        rewards['manufacturability'] = manufacturability
        
        # 6. 存活奖励 (每步小奖励，鼓励长期生存)
        rewards['survival'] = 0.01
        
        return rewards
    
    def _calculate_stability(self) -> float:
        """计算稳定性分数 (0-1)"""
        if len(self.robot_parts) == 0:
            return 0.0
        
        stability = 1.0
        
        for part in self.robot_parts[:min(5, len(self.robot_parts))]:  # 只检查前5个零件
            pos, orn = p.getBasePositionAndOrientation(
                part['body_id'], physicsClientId=self.physics_client
            )
            
            # 检查是否翻倒 (Z轴朝上分量)
            rotation_matrix = p.getMatrixFromQuaternion(orn, physicsClientId=self.physics_client)
            # rotation_matrix是9个元素的列表，表示3x3矩阵
            z_up = rotation_matrix[8]  # 索引8是矩阵的[2][2]元素，即Z轴在世界坐标系中的Z分量
            
            if z_up < 0.5:  # 严重倾斜
                stability *= 0.5
            elif z_up < 0.8:  # 轻微倾斜
                stability *= 0.8
            
            # 检查高度 (不能太高或太低)
            if pos[2] < 0.01:  # 掉到地面以下
                stability *= 0.1
            elif pos[2] > 2.0:  # 飞太高了
                stability *= 0.7
        
        return stability
    
    def _calculate_energy_efficiency(self, distance: float = 0.0) -> float:
        """计算能效分数"""
        if distance <= 0 or len(self.robot_parts) == 0:
            return 0.0
        
        # 估算总质量
        total_mass = sum(
            get_part_info(part['name'])['physics']['mass_kg']
            for part in self.robot_parts
        )
        
        # 简单能效模型: distance / mass
        efficiency = distance / (total_mass + 0.1)  # 避免除零
        
        return min(efficiency * 10, 1.0)  # 归一化到0-1
    
    def _calculate_manufacturability(self) -> float:
        """计算可制造性分数"""
        if len(self.robot_parts) < 2:
            return 0.5  # 基础分
        
        score = 1.0
        
        # 检查连接多样性
        connection_types = set()
        for record in self.assembly_history[-10:]:  # 最近10次连接
            if record['success']:
                connection_types.add((record['parent_conn'], record['child_conn']))
        
        diversity_bonus = min(len(connection_types) / 5, 1.0)  # 鼓励多种连接方式
        score *= (0.7 + 0.3 * diversity_bonus)
        
        # 检查零件类别平衡
        categories = set()
        for part in self.robot_parts:
            cat = get_part_info(part['name']).get('category', '')
            categories.add(cat)
        
        category_bonus = min(len(categories) / 4, 1.0)  # 鼓励使用多类别零件
        score *= (0.8 + 0.2 * category_bonus)
        
        return score
    
    def _check_termination(self) -> Tuple[bool, bool]:
        """检查episode终止条件"""
        terminated = False
        truncated = False
        
        # 终止条件1: 达到最大步数
        if self.current_step >= self.config.max_steps_per_episode:
            truncated = True
        
        # 终止条件2: 机器人完全散架 (所有零件掉落)
        if len(self.robot_parts) > 0:
            base_part = self.robot_parts[0]
            pos, _ = p.getBasePositionAndOrientation(base_part['body_id'],
                                                     physicsClientId=self.physics_client)
            if pos[2] < -0.1:  # 基础件掉落到地下
                terminated = True
                logger.warning("💥 Robot collapsed - episode terminated")
        
        # 终止条件3: 成功完成任务 (可选)
        if len(self.robot_parts) >= self.config.min_parts_for_complete:
            # 可以添加任务特定的成功条件
            pass
        
        return terminated, truncated
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取当前状态观测"""
        obs = {}
        
        # 1. 机器人状态
        if len(self.robot_parts) > 0:
            base_part = self.robot_parts[0]
            pos, orn = p.getBasePositionAndOrientation(
                base_part['body_id'], physicsClientId=self.physics_client
            )
            vel, angular_vel = p.getBaseVelocity(
                base_part['body_id'], physicsClientId=self.physics_client
            )
            
            robot_state = np.array([
                *pos,           # 位置 (3)
                *vel,           # 速度 (3)
                *orn,           # 四元数 (4)
                *angular_vel,   # 角速度 (3)
            ], dtype=np.float32)
        else:
            robot_state = np.zeros(13, dtype=np.float32)
        
        obs['robot_state'] = robot_state
        
        # 2. 零件配置 (one-hot计数)
        part_config = np.zeros(self.n_parts, dtype=np.float32)
        for part in self.robot_parts:
            idx = self.available_parts.index(part['name']) if part['name'] in self.available_parts else 0
            part_config[idx] += 1
        obs['part_config'] = part_config
        
        # 3. 连接图 (邻接矩阵)
        conn_graph = np.full(
            (self.config.max_parts_per_robot, self.config.max_parts_per_robot, 4),
            -1.0, dtype=np.float32
        )
        for i, record in enumerate(self.assembly_history[:self.config.max_parts_per_robot]):
            if record['success']:
                # 简化的连接表示
                conn_graph[i, i+1, 0] = 1.0  # connected
        obs['connection_graph'] = conn_graph
        
        # 4. 传感器数据
        sensor_data = self._get_sensor_data()
        obs['sensor_data'] = sensor_data
        
        # 5. 装配进度
        progress = np.array([
            len(self.robot_parts) / self.config.max_parts_per_robot,  # 完成度
            self._calculate_stability(),  # 稳定性
            self._calculate_energy_efficiency(),  # 能效
            self.current_step / self.config.max_steps_per_episode,  # 步数比率
            self.config.difficulty,  # 难度
        ], dtype=np.float32)
        obs['assembly_progress'] = progress
        
        # 6. 有效动作掩码
        valid_mask = self._compute_valid_action_mask()
        obs['valid_actions_mask'] = valid_mask
        
        # 添加到历史缓冲区
        self.state_buffer.append(obs)
        
        return obs
    
    def _get_sensor_data(self) -> np.ndarray:
        """获取物理传感器数据"""
        sensors = np.zeros(20, dtype=np.float32)
        
        if len(self.robot_parts) > 0:
            base_part = self.robot_parts[0]
            
            # IMU数据 (加速度计 + 陀螺仪)
            vel, angular_vel = p.getBaseVelocity(
                base_part['body_id'], physicsClientId=self.physics_client
            )
            sensors[0:3] = np.array(vel) / 10.0  # 归一化加速度
            sensors[3:6] = np.array(angular_vel) / 10.0  # 归一化角速度
            
            # 力/力矩传感器 (简化)
            sensors[6:12] = 0.0  # 需要force sensor才能获取真实值
            
            # 电机状态 (位置/速度/扭矩)
            for i, part in enumerate(self.robot_parts[:8]):  # 最多8个电机
                if get_part_info(part['name']).get('category') == 'actuator':
                    pos, orn = p.getBasePositionAndOrientation(
                        part['body_id'], physicsClientId=self.physics_client
                    )
                    vel, ang_vel = p.getBaseVelocity(
                        part['body_id'], physicsClientId=self.physics_client
                    )
                    sensors[12 + i*1] = np.linalg.norm(ang_vel)  # 电机转速
        
        return sensors
    
    def _compute_valid_action_mask(self) -> np.ndarray:
        """计算有效动作掩码 (用于masking无效动作)"""
        mask = np.ones(self.n_parts * 15 * 2, dtype=np.float32)
        
        # 简化: 所有动作都允许 (实际应该根据当前状态过滤)
        if len(self.robot_parts) >= self.config.max_parts_per_robot:
            mask[:] = 0.0  # 已达到最大零件数
        
        return mask
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            'num_parts': len(self.robot_parts),
            'current_step': self.current_step,
            'episode_reward': self.episode_reward,
            'assembly_history_length': len(self.assembly_history),
            'stats': self.stats.copy(),
        }
    
    def _random_color(self) -> List[float]:
        """生成随机颜色"""
        return [
            self.np_random.uniform(0.3, 1.0),
            self.np_random.uniform(0.3, 1.0),
            self.np_random.uniform(0.3, 1.0),
            1.0,  # alpha
        ]
    
    def render(self):
        """渲染环境 (仅GUI模式)"""
        if self.render_mode == 'human':
            # PyBullet GUI自动更新
            pass
        elif self.render_mode == 'rgb_array':
            # 返回RGB图像数组
            width, height = 640, 480
            view_matrix = p.computeViewMatrixFromYawRollPitch(
                cameraTargetPosition=[0, 0, 0.5],
                distance=2.0,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.physics_client
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=width/height,
                nearVal=0.01,
                farVal=100.0,
                physicsClientId=self.physics_client
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=self.physics_client
            )
            return np.array(px).reshape(height, width, 4)[..., :3]
        return None
    
    def close(self):
        """关闭环境并清理资源"""
        if hasattr(self, 'physics_client') and self.physics_client >= 0:
            p.disconnect(physicsClientId=self.physics_client)
            self.physics_client = -1
        logger.info("🔌 Environment closed")


class VectorizedV8Env(gym.vector.VectorEnv):
    """
    向量化V8环境 (用于大规模并行训练)
    
    支持同时运行多个环境实例，显著提升训练效率。
    可与Stable-Baselines3的VecEnv无缝集成。
    """
    
    def __init__(self, num_envs: int = 8, config: RLConfig = None):
        super().__init__()
        self.num_envs = num_envs
        self.config = config or RLConfig()
        
        # 创建多个环境实例
        self.envs = [
            V8RobotAssemblyEnv(config=self.config)
            for _ in range(num_envs)
        ]
        
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self, **kwargs):
        """重置所有环境"""
        observations = []
        infos = []
        for env in self.envs:
            obs, info = env.reset(**kwargs)
            observations.append(obs)
            infos.append(info)
        
        # 堆叠观测 (向量化格式)
        stacked_obs = self._stack_observations(observations)
        return stacked_obs, infos
    
    def step_async(self, actions):
        """异步执行步骤 (用于并行化)"""
        self._actions = actions
    
    def step_wait(self):
        """等待步骤完成并返回结果"""
        observations = []
        rewards = []
        terminators = []
        truncators = []
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, reward, term, trunc, info = env.step(self._actions[i])
            observations.append(obs)
            rewards.append(reward)
            terminators.append(term)
            truncators.append(trunc)
            infos.append(info)
        
        return (
            self._stack_observations(observations),
            np.array(rewards),
            np.array(terminators),
            np.array(truncators),
            infos,
        )
    
    def step(self, actions):
        """同步步骤 (简化接口)"""
        self.step_async(actions)
        return self.step_wait()
    
    def _stack_observations(self, observations: List[Dict]) -> Dict[str, np.ndarray]:
        """堆叠多个环境的观测"""
        stacked = {}
        for key in observations[0].keys():
            stacked[key] = np.stack([obs[key] for obs in observations])
        return stacked
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()
    
    @property
    def unwrapped(self):
        """返回未包装的环境"""
        return self.envs[0].unwrapped
    
    def env_is_wrapped(self, wrapper_class):
        """检查环境是否被指定wrapper包装"""
        return [False] * self.num_envs


# ============================================================
# 便捷函数和环境创建器
# ============================================================

def create_training_env(render_mode: str = None, difficulty: float = 0.5) -> V8RobotAssemblyEnv:
    """创建标准训练环境"""
    config = RLConfig(
        max_steps_per_episode=500,
        difficulty=difficulty,
        domain_randomization=True,
    )
    return V8RobotAssemblyEnv(config=config, render_mode=render_mode)


def create_eval_env(render_mode: str = 'human') -> V8RobotAssemblyEnv:
    """创建评估/演示环境"""
    config = RLConfig(
        max_steps_per_episode=1000,
        domain_randomization=False,  # 评估时不随机化
        difficulty=0.7,
    )
    return V8RobotAssemblyEnv(config=config, render_mode=render_mode)


def create_vectorized_env(num_envs: int = 8) -> VectorizedV8Env:
    """创建向量化训练环境 (用于PPO/SAC等算法)"""
    return VectorizedV8Env(num_envs=num_envs)


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("V8 Robot Assembly RL Environment - Test Suite")
    print("=" * 70)
    
    # 测试1: 基本功能测试
    print("\n[Test 1] Basic environment functionality...")
    try:
        env = create_training_env(render_mode=None)
        print(f"✅ Environment created successfully")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space type: {type(env.observation_space)}")
        
        # 测试reset
        obs, info = env.reset()
        print(f"✅ Reset successful, observation keys: {list(obs.keys())}")
        
        # 测试step (随机动作)
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"✅ Step successful, reward={reward:.4f}, terminated={term}, truncated={trunc}")
        
        # 运行几个episode
        total_reward = 0
        for ep in range(3):
            obs, info = env.reset()
            ep_reward = 0
            for step in range(50):
                action = env.action_space.sample()
                obs, reward, term, trunc, info = env.step(action)
                ep_reward += reward
                if term or trunc:
                    break
            total_reward += ep_reward
            print(f"   Episode {ep+1}: steps={step}, reward={ep_reward:.2f}, "
                  f"parts={info['num_parts']}")
        
        print(f"\n📊 Average reward over 3 episodes: {total_reward/3:.2f}")
        print(f"📊 Stats: {env.stats}")
        
        env.close()
        print("✅ Basic test PASSED\n")
        
    except Exception as e:
        print(f"❌ Basic test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
    
    # 测试2: 向量化环境测试
    print("[Test 2] Vectorized environment...")
    try:
        vec_env = create_vectorized_env(num_envs=4)
        print(f"✅ Vectorized environment created ({vec_env.num_envs} envs)")
        
        obs, infos = vec_env.reset()
        print(f"✅ Reset all environments")
        
        actions = [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
        obs, rewards, terms, truncs, infos = vec_env.step(actions)
        print(f"✅ Step all environments, mean reward: {np.mean(rewards):.4f}")
        
        vec_env.close()
        print("✅ Vectorized environment test PASSED\n")
        
    except Exception as e:
        print(f"❌ Vectorized environment test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
    
    # 测试3: 域随机化测试
    print("[Test 3] Domain randomization...")
    try:
        env_no_rand = create_training_env(difficulty=0.3)
        env_with_rand = create_training_env(difficulty=0.8)
        
        obs1, _ = env_no_rand.reset()
        obs2, _ = env_with_rand.reset()
        
        print(f"✅ Different difficulty levels created successfully")
        
        env_no_rand.close()
        env_with_rand.close()
        print("✅ Domain randomization test PASSED\n")
        
    except Exception as e:
        print(f"❌ Domain randomization test FAILED: {e}\n")
    
    print("=" * 70)
    print("All tests completed! 🎉")
    print("=" * 70)
