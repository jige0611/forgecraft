#!/usr/bin/env python3
"""
两电机竞速 RL 环境 (Two-Motor Racing Gym)
==========================================
- 仅 2× M3508 电机有驱动能力，其余零件被动
- 目标：不倒的前提下跑最快
- 模拟：MuJoCo 物理引擎
- 接口：Gymnasium

状态: IMU姿态 + 轮速 + 电机力矩 + 底盘位置
动作: 左右电机控制信号 [-1, 1]
奖励: 前进速度 × 不倒奖励 - 能耗惩罚
终止: 倾倒 (roll > 60° 或 pitch > 60°) 
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from typing import Optional
from pathlib import Path

XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "two_wheel_racer.xml"))


class TwoWheelRacerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 3000):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        
        # 加载 MuJoCo 模型 (用 from_xml_string 绕过中文路径问题)
        xml_content = Path(XML_PATH).read_text(encoding='utf-8')
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)
        
        # 渲染
        self._viewer = None
        if render_mode == "human":
            from mujoco import viewer
            self._viewer = viewer.launch_passive(self.model, self.data)
            # 调整视角：离近点看
            self._viewer.cam.distance = 0.8
            self._viewer.cam.elevation = -15
            self._viewer.cam.azimuth = 90
        
        # 动作空间: [左电机, 右电机] ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # 观测空间 (23维)
        # - quat(4): 底盘四元数
        # - angvel(3): 角速度
        # - linvel(3): 线速度
        # - imu_acc(3): 加速度计
        # - wheel_vel(2): 轮速
        # - motor_torque(2): 当前力矩  
        # - motor_action(2): 上一帧动作
        # - rel_pos_xy(2): 相对起始位置
        # - height(1): 底盘高度
        # - dot_up(1): 上方向点积
        # 实际: 4+3+3+3+2+2+2+2+1+1 = 23
        obs_size = 23
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # 状态记忆
        self._step_count = 0
        self._total_reward = 0.0
        self._max_speed = 0.0
        self._initial_qpos = None
        self._episode_history = []
        
        # 上次动作
        self._last_action = np.zeros(2, dtype=np.float32)
    
    def _get_obs(self) -> np.ndarray:
        """构建观测向量"""
        d = self.data
        
        # 传感器索引
        # imu_pos: 0-2, imu_quat: 3-6, imu_linvel: 7-9, imu_angvel: 10-12
        # left_wheel_vel: 13, right_wheel_vel: 14
        # left_torque: 15, right_torque: 16
        # chassis_acc: 17-19
        
        # 从 MuJoCo sensor 读取
        quat = d.sensordata[3:7].copy()  # 四元数 [w,x,y,z]
        angvel = d.sensordata[10:13].copy()  # 角速度
        linvel = d.sensordata[7:10].copy()  # 线速度
        wheel_vel = d.sensordata[13:15].copy()  # 轮速
        motor_torque = d.sensordata[15:17].copy()  # 力矩
        imu_acc = d.sensordata[17:20].copy()  # 加速度
        
        # 相对起始位置
        rel_xy = d.qpos[0:2].copy() - self._initial_qpos[0:2]
        
        # 底盘高度
        height = d.qpos[2] - self._initial_qpos[2]
        
        # 上方向点积 (1 = 完全竖直, 0 = 水平, -1 = 倒立)
        # 用 z 分量近似
        # 四元数到上向量: [2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx), 1-2*(qx^2+qy^2)]
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        up_z = 1 - 2 * (qx**2 + qy**2)
        
        obs = np.concatenate([
            quat,           # 4
            angvel,         # 3
            linvel,         # 3
            imu_acc,        # 3
            wheel_vel,      # 2
            motor_torque,   # 2
            self._last_action,  # 2
            rel_xy,         # 2
            [height],       # 1
            [up_z],         # 1
        ]).astype(np.float32)
        
        return obs
    
    def _get_reward(self) -> float:
        """计算奖励: 前进速度 × 不倒系数 - 能耗"""
        d = self.data
        
        # 1. 前进速度奖励 (以底盘 x 方向为准)
        linvel = d.sensordata[7:10]
        forward_speed = linvel[0]  # x方向速度
        speed_reward = forward_speed * 5.0  # 放大速度奖励
        
        # 2. 不倒奖励
        qw, qx, qy, qz = d.sensordata[3], d.sensordata[4], d.sensordata[5], d.sensordata[6]
        # 上向量 z 分量
        up_z = 1 - 2 * (qx**2 + qy**2)
        # roll/pitch 角度
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        
        # 倾斜惩罚 (roll或pitch超过30°开始惩罚)
        tilt = max(abs(roll), abs(pitch))
        if tilt < 0.3:  # < 17°
            stability = 1.0
        elif tilt < 0.6:  # 17-34°
            stability = 1.0 - (tilt - 0.3) / 0.3 * 0.5
        else:
            stability = max(0.1, 0.5 - (tilt - 0.6) / 0.4 * 0.4)
        
        # 3. 能耗惩罚 (鼓励高效)
        torque = d.sensordata[15:17]
        wheel_vel = d.sensordata[13:15]
        power = np.sum(np.abs(torque * wheel_vel))
        energy_penalty = power * 0.05
        
        # 4. 存活奖励
        alive_bonus = 0.02
        
        # 5. 轮速一致性奖励 (差速不能太大，否则容易打转)
        diff = abs(wheel_vel[0] - wheel_vel[1])
        smoothness = max(0, 1.0 - diff * 0.1)
        
        reward = (
            speed_reward * stability
            + alive_bonus
            - energy_penalty
            + smoothness * 0.1
        )
        
        return float(reward)
    
    def _check_done(self) -> tuple[bool, bool]:
        """检查终止条件"""
        d = self.data
        
        # 四元数
        qw, qx, qy, qz = d.sensordata[3], d.sensordata[4], d.sensordata[5], d.sensordata[6]
        up_z = 1 - 2 * (qx**2 + qy**2)
        
        # 1. 翻倒检测 (上方向 < cos(60°) = 0.5)
        tipped = up_z < 0.5
        
        # 2. 底盘触地 (高度太低)
        height = d.qpos[2]
        too_low = height < 0.03
        
        # 3. 步数限制
        timeout = self._step_count >= self.max_steps
        
        # 4. 飞出边界
        out_of_bounds = abs(d.qpos[0]) > 20 or abs(d.qpos[1]) > 20
        
        terminated = tipped or too_low or out_of_bounds
        truncated = timeout
        
        return terminated, truncated
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # 随机初始姿态（小扰动）
        mujoco.mj_resetData(self.model, self.data)
        
        # 初始位置: 略微离地
        self.data.qpos[0] = np.random.uniform(-1, 1)
        self.data.qpos[1] = np.random.uniform(-0.5, 0.5)
        self.data.qpos[2] = 0.12  # 悬空一点
        self.data.qpos[3:7] = [1, 0, 0, 0]  # 初始四元数
        
        # 小角度初始倾斜(让机器人落稳)
        if self.np_random.random() < 0.4:
            rand_angle = self.np_random.uniform(-0.1, 0.1)
            self.data.qpos[4] = np.sin(rand_angle/2)  # 绕x轴
            self.data.qpos[3] = np.cos(rand_angle/2)
        
        # 让物理稳定
        mujoco.mj_forward(self.model, self.data)
        self._initial_qpos = self.data.qpos.copy()
        self._step_count = 0
        self._total_reward = 0.0
        self._max_speed = 0.0
        self._last_action = np.zeros(2, dtype=np.float32)
        self._episode_history = []
        
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action: np.ndarray):
        self._step_count += 1
        
        # 应用动作
        action = np.clip(action, -1, 1).astype(np.float32)
        self.data.ctrl[0] = action[0]  # 左电机
        self.data.ctrl[1] = action[1]  # 右电机
        
        # 物理步进 (子步)
        n_substeps = 10
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        # 保存动作
        self._last_action = action.copy()
        
        # 观测 + 奖励 + 终止
        obs = self._get_obs()
        reward = self._get_reward()
        terminated, truncated = self._check_done()
        
        # 累计统计
        self._total_reward += reward
        forward_speed = self.data.sensordata[7]
        if abs(forward_speed) > self._max_speed:
            self._max_speed = abs(forward_speed)
        
        # 额外终止奖励
        if terminated and not truncated:
            tip_penalty = -5.0
            reward += tip_penalty
        
        info = {
            "forward_speed": float(self.data.sensordata[7]),
            "max_speed": self._max_speed,
            "total_reward": self._total_reward,
            "tilt": float(max(
                abs(np.arctan2(2*(self.data.sensordata[3]*self.data.sensordata[4] + 
                    self.data.sensordata[5]*self.data.sensordata[6]),
                    1-2*(self.data.sensordata[4]**2+self.data.sensordata[5]**2))),
                abs(np.arcsin(np.clip(2*(self.data.sensordata[3]*self.data.sensordata[5] - 
                    self.data.sensordata[6]*self.data.sensordata[4]), -1, 1)))
            )),
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human" and self._viewer:
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, 480, 640)
            renderer.update_scene(self.data)
            return renderer.render()
    
    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None


def test_env():
    """快速测试环境"""
    env = TwoWheelRacerEnv(render_mode=None, max_steps=500)
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}, Obs range: [{obs.min():.2f}, {obs.max():.2f}]")
    
    total_reward = 0
    for i in range(500):
        # 简单差速驱动
        action = np.array([0.8, 0.8])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode end at step {i+1}: reward={total_reward:.2f}, max_speed={info['max_speed']:.3f} m/s")
            break
    
    env.close()
    print("Environment test passed!")


if __name__ == "__main__":
    test_env()
