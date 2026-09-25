"""
ForgeCraft 通用 RL 环境

支持任意形态 + 任意任务配置。任务通过 TaskConfig.reward_components
动态定义奖励函数，不再硬编码 speed/energy/upright。
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
import logging

from forgecraft.config import SimConfig, TaskConfig, PartSpec
from forgecraft.core.morphology import MechanicalBody
from forgecraft.rl.encoder import MorphologyEncoder
from forgecraft.simulation.builder import build_mjcf_model

_logger = logging.getLogger(__name__)

__all__ = ["RunningMeanStd", "ForgeCraftEnv"]


class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_stats(batch_mean, batch_var, batch_count)

    def _update_from_stats(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


_REWARD_SENSORS = {
    "displacement": lambda env: env._step_dx * 10.0,
    "speed": lambda env: env._step_dx / max(env.sim_config.timestep * env.sim_config.substeps, 1e-6),
    "upright": lambda env: max(0.0, env._com_z - env.task_config.fall_height),
    "alive": lambda env: env.sim_config.timestep,
    "energy": lambda env: -1 * sum(abs(f * v) for f, v in zip(env._act_forces, env._act_velocities)) * env.sim_config.timestep,
    "height": lambda env: env._com_z,
    "velocity": lambda env: np.sqrt(env._com_vel[0]**2 + env._com_vel[1]**2 + env._com_vel[2]**2),
    "joint_effort": lambda env: -1 * sum(abs(env.data.actuator_force)),
    "orientation": lambda env: 1.0 - abs(env._up_dot) * 0.5,
    "action_smoothness": lambda env: -1 * np.sum((env._last_action - env.data.ctrl[:env.n_actuators])**2) * 0.1 if env._last_action is not None else 0.0,

    # 新增：移动激励传感器
    "movement_bonus": lambda env: 1.0 if abs(env._step_dx) > 0.001 else 0.0,
    "stillness_penalty": lambda env: 1.0 if abs(env._step_dx) < 0.0001 else 0.0,

    # 反常识运动指标
    "novelty_score": lambda env: env._compute_novelty(),
    "asymmetry_index": lambda env: env._compute_asymmetry(),
    "irregularity_metric": lambda env: env._compute_irregularity(),
    "bilateral_symmetry": lambda env: env._compute_bilateral_symmetry(),
    "shape_commonality_score": lambda env: env._compute_shape_commonality(),
    "passive_displacement": lambda env: abs(env._passive_dx) if hasattr(env, '_passive_dx') else 0.0,
    "total_displacement": lambda env: abs(env._initial_com_pos[0] - env.data.sensordata[env._sensor_offset_com]) if env._initial_com_pos is not None and env.model.nsensor >= 3 else 0.0,
    "energy_recycling": lambda env: env._compute_energy_recycling(),
    "mode_switching_bonus": lambda env: env._compute_mode_switching(),
    "energy_waste_penalty": lambda env: env._compute_energy_waste(),
    "efficiency_ratio": lambda env: (abs(env._step_dx) / max(sum(abs(f * v) for f, v in zip(env._act_forces, env._act_velocities)) * env.sim_config.timestep, 1e-6)) if env._act_forces else 0.0,
}
from forgecraft.evaluation.metrics import _FITNESS_SENSORS  # 统一传感器定义


class ForgeCraftEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        body: MechanicalBody,
        sim_config: Optional[SimConfig] = None,
        task_config: Optional[TaskConfig] = None,
        morph_encoder: Optional[MorphologyEncoder] = None,
        catalog: Optional[Dict[str, PartSpec]] = None,
        render_mode: Optional[str] = None,
        obs_rms: Optional["RunningMeanStd"] = None,
    ):
        super().__init__()
        self.body = body
        self.sim_config = sim_config or SimConfig()
        self.task_config = task_config or TaskConfig()
        self.morph_encoder = morph_encoder
        self.render_mode = render_mode

        if body._cached_mjcf is not None:
            xml_str, self.joint_map, self.motor_map, self.sensor_info = body._cached_mjcf
        else:
            xml_str, self.joint_map, self.motor_map, self.sensor_info = build_mjcf_model(body, catalog)
            body._cached_mjcf = (xml_str, self.joint_map, self.motor_map, self.sensor_info)
        self.xml_str = xml_str
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)

        self.n_actuators = self.model.nu
        self.n_motors = len(self.motor_map)

        if self.n_actuators == 0:
            self.n_actuators = 1

        self._sensor_offset_com = 0
        self._sensor_offset_comvel = 3
        self._sensor_offset_comacc = 6
        self._sensor_offset_joints = 9

        self._touch_sensor_ids = []
        self._imu_quat_ids = []
        self._imu_acc_ids = []
        self._imu_gyro_ids = []

        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) or ""
            if name.startswith("touch_"):
                self._touch_sensor_ids.append(i)
            elif name.startswith("imu_quat_"):
                self._imu_quat_ids.append(i)
            elif name.startswith("imu_acc_"):
                self._imu_acc_ids.append(i)
            elif name.startswith("imu_gyro_"):
                self._imu_gyro_ids.append(i)

        obs_dim = self._compute_obs_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_actuators,), dtype=np.float32
        )

        self._step_count = 0
        self._initial_com_pos: Optional[np.ndarray] = None
        self._last_com_pos: Optional[np.ndarray] = None
        self._total_energy: float = 0.0
        self._cumulative_reward: float = 0.0
        self._last_action: Optional[np.ndarray] = None
        self._step_dx: float = 0.0
        self._com_z: float = 0.5
        self._com_vel: np.ndarray = np.zeros(3)
        self._act_forces: List[float] = []
        self._act_velocities: List[float] = []
        self._up_dot: float = 1.0
        self._obs_buffer: List[np.ndarray] = []
        self.obs_rms: Optional[RunningMeanStd] = obs_rms

        obs_dim = self._compute_obs_dim()
        self._own_obs_rms = RunningMeanStd(shape=(obs_dim,))

        if render_mode == "human":
            import glfw
            glfw.init()
            self._renderer = None

    def _compute_obs_dim(self) -> int:
        base = self.n_motors * 2 + 3 + 3
        n_touch = len(self.sensor_info.get("touch_nodes", []))
        n_imu = len(self.sensor_info.get("imu_nodes", []))
        base += n_touch
        base += n_imu * 10
        return max(base, 10)

    # ── Sensor Reading ── 从 MuJoCo sensor 提取质心/关节/触地/IMU 数据
    def _read_sensors(self):
        if self.model.nsensor < 9:
            self._com_z = self.data.qpos[2] if self.model.nq >= 3 else 0.5
            self._com_vel = np.zeros(3)
            if self.model.nv >= 3:
                self._com_vel = self.data.qvel[:3]
            self._act_forces = list(self.data.actuator_force) if self.n_actuators > 0 else [0.0]
            self._act_velocities = list(self.data.actuator_velocity) if self.n_actuators > 0 else [0.0]
            return

        com_x = self.data.sensordata[self._sensor_offset_com]
        com_y = self.data.sensordata[self._sensor_offset_com + 1]
        self._com_z = self.data.sensordata[self._sensor_offset_com + 2]
        self._com_vel = np.array([
            self.data.sensordata[self._sensor_offset_comvel],
            self.data.sensordata[self._sensor_offset_comvel + 1],
            self.data.sensordata[self._sensor_offset_comvel + 2],
        ])
        self._act_forces = list(self.data.actuator_force)
        self._act_velocities = list(self.data.actuator_velocity)

        if self._initial_com_pos is not None:
            self._step_dx = com_x - self._last_com_pos[0]

    # ── Observation Encoding ── 拼接关节位置+速度+质心+触地+姿态编码
    def _get_obs(self) -> np.ndarray:
        obs_parts = []

        for i in range(min(self.n_motors, 8)):
            idx_jp = self._sensor_offset_joints + i
            idx_jv = self._sensor_offset_joints + self.model.njnt + i
            if self.model.nsensor > 0 and idx_jp < self.model.nsensor:
                obs_parts.append(float(self.data.sensordata[idx_jp]))
                if idx_jv < self.model.nsensor:
                    obs_parts.append(float(self.data.sensordata[idx_jv]))
                else:
                    obs_parts.append(0.0)
            else:
                if self.model.nq > i:
                    obs_parts.append(float(self.data.qpos[i]))
                    if self.model.nv > i:
                        obs_parts.append(float(self.data.qvel[i]))
                    else:
                        obs_parts.append(0.0)

        if self.model.nsensor >= 3:
            for i in range(3):
                obs_parts.append(float(self.data.sensordata[self._sensor_offset_com + i]))
            for i in range(3):
                obs_parts.append(float(self.data.sensordata[self._sensor_offset_comvel + i]))
        else:
            for i in range(min(3, self.model.nq)):
                obs_parts.append(float(self.data.qpos[i]))
            for i in range(min(3, self.model.nv)):
                obs_parts.append(float(self.data.qvel[i]))

        for tid in self._touch_sensor_ids:
            obs_parts.append(float(self.data.sensordata[tid]))

        for i in range(len(self._imu_quat_ids)):
            qid = self._imu_quat_ids[i]
            aid = self._imu_acc_ids[i] if i < len(self._imu_acc_ids) else None
            gid = self._imu_gyro_ids[i] if i < len(self._imu_gyro_ids) else None
            for j in range(4):
                if qid is not None and self.model.nsensor > qid:
                    idx = self.model.sensor_adr[qid] + j
                    if idx < self.model.nsensordata:
                        obs_parts.append(float(self.data.sensordata[idx]))
                    else:
                        obs_parts.append(0.0)
                else:
                    obs_parts.append(0.0)
            for j in range(3):
                if aid is not None and self.model.nsensor > aid:
                    idx = self.model.sensor_adr[aid] + j
                    if idx < self.model.nsensordata:
                        obs_parts.append(float(self.data.sensordata[idx]))
                    else:
                        obs_parts.append(0.0)
                else:
                    obs_parts.append(0.0)
            for j in range(3):
                if gid is not None and self.model.nsensor > gid:
                    idx = self.model.sensor_adr[gid] + j
                    if idx < self.model.nsensordata:
                        obs_parts.append(float(self.data.sensordata[idx]))
                    else:
                        obs_parts.append(0.0)
                else:
                    obs_parts.append(0.0)

        while len(obs_parts) < self.observation_space.shape[0]:
            obs_parts.append(0.0)

        obs = np.array(obs_parts[:self.observation_space.shape[0]], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        self._obs_buffer.append(obs.astype(np.float64))
        rms = self.obs_rms if self.obs_rms is not None else self._own_obs_rms
        if rms.mean.shape[0] != obs.shape[0]:
            return obs
        return rms.normalize(obs).astype(np.float32)

    # ── Reward Computation ── 按 TaskConfig.reward_components 加权求和
    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        self._read_sensors()

        if self._initial_com_pos is None:
            self._initial_com_pos = np.array([
                self.data.sensordata[self._sensor_offset_com] if self.model.nsensor >= 1 else 0.0,
                0.0,
                self.data.sensordata[self._sensor_offset_com + 2] if self.model.nsensor >= 3 else 0.5,
            ])
            self._last_com_pos = self._initial_com_pos.copy()
            self._step_dx = 0.0
            return 0.0, {}

        components = {}
        total_reward = 0.0

        for rc in self.task_config.reward_components:
            if rc.name in _REWARD_SENSORS:
                val = _REWARD_SENSORS[rc.name](self)
                components[rc.name] = float(val)
                total_reward += rc.weight * val
            else:
                components[rc.name] = 0.0

        if self._last_com_pos is not None and self.model.nsensor >= 1:
            self._last_com_pos = np.array([
                self.data.sensordata[self._sensor_offset_com],
                self.data.sensordata[self._sensor_offset_com + 1] if self.model.nsensor >= 2 else 0.0,
                self._com_z,
            ])

        return total_reward, components

    # ── Termination Check ── 跌落/超时/存活
    def _is_terminated(self) -> bool:
        if self._com_z < self.task_config.fall_height:
            return True
        if self._step_count >= self.task_config.max_episode_steps:
            return True
        return False

    # ── Environment Reset ── 重置 MJCF→观测→初始化状态
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        if seed is not None:
            np.random.seed(seed)

        for i in range(min(self.model.nq, self.model.njnt)):
            jnt_type = self.model.jnt_type[i] if i < self.model.njnt else 0
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            self.data.qpos[i] += np.random.uniform(-0.1, 0.1)

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._initial_com_pos = None
        self._last_com_pos = None
        self._total_energy = 0.0
        self._cumulative_reward = 0.0
        self._last_action = None
        self._step_dx = 0.0
        self._com_z = 0.5
        self._com_vel = np.zeros(3)

        return self._get_obs(), {}

    # ── Environment Step ── 动作→物理→奖励→终止→info
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        if self.data.ctrl.shape[0] > 0:
            ctrl_len = min(len(action), self.data.ctrl.shape[0])
            self.data.ctrl[:ctrl_len] = action[:ctrl_len]

        # 物理仿真 (带异常保护 — 非法 MJCF 或数值溢出时优雅降级)
        try:
            for _ in range(self.sim_config.substeps):
                mujoco.mj_step(self.model, self.data)
        except Exception:
            _logger.warning("MuJoCo step failed, returning zero reward + terminated")
            obs = self._get_obs() if hasattr(self, '_get_obs') else np.zeros(self._compute_obs_dim(), dtype=np.float32)
            return obs, -1.0, True, True, {"error": "mj_step_failed"}

        self._step_count += 1

        reward, reward_components = self._compute_reward()
        self._cumulative_reward += reward
        terminated = self._is_terminated()

        fell = self._com_z < self.task_config.fall_height
        survived = self._step_count >= self.task_config.max_episode_steps
        truncated = self._step_count >= self.task_config.max_episode_steps or fell

        if fell:
            reward += self.task_config.terminal_fall_penalty
        elif survived:
            reward += self.task_config.terminal_survival_bonus

        self._last_action = action.copy()

        displacement = 0.0
        if self._initial_com_pos is not None and self.model.nsensor >= 1:
            displacement = float(self.data.sensordata[self._sensor_offset_com] - self._initial_com_pos[0])

        info = {
            "cumulative_reward": self._cumulative_reward,
            "step": self._step_count,
            "displacement": displacement,
            "speed": float(displacement / max(self._step_count * self.sim_config.timestep * self.sim_config.substeps, 1e-6)),
            **reward_components,
            "total_energy": self._total_energy,
            "survived": survived,
            "fell": fell,
            "episode_length": self._step_count,
        }

        if self.model.nsensor >= 3:
            info["height"] = self._com_z
        if self._act_velocities:
            info["velocity"] = float(np.sqrt(sum(v*v for v in self._act_velocities[:3])) if len(self._act_velocities) >= 3 else self._act_velocities[0])
        if self._act_forces:
            info["joint_effort"] = float(sum(abs(f) for f in self._act_forces))

        if terminated:
            info["final_displacement"] = info["displacement"]
            info["episode_length"] = self._step_count
            if self._obs_buffer:
                data = np.array(self._obs_buffer)
                if data.shape[1] == self._own_obs_rms.mean.shape[0]:
                    self._own_obs_rms.update(data)
                if self.obs_rms is not None and data.shape[1] == self.obs_rms.mean.shape[0]:
                    self.obs_rms.update(data)
                self._obs_buffer = []

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            if not hasattr(self, '_renderer') or self._renderer is None:
                width, height = 800, 600
                self._renderer = mujoco.Renderer(self.model, width, height)
            self._renderer.update_scene(self.data, camera="track")
            pixels = self._renderer.render()
            return pixels

        elif self.render_mode == "rgb_array":
            if not hasattr(self, '_renderer') or self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 360)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

        return None

    def close(self):
        if hasattr(self, '_renderer') and self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _compute_novelty(self) -> float:
        asymmetry = self._compute_asymmetry()
        irregularity = self._compute_irregularity()
        return asymmetry * irregularity

    def _compute_asymmetry(self) -> float:
        if not hasattr(self.body, 'graph') or len(self.body.graph.nodes) == 0:
            return 0.0
        positions = []
        masses = []
        for node_id in self.body.graph.nodes:
            part = self.body.get_part(node_id)
            positions.append(part.position[0])
            mass = part.params.get("mass", 1.0)
            masses.append(mass)
        if not positions or sum(masses) == 0:
            return 0.0
        left_mass = sum(m * (p < 0) for m, p in zip(masses, positions))
        right_mass = sum(m * (p >= 0) for m, p in zip(masses, positions))
        total_mass = sum(masses)
        return abs(left_mass - right_mass) / max(total_mass, 1e-6)

    def _compute_irregularity(self) -> float:
        if not hasattr(self.body, 'graph') or len(self.body.graph.nodes) < 2:
            return 0.0
        positions = []
        for node_id in self.body.graph.nodes:
            part = self.body.get_part(node_id)
            positions.append(part.position)
        if not positions:
            return 0.0
        pos_array = np.array(positions)
        stds = np.std(pos_array, axis=0)
        mean_dist = np.mean(stds)
        return min(mean_dist / 0.5, 2.0)

    def _compute_bilateral_symmetry(self) -> float:
        return 1.0 - self._compute_asymmetry()

    def _compute_shape_commonality(self) -> float:
        part_types = {}
        for node_id in self.body.graph.nodes:
            part = self.body.get_part(node_id)
            ptype = part.part_type
            part_types[ptype] = part_types.get(ptype, 0) + 1
        n_parts = max(len(self.body.graph.nodes), 1)
        type_counts = list(part_types.values())
        if not type_counts:
            return 1.0
        max_count = max(type_counts)
        commonality = max_count / n_parts
        symmetry_bonus = self._compute_bilateral_symmetry() * 0.3
        return min(commonality + symmetry_bonus, 1.0)

    def _compute_energy_recycling(self) -> float:
        if not hasattr(self, '_act_forces') or not self._act_forces:
            return 0.0
        current_energy = sum(abs(f * v) for f, v in zip(self._act_forces, self._act_velocities))
        if not hasattr(self, '_prev_energy') or self._prev_energy is None:
            self._prev_energy = current_energy
            return 0.0
        recycling = max(0, self._prev_energy - current_energy) / max(self._prev_energy, 1e-6)
        self._prev_energy = current_energy
        return recycling

    def _compute_mode_switching(self) -> float:
        if not hasattr(self, '_action_history'):
            self._action_history = []
        if self.n_actuators > 0 and len(self.data.ctrl) > 0:
            current_action = np.array(self.data.ctrl[:self.n_actuators])
            self._action_history.append(current_action.copy())
            if len(self._action_history) > 20:
                self._action_history.pop(0)
        if len(self._action_history) < 10:
            return 0.0
        recent = np.array(self._action_history[-10:])
        older = np.array(self._action_history[:-10]) if len(self._action_history) > 10 else recent
        if len(older) == 0:
            return 0.0
        recent_var = np.var(recent, axis=0).mean()
        older_var = np.var(older, axis=0).mean()
        switch_score = abs(recent_var - older_var) / max(older_var + 1e-6, 1e-6)
        return min(switch_score, 2.0)

    def _compute_energy_waste(self) -> float:
        if not hasattr(self, '_act_forces') or not self._act_forces:
            return 0.0
        total_force = sum(abs(f) for f in self._act_forces)
        effective_force = sum(abs(f) for f in self._act_forces if f != 0)
        waste_ratio = (total_force - effective_force) / max(total_force, 1e-6)
        return waste_ratio

    def get_morph_embedding(self) -> np.ndarray:
        if self.morph_encoder is not None:
            import torch
            with torch.no_grad():
                embed = self.morph_encoder.encode_body(self.body)
                return embed.cpu().numpy()
        return np.zeros(64, dtype=np.float32)

    @property
    def obs_dim_without_morph(self) -> int:
        return self.observation_space.shape[0]

    @property
    def total_obs_dim(self) -> int:
        morph_dim = 64 if self.morph_encoder is None else self.morph_encoder.output_dim
        return self.observation_space.shape[0] + morph_dim
