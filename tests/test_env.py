"""
单元测试: RL 环境 (Gymnasium 接口)
"""

import numpy as np
import pytest
from forgecraft.core.morphology import MechanicalBody, Part, Joint
from forgecraft.core.catalog import get_default_catalog


class TestForgeCraftEnv:
    @pytest.fixture
    def env(self):
        """创建简单可行走形态的环境。"""
        from forgecraft.config import SimConfig, TaskConfig, RewardComponent
        from forgecraft.rl.env import ForgeCraftEnv

        cat = get_default_catalog()
        body = MechanicalBody("test_walker")
        root = Part("base", {"length": 0.1, "mass": 0.5}, np.array([0, 0, 0.5]))
        body.add_part(root)

        motor = Part("motor", {"actuated": 1.0, "max_torque": 5.0, "max_velocity": 10.0,
                                "length": 0.04, "mass": 0.12}, np.array([0.02, 0, 0.45]))
        body.add_part(motor)
        body.add_joint(Joint("hinge", root.part_id, motor.part_id,
                              anchor=np.array([0.02, 0, -0.05]),
                              axis=np.array([0, 1, 0]),
                              params={"range_min": -1.5, "range_max": 1.5, "damping": 0.5}))

        foot = Part("foot", {"length": 0.03, "mass": 0.05}, np.array([0.02, 0, 0.4]))
        body.add_part(foot)
        body.add_joint(Joint("hinge", motor.part_id, foot.part_id,
                              anchor=np.array([0, 0, -0.05]),
                              axis=np.array([0, 1, 0]),
                              params={"range_min": -1.0, "range_max": 1.0}))

        sc = SimConfig(max_steps=100, timestep=0.005, substeps=10)
        tc = TaskConfig(
            reward_components=[RewardComponent("displacement", 1.0),
                                RewardComponent("energy", -0.01)],
            max_episode_steps=100, fall_height=0.05,
        )
        return ForgeCraftEnv(body, sc, tc, catalog=cat)

    def test_observation_space(self, env):
        assert env.observation_space.shape[0] >= 10

    def test_action_space(self, env):
        assert env.action_space.shape[0] >= 1

    def test_reset(self, env):
        obs, info = env.reset()
        assert obs.shape == (env.observation_space.shape[0],)
        assert np.all(np.isfinite(obs))

    def test_step(self, env):
        env.reset()
        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (env.observation_space.shape[0],)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "displacement" in info or "speed" in info

    def test_multiple_steps_no_crash(self, env):
        env.reset()
        for _ in range(50):
            action = np.random.uniform(-1, 1, env.action_space.shape[0]).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    def test_close(self, env):
        env.close()  # 不应崩溃

    def test_no_nan_obs(self, env):
        env.reset()
        for _ in range(20):
            action = np.zeros(env.action_space.shape[0])
            obs, _, _, _, _ = env.step(action)
            assert not np.any(np.isnan(obs))


class TestEnvReward:
    def test_displacement_reward(self):
        from forgecraft.config import SimConfig, TaskConfig, RewardComponent
        from forgecraft.rl.env import ForgeCraftEnv

        cat = get_default_catalog()
        body = MechanicalBody("reward_test")
        root = Part("base", {"length": 0.1, "mass": 0.5}, np.array([0, 0, 0.5]))
        body.add_part(root)

        sc = SimConfig(max_steps=50)
        tc = TaskConfig(
            reward_components=[RewardComponent("displacement", 10.0),
                                RewardComponent("alive", 0.01)],
            max_episode_steps=50,
        )
        e = ForgeCraftEnv(body, sc, tc, catalog=cat)
        e.reset()
        # step几次
        for _ in range(10):
            e.step(np.zeros(e.action_space.shape[0]))
        e.close()

    def test_empty_body_env(self):
        from forgecraft.config import SimConfig, TaskConfig
        from forgecraft.rl.env import ForgeCraftEnv

        cat = get_default_catalog()
        body = MechanicalBody("empty_test")
        root = Part("base", {"length": 0.1}, np.array([0, 0, 0.5]))
        body.add_part(root)

        sc = SimConfig(max_steps=20)
        tc = TaskConfig(max_episode_steps=20)
        e = ForgeCraftEnv(body, sc, tc, catalog=cat)
        e.reset()
        obs, _, _, _, _ = e.step(np.zeros(e.action_space.shape[0]))
        e.close()
