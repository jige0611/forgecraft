"""集成测试共享 fixtures

为所有集成测试提供 session-scoped 对象:
  - 零件箱 (catalog)
  - 任务配置 (task_config)
  - 快速进化循环 (quick_loop)

设计原则:
  - session 级别复用 (避免重复初始化)
  - 快速参数 (3 代, 5 个体, ≤ 15s)
  - 自动清理临时文件
"""

import os
import shutil
import tempfile
from typing import Dict

import pytest
import torch
import numpy as np

from forgecraft.config import (
    EvolutionConfig,
    RLConfig,
    SimConfig,
    TaskConfig,
    PartSpec,
)
from forgecraft.core.catalog import DEFAULT_CATALOG
from forgecraft.core.morphology import MechanicalBody
__all__ = [
    "default_catalog",
    "default_sim_config",
    "default_rl_config",
    "default_task_config",
    "default_evo_config",
    "sample_body",
]




# ── 测试目录 ──

TEST_DIR = os.path.join(tempfile.gettempdir(), "forgecraft_e2e_tests")


@pytest.fixture(scope="session")
def catalog() -> Dict[str, PartSpec]:
    """共享零件箱 (session 级别, 仅初始化一次)"""
    return dict(DEFAULT_CATALOG)


@pytest.fixture(scope="session")
def task_config() -> TaskConfig:
    """快速任务配置 (500 步 max)"""
    from forgecraft.config import RewardComponent
    
    return TaskConfig(
        max_episode_steps=500,
        domain="robot",
        reward_components=[
            RewardComponent(name="speed", weight=1.0),
        ],
        fall_height=0.3,
    )


@pytest.fixture(scope="session")
def sim_config() -> SimConfig:
    """仿真配置"""
    return SimConfig(max_steps=500, timestep=0.005)


@pytest.fixture(scope="session")
def rl_config() -> RLConfig:
    """RL 配置 (快速训练) — 与默认 RLConfig 保持一致以支持断点恢复"""
    return RLConfig(
        hidden_dim=128,
        gnn_hidden=64,
        morph_embed_dim=64,
        ppo_epochs=4,
        train_iters=20,
        batch_size=64,
    )


@pytest.fixture(scope="session")
def evo_config() -> EvolutionConfig:
    """进化配置 (小种群, 快速)"""
    return EvolutionConfig(
        population_size=5,
        elite_count=2,
        generations=3,
    )


@pytest.fixture
def quick_loop(catalog, task_config, sim_config, rl_config, evo_config):
    """快速进化循环 (每个测试独立)"""
    from forgecraft.evolution.loop import EvolutionLoop

    loop = EvolutionLoop(
        evo_config=evo_config,
        sim_config=sim_config,
        rl_config=rl_config,
        task_config=task_config,
        catalog=catalog,
        device="cpu",
        n_workers=1,
        seed=42,
    )

    yield loop

    # 注意: 测试产物一律写入临时目录 (见各测试中的 tempfile.gettempdir())。
    # 这里绝不能清理仓库内目录 —— 曾用相对路径 rmtree("design_output"/"checkpoints")，
    # 会误删用户真实的产物目录。


def setup_module():
    """模块级 setup: 确保测试目录存在"""
    os.makedirs(TEST_DIR, exist_ok=True)


def teardown_module():
    """模块级 teardown: 清理"""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)
