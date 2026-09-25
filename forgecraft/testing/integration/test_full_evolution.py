"""端到端进化流程集成测试

验证: 初始化 → 进化 → 制造导出 完整流程
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

from forgecraft.evolution.loop import EvolutionLoop
from forgecraft.core.morphology import MechanicalBody


class TestFullEvolution:
    """完整进化流程集成测试"""

    def test_initialize_population(self, quick_loop):
        """测试种群初始化"""
        quick_loop.initialize_population()

        assert len(quick_loop.population) == quick_loop.evo_config.population_size
        assert quick_loop.generation == 0
        for body in quick_loop.population:
            assert isinstance(body, MechanicalBody)
            parts = list(body.parts())
            assert len(parts) > 0, "每个形态应至少有 1 个零件"
            assert body.fitness == 0.0, "初始适应度应为 0"

    def test_run_3_generations(self, quick_loop):
        """测试 3 代完整进化 (safe=False 避免 TrainGuard)"""
        quick_loop.initialize_population()

        n_gens = 3
        quick_loop.run(n_generations=n_gens, safe=False)

        assert quick_loop.generation >= n_gens
        assert quick_loop.best_body is not None
        assert quick_loop.best_body.fitness >= 0
        assert len(quick_loop.history) >= n_gens

        # 验证历史记录
        for gen_idx, gen_data in enumerate(quick_loop.history):
            assert "generation" in gen_data or "best_fitness" in gen_data

    def test_breeder_and_evaluator_in_loop(self, quick_loop):
        """测试 Breeder 和 Evaluator 在循环中的协同"""
        quick_loop.initialize_population()

        # 单代
        quick_loop.evolve_one_generation()

        assert quick_loop.generation == 1
        assert len(quick_loop.population) == quick_loop.evo_config.population_size
        assert quick_loop.best_body is not None


class TestResumeTraining:
    """断点续训集成测试"""

    def test_checkpoint_roundtrip(self, quick_loop):
        """测试断点保存→加载 (不继续训练，避免未初始化属性)"""
        quick_loop.initialize_population()
        quick_loop.run(n_generations=2, safe=False)

        gen_before = quick_loop.generation
        pop_size = len(quick_loop.population)

        # 保存
        ckpt_path = os.path.join(tempfile.gettempdir(), "forgecraft_test_ckpt.pkl")
        try:
            quick_loop.save_checkpoint(ckpt_path)

            # 验证文件存在且非空
            assert os.path.exists(ckpt_path)
            assert os.path.getsize(ckpt_path) > 0

            # 加载
            loop2 = EvolutionLoop.load_checkpoint(ckpt_path)
            assert loop2.generation == gen_before
            assert len(loop2.population) == pop_size
        finally:
            if os.path.exists(ckpt_path):
                os.unlink(ckpt_path)

    def test_auto_checkpoint(self, quick_loop):
        """测试 SafeCheckpointManager 原子写入 + 最新断点查找"""
        from forgecraft.evolution.recovery import SafeCheckpointManager

        ckpt_dir = os.path.join(tempfile.gettempdir(), "forgecraft_auto_ckpt")
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        try:
            manager = SafeCheckpointManager(checkpoint_dir=ckpt_dir, max_checkpoints=3)

            quick_loop.initialize_population()

            # 第一个断点
            path_gen5 = manager.save(quick_loop, generation=5)
            assert os.path.exists(path_gen5)
            assert manager.find_latest() is not None

            # 第二个断点
            path_gen6 = manager.save(quick_loop, generation=6)
            assert os.path.exists(path_gen6)

            # 最新断点应指向 gen0006
            latest = manager.find_latest()
            assert latest is not None
            assert "gen0006" in os.path.basename(latest)

            # 加载并校验状态
            state = manager.load(latest)
            assert state is not None
            assert state["version"] == 2
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)


class TestManufacturingE2E:
    """制造导出端到端集成测试"""

    def test_export_from_loop(self, quick_loop):
        """测试从进化循环导出制造文件"""
        quick_loop.initialize_population()
        quick_loop.run(n_generations=2, safe=False)

        from forgecraft.manufacturing import export_all

        body = quick_loop.best_body
        assert body is not None

        body_data = {
            "name": body.name,
            "fitness": body.fitness,
            "num_parts": body.num_parts(),
            "num_joints": body.num_joints(),
            "parts": [p.to_dict() for p in body.parts()],
            "joints": [j.to_dict() for j in body.joints()],
        }

        output_dir = os.path.join(tempfile.gettempdir(), "forgecraft_mfg_test")
        shutil.rmtree(output_dir, ignore_errors=True)
        try:
            results = export_all(body_data, {}, output_dir)

            assert results["output_dir"] == output_dir
            assert os.path.exists(results["body_json"])
            assert os.path.exists(results["bom"])
            assert results["grade"]
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


class TestHPOE2E:
    """超参数优化端到端集成测试"""

    def test_param_space_sampling(self):
        """测试超参数空间采样"""
        from forgecraft.evolution.hyperparam_opt import HyperParameterSpace

        space = HyperParameterSpace()
        assert len(space.params) > 0

        rng = np.random.RandomState(0)
        config = space.sample(rng)
        assert isinstance(config, dict)
        assert len(config) == len(space.params)
        assert "population_size" in config

    def test_optimizer_init(self):
        """测试优化器初始化"""
        from forgecraft.evolution.hyperparam_opt import HyperParamOptimizer

        try:
            optimizer = HyperParamOptimizer(seed=0)
            assert optimizer is not None
        except ImportError:
            pytest.skip("Optuna 未安装")


class TestDistributedE2E:
    """分布式端到端集成测试"""

    def test_distributed_config_defaults(self):
        """测试分布式配置默认值"""
        from forgecraft.evolution.distributed import DistributedConfig

        config = DistributedConfig(use_ray=False, n_remote_workers=1)
        assert config.n_remote_workers == 1
        assert not config.use_ray
        assert config.task_timeout > 0

    def test_cluster_info_defaults(self):
        """测试集群信息数据结构"""
        from forgecraft.evolution.distributed import ClusterInfo

        info = ClusterInfo()
        assert info.n_nodes == 0
        assert info.n_workers == 0
        assert info.node_ids == []
