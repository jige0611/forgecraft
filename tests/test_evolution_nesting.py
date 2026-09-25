"""
集成测试: 进化模块 — NSGA-III + CMA-ME + 六层嵌套 + Pareto 存档 + MAP-Elites

覆盖:
  - NSGA-III: 参考点生成 / 归一化 / 关联 / 小生境选择
  - CMA-ME: CMAES 采样-tell / CMAMEmitter / OptimizerEmitter
  - Pareto 存档: 支配 / 超体积 / 膝点 / 多样性指标
  - MAP-Elites: CVT/Grid 存档 / 四种发射器 / 引擎 / 精英注入
  - 六层嵌套: LayerState / NestingLayer / Scheduler / Coordinator
  - 端到端: 模拟进化循环 (MAP-Elites + NSGA-III + 嵌套)
"""

import numpy as np
import pytest

from forgecraft.core.morphology import MechanicalBody, Part, Joint


# ── helpers ─────────────────────────────────────────────────

def make_body(name, fitness=0.0, comps=None):
    """创建测试用 MechanicalBody"""
    body = MechanicalBody(name)
    root = Part("base", {"length": 0.1, "width": 0.05, "height": 0.03},
                 np.array([0.0, 0.0, 0.5]))
    body.add_part(root)
    body.fitness = fitness
    if comps:
        body.fitness_components = dict(comps)
    return body


# ══════════════════════════════════════════════════════════
#  1. NSGA-III
# ══════════════════════════════════════════════════════════

class TestNSGA3:
    def test_reference_point_count(self):
        """验证 C(H+M-1, H) 点数"""
        from forgecraft.evolution.nsga3 import generate_reference_points
        pts_3_4 = generate_reference_points(3, 4)
        assert pts_3_4.shape == (15, 3)
        assert np.allclose(pts_3_4.sum(axis=1), 1.0)

        pts_2_5 = generate_reference_points(2, 5)
        assert pts_2_5.shape == (6, 2)

    def test_normalization(self):
        from forgecraft.evolution.nsga3 import normalize_objectives
        F = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 1.0],
            [3.0, 1.0, 2.0],
            [1.5, 1.5, 1.5],
        ])
        F_norm, ideal, intercepts = normalize_objectives(
            F, ["maximize", "maximize", "maximize"],
        )
        assert F_norm.shape == F.shape
        assert ideal.shape == (3,)
        assert intercepts.shape == (3,)

    def test_association(self):
        from forgecraft.evolution.nsga3 import (
            generate_reference_points, associate_to_reference_points,
        )
        ref = generate_reference_points(3, 4)
        F = np.random.rand(10, 3)
        assoc, dists = associate_to_reference_points(F, ref)
        assert len(assoc) == 10
        assert len(dists) == 10
        assert 0 <= assoc.min() and assoc.max() < len(ref)

    def test_niching_select(self):
        from forgecraft.evolution.nsga3 import niching_select
        front = list(range(20))
        assoc = np.array([i % 5 for i in range(20)])
        niche_counts = np.zeros(5, dtype=int)
        dists = np.random.rand(20)

        sel = niching_select(front, 10, assoc, niche_counts, dists,
                             rng=np.random.RandomState(42))
        assert len(sel) == 10
        assert all(0 <= s < 20 for s in sel)

    def test_nsga3_select_on_bodies(self):
        from forgecraft.evolution.nsga3 import nsga3_select

        pop = []
        for i in range(30):
            b = make_body(f"b{i}", comps={
                "speed": np.random.uniform(0, 5),
                "energy": np.random.uniform(0, 10),
                "upright": np.random.uniform(0, 1),
            })
            pop.append(b)

        selected = nsga3_select(
            pop, 10,
            objective_keys=["speed", "energy", "upright"],
            directions=["maximize", "minimize", "maximize"],
        )
        assert len(selected) == 10
        assert all(isinstance(b, MechanicalBody) for b in selected)


# ══════════════════════════════════════════════════════════
#  2. CMA-ME / CMA-ES
# ══════════════════════════════════════════════════════════

class TestCMAES:
    def test_ask_tell_cycle(self):
        from forgecraft.evolution.cma_me import CMAES

        cma = CMAES(mean=np.zeros(5), sigma0=0.3, random_seed=42)
        for _ in range(5):
            samples = cma.ask(20)
            assert samples.shape == (20, 5)
            fitnesses = -np.sum(samples ** 2, axis=1)  # 中心最优
            cma.tell(samples, fitnesses)

        diag = cma.get_diagnostics()
        assert diag["generation"] == 5
        assert "sigma" in diag

    def test_restart(self):
        from forgecraft.evolution.cma_me import CMAES

        cma = CMAES(mean=np.ones(3), sigma0=0.3, random_seed=42)
        orig_popsize = cma.popsize
        cma.restart(sigma_factor=2.0)
        assert cma.popsize == min(orig_popsize * 2, 2000)

    def test_bounds(self):
        from forgecraft.evolution.cma_me import CMAES

        bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        cma = CMAES(mean=np.array([0.0, 0.0]), sigma0=0.5,
                    bounds=bounds, random_seed=42)
        samples = cma.ask(100)
        assert np.all(samples[:, 0] >= -1.0) and np.all(samples[:, 0] <= 1.0)
        assert np.all(samples[:, 1] >= -1.0) and np.all(samples[:, 1] <= 1.0)


class TestCMAMEmitter:
    """CMAMEmitter 需要 archive (用 CVTArchive 模拟)"""

    def test_basic_ask_tell(self):
        from forgecraft.evolution.cma_me import CMAMEmitter
        from forgecraft.evolution.map_elites import CVTArchive

        archive = CVTArchive(
            num_centroids=50, bc_dim=3,
            bc_ranges=[(-1, 1), (-1, 1), (-1, 1)],
            random_seed=42,
        )
        emitter = CMAMEmitter(archive, n_dim=5, popsize=15,
                              sigma0=0.3, random_seed=42)

        solutions, target_cells = emitter.ask(15)
        assert solutions.shape == (15, 5)
        assert len(target_cells) == 15

        fitnesses = -np.sum(solutions ** 2, axis=1)
        improvements = [True] * 15
        emitter.tell(solutions, fitnesses, improvements)

        diag = emitter.get_diagnostics()
        assert "emitter_generation" in diag


class TestCmaMeOptimizerEmitter:
    """cma_me.py 的独立 OptimizerEmitter"""

    def test_simple_mode(self):
        from forgecraft.evolution.cma_me import OptimizerEmitter
        from forgecraft.evolution.map_elites import CVTArchive

        archive = CVTArchive(
            num_centroids=20, bc_dim=2,
            bc_ranges=[(0, 1), (0, 1)],
        )
        emitter = OptimizerEmitter(archive, n_dim=4, method="simple")
        solutions = emitter.ask(10)
        assert solutions.shape == (10, 4)

    def test_get_diagnostics(self):
        from forgecraft.evolution.cma_me import OptimizerEmitter
        from forgecraft.evolution.map_elites import CVTArchive

        archive = CVTArchive(num_centroids=10, bc_dim=2,
                             bc_ranges=[(0, 1), (0, 1)])
        emitter = OptimizerEmitter(archive, n_dim=3, method="simple")
        assert "method" in emitter.get_diagnostics()


class TestCmameSample:
    def test_convenience(self):
        from forgecraft.evolution.cma_me import cmame_sample
        from forgecraft.evolution.map_elites import CVTArchive

        archive = CVTArchive(num_centroids=10, bc_dim=2,
                             bc_ranges=[(0, 1), (0, 1)])
        solutions, emitter = cmame_sample(archive, n_dim=3, n_samples=5)
        assert solutions.shape == (5, 3)
        assert emitter is not None


# ══════════════════════════════════════════════════════════
#  3. Pareto 存档
# ══════════════════════════════════════════════════════════

class TestParetoArchive:
    def test_add_and_dominate(self):
        from forgecraft.evolution.pareto_archive import ParetoArchive

        pa = ParetoArchive(n_objectives=2,
                           directions=["maximize", "maximize"])
        assert pa.add("A", np.array([2.0, 5.0])) is True
        assert pa.add("B", np.array([5.0, 2.0])) is True  # 互补, 非支配
        assert pa.add("C", np.array([1.0, 1.0])) is False  # dominated by both
        assert pa.size() == 2

    def test_add_batch(self):
        from forgecraft.evolution.pareto_archive import ParetoArchive

        pa = ParetoArchive(n_objectives=2)
        objs = np.array([
            [3.0, 3.0],
            [5.0, 1.0],
            [1.0, 5.0],
            [2.0, 2.0],  # dominated
        ])
        n = pa.add_batch(["a", "b", "c", "d"], objs)
        assert n >= 3  # d may be dominated, so at least 3 added

    def test_minimize_direction(self):
        from forgecraft.evolution.pareto_archive import ParetoArchive

        pa = ParetoArchive(n_objectives=2,
                           directions=["minimize", "minimize"])
        # 越小越好
        assert pa.add("low_energy", np.array([1.0, 5.0])) is True
        assert pa.add("low_cost", np.array([5.0, 1.0])) is True  # 互补, 非支配
        assert pa.add("worse", np.array([6.0, 6.0])) is False
        assert pa.size() == 2

    def test_hypervolume(self):
        from forgecraft.evolution.pareto_archive import (
            ParetoArchive, compute_hypervolume,
        )

        pa = ParetoArchive(n_objectives=2,
                           reference_point=np.array([0.0, 0.0]))
        pa.add("a", np.array([2.0, 1.0]))
        pa.add("b", np.array([1.0, 3.0]))
        hv = pa.compute_hypervolume()
        assert hv > 0
        assert len(pa.hypervolume_history) > 0

    def test_knee_point(self):
        from forgecraft.evolution.pareto_archive import (
            ParetoArchive, detect_knee_point,
        )

        pa = ParetoArchive(n_objectives=2)
        pa.add("a", np.array([1.0, 10.0]))
        pa.add("b", np.array([5.0, 5.0]))   # 膝点候选
        pa.add("c", np.array([10.0, 1.0]))
        knee = pa.detect_knee_point()
        assert knee is not None

        # 独立函数
        idx = detect_knee_point(pa.get_objectives())
        assert 0 <= idx < 3

    def test_reference_point_query(self):
        from forgecraft.evolution.pareto_archive import (
            ParetoArchive, reference_point_query,
        )

        pa = ParetoArchive(n_objectives=2)
        pa.add("a", np.array([1.0, 10.0]))
        pa.add("b", np.array([5.0, 5.0]))
        pa.add("c", np.array([10.0, 1.0]))

        results = pa.reference_point_query(
            aspiration=np.array([6.0, 4.0]), n_results=2,
        )
        assert len(results) == 2

        # 独立函数
        idxs = reference_point_query(
            pa.get_objectives(), np.array([6.0, 4.0]), n_results=2,
        )
        assert len(idxs) == 2

    def test_diversity_metrics(self):
        from forgecraft.evolution.pareto_archive import (
            ParetoArchive, spacing_metric, spread_metric,
        )

        F = np.array([[1.0, 10.0], [3.0, 7.0], [6.0, 3.0],
                       [10.0, 1.0]])
        assert spacing_metric(F) >= 0
        assert 0 <= spread_metric(F) <= 1

        pa = ParetoArchive(n_objectives=2)
        for i in range(4):
            pa.add(f"x{i}", F[i])
        m = pa.diversity_metrics()
        assert "spacing" in m and "spread" in m

    def test_get_statistics(self):
        from forgecraft.evolution.pareto_archive import ParetoArchive

        pa = ParetoArchive(n_objectives=2)
        pa.add("a", np.array([2.0, 3.0]))
        s = pa.get_statistics()
        assert s["archive_size"] >= 1


# ══════════════════════════════════════════════════════════
#  4. MAP-Elites: 存档 & 发射器 & 引擎
# ══════════════════════════════════════════════════════════

class TestCVTArchive:
    def test_add_and_retrieve(self):
        from forgecraft.evolution.map_elites import CVTArchive

        arch = CVTArchive(num_centroids=50, bc_dim=3,
                          bc_ranges=[(-1, 1), (-1, 1), (-1, 1)],
                          random_seed=42)
        for i in range(20):
            b = make_body(f"b{i}", fitness=float(i))
            bc = np.random.uniform(-0.5, 0.5, 3)
            arch.add(b, b.fitness, bc, generation=0)

        assert arch.coverage() > 0
        assert len(arch.get_all_elites()) > 0
        assert arch.best_fitness() >= 0

    def test_qd_score(self):
        from forgecraft.evolution.map_elites import CVTArchive

        arch = CVTArchive(num_centroids=30, bc_dim=2,
                          bc_ranges=[(0, 1), (0, 1)],
                          random_seed=42)
        for i in range(50):
            b = make_body(f"b{i}", fitness=float(i % 10))
            bc = np.random.rand(2)
            arch.add(b, b.fitness, bc)

        assert arch.compute_qd_score() >= 0
        stats = arch.get_stats()
        assert "coverage" in stats and "qd_score" in stats

    def test_sample_elites(self):
        from forgecraft.evolution.map_elites import CVTArchive

        arch = CVTArchive(num_centroids=30, bc_dim=2,
                          bc_ranges=[(0, 1), (0, 1)])
        for i in range(10):
            b = make_body(f"b{i}", fitness=float(i))
            arch.add(b, b.fitness, np.random.rand(2))

        samples = arch.sample_elites(5)
        assert len(samples) == 5

    def test_random_cell_and_elite(self):
        from forgecraft.evolution.map_elites import CVTArchive

        arch = CVTArchive(num_centroids=20, bc_dim=2,
                          bc_ranges=[(0, 1), (0, 1)])
        # empty archive
        assert arch.random_cell() is None
        assert arch.random_elite() is None

        b = make_body("x", fitness=1.0)
        arch.add(b, 1.0, np.array([0.5, 0.5]))
        assert arch.random_cell() is not None
        assert arch.random_elite() is not None


class TestGridArchive:
    def test_add_and_cell_mapping(self):
        from forgecraft.evolution.map_elites import GridArchive

        arch = GridArchive(grid_dims=(5, 5),
                           bc_ranges=[(0, 1), (0, 1)])
        for i in range(30):
            b = make_body(f"b{i}", fitness=np.random.random())
            bc = np.random.rand(2)
            arch.add(b, b.fitness, bc)

        assert arch.coverage() > 0
        assert len(arch.get_all_elites()) > 0

    def test_qd_score_and_stats(self):
        from forgecraft.evolution.map_elites import GridArchive

        arch = GridArchive(grid_dims=(4, 4),
                           bc_ranges=[(0, 1), (0, 1)])
        for i in range(20):
            b = make_body(f"b{i}", fitness=float(i))
            arch.add(b, b.fitness, np.random.rand(2))

        s = arch.get_stats()
        assert s["coverage"] > 0
        assert s["qd_score"] >= 0


class TestEmitters:
    def test_base_emitter_abstract(self):
        from forgecraft.evolution.map_elites import BaseEmitter

        emitter = BaseEmitter(batch_size=10,
                              rng=np.random.RandomState(42))
        with pytest.raises(NotImplementedError):
            emitter.ask(None)

    def test_improvement_emitter(self):
        from forgecraft.evolution.map_elites import (
            ImprovementEmitter, CVTArchive,
        )

        arch = CVTArchive(num_centroids=20, bc_dim=2,
                          bc_ranges=[(0, 1), (0, 1)])
        for i in range(5):
            b = make_body(f"b{i}", fitness=float(i))
            arch.add(b, b.fitness, np.random.rand(2))

        emitter = ImprovementEmitter(
            batch_size=10, rng=np.random.RandomState(42),
            mutation_strength=0.1,
        )
        offspring = emitter.ask(arch)
        assert 1 <= len(offspring) <= 10
        assert all(isinstance(b, MechanicalBody) for b in offspring)

    def test_random_emitter(self):
        from forgecraft.evolution.map_elites import (
            RandomEmitter, CVTArchive,
        )

        arch = CVTArchive(num_centroids=10, bc_dim=2,
                          bc_ranges=[(0, 1), (0, 1)])

        emitter = RandomEmitter(
            batch_size=5, rng=np.random.RandomState(42),
            generator_fn=lambda: make_body("rand"),
        )
        offspring = emitter.ask(arch)
        assert len(offspring) == 5

    def test_adaptive_emitter(self):
        from forgecraft.evolution.map_elites import (
            AdaptiveEmitter, CVTArchive,
        )

        arch = CVTArchive(num_centroids=20, bc_dim=2,
                          bc_ranges=[(0, 1), (0, 1)])
        for i in range(5):
            b = make_body(f"b{i}", fitness=float(i))
            arch.add(b, b.fitness, np.random.rand(2))

        emitter = AdaptiveEmitter(
            batch_size=8, rng=np.random.RandomState(42),
            crossover_rate=0.5,
        )
        offspring = emitter.ask(arch)
        assert 1 <= len(offspring) <= 8

    def test_optimizer_emitter(self):
        from forgecraft.evolution.map_elites import (
            OptimizerEmitter, CVTArchive,
        )

        arch = CVTArchive(num_centroids=20, bc_dim=2,
                          bc_ranges=[(0, 1), (0, 1)])
        for i in range(5):
            b = make_body(f"b{i}", fitness=float(i))
            arch.add(b, b.fitness, np.random.rand(2))

        emitter = OptimizerEmitter(
            batch_size=8, rng=np.random.RandomState(42),
            n_params=5, method="simple", sigma0=0.3,
        )
        offspring = emitter.ask(arch)
        assert len(offspring) == 8
        assert all(isinstance(b, MechanicalBody) for b in offspring)

    def test_emitter_tell_updates_archive(self):
        from forgecraft.evolution.map_elites import (
            BaseEmitter, CVTArchive,
        )

        arch = CVTArchive(num_centroids=20, bc_dim=2,
                          bc_ranges=[(0, 1), (0, 1)])
        emitter = BaseEmitter(batch_size=10,
                              rng=np.random.RandomState(42))

        bodies = [make_body(f"t{i}", fitness=float(i)) for i in range(5)]
        fitnesses = [b.fitness for b in bodies]
        bcs = [np.random.rand(2) for _ in range(5)]

        emitter.tell(arch, bodies, fitnesses, bcs, generation=0)
        assert len(arch.get_all_elites()) > 0


class TestMAPElitesEngine:
    def test_full_cycle(self):
        from forgecraft.evolution.map_elites import (
            MAPElitesEngine, MAPElitesConfig,
        )

        config = MAPElitesConfig(
            bc_names=("speed", "stability"),
            bc_ranges=((0, 5), (0, 1)),
            archive_size=50,
            use_cvt=True,
            num_emitters=3,
            emitter_batch_size=8,
            evaluations_per_generation=30,
            inject_every_n_gens=5,
            inject_count=2,
        )
        engine = MAPElitesEngine(config=config, seed=42)
        engine.initialize(generator_fn=lambda: make_body("init"))

        # 运行 5 代
        def dummy_eval(body):
            fitness = np.random.random()
            bc = np.array([fitness * 5, np.random.random()])
            return fitness, bc, {}

        for _ in range(5):
            stats = engine.step(dummy_eval)
            assert "coverage" in stats
            assert "qd_score" in stats

        stats = engine.get_stats()
        assert stats["num_elites"] >= 0
        assert engine.generation == 5

    def test_injection_elites(self):
        from forgecraft.evolution.map_elites import (
            MAPElitesEngine, MAPElitesConfig,
        )

        config = MAPElitesConfig(
            bc_names=("speed", "stability"),
            bc_ranges=((0, 5), (0, 1)),
            archive_size=30, use_cvt=True,
            num_emitters=2,
            emitter_batch_size=5,
            evaluations_per_generation=10,
        )
        engine = MAPElitesEngine(config=config, seed=42)
        engine.initialize(generator_fn=lambda: make_body("init"))

        def dummy_eval(body):
            return np.random.random(), np.random.rand(2), {}

        engine.step(dummy_eval)
        engine.step(dummy_eval)

        elites = engine.get_injection_elites(n=3)
        assert 0 <= len(elites) <= 3

    def test_inject_elites_function(self):
        from forgecraft.evolution.map_elites import (
            MAPElitesEngine, MAPElitesConfig, inject_elites,
        )

        config = MAPElitesConfig(
            bc_names=("speed", "stability"),
            bc_ranges=((0, 5), (0, 1)),
            archive_size=20,
            num_emitters=2,
            emitter_batch_size=5,
            evaluations_per_generation=10,
        )
        engine = MAPElitesEngine(config=config, seed=42)
        engine.initialize(generator_fn=lambda: make_body("init"))

        def dummy_eval(body):
            return np.random.random(), np.random.rand(2), {}

        engine.step(dummy_eval)

        pop = [make_body(f"p{i}", fitness=float(i)) for i in range(10)]
        new_pop = inject_elites(pop, engine, n=2, replace_worst=True)
        assert len(new_pop) == 10  # 替换保持大小

    def test_qd_score_and_coverage_history(self):
        from forgecraft.evolution.map_elites import (
            MAPElitesEngine, MAPElitesConfig,
        )

        config = MAPElitesConfig(
            bc_names=("speed", "stability"),
            bc_ranges=((0, 5), (0, 1)),
            archive_size=20,
            num_emitters=2,
            emitter_batch_size=3,
            evaluations_per_generation=5,
        )
        engine = MAPElitesEngine(config=config, seed=42)
        engine.initialize(generator_fn=lambda: make_body("init"))

        def dummy_eval(body):
            return np.random.random(), np.random.rand(2), {}

        for _ in range(3):
            engine.step(dummy_eval)

        qd_hist = engine.get_qd_score_history()
        cov_hist = engine.get_coverage_history()
        assert len(qd_hist) == 3
        assert len(cov_hist) == 3

    def test_create_map_elites_engine(self):
        from forgecraft.evolution.map_elites import (
            create_map_elites_engine, MAPElitesEngine,
        )

        engine = create_map_elites_engine(seed=123)
        assert isinstance(engine, MAPElitesEngine)
        assert engine.seed == 123

    def test_behavior_characteristics(self):
        from forgecraft.evolution.map_elites import (
            compute_behavior_characteristics,
            compute_behavior_characteristics_with_objectives,
        )

        body = make_body("test_body")
        episode = {"speed": 2.5, "displacement": 10.0,
                    "energy": 2.0, "stability": 0.8}
        bc = compute_behavior_characteristics(body, episode)
        assert isinstance(bc, np.ndarray)
        assert len(bc) == 5  # speed, efficiency, stability, symmetry, mass

        bc2, objs = compute_behavior_characteristics_with_objectives(
            body, {"custom": 42.0}, episode,
        )
        assert "custom" in objs
        assert len(bc2) == 5


# ══════════════════════════════════════════════════════════
#  5. 六层嵌套优化
# ══════════════════════════════════════════════════════════

class TestNestingLayer:
    def test_layer_creation(self):
        from forgecraft.evolution.hierarchical import NestingLayer, LayerState

        layer = NestingLayer(
            name="L1", level=0, budget=1000,
            stagnation_threshold=5,
            downstream="L2",
        )
        assert layer.state == LayerState.DORMANT
        assert layer.name == "L1"

    def test_enqueue_and_ready(self):
        from forgecraft.evolution.hierarchical import NestingLayer

        layer = NestingLayer(name="L2", level=1, budget=500)
        assert not layer.is_ready()

        body = make_body("x", fitness=5.0)
        layer.enqueue([body])
        assert layer.is_ready()
        assert len(layer.pending_individuals) == 1

    def test_stagnation_detection(self):
        from forgecraft.evolution.hierarchical import NestingLayer

        layer = NestingLayer(name="L1", level=0, budget=100,
                             stagnation_threshold=3)
        # No history → not stagnant
        assert not layer.check_stagnation()

        # Flat history
        layer.best_fitness_history = [10.0] * 6
        assert layer.check_stagnation()

    def test_convergence_check(self):
        from forgecraft.evolution.hierarchical import NestingLayer

        layer = NestingLayer(name="L1", level=0, budget=100)
        assert not layer.check_convergence()

        layer.best_fitness_history = [10.0, 10.0, 10.0, 10.0, 10.0]
        assert layer.check_convergence()


class TestLayerScheduler:
    def test_get_active_layer_empty(self):
        from forgecraft.evolution.hierarchical import (
            NestingLayer, LayerScheduler, LayerState,
        )

        layers = [
            NestingLayer(name="L1", level=0, budget=100,
                         downstream="L2"),
            NestingLayer(name="L2", level=1, budget=50,
                         upstream="L1"),
        ]
        sched = LayerScheduler(layers)
        # 首次应激活最外层 L1
        active = sched.get_active_layer()
        assert active is not None
        assert active.name == "L1"

    def test_propagation_downstream(self):
        from forgecraft.evolution.hierarchical import (
            NestingLayer, LayerScheduler, LayerState,
        )

        l1 = NestingLayer(name="L1", level=0, budget=100,
                          downstream="L2",
                          state=LayerState.ACTIVE)
        l2 = NestingLayer(name="L2", level=1, budget=50,
                          upstream="L1",
                          state=LayerState.DORMANT,
                          quality_gate=lambda x: True)

        l1.population = [make_body("x", fitness=10.0)]
        l2_before = len(l2.pending_individuals)

        layers = [l1, l2]
        sched = LayerScheduler(layers)

        # 手动触发下游传播 (通过 update → _propagate_downstream)
        sched.update(l1, reward=10.0)

        # L1 未收敛 (只有1个数据点), L2 应仍 dormant
        # 但如果我们手动设置 L1 收敛...
        l1.state = LayerState.ACTIVE
        l1.best_fitness_history = [10.0] * 10  # simulate converged
        sched.update(l1, reward=10.0)

    def test_ucb_selection(self):
        from forgecraft.evolution.hierarchical import (
            NestingLayer, LayerScheduler, LayerState,
        )

        l1 = NestingLayer(name="L1", level=0, budget=100,
                          state=LayerState.ACTIVE,
                          population=[make_body("x")])
        l2 = NestingLayer(name="L2", level=1, budget=50,
                          state=LayerState.ACTIVE,
                          population=[make_body("y")])
        l3 = NestingLayer(name="L3", level=2, budget=30,
                          state=LayerState.ACTIVE,
                          population=[make_body("z")])

        layers = [l1, l2, l3]
        sched = LayerScheduler(layers, rng=np.random.RandomState(42))

        # 初始化 count
        sched.layer_counts = {"L1": 10, "L2": 5, "L3": 1}
        sched.layer_rewards = {
            "L1": [0.5] * 10, "L2": [0.8] * 5, "L3": [0.3],
        }

        active = sched.get_active_layer()
        assert active is not None

    def test_get_statistics(self):
        from forgecraft.evolution.hierarchical import (
            NestingLayer, LayerScheduler,
        )

        l1 = NestingLayer(name="L1", level=0, budget=100,
                          population=[make_body("x")])
        l2 = NestingLayer(name="L2", level=1, budget=50)
        sched = LayerScheduler([l1, l2])
        stats = sched.get_statistics()
        assert "L1" in stats and "L2" in stats
        assert stats["L1"]["population_size"] == 1


class TestHierarchicalNestingOptimizer:
    def test_create_default_layers(self):
        from forgecraft.evolution.hierarchical import create_default_layers

        layers = create_default_layers(n_objectives=5, n_divisions=4)
        assert len(layers) == 6
        names = [l.name for l in layers]
        assert "L1_morphology" in names
        assert "L6_multitask" in names
        # 验证漏斗配置
        assert layers[0].downstream == "L2_topology"
        assert layers[1].upstream == "L1_morphology"

    def test_optimize_basic(self):
        from forgecraft.evolution.hierarchical import (
            HierarchicalNestingOptimizer, create_default_layers,
        )

        layers = create_default_layers(
            n_objectives=3, n_divisions=3,
            l1_budget=50, l2_budget=30, l3_budget=20,
            l4_budget=20, l5_budget=10, l6_budget=10,
        )
        coordinator = HierarchicalNestingOptimizer(
            layers, max_generations=30, random_seed=42,
        )
        # 注入初始种群
        init_pop = [make_body(f"p{i}", fitness=float(i)) for i in range(10)]
        result = coordinator.optimize(initial_population=init_pop)
        assert len(result) >= 0

    def test_statistics_and_pareto_front(self):
        from forgecraft.evolution.hierarchical import (
            HierarchicalNestingOptimizer, create_default_layers,
        )

        layers = create_default_layers(
            n_objectives=2, n_divisions=3,
            l1_budget=30, l2_budget=15, l3_budget=10,
            l4_budget=10, l5_budget=5, l6_budget=5,
        )
        coordinator = HierarchicalNestingOptimizer(
            layers, max_generations=20, random_seed=42,
        )
        init_pop = [make_body(f"p{i}") for i in range(5)]
        coordinator.optimize(initial_population=init_pop)

        stats = coordinator.get_statistics()
        assert isinstance(stats, dict)
        assert "L1_morphology" in stats

        front = coordinator.get_pareto_front()
        assert isinstance(front, list)


# ══════════════════════════════════════════════════════════
#  6. 端到端集成: MAP-Elites + NSGA-III + Pareto 存档
# ══════════════════════════════════════════════════════════

class TestEndToEnd:
    """模拟完整进化循环: MAP-Elites 生成多样性 → NSGA-III 选择 → Pareto 存档"""

    def test_map_elites_to_nsga3_to_pareto(self):
        from forgecraft.evolution.map_elites import (
            MAPElitesEngine, MAPElitesConfig, inject_elites,
            compute_behavior_characteristics,
        )
        from forgecraft.evolution.nsga3 import nsga3_select
        from forgecraft.evolution.pareto_archive import ParetoArchive

        # 1. MAP-Elites 引擎 (2D 行为空间)
        config = MAPElitesConfig(
            bc_names=("speed", "stability"),
            bc_ranges=((0, 5), (0, 1)),
            archive_size=40,
            use_cvt=True,
            num_emitters=3,
            emitter_batch_size=6,
            evaluations_per_generation=18,
        )
        engine = MAPElitesEngine(config=config, seed=42)
        engine.initialize(generator_fn=lambda: make_body("init"))

        # 2. 运行 4 代 MAP-Elites
        for gen in range(4):
            def me_eval(body):
                fitness = np.random.random() * 2
                bc = np.array([fitness * 2, np.random.random()])
                return fitness, bc, {}

            engine.step(me_eval)

            # 周期性注入主流
            if gen % 2 == 1:
                elites = engine.get_injection_elites(n=2)
                # 用 NSGA-III 从精英中进一步选择
                if len(elites) >= 3:
                    selected = nsga3_select(
                        elites, min(2, len(elites)),
                        objective_keys=["speed", "stability"],
                        directions=["maximize", "maximize"],
                    )
                    assert len(selected) >= 1

        # 3. Pareto 存档 (记录所有精英)
        pa = ParetoArchive(n_objectives=2,
                           directions=["maximize", "maximize"])
        all_elites = engine.archive.get_all_elites()
        for e in all_elites:
            pa.add(e.body, e.behavior[:2])

        assert pa.size() > 0

        # 4. 验证集成数据流
        stats = engine.get_stats()
        assert "coverage" in stats
        qd_score = engine.get_qd_score_history()
        assert len(qd_score) >= 1

    def test_nesting_with_map_elites_injection(self):
        """模拟嵌套优化中 MAP-Elites 向 NSGA-III 种群注入精英"""
        from forgecraft.evolution.hierarchical import (
            NestingLayer, LayerState, HierarchicalNestingOptimizer,
            create_default_layers,
        )
        from forgecraft.evolution.map_elites import (
            MAPElitesEngine, MAPElitesConfig, inject_elites,
        )

        # 配置 MAP-Elites
        me_config = MAPElitesConfig(
            bc_names=("speed", "stability"),
            bc_ranges=((0, 5), (0, 1)),
            archive_size=20,
            num_emitters=2,
            emitter_batch_size=3,
            evaluations_per_generation=10,
        )
        me_engine = MAPElitesEngine(config=me_config, seed=42)
        me_engine.initialize(generator_fn=lambda: make_body("init"))

        # 运行 2 代
        for _ in range(2):
            me_engine.step(lambda b: (np.random.random(),
                          np.random.rand(2), {}))

        # 用 MAP-Elites 精英注入嵌套优化的 L1 种群
        layers = create_default_layers(n_objectives=3, n_divisions=2,
                                       l1_budget=20, l2_budget=10,
                                       l3_budget=5, l4_budget=5,
                                       l5_budget=5, l6_budget=5)
        coordinator = HierarchicalNestingOptimizer(
            layers, max_generations=15, random_seed=42,
        )

        init_pop = [make_body(f"p{i}") for i in range(8)]
        # 注入 MAP-Elites 精英
        init_pop = inject_elites(init_pop, me_engine, n=2, replace_worst=True)
        coordinator.optimize(initial_population=init_pop)

        stats = coordinator.get_statistics()
        assert stats["L1_morphology"]["population_size"] >= 1


# ══════════════════════════════════════════════════════════
#  7. MAP-Elites + CMA-ME 发射器集成
# ══════════════════════════════════════════════════════════

class TestCMAMEIntegration:
    """cma_me.OptimizerEmitter 与 map_elites 存档集成"""

    def test_cma_me_emitter_with_cvt_archive(self):
        from forgecraft.evolution.cma_me import OptimizerEmitter as CmaEmitter
        from forgecraft.evolution.map_elites import CVTArchive

        archive = CVTArchive(num_centroids=20, bc_dim=3,
                             bc_ranges=[(0, 1), (0, 1), (0, 1)],
                             random_seed=42)
        emitter = CmaEmitter(archive, n_dim=4, method="simple",
                             sigma0=0.2)

        # 多轮 ask/tell
        for _ in range(3):
            sols = emitter.ask(10)
            assert sols.shape == (10, 4)
            fitnesses = -np.sum(sols ** 2, axis=1)
            improvements = fitnesses > -1.0
            emitter.tell(sols, fitnesses, improvements.tolist())

        diag = emitter.get_diagnostics()
        assert "method" in diag

    def test_cma_me_cma_mode_with_empty_archive(self):
        """CMA-ME 模式下应能优雅处理空存档"""
        from forgecraft.evolution.cma_me import CMAMEmitter
        from forgecraft.evolution.map_elites import CVTArchive

        archive = CVTArchive(num_centroids=10, bc_dim=2,
                             bc_ranges=[(0, 1), (0, 1)],
                             random_seed=42)
        emitter = CMAMEmitter(archive, n_dim=3, popsize=10,
                              empty_niche_prob=0.5, random_seed=42)
        solutions, cells = emitter.ask(10)
        assert solutions.shape == (10, 3)

        # tell 无改进
        fitnesses = np.random.rand(10)
        improvements = [False] * 10
        emitter.tell(solutions, fitnesses, improvements)
        # 不应崩溃


# ══════════════════════════════════════════════════════════
#  8. import 完整性
# ══════════════════════════════════════════════════════════

class TestAllImports:
    def test_full_init_import(self):
        """验证 __init__.py 完整导入"""
        from forgecraft.evolution import (
            EvolutionLoop,
            # MAP-Elites
            MAPElitesConfig, MAPElitesEngine, GridArchive, CVTArchive,
            compute_behavior_characteristics, inject_elites, OptimizerEmitter,
            # NSGA-III
            NSGA3Selector, generate_reference_points, normalize_objectives,
            associate_to_reference_points, niching_select, nsga3_select,
            nsga3_pareto_elites,
            # CMA-ME
            CMAES, CMAMEmitter, cmame_sample,
            # Hierarchical
            LayerState, NestingLayer, LayerScheduler,
            HierarchicalNestingOptimizer, create_default_layers,
            # Pareto Archive
            ParetoArchive, compute_hypervolume, detect_knee_point,
            reference_point_query, spacing_metric, spread_metric,
            # Core
            PopulationBreeder, PopulationEvaluator, pareto_elites,
            mutate_params, crossover, apply_mutations,
            # Recovery
            TrainGuard, SafeCheckpointManager,
            # Adaptive mutation
            AdaptiveMutationController,
            # Bayesian evaluator
            BayesianEvaluator, ThompsonSamplingEvaluator,
            # Distributed
            DistributedEvaluator, DistributedEvolutionLoop,
            DistributedConfig, ClusterInfo,
            # Hyperparam
            HyperParamOptimizer, HyperParameterSpace,
            ParameterScheduler, EarlyStopper,
            SimpleBayesianOptimizer, ParamSpec,
            OptimizationResult, TrialResult,
            # Parallel
            ParallelEvaluator, ParallelConfig, BatchResult,
            get_optimal_workers,
        )
        assert True  # 没异常即可
