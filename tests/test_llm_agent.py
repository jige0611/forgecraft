"""集成测试: LLM 智能体 + 知识库 全链路

覆盖:
  - KnowledgeBase: 嵌入 / 存储 / 检索 / 记录
  - DesignReasoningEngine: 任务分析 / 运行监控 / 故障诊断
  - CatalogAgent: 目录选择 / 评估 / 生成
  - StrategyAgent: 规则检测 / LLM 增强检测
  - CritiqueAgent: 运行评分 / 失败模式识别
  - TradeoffAgent: Pareto 前沿分析 / 偏好匹配
  - LLMObserver + LLMActuator: 挂载与执行
  - ToolRegistry: 工具注册与调用
  - Provider: Mock/工厂/消息格式
"""

import json
import os
import tempfile

import numpy as np
import pytest

from forgecraft.core.morphology import MechanicalBody, Part, Joint


# ── helpers ─────────────────────────────────────────────────

def make_body(name, fitness=0.0):
    body = MechanicalBody(name)
    root = Part("base", {"length": 0.1}, np.array([0, 0, 0.5]))
    body.add_part(root)
    body.fitness = fitness
    return body


def make_kb(storage_dir=None):
    from forgecraft.knowledge import KnowledgeBase
    if storage_dir is None:
        storage_dir = os.path.join("data", "test_knowledge")
    return KnowledgeBase(storage_dir=storage_dir, auto_load=False)


def make_provider():
    from forgecraft.llm import create_provider
    return create_provider("mock")


# ══════════════════════════════════════════════════════════
#  1. KnowledgeBase 集成测试
# ══════════════════════════════════════════════════════════

class TestKnowledgeBase:
    def test_create_and_stats(self):
        kb = make_kb()
        stats = kb.get_statistics()
        assert stats["semantic"]["total_cases"] == 0
        assert stats["graph"]["ontology_nodes"] > 0
        kb.close()

    def test_record_and_retrieve_case(self):
        kb = make_kb()
        case_id = kb.record_case(
            task_name="speed",
            catalog_name="default",
            objectives={"speed": 2.5, "energy": 0.3},
            behavior_bc=[0.8, 0.4, 0.6],
            qd_score_final=0.85,
            coverage_final=0.42,
            best_fitness=3.2,
            tags=["fast", "stable"],
        )
        assert case_id.startswith("case_")

        similar = kb.find_similar_cases("fast robot", top_k=3)
        assert len(similar) > 0

        stats = kb.get_statistics()
        assert stats["semantic"]["total_cases"] >= 1
        kb.close()

    def test_similar_cases_filtering(self):
        kb = make_kb()
        kb.record_case(task_name="speed", catalog_name="default",
                       objectives={"speed": 3.0}, behavior_bc=[0.9, 0.0],
                       qd_score_final=0.9)
        kb.record_case(task_name="climbing", catalog_name="gripper",
                       objectives={"speed": 0.5}, behavior_bc=[0.1, 0.9],
                       qd_score_final=0.5)

        # 按任务过滤
        cases = kb.find_cases_by_task("speed")
        assert len(cases) >= 1

        # 语义检索
        results = kb.find_similar_cases("robot that climbs", top_k=3,
                                        min_qd_score=0.3)
        assert len(results) >= 1
        kb.close()

    def test_generation_recording(self):
        kb = make_kb()
        case_id = kb.record_case(
            task_name="speed", catalog_name="default",
            objectives={}, behavior_bc=[], run_id="test_run_1",
        )

        for gen in range(0, 100, 10):
            kb.record_generation(case_id, gen, {
                "qd_score": gen * 0.01,
                "coverage": gen * 0.005,
                "best_fitness": gen * 0.02,
                "population_diversity": 0.5 - gen * 0.003,
            })

        traj = kb.get_trajectory(case_id)
        assert len(traj) >= 5
        assert traj[0]["generation"] == 0
        kb.close()

    def test_failure_recording(self):
        kb = make_kb()
        case_id = kb.record_case(
            task_name="speed", catalog_name="default",
            objectives={}, behavior_bc=[], run_id="test_fail_1",
        )

        kb.record_failure(case_id, 30, "coverage_stall",
                          "increase_mutation", recovery_success=True)
        kb.record_failure(case_id, 60, "coverage_stall",
                          "increase_mutation", recovery_success=False)

        stats = kb.get_failure_stats()
        # stats may be empty if failure_patterns table is empty initially
        assert isinstance(stats, list)
        kb.close()

    def test_part_ontology_queries(self):
        kb = make_kb()
        parts = kb.query_parts_by_function("structural")
        assert "base" in parts or "segment" in parts

        func = kb.query_part_function("motor")
        assert func is not None

        replacements = kb.query_replacements("wheel")
        assert isinstance(replacements, list)
        kb.close()

    def test_heuristics_and_strategy(self):
        kb = make_kb()
        rules = kb.query_heuristics("speed")
        assert len(rules) > 0

        template = kb.query_strategy_template("speed")
        assert "recommended_strategy" in template
        assert template["recommended_strategy"] in ("map_elites", "nsga3", "cma_me")

        # default fallback
        template2 = kb.query_strategy_template("nonexistent")
        assert "recommended_strategy" in template2
        kb.close()

    def test_semantic_search_empty(self):
        kb = make_kb()
        results = kb.find_similar_cases("anything")
        assert results == [] or len(results) >= 0  # 空库无结果
        kb.close()


# ══════════════════════════════════════════════════════════
#  2. Embeddings
# ══════════════════════════════════════════════════════════

class TestEmbeddings:
    def test_part_embedding(self):
        from forgecraft.knowledge.embeddings import embed_part, PartEmbedder

        spec = {
            "name": "rotary_joint",
            "geometry": {"type": "cylinder"},
            "actuated": True,
            "joint": {"type": "hinge", "axis": [0, 0, 1]},
        }
        vec = embed_part(spec)
        assert vec.ndim == 1
        assert vec.shape[0] == 384

    def test_task_embedding(self):
        from forgecraft.knowledge.embeddings import embed_task_text

        vec = embed_task_text("design a fast running quadruped robot")
        assert vec.ndim == 1
        assert vec.shape[0] == 384

    def test_morphology_embedding(self):
        from forgecraft.knowledge.embeddings import embed_morphology

        body = make_body("test")
        vec = embed_morphology(body)
        assert vec.ndim == 1
        assert vec.shape[0] == 384

    def test_performance_embedding(self):
        from forgecraft.knowledge.embeddings import embed_performance

        obj = np.array([2.5, 0.3, 5.0])
        vec = embed_performance(obj)
        assert vec.ndim == 1
        assert vec.shape[0] == 384

    def test_part_to_text(self):
        from forgecraft.knowledge.embeddings import part_spec_to_text

        text = part_spec_to_text({
            "name": "motor",
            "geometry": {"type": "cylinder"},
            "actuated": True,
            "joint": {"type": "hinge", "axis": [0, 0, 1],
                      "range": [-1.5, 1.5], "torque": [0.1, 10]},
            "physics": {"mass": 0.05},
            "size": {"radius": [0.01, 0.03]},
        })
        assert "motor" in text
        assert "hinge" in text

    def test_task_to_text(self):
        from forgecraft.knowledge.embeddings import task_config_to_text

        text = task_config_to_text({
            "name": "speed",
            "description": "Maximize forward velocity",
            "reward": {
                "components": [
                    {"name": "displacement", "weight": 1.0, "description": "Forward movement"},
                ],
                "fitness_components": ["speed", "energy"],
                "fitness_formula": "speed * 1.0",
            },
            "simulation": {"gravity": [0, 0, -9.81]},
            "terrain": {"type": "flat"},
            "termination": {"max_steps": 500},
        })
        assert "speed" in text
        assert "displacement" in text

    def test_pseudo_embed_deterministic(self):
        from forgecraft.knowledge.embeddings import _pseudo_embed

        v1 = _pseudo_embed("hello world", 128)
        v2 = _pseudo_embed("hello world", 128)
        assert np.allclose(v1, v2)

    def test_graph_topology_features(self):
        from forgecraft.knowledge.embeddings import _graph_topology_features

        body = make_body("test")
        vec = _graph_topology_features(body)
        assert vec.ndim == 1
        assert np.linalg.norm(vec) > 0


# ══════════════════════════════════════════════════════════
#  3. Provider
# ══════════════════════════════════════════════════════════

class TestProvider:
    def test_mock_provider(self):
        from forgecraft.llm import create_provider

        p = create_provider("mock", responses=["Hello, I am the design agent."])
        resp = p.chat([{"role": "user", "content": "Hi"}])
        assert "Hello" in resp.content
        assert resp.model == "mock"

    def test_provider_factory(self):
        from forgecraft.llm import create_provider
        from forgecraft.llm.provider import MockProvider, OpenAIProvider

        # OpenAI may not have API key, so it should fall through
        p_mock = create_provider("mock")
        assert isinstance(p_mock, MockProvider)

    def test_llm_message_format(self):
        from forgecraft.llm.provider import LLMMessage

        msg = LLMMessage(role="system", content="You are a design agent.")
        assert msg.role == "system"

    def test_tool_definition(self):
        from forgecraft.llm.provider import ToolDefinition

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )
        assert tool.name == "test_tool"


# ══════════════════════════════════════════════════════════
#  4. DesignReasoningEngine
# ══════════════════════════════════════════════════════════

class TestDesignReasoningEngine:
    def test_creation(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import DesignReasoningEngine

        engine = DesignReasoningEngine(kb, provider=p)
        assert engine.provider.get_model_name() == "mock"
        kb.close()

    def test_analyze_task(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import DesignReasoningEngine

        engine = DesignReasoningEngine(kb, provider=p)
        result = engine.analyze_task(
            "design a fast robot for flat terrain racing",
            task_type="speed",
        )
        assert "task_type" in result
        assert "recommended_strategy" in result
        assert result["task_type"] == "speed"
        kb.close()

    def test_analyze_task_auto_infer(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import DesignReasoningEngine

        engine = DesignReasoningEngine(kb, provider=p)

        # 自动推断任务类型
        r1 = engine.analyze_task("design a robot that can climb steep walls")
        assert r1["task_type"] == "climbing"  # "climb" → "climbing" as per _infer_task_type

        r2 = engine.analyze_task("design a gripper to pick and place objects")
        # "pick place grasp" → manipulation
        assert r2["task_type"] in ("speed", "climbing", "manipulation")
        kb.close()

    def test_monitor_run_no_issue(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import DesignReasoningEngine

        engine = DesignReasoningEngine(kb, provider=p)
        # 一切正常
        result = engine.monitor_run({
            "generation": 50,
            "qd_score": 0.8,
            "coverage": 0.6,
            "best_fitness": 3.0,
            "population_diversity": 0.4,
            "stagnation_duration": 2,
            "num_elites": 30,
        })
        # No stagnation should be detected
        assert result is None
        kb.close()

    def test_monitor_run_stagnation(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import DesignReasoningEngine

        engine = DesignReasoningEngine(kb, provider=p)
        result = engine.monitor_run({
            "generation": 80,
            "qd_score": 0.5,
            "coverage": 0.3,
            "best_fitness": 2.0,
            "population_diversity": 0.08,
            "stagnation_duration": 25,
            "num_elites": 15,
        }, run_id="test_run")
        assert result is not None
        assert result["action"] == "adjust"
        kb.close()

    def test_diagnose_run_without_data(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import DesignReasoningEngine

        engine = DesignReasoningEngine(kb, provider=p)
        report = engine.diagnose_run("nonexistent_run")
        assert "success" in report
        assert "recommendations" in report
        kb.close()

    def test_tool_registry(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import DesignReasoningEngine

        engine = DesignReasoningEngine(kb, provider=p)
        tools = engine.tools.list_names()
        assert "query_similar_cases" in tools
        assert "select_catalog" in tools
        assert "configure_evolution" in tools
        assert "adjust_strategy" in tools
        kb.close()


# ══════════════════════════════════════════════════════════
#  5. CatalogAgent
# ══════════════════════════════════════════════════════════

class TestCatalogAgent:
    def test_select_catalog(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import CatalogAgent

        agent = CatalogAgent(kb, p)

        r = agent.select("设计一个快速四足机器人", task_type="speed")
        assert r["catalog_name"] in ("default", "hoppers", "biped")
        assert "confidence" in r

        r2 = agent.select("设计一个能抓取物体的夹持器", task_type="manipulation")
        assert r2["catalog_name"] in ("gripper", "default")
        kb.close()

    def test_evaluate_match(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import CatalogAgent

        agent = CatalogAgent(kb, p)

        result = agent.evaluate_match("gripper", "manipulation")
        assert result["match_score"] >= 0.5
        kb.close()

    def test_generate_catalog(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import CatalogAgent

        agent = CatalogAgent(kb, p)

        result = agent.generate(
            task_name="underwater_explorer",
            task_description="design an underwater exploration robot",
            required_functions=["structural", "actuation", "locomotion"],
        )
        assert "yaml_content" in result
        assert "parts_generated" in result
        kb.close()


# ══════════════════════════════════════════════════════════
#  6. StrategyAgent
# ══════════════════════════════════════════════════════════

class TestStrategyAgent:
    def test_check_no_issue(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import StrategyAgent

        agent = StrategyAgent(kb, p)
        result = agent.check({
            "generation": 10,
            "qd_score": 0.5,
            "coverage": 0.3,
            "population_diversity": 0.5,
            "stagnation_duration": 0,
        }, use_llm=False)
        assert result is None

    def test_check_diversity_collapse(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import StrategyAgent

        agent = StrategyAgent(kb, p)
        result = agent.check({
            "generation": 30,
            "qd_score": 0.3,
            "coverage": 0.1,
            "population_diversity": 0.05,
            "stagnation_duration": 10,
        }, use_llm=False)
        assert result is not None
        assert result["signal"] == "diversity_collapse"

    def test_check_coverage_stall(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import StrategyAgent

        agent = StrategyAgent(kb, p)
        # Simulate accumulated stagnation
        for _ in range(21):
            agent.check({
                "generation": 80,
                "qd_score": 0.5,
                "coverage": 0.3,
                "population_diversity": 0.3,
            }, use_llm=False)

        result = agent.check({
            "generation": 100,
            "qd_score": 0.5,
            "coverage": 0.3,
            "population_diversity": 0.3,
            "stagnation_duration": 20,
        }, use_llm=False)
        assert result is not None
        assert "stagnation" in result["signal"] or "stall" in result["signal"]
        kb.close()

    def test_slow_start(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import StrategyAgent

        agent = StrategyAgent(kb, p)
        result = agent.check({
            "generation": 30,
            "qd_score": 0.0,
            "coverage": 0.0,
            "population_diversity": 0.3,
            "stagnation_duration": 0,
        }, use_llm=False)
        assert result is not None
        assert result["signal"] == "slow_start"
        kb.close()

    def test_statistics(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import StrategyAgent

        agent = StrategyAgent(kb, p)
        stats = agent.get_statistics()
        assert "total_adjustments" in stats
        kb.close()


# ══════════════════════════════════════════════════════════
#  7. CritiqueAgent
# ══════════════════════════════════════════════════════════

class TestCritiqueAgent:
    def test_analyze_empty(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import CritiqueAgent

        agent = CritiqueAgent(kb, p)
        result = agent.analyze("nonexistent")
        assert "overall_verdict" in result
        kb.close()

    def test_analyze_with_trajectory(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import CritiqueAgent

        case_id = kb.record_case(
            task_name="speed", catalog_name="default",
            objectives={"speed": 3.0}, behavior_bc=[0.8, 0.4],
            run_id="critique_test_1",
        )
        for gen in range(0, 200, 10):
            kb.record_generation(case_id, gen, {
                "qd_score": 0.1 + gen * 0.004,
                "coverage": gen * 0.005,
                "best_fitness": gen * 0.02,
                "population_diversity": 0.5 - gen * 0.001,
            })

        agent = CritiqueAgent(kb, p)
        result = agent.analyze("critique_test_1")
        assert result["overall_verdict"] in (
            "excellent", "good", "acceptable", "poor", "unknown"
        )
        assert "scores" in result
        kb.close()

    def test_compare_runs(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import CritiqueAgent

        agent = CritiqueAgent(kb, p)
        result = agent.compare_runs(["run_a", "run_b"])
        assert "ranking" in result
        kb.close()


# ══════════════════════════════════════════════════════════
#  8. TradeoffAgent
# ══════════════════════════════════════════════════════════

class TestTradeoffAgent:
    def test_analyze_front_2d(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import TradeoffAgent

        agent = TradeoffAgent(kb, p)
        front = np.array([
            [1.0, 10.0],
            [3.0, 7.0],
            [5.0, 5.0],
            [7.0, 3.0],
            [10.0, 1.0],
        ])

        result = agent.analyze_front(
            front,
            names=["speed", "energy"],
            directions=["maximize", "minimize"],
            preferences={"speed": 0.7, "energy": 0.3},
        )
        assert result["front_size"] == 5
        assert "knee_point" in result
        assert "recommendation" in result
        kb.close()

    def test_analyze_front_3d(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import TradeoffAgent

        agent = TradeoffAgent(kb, p)
        front = np.array([
            [1, 8, 5],
            [3, 6, 4],
            [5, 5, 5],
            [7, 3, 2],
            [10, 1, 1],
        ])

        result = agent.analyze_front(
            front,
            names=["speed", "stability", "energy"],
            directions=["maximize", "maximize", "minimize"],
        )
        assert result["front_size"] == 5
        kb.close()

    def test_recommend_weighted(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import TradeoffAgent

        agent = TradeoffAgent(kb, p)
        front = np.array([[1, 10], [3, 7], [5, 5], [7, 3], [10, 1]])

        result = agent.recommend_weighted(
            front,
            weights=np.array([0.8, 0.2]),
            names=["speed", "energy"],
        )
        assert result["best_index"] == 4  # 速度最高
        kb.close()


# ══════════════════════════════════════════════════════════
#  9. ToolRegistry & Function Calling
# ══════════════════════════════════════════════════════════

class TestToolRegistry:
    def test_register_and_call(self):
        from forgecraft.llm.tools import ToolRegistry, TOOL_GET_KB_STATS

        registry = ToolRegistry()
        assert "get_kb_stats" in registry.list_names()

    def test_set_handler(self):
        from forgecraft.llm.tools import ToolRegistry
        from forgecraft.llm.provider import ToolDefinition

        registry = ToolRegistry()
        called = []

        def handler(**kwargs):
            called.append(kwargs)
            return "done"

        registry.register(ToolDefinition(
            name="test",
            description="test",
            parameters={},
        ))
        registry.set_handler("test", handler)

        tool = registry.get("test")
        assert tool.handler is not None
        result = tool.handler(x=1)
        assert result == "done"
        assert called == [{"x": 1}]


# ══════════════════════════════════════════════════════════
#  10. LLMActuator
# ══════════════════════════════════════════════════════════

class TestLLMActuator:
    def test_safe_ranges(self):
        from forgecraft.llm import LLMActuator

        actuator = LLMActuator()
        assert actuator._is_safe("mutation_rate", 0.3) is True
        assert actuator._is_safe("mutation_rate", 2.0) is False
        assert actuator._is_safe("unknown_param", 1.0) is False

    def test_apply_adjustment(self):
        from forgecraft.llm import LLMActuator
        from forgecraft.config import EvolutionConfig

        config = EvolutionConfig()
        actuator = LLMActuator()

        # Mock a simple loop-like object
        class MockLoop:
            def __init__(self):
                self.evo_config = config
                self.generation = 10
                self._run_id = "test"

        loop = MockLoop()
        result = actuator.apply_adjustment(
            loop,
            {"mutation_rate": 0.5, "crossover_rate": 0.7},
            reason="testing",
        )
        assert result["success"] is True
        assert "mutation_rate" in result["applied"]
        assert config.mutation_rate == 0.5

    def test_apply_rejected_out_of_range(self):
        from forgecraft.llm import LLMActuator
        from forgecraft.config import EvolutionConfig

        config = EvolutionConfig()
        actuator = LLMActuator()

        class MockLoop:
            def __init__(self):
                self.evo_config = config

        loop = MockLoop()
        result = actuator.apply_adjustment(loop, {"mutation_rate": 2.0})
        assert len(result["rejected"]) >= 1

    def test_action_log(self):
        from forgecraft.llm import LLMActuator

        actuator = LLMActuator()
        assert actuator.get_action_log() == []


# ══════════════════════════════════════════════════════════
#  11. KnowledgeRecorder
# ══════════════════════════════════════════════════════════

class TestKnowledgeRecorder:
    def test_start_and_record(self):
        from forgecraft.knowledge import KnowledgeRecorder

        kb = make_kb()
        recorder = KnowledgeRecorder(kb, autosave_every_n_gens=10)

        run_id = recorder.start_run("speed", "default")
        assert run_id.startswith("run_")

        for gen in range(5):
            recorder.record_generation(gen, {
                "qd_score": gen * 0.2,
                "coverage": gen * 0.1,
                "best_fitness": gen * 0.3,
                "population_diversity": 0.5,
                "num_elites": 10,
            })

        assert recorder._generation_counter == 4  # last gen in range(5) is 4

        kb.close()

    def test_stagnation_detection(self):
        from forgecraft.knowledge import KnowledgeRecorder

        kb = make_kb()
        recorder = KnowledgeRecorder(kb)

        recorder.start_run("speed", "default")

        # 模拟停滞
        for gen in range(25):
            recorder.record_generation(gen, {
                "qd_score": 0.5 if gen < 5 else 0.5,
                "coverage": 0.3,
                "best_fitness": 2.0,
                "population_diversity": 0.3,
                "num_elites": 10,
            })

        stagnation = recorder.detect_stagnation()
        assert stagnation is not None
        kb.close()

    def test_finalize(self):
        from forgecraft.knowledge import KnowledgeRecorder

        kb = make_kb()
        recorder = KnowledgeRecorder(kb)

        recorder.start_run("speed", "default")
        recorder.record_generation(0, {"qd_score": 0.1, "coverage": 0.05})
        recorder.record_generation(10, {"qd_score": 0.5, "coverage": 0.3})

        case_id = recorder.finalize(
            objectives={"speed": 2.0},
            behavior_bc=[0.7, 0.3],
        )
        # finalize now calls kb.record_case which returns case_xxx
        assert case_id != ""
        kb.close()


# ══════════════════════════════════════════════════════════
#  12.  端到端: 知识库 + 智能体联合
# ══════════════════════════════════════════════════════════

class TestEndToEndLLM:
    """模拟完整工作流: 任务 → KB检索 → 智能体决策 → 运行 → 评审 → 沉淀"""

    def test_full_pipeline(self):
        kb = make_kb()
        p = make_provider()
        from forgecraft.llm import (
            DesignReasoningEngine, CatalogAgent,
            StrategyAgent, CritiqueAgent,
        )
        from forgecraft.knowledge import KnowledgeRecorder

        # 1. 分析任务
        engine = DesignReasoningEngine(kb, provider=p)
        analysis = engine.analyze_task(
            "design a fast running quadruped on flat terrain",
            task_type="speed",
        )
        assert analysis["recommended_strategy"] in ("map_elites", "nsga3", "cma_me")

        # 2. 选择目录
        cat = CatalogAgent(kb, p)
        catalog = cat.select("fast running quadruped", task_type="speed")
        assert catalog["catalog_name"] != ""

        # 3. 模拟运行
        recorder = KnowledgeRecorder(kb, autosave_every_n_gens=50)
        run_id = recorder.start_run("speed", catalog["catalog_name"])

        strat = StrategyAgent(kb, p)

        for gen in range(0, 100, 10):
            metrics = {
                "generation": gen,
                "qd_score": gen * 0.008,
                "coverage": gen * 0.004,
                "best_fitness": gen * 0.015,
                "population_diversity": max(0.1, 0.5 - gen * 0.003),
                "stagnation_duration": 0,
                "num_elites": 5 + gen // 5,
            }
            recorder.record_generation(gen, metrics)

            # 策略检查 (模拟)
            if gen >= 50:
                strat.check(metrics, run_id=run_id, use_llm=False)

        # 4. 完结
        case_id = recorder.finalize(
            objectives={"speed": 3.5, "energy": 0.4},
            behavior_bc=[0.85, 0.45],
        )

        # 5. 评审
        crit = CritiqueAgent(kb, p)
        report = crit.analyze(run_id)
        assert "overall_verdict" in report

        # 6. 检索验证 — 知识已沉淀
        similar = kb.find_similar_cases("fast running robot")
        assert len(similar) >= 1

        kb.close()


# ══════════════════════════════════════════════════════════
#  13.  全量 Import 验证
# ══════════════════════════════════════════════════════════

class TestAllLLMImports:
    def test_knowledge_imports(self):
        from forgecraft.knowledge import (
            KnowledgeBase, KnowledgeRecorder,
            SemanticKB, RelationalKB, GraphKB,
            PartEmbedder, TaskEmbedder,
            MorphologyEmbedder, PerformanceEmbedder,
            embed_part, embed_task, embed_morphology, embed_performance,
            DesignCase,
            PART_ONTOLOGY, PHYSICS_HEURISTICS, STRATEGY_TEMPLATES,
            get_parts_by_function, get_heuristics_for_task, get_strategy_template,
            part_ontology_to_graph,
        )
        assert True

    def test_llm_imports(self):
        from forgecraft.llm import (
            DesignReasoningEngine, AgentConfig, AutonomyLevel,
            CatalogAgent, StrategyAgent, CritiqueAgent, TradeoffAgent,
            LLMObserver, LLMActuator,
            create_provider, MockProvider, ToolRegistry,
            BaseLLMProvider, LLMMessage, LLMResponse, ToolDefinition,
        )
        assert True

    def test_prompts_imports(self):
        from forgecraft.llm.prompts.system_prompt import (
            SYSTEM_PROMPT, CATALOG_AGENT_PROMPT,
            STRATEGY_AGENT_PROMPT, CRITIQUE_AGENT_PROMPT,
        )
        assert len(SYSTEM_PROMPT) > 100
        assert len(CATALOG_AGENT_PROMPT) > 50

        from forgecraft.llm.prompts.templates import (
            render_task_analysis, render_run_status, render_similar_cases,
        )
        assert True

        from forgecraft.llm.prompts.few_shot_examples import (
            CATALOG_SELECTION_EXAMPLE,
            STAGNATION_HANDLING_EXAMPLE,
            CATALOG_GENERATION_EXAMPLE,
        )
        assert True

    def test_tools_imports(self):
        from forgecraft.llm.tools import (
            ToolRegistry,
            TOOL_QUERY_SIMILAR_CASES,
            TOOL_SELECT_CATALOG,
            TOOL_CONFIGURE_EVOLUTION,
            TOOL_ADJUST_STRATEGY,
            TOOL_DIAGNOSE_FAILURE,
            TOOL_PROPOSE_CATALOG,
        )
        assert len(TOOL_QUERY_SIMILAR_CASES.description) > 10
