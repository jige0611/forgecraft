"""统一知识库 — 编排三个子引擎

使用方式:
    kb = KnowledgeBase(storage_dir="data/knowledge")
    kb.record_case(task_name="speed", catalog_name="default", ...)
    similar = kb.find_similar_cases("设计一个快速移动的机器人")
    stats = kb.get_statistics()
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from forgecraft.knowledge.embeddings import (
    PartEmbedder,
    TaskEmbedder,
    MorphologyEmbedder,
    PerformanceEmbedder,
)
from forgecraft.knowledge.vector_store import SemanticKB, DesignCase
from forgecraft.knowledge.relational_store import RelationalKB
from forgecraft.knowledge.graph_store import GraphKB
from forgecraft.knowledge.ontology import (
    PART_ONTOLOGY,
    PHYSICS_HEURISTICS,
    STRATEGY_TEMPLATES,
    get_parts_by_function,
    get_heuristics_for_task,
    get_strategy_template,
)
from forgecraft.logging import get_logger

_logger = get_logger(__name__)


class KnowledgeBase:
    """统一知识库入口

    协调:
      - semantic_kb:  语义向量检索 (FAISS)
      - relational_kb: 结构化查询 (SQLite)
      - graph_kb:      图推理 (NetworkX)
      - 嵌入器:        Part/Task/Morphology/Performance Embedder
    """

    def __init__(
        self,
        storage_dir: str = "data/knowledge",
        embed_dim: int = 384,
        auto_load: bool = True,
    ):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        # 三个子引擎
        self.semantic_kb = SemanticKB(
            dim=embed_dim,
            storage_path=os.path.join(storage_dir, "semantic.pkl"),
        )
        self.relational_kb = RelationalKB(
            db_path=os.path.join(storage_dir, "knowledge.db"),
        )
        self.graph_kb = GraphKB(ontology=PART_ONTOLOGY)

        # 嵌入器 (lazy init)
        self._part_embedder: Optional[PartEmbedder] = None
        self._task_embedder: Optional[TaskEmbedder] = None
        self._morph_embedder: Optional[MorphologyEmbedder] = None
        self._perf_embedder: Optional[PerformanceEmbedder] = None

        # 尝试加载已有数据
        if auto_load:
            self._try_load()

    def _try_load(self):
        path = os.path.join(self.storage_dir, "semantic.pkl")
        if os.path.exists(path):
            try:
                loaded = SemanticKB.load(path)
                self.semantic_kb = loaded
                _logger.info(f"Loaded semantic KB: {loaded.size} cases")
            except Exception as e:
                _logger.warning(f"Failed to load semantic KB: {e}")

    # ── 设计案例记录 (统一入口) ──────────────────────────────

    def record_case(
        self,
        task_name: str,
        catalog_name: str,
        objectives: Dict[str, float],
        behavior_bc: List[float],
        morphology_hash: str = "",
        evolution_config: Optional[Dict] = None,
        generation_count: int = 0,
        wall_time_seconds: float = 0.0,
        qd_score_final: float = 0.0,
        coverage_final: float = 0.0,
        best_fitness: float = 0.0,
        best_individual: Optional[Any] = None,
        best_body_graph: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        run_id: str = "",
    ) -> str:
        """记录一次完整的进化设计案例

        同时写入三个子引擎:
          1. SemanticKB: 嵌入 + 检索
          2. RelationalKB: 结构化查询
          3. GraphKB: 成功形态图索引

        Returns:
            case_id
        """
        # 生成案例 ID
        import uuid
        case_id = f"case_{uuid.uuid4().hex[:12]}"

        # 1. 语义向量库
        case = DesignCase(
            case_id=case_id,
            task_name=task_name,
            catalog_name=catalog_name,
            objectives=objectives,
            behavior_bc=behavior_bc,
            morphology_hash=morphology_hash,
            evolution_config=evolution_config or {},
            generation_count=generation_count,
            wall_time_seconds=wall_time_seconds,
            qd_score_final=qd_score_final,
            coverage_final=coverage_final,
            best_fitness=best_fitness,
            tags=tags or [],
            notes=notes,
        )

        # 计算复合嵌入 (任务 + 性能)
        task_emb = self._get_task_embedder().embed_task_from_name(task_name)
        perf_emb = self._get_perf_embedder().embed(
            np.array(list(objectives.values()))
        )
        combined_emb = (task_emb + perf_emb) / 2.0

        self.semantic_kb.add(case, combined_emb)

        # 2. 关系库
        case_data = {
            "case_id": case_id,
            "task_name": task_name,
            "catalog_name": catalog_name,
            "objectives": objectives,
            "behavior_bc": behavior_bc,
            "morphology_hash": morphology_hash,
            "evolution_config": evolution_config or {},
            "generation_count": generation_count,
            "wall_time_seconds": wall_time_seconds,
            "qd_score_final": qd_score_final,
            "coverage_final": coverage_final,
            "best_fitness": best_fitness,
            "best_individual": _serialize_individual(best_individual),
            "tags": tags or [],
            "notes": notes,
            "run_id": run_id,
        }
        self.relational_kb.insert_design_case(case_data)

        # 3. 图库 (索引成功形态)
        if best_body_graph is not None:
            try:
                self.graph_kb.index_morphology(
                    case_id, best_body_graph,
                    performance=objectives, tags=tags,
                )
            except Exception as e:
                _logger.warning(f"Failed to index morphology: {e}")

        _logger.info(
            f"Recorded case {case_id}: task={task_name}, "
            f"catalog={catalog_name}, qd={qd_score_final:.3f}"
        )
        return case_id

    # ── 进化轨迹记录 ─────────────────────────────────────────

    def record_generation(self, case_id: str, generation: int, metrics: Dict):
        """记录一代进化的指标"""
        self.relational_kb.record_generation(case_id, generation, metrics)

    def get_trajectory(self, case_id: str) -> List[Dict]:
        return self.relational_kb.get_trajectory(case_id)

    # ── 检索 ─────────────────────────────────────────────────

    def find_similar_cases(
        self,
        query_text: str,
        top_k: int = 5,
        task_filter: Optional[str] = None,
        min_qd_score: Optional[float] = None,
    ) -> List[Tuple[DesignCase, float]]:
        """根据自然语言描述检索相似设计案例

        Example:
            kb.find_similar_cases("设计一个能快速爬楼梯的轻量四足机器人")
        """
        # 嵌入查询文本
        query_emb = self._get_task_embedder().embed_text(query_text)

        # 混合检索
        return self.semantic_kb.search_with_filter(
            query_emb,
            top_k=top_k,
            task_name=task_filter,
            min_qd_score=min_qd_score,
        )

    def find_cases_by_task(self, task_name: str, top_k: int = 10) -> List[DesignCase]:
        """按任务名精确查询"""
        return self.semantic_kb.search_by_task(task_name, top_k)

    def find_similar_morphologies(self, body_graph, top_k: int = 5) -> List[str]:
        """查找与给定形态图相似的成功案例"""
        return self.graph_kb.find_similar_morphologies(body_graph, top_k)

    # ── 知识查询 ─────────────────────────────────────────────

    def query_heuristics(self, task_type: str) -> List[Dict]:
        """查询任务相关的物理启发式规则"""
        return get_heuristics_for_task(task_type)

    def query_strategy_template(self, task_type: str) -> Dict:
        """查询任务推荐策略"""
        return get_strategy_template(task_type)

    def query_part_function(self, part_name: str) -> Optional[str]:
        return self.graph_kb.get_function_of_part(part_name)

    def query_compatible_parts(self, part_name: str) -> List:
        return self.graph_kb.get_compatible_parts(part_name)

    def query_replacements(self, part_name: str) -> List[str]:
        return self.graph_kb.find_replacements(part_name)

    def query_parts_by_function(self, function: str) -> List[str]:
        return get_parts_by_function(function)

    def query_task_types(self) -> List[str]:
        from forgecraft.knowledge.ontology import get_all_task_types
        return get_all_task_types()

    # ── 失败模式 ─────────────────────────────────────────────

    def record_failure(self, case_id: str, generation: int,
                       pattern_name: str, action_taken: str,
                       recovery_success: bool):
        self.relational_kb.record_failure_event(
            case_id, generation, pattern_name, action_taken, recovery_success,
        )

    def get_failure_stats(self) -> List[Dict]:
        return self.relational_kb.get_failure_stats()

    # ── 参数历史 ─────────────────────────────────────────────

    def record_param_change(self, case_id: str, generation: int,
                            param_name: str, old_value: float,
                            new_value: float, reason: str = ""):
        self.relational_kb.record_param_change(
            case_id, generation, param_name, old_value, new_value, reason,
        )

    # ── 统计 ─────────────────────────────────────────────────

    def get_statistics(self) -> Dict:
        return {
            "semantic": self.semantic_kb.get_statistics(),
            "relational": self.relational_kb.get_aggregate_stats(),
            "graph": self.graph_kb.get_statistics(),
        }

    def get_task_aggregate(self, task_name: str) -> Dict:
        return self.relational_kb.get_aggregate_stats(task_name)

    # ── 持久化 ───────────────────────────────────────────────

    def save(self):
        path = os.path.join(self.storage_dir, "semantic.pkl")
        self.semantic_kb.save(path)
        _logger.info(f"KnowledgeBase saved to {self.storage_dir}")

    def close(self):
        self.save()
        self.relational_kb.close()

    # ── 嵌入器 (lazy access) ──────────────────────────────────

    def _get_task_embedder(self) -> TaskEmbedder:
        if self._task_embedder is None:
            self._task_embedder = TaskEmbedder()
            # 扩展: embed_task_from_name 用预置文本
            self._task_embedder.embed_task_from_name = self._embed_task_by_name
        return self._task_embedder

    def _get_perf_embedder(self) -> PerformanceEmbedder:
        if self._perf_embedder is None:
            self._perf_embedder = PerformanceEmbedder()
        return self._perf_embedder

    def _embed_task_by_name(self, task_name: str) -> np.ndarray:
        """用预置任务描述代替 config 文件"""
        from forgecraft.knowledge.ontology import STRATEGY_TEMPLATES

        template = STRATEGY_TEMPLATES.get(task_name, STRATEGY_TEMPLATES["default"])
        desc = f"Task: {task_name}. " + template.get("reason", "")
        return self._get_task_embedder().embed_text(desc)


def _serialize_individual(individual) -> Dict:
    """安全序列化 MechanicalBody 个体"""
    if individual is None:
        return {}
    try:
        if hasattr(individual, '__dict__'):
            return {
                "name": getattr(individual, "name", str(individual)),
                "fitness": getattr(individual, "fitness", None),
                "n_parts": (
                    len(individual.graph.nodes)
                    if hasattr(individual, 'graph') else 0
                ),
            }
    except Exception:
        pass
    return {"name": str(individual)}
