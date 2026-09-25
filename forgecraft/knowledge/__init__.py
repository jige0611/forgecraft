"""ForgeCraft 知识库 — LLM 智能体的记忆与推理基础

双引擎架构:
  - SemanticKB (向量库):   嵌入相似性检索, 用于启发式匹配
  - RelationalKB (关系库): SQLite 结构化查询, 用于精确过滤与聚合
  - GraphKB (图库):       零件本体图, 用于组合推理与约束传播
  - Ontology (本体):       物理启发式规则 + 零件语义分类

整合入口:
  - KnowledgeBase: 统一接口, 协调三个子引擎
  - KnowledgeRecorder: 自动记录进化执行 → 知识沉淀
"""

from forgecraft.knowledge.embeddings import (
    PartEmbedder,
    TaskEmbedder,
    MorphologyEmbedder,
    PerformanceEmbedder,
    embed_part,
    embed_task,
    embed_morphology,
    embed_performance,
)
from forgecraft.knowledge.vector_store import SemanticKB, DesignCase
from forgecraft.knowledge.relational_store import RelationalKB
from forgecraft.knowledge.graph_store import GraphKB, part_ontology_to_graph
from forgecraft.knowledge.ontology import (
    PART_ONTOLOGY,
    PHYSICS_HEURISTICS,
    STRATEGY_TEMPLATES,
    get_parts_by_function,
    get_heuristics_for_task,
    get_strategy_template,
)
from forgecraft.knowledge.store import KnowledgeBase
from forgecraft.knowledge.recorder import KnowledgeRecorder

__all__ = [
    # Embedders
    "PartEmbedder",
    "TaskEmbedder",
    "MorphologyEmbedder",
    "PerformanceEmbedder",
    "embed_part",
    "embed_task",
    "embed_morphology",
    "embed_performance",
    # Stores
    "SemanticKB",
    "DesignCase",
    "RelationalKB",
    "GraphKB",
    "part_ontology_to_graph",
    # Ontology
    "PART_ONTOLOGY",
    "PHYSICS_HEURISTICS",
    "STRATEGY_TEMPLATES",
    "get_parts_by_function",
    "get_heuristics_for_task",
    "get_strategy_template",
    # Unified
    "KnowledgeBase",
    "KnowledgeRecorder",
]
