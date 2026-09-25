"""工具定义 — Function Calling 工具注册表

每个工具对应一个 LLM 可调用的函数。工具定义包含:
  - name: 工具名称
  - description: 自然语言描述 (LLM 用于理解何时调用)
  - parameters: JSON Schema (定义参数类型和约束)
  - handler: Python 执行函数 (可选, 用于实际执行)
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from forgecraft.llm.provider import ToolDefinition


# ══════════════════════════════════════════════════════════════
#  工具注册表
# ══════════════════════════════════════════════════════════════

class ToolRegistry:
    """工具注册表 — 管理所有可用的 Function Calling 工具"""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_defaults()

    def _register_defaults(self):
        """注册所有默认工具"""
        self.register(TOOL_QUERY_SIMILAR_CASES)
        self.register(TOOL_SELECT_CATALOG)
        self.register(TOOL_CONFIGURE_EVOLUTION)
        self.register(TOOL_ADJUST_STRATEGY)
        self.register(TOOL_DIAGNOSE_FAILURE)
        self.register(TOOL_PROPOSE_CATALOG)
        self.register(TOOL_QUERY_HEURISTICS)
        self.register(TOOL_GET_KB_STATS)

    def register(self, tool: ToolDefinition):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_all(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def set_handler(self, name: str, handler: Callable):
        if name in self._tools:
            self._tools[name].handler = handler

    def set_handlers(self, handler_map: Dict[str, Callable]):
        for name, handler in handler_map.items():
            self.set_handler(name, handler)


# ══════════════════════════════════════════════════════════════
#  工具定义
# ══════════════════════════════════════════════════════════════


TOOL_QUERY_SIMILAR_CASES = ToolDefinition(
    name="query_similar_cases",
    description=(
        "检索与当前设计任务最相似的历史案例。"
        "用于了解什么方案在类似任务上成功/失败了。"
        "返回: 案例列表, 每个包含任务名、目录名、目标值、QD-score 等。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query_text": {
                "type": "string",
                "description": "自然语言描述当前设计任务",
            },
            "top_k": {
                "type": "integer",
                "description": "返回案例数量 (默认 5)",
                "default": 5,
            },
            "task_filter": {
                "type": "string",
                "description": "按任务名过滤 (如 'speed', 'climbing'), 可选",
            },
            "min_qd_score": {
                "type": "number",
                "description": "最低 QD-Score 过滤, 可选",
            },
        },
        "required": ["query_text"],
    },
)

TOOL_SELECT_CATALOG = ToolDefinition(
    name="select_catalog",
    description=(
        "为设计任务选择最合适的零件目录。"
        "分析任务需求 → 匹配目录 → 可选: 生成新目录。"
        "返回: 目录名 + 匹配度 + 推荐理由。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_description": {
                "type": "string",
                "description": "任务的自然语言描述",
            },
            "task_type": {
                "type": "string",
                "description": "任务类型: speed | climbing | manipulation | efficiency | structure",
            },
            "constraints": {
                "type": "object",
                "description": "物理约束 (如 max_parts, max_depth, terrain_type)",
                "properties": {
                    "max_parts": {"type": "integer"},
                    "max_depth": {"type": "integer"},
                    "terrain_type": {"type": "string"},
                    "prefer_lightweight": {"type": "boolean"},
                    "require_symmetry": {"type": "boolean"},
                },
            },
        },
        "required": ["task_description", "task_type"],
    },
)

TOOL_CONFIGURE_EVOLUTION = ToolDefinition(
    name="configure_evolution",
    description=(
        "生成进化算法的完整配置。"
        "根据任务类型和所选目录, 推荐最优的进化策略和超参数。"
        "返回: EvolutionConfig 字典。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_type": {
                "type": "string",
                "description": "任务类型",
            },
            "catalog_name": {
                "type": "string",
                "description": "使用的零件目录名",
            },
            "strategy": {
                "type": "string",
                "description": "进化策略: map_elites | nsga3 | cma_me | hierarchical",
            },
            "population_size": {
                "type": "integer",
                "description": "种群规模 (默认根据任务推荐)",
            },
            "total_generations": {
                "type": "integer",
                "description": "总进化代数 (默认 200-400)",
            },
        },
        "required": ["task_type", "catalog_name"],
    },
)

TOOL_ADJUST_STRATEGY = ToolDefinition(
    name="adjust_strategy",
    description=(
        "运行时调整进化策略。用于应对停滞、多样性崩溃等异常。"
        "只能调整安全参数 (mutation rate, emitter 类型等)。"
        "返回: 调整后的新参数 + 期望效果。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "当前运行的 ID",
            },
            "detected_signal": {
                "type": "string",
                "description": "检测到的异常信号: coverage_stall | diversity_collapse | early_convergence | slow_start",
            },
            "action": {
                "type": "object",
                "description": "调整动作",
                "properties": {
                    "mutation_rate_multiplier": {
                        "type": "number",
                        "description": "变异率乘法系数 (如 1.5 = 增加 50%)",
                    },
                    "switch_emitter_type": {
                        "type": "string",
                        "description": "切换发射器类型: improvement | random | optimizer | adaptive",
                    },
                    "random_emitter_ratio": {
                        "type": "number",
                        "description": "随机发射器占比 (0.0-1.0)",
                    },
                    "inject_diversity": {
                        "type": "boolean",
                        "description": "是否注入随机新个体",
                    },
                    "switch_strategy": {
                        "type": "string",
                        "description": "切换进化策略: cma_me | nsga3 | map_elites",
                    },
                },
            },
            "reason": {
                "type": "string",
                "description": "调整理由",
            },
        },
        "required": ["run_id", "detected_signal", "action", "reason"],
    },
)

TOOL_DIAGNOSE_FAILURE = ToolDefinition(
    name="diagnose_failure",
    description=(
        "分析进化失败的原因。"
        "检查进化轨迹, 匹配已知失败模式, 给出可操作的修复建议。"
        "返回: 失败模式 + 证据 + 建议修复 + 历史成功率。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "运行 ID",
            },
            "failure_description": {
                "type": "string",
                "description": "失败的描述 (可选, 用于辅助诊断)",
            },
        },
        "required": ["run_id"],
    },
)

TOOL_PROPOSE_CATALOG = ToolDefinition(
    name="propose_catalog",
    description=(
        "为全新任务类型生成零件目录 YAML 配置。"
        "当现有目录都不适合时使用。"
        "返回: YAML 格式的目录定义。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_description": {
                "type": "string",
                "description": "新任务的详细自然语言描述",
            },
            "task_name": {
                "type": "string",
                "description": "任务名称 (蛇形命名)",
            },
            "num_parts": {
                "type": "integer",
                "description": "期望零件种类数 (4-10 推荐)",
            },
            "physical_constraints": {
                "type": "object",
                "description": "物理约束 (环境, 尺寸, 质量限制等)",
            },
            "required_functions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "必需的功能: structural, actuation, locomotion, manipulation, energy, sensing",
            },
        },
        "required": ["task_description", "task_name", "required_functions"],
    },
)

TOOL_QUERY_HEURISTICS = ToolDefinition(
    name="query_heuristics",
    description=(
        "查询任务相关的物理启发式规则。"
        "返回: 适用规则列表。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_type": {
                "type": "string",
                "description": "任务类型",
            },
        },
        "required": ["task_type"],
    },
)

TOOL_GET_KB_STATS = ToolDefinition(
    name="get_kb_stats",
    description=(
        "获取知识库的统计信息。"
        "返回: 案例总数、各任务分布、平均性能等。"
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
)
