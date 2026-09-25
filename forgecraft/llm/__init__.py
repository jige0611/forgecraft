"""ForgeCraft LLM 智能体 — 自主设计决策引擎

架构:
  - DesignReasoningEngine (核心): Plan → Retrieve → Reason → Decide → Execute
  - CatalogAgent:           任务 → 零件目录选择/生成
  - StrategyAgent:          运行时监控 → 异常检测 → 参数调整
  - CritiqueAgent:          事后分析 → 失败诊断 → 改进建议
  - TradeoffAgent:          Pareto 前沿分析 → 多目标权衡推荐

集成:
  - LLMObserver:  非侵入式挂载到 EvolutionLoop
  - LLMActuator:  将 LLM 决策转化为配置修改

Provider 支持:
  - OpenAI (gpt-4o, gpt-4o-mini)
  - Anthropic (claude-3.5-sonnet)
  - Ollama (本地 llama3, qwen2.5)
  - Mock (测试)
"""

from forgecraft.llm.provider import (
    BaseLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    MockProvider,
    create_provider,
    LLMMessage,
    LLMResponse,
    ToolDefinition,
    ProviderType,
)
from forgecraft.llm.tools import ToolRegistry
from forgecraft.llm.agent import (
    DesignReasoningEngine,
    AgentConfig,
    AgentContext,
    AutonomyLevel,
)
from forgecraft.llm.subagents.catalog_agent import CatalogAgent
from forgecraft.llm.subagents.strategy_agent import StrategyAgent
from forgecraft.llm.subagents.critique_agent import CritiqueAgent
from forgecraft.llm.subagents.tradeoff_agent import TradeoffAgent
from forgecraft.llm.observer import LLMObserver
from forgecraft.llm.actuator import LLMActuator

__all__ = [
    # Provider
    "BaseLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "MockProvider",
    "create_provider",
    "LLMMessage",
    "LLMResponse",
    "ToolDefinition",
    "ProviderType",
    # Tools
    "ToolRegistry",
    # Agent
    "DesignReasoningEngine",
    "AgentConfig",
    "AgentContext",
    "AutonomyLevel",
    # Sub-agents
    "CatalogAgent",
    "StrategyAgent",
    "CritiqueAgent",
    "TradeoffAgent",
    # Integration
    "LLMObserver",
    "LLMActuator",
]
