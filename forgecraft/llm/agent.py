"""设计推理引擎 — LLM 智能体核心

Plan → Observe → Retrieve → Reason → Decide → Execute → Reflect

核心循环:
  1. 接收设计任务或当前进化状态
  2. 从知识库检索相似案例 + 物理启发式规则
  3. LLM 推理: 生成 2-3 个候选方案
  4. Function Calling 执行: 直接调用工具配置/调整
  5. 观察结果 → 反思 → 更新知识库

自主性层级:
  Level 0 - 建议模式: 输出方案, 等人类确认
  Level 1 - 安全自主: 自动调 mutation_rate ±20%, 切换 emitter
  Level 2 - 半自主: 自动选择 catalog + 配置, 异常时暂停
  Level 3 - 全自主: 端到端 (需要充分的知识库积累)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from forgecraft.llm.provider import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    ToolDefinition,
    create_provider,
)
from forgecraft.llm.tools import ToolRegistry
from forgecraft.llm.prompts.system_prompt import (
    SYSTEM_PROMPT,
    CATALOG_AGENT_PROMPT,
    STRATEGY_AGENT_PROMPT,
    CRITIQUE_AGENT_PROMPT,
)
from forgecraft.llm.prompts.templates import (
    render_task_analysis,
    render_run_status,
    render_similar_cases,
)
from forgecraft.llm.prompts.few_shot_examples import (
    CATALOG_SELECTION_EXAMPLE,
    STAGNATION_HANDLING_EXAMPLE,
)

from forgecraft.logging import get_logger

_logger = get_logger(__name__)


class AutonomyLevel(Enum):
    ADVISORY = 0    # 仅建议
    SAFE_AUTO = 1   # 安全参数自动
    SEMI_AUTO = 2   # 半自主
    FULL_AUTO = 3   # 全自主


@dataclass
class AgentConfig:
    """智能体配置"""
    autonomy_level: AutonomyLevel = AutonomyLevel.ADVISORY
    provider_type: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_reasoning_rounds: int = 5   # 最大推理轮次 (防止无限循环)
    knowledge_base_dir: str = "data/knowledge"


@dataclass
class AgentContext:
    """智能体上下文 — 每次调用的状态"""
    task_description: str = ""
    task_type: str = ""
    catalog_name: str = ""
    run_id: str = ""
    messages: List[LLMMessage] = field(default_factory=list)
    current_metrics: Dict = field(default_factory=dict)
    similar_cases: List[Dict] = field(default_factory=list)
    heuristics: List[Dict] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    pending_tool_calls: List[Dict] = field(default_factory=list)


class DesignReasoningEngine:
    """设计推理引擎 — LLM 智能体主管

    负责:
      - 任务分析 → 推荐 catalog + 策略
      - 运行时监控 → 检测异常 → 调整参数
      - 失败诊断 → 分析轨迹 → 给出改进建议
      - 知识沉淀 → 将每次交互学习到知识库

    使用方式:
        engine = DesignReasoningEngine(kb, provider)
        result = engine.analyze_task("设计一个快速四足机器人")
        # result.recommendation.catalog = "biped"
        # result.recommendation.strategy = "map_elites"
    """

    def __init__(
        self,
        knowledge_base: "KnowledgeBase",  # noqa: F821
        config: Optional[AgentConfig] = None,
        provider: Optional[BaseLLMProvider] = None,
    ):
        self.kb = knowledge_base
        self.config = config or AgentConfig()

        # LLM Provider
        if provider is not None:
            self.provider = provider
        else:
            self.provider = create_provider(
                provider_type=self.config.provider_type,
                model=self.config.model,
            )

        # 工具注册
        self.tools = ToolRegistry()
        self._bind_tool_handlers()

        # 上下文
        self.context = AgentContext()
        self._conversation_history: List[Dict] = []

        _logger.info(
            f"DesignReasoningEngine initialized: "
            f"provider={self.config.provider_type}, model={self.config.model}"
        )

    def _bind_tool_handlers(self):
        """将工具定义绑定到实际执行函数"""
        self.tools.set_handlers({
            "query_similar_cases": self._handle_query_similar_cases,
            "select_catalog": self._handle_select_catalog,
            "configure_evolution": self._handle_configure_evolution,
            "adjust_strategy": self._handle_adjust_strategy,
            "diagnose_failure": self._handle_diagnose_failure,
            "propose_catalog": self._handle_propose_catalog,
            "query_heuristics": self._handle_query_heuristics,
            "get_kb_stats": self._handle_get_kb_stats,
        })

    # ══════════════════════════════════════════════════════════
    #  高层 API
    # ══════════════════════════════════════════════════════════

    def analyze_task(
        self,
        task_description: str,
        task_type: Optional[str] = None,
        constraints: Optional[Dict] = None,
    ) -> Dict:
        """分析设计任务 → 推荐方案

        Args:
            task_description: 自然语言任务描述
            task_type: 任务类型 (自动推断如不指定)
            constraints: 物理约束

        Returns:
            {
                "task_type": str,
                "recommended_catalog": str,
                "recommended_strategy": str,
                "evolution_config": Dict,
                "reasoning": str,
                "confidence": float,
                "similar_cases": [...],
            }
        """
        # 推断任务类型
        if task_type is None:
            task_type = self._infer_task_type(task_description)

        self.context.task_description = task_description
        self.context.task_type = task_type
        self.context.messages = []

        # 1. 知识库检索
        similar_cases = self.kb.find_similar_cases(task_description, top_k=5)
        heuristics = self.kb.query_heuristics(task_type)
        strategy_template = self.kb.query_strategy_template(task_type)
        kb_stats = self.kb.get_statistics()

        # 2. 构建上下文
        task_ctx = render_task_analysis(
            task_description, kb_stats, heuristics, strategy_template,
        )
        cases_ctx = render_similar_cases([
            {"task_name": c.task_name, "catalog_name": c.catalog_name,
             "qd_score_final": c.qd_score_final, "best_fitness": c.best_fitness,
             "objectives": c.objectives}
            for c, _ in similar_cases
        ])

        # 3. LLM 推理
        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="system", content=CATALOG_AGENT_PROMPT.format(
                available_catalogs=self._get_available_catalogs(),
                part_ontology_summary=self._get_part_ontology_summary(),
            )),
            LLMMessage(role="user", content=CATALOG_SELECTION_EXAMPLE),
            LLMMessage(role="user", content=task_ctx),
            LLMMessage(role="user", content=cases_ctx),
            LLMMessage(
                role="user",
                content=(
                    f"请为以下任务推荐最佳方案:\n\n"
                    f"任务描述: {task_description}\n"
                    f"任务类型: {task_type}\n"
                    f"约束: {json.dumps(constraints) if constraints else '无特殊约束'}\n\n"
                    f"请使用 select_catalog 和 configure_evolution 工具给出具体配置。"
                ),
            ),
        ]

        response = self._reason(messages, tool_choice="auto")

        # 4. 解析结果
        result = {
            "task_type": task_type,
            "recommended_catalog": "",
            "recommended_strategy": "",
            "evolution_config": {},
            "reasoning": response.content,
            "confidence": 0.0,
            "similar_cases": [
                {"name": c.task_name, "qd_score": c.qd_score_final}
                for c, _ in similar_cases
            ],
        }

        # 从工具调用提取具体配置
        if response.tool_calls:
            for tc in response.tool_calls:
                if tc["function"]["name"] == "select_catalog":
                    args = json.loads(tc["function"]["arguments"])
                    result["recommended_catalog"] = args.get("catalog_name", "")
                elif tc["function"]["name"] == "configure_evolution":
                    args = json.loads(tc["function"]["arguments"])
                    result["recommended_strategy"] = args.get("strategy", "")
                    result["evolution_config"] = args

        # 回退: 使用知识库模板
        if not result["recommended_catalog"]:
            result["recommended_strategy"] = strategy_template.get(
                "recommended_strategy", "map_elites"
            )
            result["evolution_config"] = strategy_template.get(
                "evolution_config", {}
            )

        return result

    def monitor_run(
        self,
        metrics: Dict,
        run_id: str = "",
    ) -> Optional[Dict]:
        """监控进化运行 → 检测异常 → 给出调整建议

        Args:
            metrics: 当前代指标
            run_id: 运行 ID

        Returns:
            {"action": "adjust", "params": {...}, "reason": "..."} or None
        """
        self.context.current_metrics = metrics
        self.context.run_id = run_id

        # 检测停滞
        stagnation = self._detect_stagnation(metrics)
        if stagnation is None:
            return None  # 一切正常, 不需要调整

        # 知识库查询: 这个失败模式之前怎么处理的?
        kb_suggestions = None
        if self.kb:
            try:
                pattern_name = stagnation.get("pattern", "")
                kb_suggestions = self.kb.relational_kb.get_best_recovery_action(
                    pattern_name
                )
            except Exception:
                pass

        # 构建 LLM 上下文
        status_ctx = render_run_status(metrics)
        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="system", content=STRATEGY_AGENT_PROMPT.format(
                current_metrics=json.dumps(metrics, indent=2, default=str),
                trajectory_summary=self._get_trajectory_summary(),
            )),
            LLMMessage(role="user", content=STAGNATION_HANDLING_EXAMPLE),
            LLMMessage(role="user", content=status_ctx),
            LLMMessage(
                role="user",
                content=(
                    f"检测到停滞信号: {stagnation['pattern']}, "
                    f"已持续 {stagnation.get('duration', 0)} 代。\n"
                    f"知识库建议: {kb_suggestions or '无历史记录'}\n"
                    f"请使用 adjust_strategy 工具给出具体调整方案。"
                ),
            ),
        ]

        response = self._reason(messages, tool_choice="auto")

        # 解析调整方案
        if response.tool_calls:
            for tc in response.tool_calls:
                if tc["function"]["name"] == "adjust_strategy":
                    args = json.loads(tc["function"]["arguments"])
                    return {
                        "action": "adjust",
                        "params": args.get("action", {}),
                        "reason": args.get("reason", ""),
                        "reasoning": response.content,
                    }

        # LLM 未调用工具 → 回退到规则推荐
        return {
            "action": "adjust",
            "params": stagnation.get("suggested_params", {}),
            "reason": stagnation.get("description", "Auto-detected stagnation"),
            "reasoning": response.content,
        }

    def diagnose_run(self, run_id: str) -> Dict:
        """分析已完成运行 → 失败原因 + 改进建议

        Returns:
            {
                "success": bool,
                "failure_patterns": [...],
                "recommendations": [...],
                "summary": str,
            }
        """
        trajectory = self.kb.get_trajectory(run_id) if self.kb else []

        # 识别失败模式
        failure_patterns = self._identify_failure_patterns(trajectory)

        # 构建上下文
        trajectory_ctx = self._format_trajectory(trajectory)

        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="system", content=CRITIQUE_AGENT_PROMPT.format(
                run_summary=trajectory_ctx,
            )),
            LLMMessage(
                role="user",
                content=(
                    f"请分析以下进化运行的成败:\n\n{trajectory_ctx}\n\n"
                    f"自动检测的失败模式: {json.dumps(failure_patterns, indent=2)}\n"
                    f"请使用 diagnose_failure 工具分析并给出建议。"
                ),
            ),
        ]

        response = self._reason(messages)

        return {
            "success": len(failure_patterns) == 0,
            "failure_patterns": failure_patterns,
            "recommendations": self._extract_recommendations(response.content),
            "summary": response.content,
        }

    # ══════════════════════════════════════════════════════════
    #  工具处理器 (Function Calling 执行端)
    # ══════════════════════════════════════════════════════════

    def _handle_query_similar_cases(self, **kwargs) -> str:
        query_text = kwargs.get("query_text", "")
        top_k = kwargs.get("top_k", 5)
        task_filter = kwargs.get("task_filter")
        min_qd_score = kwargs.get("min_qd_score")

        results = self.kb.find_similar_cases(
            query_text, top_k=top_k,
            task_filter=task_filter,
            min_qd_score=min_qd_score,
        )

        if not results:
            return "No similar cases found."

        lines = []
        for case, score in results:
            obj_str = ", ".join(
                f"{k}={v:.2f}" for k, v in case.objectives.items()
            )
            lines.append(
                f"- [{case.task_name}] catalog={case.catalog_name} "
                f"qd_score={case.qd_score_final:.3f} "
                f"fitness={case.best_fitness:.3f} "
                f"objectives={{{obj_str}}} "
                f"similarity={score:.3f}"
            )
        return "\n".join(lines)

    def _handle_select_catalog(self, **kwargs) -> str:
        task_type = kwargs.get("task_type", "")
        strategy = self.kb.query_strategy_template(task_type)

        available = self._get_available_catalogs()

        # 按任务推荐
        task_catalog_map = {
            "speed": "default",
            "climbing": "gripper",
            "manipulation": "gripper",
            "efficiency": "default",
            "structure": "default",
        }

        recommended = task_catalog_map.get(task_type, "default")
        reason = strategy.get("reason", "Default recommendation")

        return json.dumps({
            "catalog_name": recommended,
            "available_catalogs": available,
            "confidence": 0.7,
            "reason": reason,
        })

    def _handle_configure_evolution(self, **kwargs) -> str:
        task_type = kwargs.get("task_type", "default")
        strategy = self.kb.query_strategy_template(task_type)

        if strategy is None:
            strategy = self.kb.query_strategy_template("default")

        config = {
            "strategy": strategy["recommended_strategy"],
            "reason": strategy["reason"],
        }
        config.update(strategy.get("evolution_config", {}))

        return json.dumps(config, indent=2)

    def _handle_adjust_strategy(self, **kwargs) -> str:
        action = kwargs.get("action", {})
        reason = kwargs.get("reason", "")

        # 根据自主性级别决定是否自动执行
        if self.config.autonomy_level.value >= AutonomyLevel.SAFE_AUTO.value:
            # 自动执行
            self._apply_strategy_adjustment(action)
            result = "applied"
        else:
            result = "pending_confirmation"

        return json.dumps({
            "status": result,
            "action": action,
            "reason": reason,
            "autonomy_level": self.config.autonomy_level.name,
        })

    def _handle_diagnose_failure(self, **kwargs) -> str:
        run_id = kwargs.get("run_id", "")

        trajectory = []
        if self.kb and run_id:
            trajectory = self.kb.get_trajectory(run_id)

        patterns = self._identify_failure_patterns(trajectory)
        stats = self.kb.get_failure_stats() if self.kb else []

        return json.dumps({
            "detected_patterns": patterns,
            "global_failure_stats": stats,
            "recommendation": (
                "Review trajectory for specific failure modes" if not patterns
                else f"Found {len(patterns)} failure patterns"
            ),
        }, indent=2)

    def _handle_propose_catalog(self, **kwargs) -> str:
        """LLM 已生成 catalog 描述, 这里确认并保存"""
        task_name = kwargs.get("task_name", "custom")
        parts = kwargs.get("required_functions", [])

        return json.dumps({
            "status": "proposed",
            "task_name": task_name,
            "functions": parts,
            "message": (
                f"Catalog proposal for '{task_name}' received. "
                f"Requires {len(parts)} functional groups. "
                f"Catalog will be saved to configs/catalogs/{task_name}.yaml"
            ),
        })

    def _handle_query_heuristics(self, **kwargs) -> str:
        task_type = kwargs.get("task_type", "")
        rules = self.kb.query_heuristics(task_type)
        return json.dumps(rules, indent=2, ensure_ascii=False)

    def _handle_get_kb_stats(self, **kwargs) -> str:
        stats = self.kb.get_statistics()
        return json.dumps(stats, indent=2)

    # ══════════════════════════════════════════════════════════
    #  内部方法
    # ══════════════════════════════════════════════════════════

    def _reason(
        self,
        messages: List[LLMMessage],
        tool_choice: str = "auto",
        max_rounds: Optional[int] = None,
    ) -> LLMResponse:
        """执行 LLM 推理, 支持多轮 function calling

        循环:
          1. 发送 messages + tools → LLM
          2. 如果 LLM 返回 tool_calls → 执行 tools → 将结果追加到 messages
          3. 重复直到无 tool_calls 或达到 max_rounds
        """
        max_rounds = max_rounds or self.config.max_reasoning_rounds

        tools = self.tools.list_all() if self.provider.supports_tools() else None

        for round_idx in range(max_rounds):
            response = self.provider.chat(
                messages=messages,
                tools=tools,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            # 无 tool_calls → 推理完成
            if not response.tool_calls or response.finish_reason == "stop":
                return response

            # 执行 tool_calls
            messages.append(LLMMessage(
                role="assistant",
                content=response.content or "",
                tool_calls=response.tool_calls,
            ))

            for tc in response.tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}

                tool = self.tools.get(fn_name)
                if tool and tool.handler:
                    result = tool.handler(**fn_args)
                else:
                    result = f"Tool '{fn_name}' not found or has no handler."

                messages.append(LLMMessage(
                    role="tool",
                    content=str(result)[:4000],  # 截断过长结果
                    tool_call_id=tc["id"],
                    name=fn_name,
                ))

                self.context.reasoning_trace.append(
                    f"[Tool] {fn_name}({fn_args}) → {str(result)[:200]}"
                )

        return response

    def _infer_task_type(self, description: str) -> str:
        """从描述推断任务类型"""
        desc_lower = description.lower()

        if any(w in desc_lower for w in ["速度", "speed", "快", "fast", "冲刺"]):
            return "speed"
        if any(w in desc_lower for w in ["攀爬", "爬", "climb", "爬升"]):
            return "climbing"
        if any(w in desc_lower for w in ["抓取", "夹", "grip", "操作", "manipulate"]):
            return "manipulation"
        if any(w in desc_lower for w in ["效率", "节能", "efficien", "省电"]):
            return "efficiency"
        if any(w in desc_lower for w in ["结构", "承重", "structur", "支撑"]):
            return "structure"

        return "speed"  # default

    def _detect_stagnation(self, metrics: Dict) -> Optional[Dict]:
        """基于规则检测停滞"""
        stagnation_duration = metrics.get("stagnation_duration", 0)
        coverage = metrics.get("coverage", 0)
        diversity = metrics.get("population_diversity", 0.5)
        generation = metrics.get("generation", 0)

        if stagnation_duration >= 20:
            return {
                "pattern": "coverage_stall",
                "duration": stagnation_duration,
                "description": "QD-score 停滞超过 20 代",
                "suggested_params": {"mutation_rate_multiplier": 1.5},
            }

        if diversity < 0.1 and generation > 20:
            return {
                "pattern": "diversity_collapse",
                "duration": 0,
                "description": "种群多样性过低",
                "suggested_params": {"random_emitter_ratio": 0.5},
            }

        if generation >= 50 and stagnation_duration >= 20:
            return {
                "pattern": "early_convergence",
                "duration": stagnation_duration,
                "description": "早期收敛, 可能陷入局部最优",
                "suggested_params": {"switch_strategy": "cma_me"},
            }

        if generation == 30 and coverage == 0:
            return {
                "pattern": "slow_start",
                "duration": 0,
                "description": "前 30 代覆盖率为 0",
                "suggested_params": {},
            }

        return None

    def _identify_failure_patterns(self, trajectory: List[Dict]) -> List[Dict]:
        """从进化轨迹识别失败模式"""
        if not trajectory:
            return []

        patterns = []

        # 检查最终 QD-score
        final = trajectory[-1] if trajectory else {}
        if final.get("qd_score", 0) < 0.1:
            patterns.append({
                "pattern": "poor_convergence",
                "evidence": f"Final QD-score: {final.get('qd_score', 0):.4f}",
            })

        # 检查是否有 QD-score 下降
        scores = [g.get("qd_score", 0) for g in trajectory if g.get("qd_score")]
        if len(scores) >= 20:
            first_half = sum(scores[:len(scores)//2]) / max(len(scores)//2, 1)
            second_half = sum(scores[len(scores)//2:]) / max(len(scores) - len(scores)//2, 1)
            if second_half < first_half * 0.9:
                patterns.append({
                    "pattern": "performance_degradation",
                    "evidence": f"QD-score dropped from {first_half:.4f} to {second_half:.4f}",
                })

        return patterns

    def _apply_strategy_adjustment(self, action: Dict):
        """实际执行策略调整 (通过回调或直接修改)"""
        _logger.info(f"Applying strategy adjustment: {action}")

        # 记录参数变化到知识库
        if self.kb and self.context.run_id:
            for param, value in action.items():
                if isinstance(value, (int, float)):
                    self.kb.record_param_change(
                        self.context.run_id,
                        self.context.current_metrics.get("generation", 0),
                        param, 0, value,
                        reason="llm_auto_adjust",
                    )

    def _get_available_catalogs(self) -> str:
        """获取可用目录列表"""
        import os
        catalog_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "forgecraft", "configs", "catalogs",
        )
        if os.path.isdir(catalog_dir):
            files = [f.replace(".yaml", "") for f in os.listdir(catalog_dir) if f.endswith(".yaml")]
            return ", ".join(files[:15])
        return "default, biped, hexapod, gripper, hoppers"

    def _get_part_ontology_summary(self) -> str:
        """零件本体摘要"""
        from forgecraft.knowledge.ontology import PART_ONTOLOGY
        groups = PART_ONTOLOGY.get("functional_groups", {})
        return "\n".join(
            f"- {name}: {data['description']} ({len(data['parts'])} parts)"
            for name, data in groups.items()
        )

    def _get_trajectory_summary(self) -> str:
        """轨迹摘要"""
        if not self.context.run_id or not self.kb:
            return "(无轨迹数据)"
        try:
            traj = self.kb.get_trajectory(self.context.run_id)
            if not traj:
                return "(无轨迹数据)"
            first = traj[0]
            last = traj[-1]
            return (
                f"从 gen {first.get('generation', 0)} 到 gen {last.get('generation', 0)}, "
                f"QD-score: {first.get('qd_score', 0):.4f} → {last.get('qd_score', 0):.4f}, "
                f"覆盖率: {first.get('coverage', 0):.2%} → {last.get('coverage', 0):.2%}"
            )
        except Exception:
            return "(无法获取轨迹)"

    def _format_trajectory(self, trajectory: List[Dict]) -> str:
        """格式化进化轨迹为文本"""
        if not trajectory:
            return "(无轨迹数据)"

        lines = ["| Gen | QD-Score | Coverage | Best Fitness | Stagnation |"]
        lines.append("|-----|----------|----------|-------------|------------|")
        step = max(1, len(trajectory) // 20)
        for g in trajectory[::step]:
            lines.append(
                f"| {g.get('generation', 0):4d} "
                f"| {g.get('qd_score', 0):8.4f} "
                f"| {g.get('coverage', 0):8.2%} "
                f"| {g.get('best_fitness', 0):11.4f} "
                f"| {g.get('stagnation_flag', False)} |"
            )
        return "\n".join(lines)

    def _extract_recommendations(self, text: str) -> List[str]:
        """从 LLM 输出中提取建议列表"""
        recs = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                recs.append(line[2:])
            elif line.startswith("1. ") or line.startswith("2. ") or line.startswith("3. "):
                recs.append(line[3:])
        return recs[:5]
