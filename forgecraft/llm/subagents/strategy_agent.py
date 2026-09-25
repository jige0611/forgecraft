"""策略智能体 — 运行时监控 & 自动调整

职责:
  - 每代或每隔 N 代检查进化状态
  - 检测停滞/退化/多样性崩溃
  - 推荐并执行参数调整 (在安全范围内)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from forgecraft.llm.provider import BaseLLMProvider, LLMMessage, create_provider
from forgecraft.llm.prompts.system_prompt import STRATEGY_AGENT_PROMPT
from forgecraft.llm.prompts.few_shot_examples import STAGNATION_HANDLING_EXAMPLE
from forgecraft.llm.prompts.templates import render_run_status
from forgecraft.logging import get_logger

_logger = get_logger(__name__)


class StrategyAgent:
    """策略智能体 — 进化运行时监控与调整

    使用方式:
        agent = StrategyAgent(kb, provider)
        suggestion = agent.check(metrics)
        if suggestion:
            agent.apply(suggestion)  # 需要 actuator 支持
    """

    # 安全调整范围 (Level 1 自动执行的参数)
    SAFE_PARAM_RANGES = {
        "mutation_rate": (0.05, 0.80),
        "crossover_rate": (0.10, 0.90),
        "topo_mutation_prob": (0.02, 0.50),
        "param_mutation_prob": (0.05, 0.60),
        "param_mutation_scale": (0.01, 0.50),
        "selection_pressure": (1.0, 5.0),
    }

    def __init__(
        self,
        knowledge_base: "KnowledgeBase",  # noqa: F821
        provider: Optional[BaseLLMProvider] = None,
    ):
        self.kb = knowledge_base
        self.provider = provider or create_provider("mock")
        self._history: List[Dict] = []
        self._stagnation_counter = 0
        self._last_qd_score = 0.0

    def check(
        self,
        metrics: Dict,
        run_id: str = "",
        use_llm: bool = True,
    ) -> Optional[Dict]:
        """检查进化状态, 返回调整方案 (或 None)

        Args:
            metrics: 当前代指标
            run_id: 运行 ID
            use_llm: 是否使用 LLM 推理 (False = 仅规则)

        Returns:
            {"action": "adjust", "params": {...}, "reason": "..."} or None
        """
        # 更新停滞计数
        qd = metrics.get("qd_score", 0.0)
        generation = metrics.get("generation", 0)

        if generation > 0 and qd > 0:
            if qd <= self._last_qd_score * 1.01:
                self._stagnation_counter += 1
            else:
                self._stagnation_counter = 0
        self._last_qd_score = qd

        metrics["stagnation_duration"] = self._stagnation_counter

        # 规则检测
        rule_result = self._rule_based_check(metrics)
        if rule_result is None:
            return None

        if not use_llm:
            return rule_result

        # LLM 增强推理
        return self._llm_enhanced_check(metrics, rule_result, run_id)

    def _rule_based_check(self, metrics: Dict) -> Optional[Dict]:
        """纯规则检测"""
        stagnation = metrics.get("stagnation_duration", 0)
        coverage = metrics.get("coverage", 0.0)
        diversity = metrics.get("population_diversity", 0.5)
        generation = metrics.get("generation", 0)

        # ST-01: coverage_stall
        if stagnation >= 20:
            return {
                "action": "adjust",
                "signal": "coverage_stall",
                "params": {"mutation_rate_multiplier": 1.5},
                "reason": f"QD-score stagnated for {stagnation} generations",
                "severity": "high" if stagnation >= 30 else "medium",
            }

        # ST-02: diversity_collapse
        if diversity < 0.1 and generation > 20:
            return {
                "action": "adjust",
                "signal": "diversity_collapse",
                "params": {
                    "random_emitter_ratio": 0.5,
                    "mutation_rate_multiplier": 1.3,
                },
                "reason": f"Population diversity critically low ({diversity:.3f})",
                "severity": "high",
            }

        # ST-03: early_convergence
        if generation >= 50 and stagnation >= 20:
            return {
                "action": "adjust",
                "signal": "early_convergence",
                "params": {"switch_strategy": "cma_me"},
                "reason": "Early convergence detected, switching to CMA-ME for local refinement",
                "severity": "medium",
            }

        # ST-04: slow_start
        if generation == 30 and coverage < 0.02:
            return {
                "action": "adjust",
                "signal": "slow_start",
                "params": {
                    "mutation_rate_multiplier": 2.0,
                    "random_emitter_ratio": 0.7,
                },
                "reason": "No coverage after 30 generations, boosting exploration",
                "severity": "high",
            }

        return None

    def _llm_enhanced_check(
        self, metrics: Dict, rule_result: Dict, run_id: str,
    ) -> Optional[Dict]:
        """LLM 增强推理"""
        # 查询知识库: 这个信号的历史修复成功率
        kb_actions = None
        try:
            kb_actions = self.kb.relational_kb.get_best_recovery_action(
                rule_result["signal"]
            )
        except Exception:
            pass

        status_ctx = render_run_status(metrics)
        messages = [
            LLMMessage(role="system", content=STRATEGY_AGENT_PROMPT.format(
                current_metrics=str(metrics),
                trajectory_summary=self._get_trajectory_summary(run_id),
            )),
            LLMMessage(role="user", content=STAGNATION_HANDLING_EXAMPLE),
            LLMMessage(role="user", content=status_ctx),
            LLMMessage(
                role="user",
                content=(
                    f"检测到: {rule_result['signal']} (严重程度: {rule_result['severity']})\n"
                    f"规则建议: {rule_result['params']}\n"
                    f"知识库最有效修复: {kb_actions or '无历史数据'}\n\n"
                    f"请确认或优化调整方案。仅输出具体参数, "
                    f"确保在安全范围: {self.SAFE_PARAM_RANGES}"
                ),
            ),
        ]

        try:
            response = self.provider.chat(messages, temperature=0.5)
            # 提取 LLM 建议的参数
            llm_params = self._extract_params(response.content)
            if llm_params:
                rule_result["params"] = {
                    **rule_result["params"],
                    **llm_params,
                }
                rule_result["llm_reasoning"] = response.content[:500]
        except Exception as e:
            _logger.warning(f"LLM strategy check failed, using rules: {e}")

        return rule_result

    def get_history(self) -> List[Dict]:
        return self._history

    def get_statistics(self) -> Dict:
        """统计调整历史"""
        if not self._history:
            return {"total_adjustments": 0, "most_common_signal": "none"}

        signals = {}
        for h in self._history:
            sig = h.get("signal", "unknown")
            signals[sig] = signals.get(sig, 0) + 1

        return {
            "total_adjustments": len(self._history),
            "most_common_signal": max(signals, key=signals.get),
            "signals": signals,
        }

    # ── 内部 ──

    def _get_trajectory_summary(self, run_id: str) -> str:
        if not run_id or not self.kb:
            return "(无轨迹数据)"
        try:
            traj = self.kb.get_trajectory(run_id)
            if not traj or len(traj) < 2:
                return "(无轨迹数据)"
            first, last = traj[0], traj[-1]
            return (
                f"Gen {first.get('generation', 0)}→{last.get('generation', 0)}, "
                f"QD: {first.get('qd_score', 0):.4f}→{last.get('qd_score', 0):.4f}, "
                f"Coverage: {first.get('coverage', 0):.2%}→{last.get('coverage', 0):.2%}"
            )
        except Exception:
            return "(无法获取轨迹)"

    def _extract_params(self, text: str) -> Optional[Dict]:
        """从 LLM 输出中提取参数建议"""
        params = {}
        for line in text.split("\n"):
            line = line.strip()
            for param in self.SAFE_PARAM_RANGES:
                if param in line and ":" in line:
                    try:
                        val = float(line.split(":")[-1].strip().split()[0].rstrip(","))
                        lo, hi = self.SAFE_PARAM_RANGES[param]
                        if lo <= val <= hi:
                            params[param] = val
                    except (ValueError, IndexError):
                        pass
        return params if params else None
