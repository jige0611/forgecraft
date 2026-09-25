"""LLM 观察器 — 非侵入式挂载到进化管线

将 LLM 推理能力注入现有 EvolutionLoop / MAPElitesEngine，
不修改其内部逻辑。

使用方式:
    loop = EvolutionLoop(...)
    observer = LLMObserver(agent, kb, recorder)
    observer.attach_to(loop)
    loop.run()  # observer 自动在每代结束时触发
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from forgecraft.llm.agent import DesignReasoningEngine
from forgecraft.llm.subagents.strategy_agent import StrategyAgent
from forgecraft.logging import get_logger

_logger = get_logger(__name__)


class LLMObserver:
    """LLM 观察器 — 进化管线的事件监听器

    挂载点:
      - on_generation_end: 每代结束 → 检查停滞 → 推荐调整
      - on_run_start:     运行开始 → 记录初始状态
      - on_run_end:       运行结束 → 自动评审 → 更新知识库
    """

    def __init__(
        self,
        agent: DesignReasoningEngine,
        knowledge_base: "KnowledgeBase",  # noqa: F821
        recorder: Optional["KnowledgeRecorder"] = None,  # noqa: F821
        strategy_agent: Optional[StrategyAgent] = None,
        check_interval: int = 5,  # 每隔 N 代检查一次
        auto_adjust: bool = False,  # 是否自动执行调整
    ):
        self.agent = agent
        self.kb = knowledge_base
        self.recorder = recorder
        self.strategy_agent = strategy_agent or StrategyAgent(knowledge_base)
        self.check_interval = check_interval
        self.auto_adjust = auto_adjust

        self._run_id: Optional[str] = None
        self._generation = 0
        self._metrics_history: List[Dict] = []
        self._adjustment_history: List[Dict] = []
        self._last_check_gen = 0

    # ── 挂载 ─────────────────────────────────────────────────

    def attach_to(self, evolution_loop):
        """挂载到 EvolutionLoop (monkey-patch 方式, 非侵入)"""
        original_step = evolution_loop.step
        original_run = evolution_loop.run

        observer = self

        def observed_step(*args, **kwargs):
            observer._on_generation_start()
            result = original_step(*args, **kwargs)
            observer._on_generation_end(result)
            return result

        def observed_run(*args, **kwargs):
            observer.on_run_start()
            result = original_run(*args, **kwargs)
            observer.on_run_end(result)
            return result

        evolution_loop.step = observed_step
        evolution_loop.run = observed_run
        _logger.info("LLMObserver attached to EvolutionLoop")

    def attach_to_map_elites(self, engine):
        """挂载到 MAPElitesEngine"""
        original_step = engine.step

        observer = self

        def observed_step(eval_fn, *args, **kwargs):
            observer._on_generation_start()
            result = original_step(eval_fn, *args, **kwargs)
            observer._on_generation_end(result)
            return result

        engine.step = observed_step
        _logger.info("LLMObserver attached to MAPElitesEngine")

    # ── 事件处理 ─────────────────────────────────────────────

    def on_run_start(self, task_name: str = "", catalog_name: str = ""):
        """运行开始"""
        if self.recorder:
            self._run_id = self.recorder.start_run(
                task_name=task_name,
                catalog_name=catalog_name,
            )
        else:
            import uuid
            self._run_id = f"run_{uuid.uuid4().hex[:8]}"

        self._generation = 0
        self._metrics_history = []
        self._adjustment_history = []
        self._last_check_gen = 0

        _logger.info(f"LLMObserver run started: {self._run_id}")

    def on_run_end(self, result: Any = None):
        """运行结束 — 自动评审"""
        if not self._run_id:
            return

        # 记录最终案例
        if self.recorder:
            self.recorder.finalize(
                behavior_bc=self._metrics_history[-1].get("behavior_bc", [])
                if self._metrics_history else [],
            )

        # 自动评审
        try:
            critique = self.agent.diagnose_run(self._run_id)
            verdict = critique.get("overall_verdict", "unknown")
            n_issues = len(critique.get("failure_patterns", []))
            _logger.info(
                f"Run {self._run_id} completed. "
                f"Verdict: {verdict}, Issues: {n_issues}"
            )
        except Exception as e:
            _logger.warning(f"Auto-critique failed: {e}")

    # ── 内部钩子 ─────────────────────────────────────────────

    def _on_generation_start(self):
        self._generation += 1

    def _on_generation_end(self, step_result: Any):
        """每代结束, 收集指标并选择性触发检查"""
        # 收集指标
        metrics = self._extract_metrics(step_result)
        metrics["generation"] = self._generation
        self._metrics_history.append(metrics)

        # 记录
        if self.recorder and self._run_id:
            self.recorder.record_generation(self._generation, metrics)

        # 是否该检查了?
        if self._generation - self._last_check_gen < self.check_interval:
            return
        self._last_check_gen = self._generation

        # 策略检查
        suggestion = self.strategy_agent.check(
            metrics, run_id=self._run_id or "", use_llm=True,
        )

        if suggestion:
            self._adjustment_history.append({
                "generation": self._generation,
                "suggestion": suggestion,
            })

            _logger.info(
                f"Gen {self._generation}: Strategy suggestion - {suggestion.get('signal')} "
                f"(severity: {suggestion.get('severity')})"
            )

            if self.auto_adjust:
                self._apply_suggestion(suggestion)

    # ── 调整执行 ─────────────────────────────────────────────

    def _apply_suggestion(self, suggestion: Dict):
        """执行策略调整 (通过回调)"""
        params = suggestion.get("params", {})
        _logger.info(f"Auto-applying adjustment: {params}")

        # 记录到知识库
        if self.kb and self._run_id:
            for name, value in params.items():
                if isinstance(value, (int, float)):
                    self.kb.record_param_change(
                        self._run_id, self._generation,
                        name, 0, value,
                        reason=suggestion.get("reason", "auto"),
                    )

    # ── 指标提取 ─────────────────────────────────────────────

    def _extract_metrics(self, step_result: Any) -> Dict:
        """从 step 返回值中提取指标"""
        metrics = {}

        if isinstance(step_result, dict):
            # MAPElitesEngine.step 返回 dict
            metrics = {
                "qd_score": step_result.get("qd_score", 0),
                "coverage": step_result.get("coverage", 0),
                "best_fitness": step_result.get("best_fitness", 0),
                "num_elites": step_result.get("num_elites", 0),
                "population_diversity": step_result.get("diversity", 0.5),
                "active_strategy": step_result.get("strategy", "map_elites"),
                "behavior_bc": step_result.get("behavior_bc", []),
            }
        elif hasattr(step_result, '__dict__'):
            # 可能是个 dataclass
            metrics = {
                "qd_score": getattr(step_result, "qd_score", 0),
                "coverage": getattr(step_result, "coverage", 0),
                "best_fitness": getattr(step_result, "best_fitness", 0),
                "num_elites": getattr(step_result, "num_elites", 0),
                "population_diversity": getattr(step_result, "diversity", 0.5),
                "active_strategy": getattr(step_result, "strategy", "map_elites"),
                "behavior_bc": getattr(step_result, "behavior_bc", []),
            }

        return metrics

    def get_run_summary(self) -> Dict:
        """获取运行摘要"""
        return {
            "run_id": self._run_id,
            "generations": self._generation,
            "adjustments": len(self._adjustment_history),
            "adjustment_history": self._adjustment_history[-10:],
        }
