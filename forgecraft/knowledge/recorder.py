"""知识记录器 — 自动从进化执行中提取并沉淀知识

用途:
  - 挂载到 EvolutionLoop / MAPElitesEngine, 自动记录每代状态
  - 运行结束时自动生成设计案例
  - 检测失败模式并记录修复历史
  - 异步写入 (不影响进化性能)

使用:
    recorder = KnowledgeRecorder(kb)
    recorder.attach_to(evolution_loop)
    # ... 进化运行 ...
    recorder.finalize(case_id, final_population)
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from forgecraft.logging import get_logger

_logger = get_logger(__name__)


class KnowledgeRecorder:
    """知识自动记录器

    非侵入式观察者: 通过钩子函数收集数据, 不修改进化循环内部逻辑。
    """

    def __init__(
        self,
        knowledge_base: "KnowledgeBase",  # noqa: F821
        autosave_every_n_gens: int = 50,
    ):
        self.kb = knowledge_base
        self.autosave_every_n_gens = autosave_every_n_gens

        # 运行时缓冲
        self._current_case_id: Optional[str] = None
        self._start_time: Optional[float] = None
        self._generation_counter: int = 0
        self._generation_buffer: List[Dict] = []
        self._stagnation_counter: int = 0
        self._last_qd_score: float = 0.0

        # 回调注册
        self._on_generation_end: Optional[Callable] = None
        self._on_run_start: Optional[Callable] = None
        self._on_run_end: Optional[Callable] = None

    # ── 生命周期 ────────────────────────────────────────────

    def start_run(
        self,
        task_name: str,
        catalog_name: str,
        evolution_config: Optional[Dict] = None,
        run_id: str = "",
    ) -> str:
        """开始一次进化运行

        Returns:
            case_id
        """
        import uuid

        self._current_case_id = f"run_{uuid.uuid4().hex[:8]}"
        self._start_time = time.time()
        self._generation_counter = 0
        self._generation_buffer = []
        self._stagnation_counter = 0
        self._last_qd_score = 0.0

        # 插入占位 case 记录，确保外键约束满足
        self.kb.relational_kb.insert_design_case({
            "case_id": self._current_case_id,
            "task_name": task_name,
            "catalog_name": catalog_name,
            "evolution_config": evolution_config or {},
            "generation_count": 0,
            "wall_time_seconds": 0.0,
            "qd_score_final": 0.0,
            "coverage_final": 0.0,
            "best_fitness": 0.0,
            "tags": [],
            "notes": "",
            "run_id": self._current_case_id,
        })

        _logger.info(
            f"Recorder started: case={self._current_case_id}, "
            f"task={task_name}, catalog={catalog_name}"
        )

        if self._on_run_start:
            self._on_run_start(self._current_case_id, task_name, catalog_name)

        return self._current_case_id

    def record_generation(self, gen: int, metrics: Dict):
        """记录一代进化

        metrics 应包含:
          - qd_score (if using MAP-Elites)
          - coverage
          - best_fitness
          - population_diversity
          - num_elites
          - emitter_usage: {emitter_type: count}
          - active_strategy
        """
        self._generation_counter = gen

        # 缓存 (批量写入)
        self._generation_buffer.append({
            "generation": gen,
            "metrics": metrics,
        })

        # 检测停滞
        qd = metrics.get("qd_score", 0.0)
        if gen > 0 and qd > 0:
            if qd <= self._last_qd_score * 1.01:
                self._stagnation_counter += 1
            else:
                self._stagnation_counter = 0
        self._last_qd_score = qd

        # 更新停滞标记
        if self._current_case_id:
            self.kb.record_generation(self._current_case_id, gen, {
                **metrics,
                "stagnation_flag": self._stagnation_counter >= 10,
                "stagnation_duration": self._stagnation_counter,
            })

        # 定期持久化
        if gen % self.autosave_every_n_gens == 0 and gen > 0:
            self._flush()

        if self._on_generation_end:
            self._on_generation_end(gen, metrics)

    def finalize(
        self,
        objectives: Optional[Dict[str, float]] = None,
        behavior_bc: Optional[List[float]] = None,
        best_body_graph=None,
        best_individual=None,
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> str:
        """结束运行, 记录完整案例

        Returns:
            case_id
        """
        self._flush()

        if self._current_case_id is None:
            _logger.warning("finalize called without start_run")
            return ""

        wall_time = time.time() - (self._start_time or time.time())

        # 收集最终指标
        final_metrics = self._aggregate_metrics()

        case_id = self.kb.record_case(
            task_name=final_metrics.get("task_name", "unknown"),
            catalog_name=final_metrics.get("catalog_name", "default"),
            objectives=objectives or {},
            behavior_bc=behavior_bc or [],
            generation_count=self._generation_counter,
            wall_time_seconds=wall_time,
            qd_score_final=final_metrics.get("qd_score", 0.0),
            coverage_final=final_metrics.get("coverage", 0.0),
            best_fitness=final_metrics.get("best_fitness", 0.0),
            best_individual=best_individual,
            best_body_graph=best_body_graph,
            tags=tags or [],
            notes=notes,
            run_id=self._current_case_id,
        )

        self.kb.save()

        _logger.info(
            f"Run finalized: case={case_id}, "
            f"gens={self._generation_counter}, time={wall_time:.1f}s"
        )

        if self._on_run_end:
            self._on_run_end(case_id, final_metrics)

        return case_id

    # ── 失败检测 ────────────────────────────────────────────

    def detect_stagnation(self) -> Optional[Dict]:
        """检测当前运行是否停滞

        Returns:
            {"pattern": "coverage_stall", "severity": "high", ...} or None
        """
        if self._stagnation_counter >= 20:
            return {
                "pattern": "coverage_stall",
                "duration": self._stagnation_counter,
                "severity": "high" if self._stagnation_counter >= 30 else "medium",
                "suggested_actions": ["increase_mutation_rate", "switch_emitter_to_random"],
            }

        if self._stagnation_counter >= 10:
            return {
                "pattern": "early_stagnation",
                "duration": self._stagnation_counter,
                "severity": "low",
                "suggested_actions": ["increase_mutation_rate"],
            }

        return None

    def record_failure_and_action(
        self, pattern_name: str, action: str, success: bool,
    ):
        """记录一次失败检测与修复动作"""
        if self._current_case_id:
            self.kb.record_failure(
                self._current_case_id,
                self._generation_counter,
                pattern_name, action, success,
            )

    # ── 回调注册 ────────────────────────────────────────────

    def on_generation_end(self, callback: Callable):
        """注册每代结束回调"""
        self._on_generation_end = callback

    def on_run_start(self, callback: Callable):
        self._on_run_start = callback

    def on_run_end(self, callback: Callable):
        self._on_run_end = callback

    # ── 内部方法 ────────────────────────────────────────────

    def _flush(self):
        """批量写入缓冲数据"""
        if not self._generation_buffer or not self._current_case_id:
            return
        # 当前 record_generation 已实时写入, buffer 仅作备份
        self._generation_buffer.clear()

    def _aggregate_metrics(self) -> Dict:
        """从缓冲数据聚合最终指标"""
        result = {
            "task_name": "unknown",
            "catalog_name": "default",
            "qd_score": self._last_qd_score,
            "coverage": 0.0,
            "best_fitness": 0.0,
        }

        for entry in self._generation_buffer:
            m = entry["metrics"]
            result["qd_score"] = max(result["qd_score"], m.get("qd_score", 0))
            result["coverage"] = max(result["coverage"], m.get("coverage", 0))
            result["best_fitness"] = max(
                result["best_fitness"], m.get("best_fitness", 0),
            )
            if m.get("task_name"):
                result["task_name"] = m["task_name"]
            if m.get("catalog_name"):
                result["catalog_name"] = m["catalog_name"]

        return result
