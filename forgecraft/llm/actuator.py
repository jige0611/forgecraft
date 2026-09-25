"""LLM 执行器 — 将 LLM 决策转换为实际参数修改

使用方式:
    actuator = LLMActuator(kb)
    actuator.apply_adjustment(loop, {"mutation_rate_multiplier": 1.5})
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from forgecraft.logging import get_logger

_logger = get_logger(__name__)


class LLMActuator:
    """LLM 执行器 — 将决策转化为动作

    安全原则:
      - 只能修改"安全参数" (mutation rate, emitter ratio 等)
      - 不能修改"结构参数" (population size, graph structure)
      - 所有修改有记录可追溯
      - 超出安全范围的修改需要确认
    """

    # 安全参数范围
    SAFE_RANGES = {
        "mutation_rate": (0.05, 0.80),
        "mutation_rate_multiplier": (0.5, 2.0),
        "crossover_rate": (0.10, 0.90),
        "topo_mutation_prob": (0.02, 0.50),
        "param_mutation_prob": (0.05, 0.60),
        "param_mutation_scale": (0.01, 0.50),
        "selection_pressure": (1.0, 5.0),
        "random_emitter_ratio": (0.0, 1.0),
    }

    def __init__(self, knowledge_base: Optional["KnowledgeBase"] = None):  # noqa: F821
        self.kb = knowledge_base
        self._action_log: List[Dict] = []

    def apply_adjustment(
        self,
        evolution_loop,
        params: Dict,
        reason: str = "",
    ) -> Dict:
        """将参数调整应用到 EvolutionLoop

        Args:
            evolution_loop: EvolutionLoop 实例
            params: 参数映射 {"mutation_rate_multiplier": 1.5, ...}
            reason: 调整理由

        Returns:
            {"success": bool, "applied": {...}, "rejected": {...}, "warnings": [...]}
        """
        result = {"success": True, "applied": {}, "rejected": {}, "warnings": []}

        evo_config = evolution_loop.evo_config
        changed = False

        for param, value in params.items():
            safe = self._is_safe(param, value)

            if not safe:
                result["rejected"][param] = value
                result["warnings"].append(
                    f"Parameter '{param}'={value} out of safe range "
                    f"{self.SAFE_RANGES.get(param, 'unknown')}"
                )
                continue

            old_val = getattr(evo_config, param, None)
            setattr(evo_config, param, value)
            result["applied"][param] = {"old": old_val, "new": value}
            changed = True

        # 日志
        if changed:
            action = {
                "params": params,
                "applied": result["applied"],
                "rejected": result["rejected"],
                "reason": reason,
            }
            self._action_log.append(action)

            _logger.info(f"Actuator applied: {result['applied']}")

            # 记录到知识库
            if self.kb:
                for param, vals in result["applied"].items():
                    self.kb.record_param_change(
                        case_id=getattr(evolution_loop, '_run_id', 'unknown'),
                        generation=getattr(evolution_loop, 'generation', 0),
                        param_name=param,
                        old_value=float(vals["old"]) if vals["old"] else 0,
                        new_value=float(vals["new"]),
                        reason=reason,
                    )

        return result

    def apply_map_elites_adjustment(
        self,
        engine,
        params: Dict,
        reason: str = "",
    ) -> Dict:
        """将调整应用到 MAPElitesEngine

        MAPElitesEngine 的参数分布在 engine.config 和各 emitter 中
        """
        result = {"success": True, "applied": {}, "rejected": {}, "warnings": []}

        # emitter 类型切换
        if "random_emitter_ratio" in params:
            ratio = params["random_emitter_ratio"]
            if 0 <= ratio <= 1:
                n_random = max(1, int(len(engine.emitters) * ratio))
                for i in range(min(n_random, len(engine.emitters))):
                    engine.emitters[i].method = "random"
                result["applied"]["random_emitter_ratio"] = ratio
            else:
                result["rejected"]["random_emitter_ratio"] = ratio

        if "switch_strategy" in params:
            strategy = params["switch_strategy"]
            result["applied"]["switch_strategy"] = strategy
            result["warnings"].append(
                f"Strategy switch to '{strategy}' requires re-initialization. "
                f"Applied as suggestion only."
            )

        # 记录
        action = {
            "params": params,
            "applied": result["applied"],
            "rejected": result["rejected"],
            "reason": reason,
        }
        self._action_log.append(action)

        return result

    def save_catalog(
        self,
        yaml_content: str,
        catalog_name: str,
    ) -> str:
        """保存 LLM 生成的目录到 configs/catalogs/"""
        catalog_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "forgecraft", "configs", "catalogs",
        )
        os.makedirs(catalog_dir, exist_ok=True)

        path = os.path.join(catalog_dir, f"{catalog_name}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        _logger.info(f"Catalog saved: {path}")
        return path

    def inject_elite(
        self,
        evolution_loop,
        body,
        reason: str = "",
    ):
        """注入 LLM 设计的具体形态到种群"""
        if hasattr(evolution_loop, 'population'):
            # 替换最差个体
            worst_idx = min(
                range(len(evolution_loop.population)),
                key=lambda i: evolution_loop.population[i].fitness,
            )
            evolution_loop.population[worst_idx] = body
            _logger.info(f"Injected elite design at index {worst_idx}")

            if self.kb:
                self.kb.record_param_change(
                    case_id=getattr(evolution_loop, '_run_id', 'unknown'),
                    generation=getattr(evolution_loop, 'generation', 0),
                    param_name="injected_elite",
                    old_value=0, new_value=1,
                    reason=reason,
                )

    def configure_from_llm(
        self,
        llm_response: Dict,
    ) -> Dict:
        """从 LLM 响应构建 EvolutionConfig

        Args:
            llm_response: DesignReasoningEngine.analyze_task 的返回

        Returns:
            可用于 EvolutionConfig 的参数字典
        """
        config = llm_response.get("evolution_config", {})

        return {
            "population_size": config.get("population_size", 50),
            "generations": config.get("generations", 200),
            "elite_count": config.get("elite_count", 8),
            "mutation_rate": config.get("mutation_rate", 0.35),
            "crossover_rate": config.get("crossover_rate", 0.60),
            "topo_mutation_prob": config.get("topo_mutation_prob", 0.18),
            "param_mutation_prob": config.get("param_mutation_prob", 0.35),
            "param_mutation_scale": config.get("param_mutation_scale", 0.12),
            "selection_pressure": config.get("selection_pressure", 2.5),
        }

    def get_action_log(self) -> List[Dict]:
        return self._action_log

    def _is_safe(self, param: str, value: float) -> bool:
        range_val = self.SAFE_RANGES.get(param)
        if range_val is None:
            return False  # 未知参数拒绝
        lo, hi = range_val
        return lo <= value <= hi
