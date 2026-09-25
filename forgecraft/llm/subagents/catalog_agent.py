"""目录选择智能体 — 任务 → 最优零件目录

职责:
  - 分析任务需求 → 推荐目录或生成新目录
  - 评估目录-任务匹配度
  - 目录修改建议 (添加/删除零件)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from forgecraft.llm.provider import BaseLLMProvider, LLMMessage, create_provider
from forgecraft.llm.prompts.system_prompt import CATALOG_AGENT_PROMPT
from forgecraft.llm.prompts.few_shot_examples import (
    CATALOG_SELECTION_EXAMPLE,
    CATALOG_GENERATION_EXAMPLE,
)
from forgecraft.logging import get_logger

_logger = get_logger(__name__)


class CatalogAgent:
    """目录选择智能体

    使用方式:
        agent = CatalogAgent(kb, provider)
        result = agent.select(task_description="设计一个快速四足机器人")
        # result = {"catalog": "default", "confidence": 0.85, ...}
    """

    def __init__(
        self,
        knowledge_base: "KnowledgeBase",  # noqa: F821
        provider: Optional[BaseLLMProvider] = None,
    ):
        self.kb = knowledge_base
        self.provider = provider or create_provider("mock")

    def select(
        self,
        task_description: str,
        task_type: Optional[str] = None,
        constraints: Optional[Dict] = None,
    ) -> Dict:
        """为任务选择最佳目录

        Returns:
            {
                "catalog_name": str,
                "confidence": float,
                "reasoning": str,
                "modifications": Optional[List[Dict]],
                "alternative_catalogs": List[str],
            }
        """
        # 推断任务类型
        if task_type is None:
            task_type = self._infer(task_description)

        # 知识库: 相似案例用了什么目录?
        similar = self.kb.find_similar_cases(task_description, top_k=5)
        used_catalogs = {}
        for case, score in similar:
            cat = case.catalog_name
            used_catalogs[cat] = used_catalogs.get(cat, 0) + score

        # 任务-目录映射表
        task_catalog_map = {
            "speed": ["default", "hoppers", "biped"],
            "climbing": ["gripper", "biped", "hexapod"],
            "manipulation": ["gripper", "default"],
            "efficiency": ["default", "primitives"],
            "structure": ["default", "advanced"],
        }

        candidates = task_catalog_map.get(task_type, ["default"])

        # 如果知识库有历史偏好, 优先使用
        if used_catalogs:
            best_cat = max(used_catalogs, key=used_catalogs.get)
            if best_cat not in candidates:
                candidates.insert(0, best_cat)

        # 评估每个候选
        from forgecraft.knowledge.ontology import (
            PART_ONTOLOGY, get_heuristics_for_task,
        )

        task_config = PART_ONTOLOGY.get("task_specific_configs", {}).get(task_type, {})
        heuristics = get_heuristics_for_task(task_type)

        recommended = candidates[0]
        confidence = 0.7

        # 知识库加权
        if used_catalogs and recommended in used_catalogs:
            confidence = min(0.95, 0.7 + used_catalogs[recommended] * 0.1)

        return {
            "catalog_name": recommended,
            "confidence": confidence,
            "reasoning": (
                f"Task '{task_type}' maps to catalog '{recommended}'. "
                f"Similar cases used: {list(used_catalogs.keys())[:3]}. "
                f"Relevant heuristics: {len(heuristics)} rules."
            ),
            "modifications": self._suggest_modifications(task_type, recommended),
            "alternative_catalogs": [c for c in candidates if c != recommended][:2],
            "task_config": task_config,
        }

    def generate(
        self,
        task_name: str,
        task_description: str,
        required_functions: List[str],
        physical_constraints: Optional[Dict] = None,
    ) -> Dict:
        """为全新任务生成目录 YAML

        Returns:
            {
                "yaml_content": str,
                "parts_generated": int,
                "warnings": List[str],
            }
        """
        available = self._list_available_catalogs()

        messages = [
            LLMMessage(role="system", content=CATALOG_AGENT_PROMPT.format(
                available_catalogs=", ".join(available),
                part_ontology_summary=self._ontology_summary(),
            )),
            LLMMessage(role="user", content=CATALOG_GENERATION_EXAMPLE),
            LLMMessage(
                role="user",
                content=(
                    f"生成一个名为 '{task_name}' 的新零件目录。\n\n"
                    f"任务描述: {task_description}\n"
                    f"必需功能: {required_functions}\n"
                    f"物理约束: {json.dumps(physical_constraints) if physical_constraints else '无'}\n\n"
                    f"请生成完整的 YAML 格式目录定义, 包含 parts 和 constraints 部分。"
                    f"参考现有的 {', '.join(available[:5])} 目录格式。"
                ),
            ),
        ]

        try:
            response = self.provider.chat(messages)
            yaml_content = self._extract_yaml(response.content)
        except Exception as e:
            _logger.warning(f"Catalog generation failed: {e}")
            yaml_content = self._generate_fallback(task_name, required_functions)

        return {
            "yaml_content": yaml_content,
            "parts_generated": yaml_content.count("geometry:"),
            "warnings": [],
            "save_path": os.path.join(
                "forgecraft", "configs", "catalogs", f"{task_name}.yaml",
            ),
        }

    def evaluate_match(self, catalog_name: str, task_type: str) -> Dict:
        """评估目录与任务的匹配度

        Returns:
            {"match_score": 0.0-1.0, "strengths": [...], "weaknesses": [...]}
        """
        from forgecraft.knowledge.ontology import PART_ONTOLOGY

        task_config = PART_ONTOLOGY.get("task_specific_configs", {}).get(task_type, {})

        strengths = []
        weaknesses = []
        score = 0.5

        # 检查目录是否包含任务优先零件
        priority_funcs = [
            p.get("function") for p in task_config.get("priority_parts", [])
        ]

        # 这里需要实际加载目录验证, 简化为基于名称的启发式评分
        catalog_task_map = {
            "speed": {"default": 0.9, "hoppers": 0.7, "biped": 0.6},
            "climbing": {"gripper": 0.85, "biped": 0.6, "hexapod": 0.7},
            "manipulation": {"gripper": 0.9, "default": 0.5},
            "efficiency": {"default": 0.8, "primitives": 0.6},
            "structure": {"default": 0.7, "advanced": 0.8},
        }

        score = catalog_task_map.get(task_type, {}).get(catalog_name, 0.5)

        if score > 0.7:
            strengths.append(f"Good match for '{task_type}' tasks")
        else:
            weaknesses.append(f"Suboptimal for '{task_type}' tasks")

        return {"match_score": score, "strengths": strengths, "weaknesses": weaknesses}

    # ── 内部 ──

    def _infer(self, description: str) -> str:
        desc = description.lower()
        if any(w in desc for w in ["速度", "speed", "快", "fast"]):
            return "speed"
        if any(w in desc for w in ["攀爬", "climb", "爬"]):
            return "climbing"
        if any(w in desc for w in ["抓取", "grip", "操作", "夹"]):
            return "manipulation"
        if any(w in desc for w in ["效率", "节能", "efficien"]):
            return "efficiency"
        return "speed"

    def _suggest_modifications(self, task_type: str, catalog: str) -> List[Dict]:
        """建议目录修改"""
        suggestions = []
        from forgecraft.knowledge.ontology import PART_ONTOLOGY

        task_config = PART_ONTOLOGY.get("task_specific_configs", {}).get(task_type, {})

        # 检查避免的零件
        avoid_funcs = [a.get("function") for a in task_config.get("avoid", [])]
        for func in avoid_funcs:
            suggestions.append({
                "action": "consider_removing",
                "function": func,
                "reason": f"Task '{task_type}' typically doesn't need {func} parts",
            })

        return suggestions

    def _list_available_catalogs(self) -> List[str]:
        catalog_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "forgecraft", "configs", "catalogs",
        )
        if os.path.isdir(catalog_dir):
            return sorted([
                f.replace(".yaml", "") for f in os.listdir(catalog_dir)
                if f.endswith(".yaml")
            ])
        return ["default", "biped", "hexapod", "gripper", "hoppers"]

    def _ontology_summary(self) -> str:
        from forgecraft.knowledge.ontology import PART_ONTOLOGY
        groups = PART_ONTOLOGY.get("functional_groups", {})
        return "\n".join(
            f"- {name}: {data['description']} ({len(data['parts'])} parts)"
            for name, data in groups.items()
        )

    def _extract_yaml(self, content: str) -> str:
        """从 LLM 输出中提取 YAML 块"""
        if "```yaml" in content:
            start = content.index("```yaml") + 7
            end = content.index("```", start)
            return content[start:end].strip()
        elif "```" in content:
            start = content.index("```") + 3
            end = content.index("```", start)
            return content[start:end].strip()
        return content

    def _generate_fallback(self, task_name: str, functions: List[str]) -> str:
        """回退: 生成最小化目录"""
        lines = [
            f"# Auto-generated catalog for '{task_name}'",
            f"# Functions: {functions}",
            "",
            "constraints:",
            "  max_depth: 10",
            "  max_parts: 30",
            "  require_root: true",
            "",
            "parts:",
        ]
        for func in functions:
            from forgecraft.knowledge.ontology import get_parts_by_function
            parts = get_parts_by_function(func)[:2]
            for part in parts:
                lines.append(f"  # include {part} for {func}")
        return "\n".join(lines)
