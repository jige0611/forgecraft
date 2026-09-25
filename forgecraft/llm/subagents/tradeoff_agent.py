"""权衡智能体 — Pareto 前沿分析与多目标决策

职责:
  - Pareto 前沿分析: 识别拐点 (knee point)
  - 多目标权衡推荐: 根据用户偏好选最优折衷
  - 目标冲突分析: 量化目标间的 trade-off 关系
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from forgecraft.llm.provider import BaseLLMProvider, LLMMessage, create_provider
from forgecraft.llm.prompts.system_prompt import TRADEOFF_AGENT_PROMPT
from forgecraft.logging import get_logger

_logger = get_logger(__name__)


class TradeoffAgent:
    """权衡智能体 — 多目标决策支持

    使用方式:
        agent = TradeoffAgent(kb, provider)
        result = agent.analyze_front(pareto_front, preferences)
        # result = {"knee_point": {...}, "recommendation": {...}, "tradeoff_analysis": "..."}
    """

    def __init__(
        self,
        knowledge_base: "KnowledgeBase",  # noqa: F821
        provider: Optional[BaseLLMProvider] = None,
    ):
        self.kb = knowledge_base
        self.provider = provider or create_provider("mock")

    def analyze_front(
        self,
        objectives: np.ndarray,
        names: Optional[List[str]] = None,
        directions: Optional[List[str]] = None,
        preferences: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """分析 Pareto 前沿

        Args:
            objectives: [n_solutions, n_objectives] 目标值矩阵
            names: 目标名称列表
            directions: 优化方向 ["maximize", "minimize", ...]
            preferences: 用户偏好 {"speed": 0.7, "energy": 0.3}

        Returns:
            {
                "front_size": int,
                "knee_point": {"index": int, "objectives": [...], "description": str},
                "extreme_points": [...],
                "recommendation": {...},
                "tradeoff_analysis": str,
                "hypervolume": float,
            }
        """
        objectives = np.asarray(objectives, dtype=np.float64)
        n_solutions, n_obj = objectives.shape

        if names is None:
            names = [f"obj_{i}" for i in range(n_obj)]
        if directions is None:
            directions = ["maximize"] * n_obj
        if preferences is None:
            preferences = {name: 1.0 / n_obj for name in names}

        # 归一化
        obj_norm = self._normalize(objectives, directions)

        # 拐点检测
        knee_idx = self._detect_knee_point(obj_norm)

        # 极端点
        extremes = self._find_extremes(objectives, directions)

        # 用户偏好匹配
        recommendation = self._match_preferences(
            objectives, obj_norm, preferences, names,
        )

        # 超体积 (简化)
        hv = self._compute_hypervolume(objectives, directions)

        # 生成分析文本
        tradeoff_text = self._generate_tradeoff_text(
            objectives, names, preferences, knee_idx, recommendation,
        )

        return {
            "front_size": n_solutions,
            "knee_point": {
                "index": int(knee_idx),
                "objectives": {
                    names[i]: float(objectives[knee_idx, i])
                    for i in range(n_obj)
                },
                "description": (
                    f"Solution {knee_idx} offers the best balance across all objectives. "
                    f"It sits at the point where improving one objective would significantly "
                    f"degrade another."
                ),
            },
            "extreme_points": [
                {
                    "index": int(idx),
                    "focus": names[i],
                    "objectives": {
                        names[j]: float(objectives[idx, j])
                        for j in range(n_obj)
                    },
                }
                for i, idx in enumerate(extremes)
            ],
            "recommendation": recommendation,
            "tradeoff_analysis": tradeoff_text,
            "hypervolume": float(hv),
        }

    def recommend_weighted(
        self,
        objectives: np.ndarray,
        weights: np.ndarray,
        names: Optional[List[str]] = None,
    ) -> Dict:
        """按加权和推荐最优解"""
        objectives = np.asarray(objectives, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        # 归一化
        obj_norm = (objectives - objectives.min(axis=0)) / (
            objectives.max(axis=0) - objectives.min(axis=0) + 1e-8
        )

        weighted = obj_norm @ (weights / weights.sum())
        best_idx = int(np.argmax(weighted))

        if names is None:
            names = [f"obj_{i}" for i in range(objectives.shape[1])]

        return {
            "best_index": best_idx,
            "objectives": {
                names[i]: float(objectives[best_idx, i])
                for i in range(objectives.shape[1])
            },
            "weighted_score": float(weighted[best_idx]),
            "all_scores": [
                {"index": i, "score": float(s), "objectives": {
                    names[j]: float(objectives[i, j])
                    for j in range(objectives.shape[1])
                }}
                for i, s in enumerate(weighted)
            ][:10],
        }

    # ── 内部算法 ─────────────────────────────────────────────

    def _normalize(self, obj: np.ndarray, directions: List[str]) -> np.ndarray:
        obj = obj.copy()
        for i, d in enumerate(directions):
            if d == "minimize":
                obj[:, i] = -obj[:, i]

        # Min-max normalize to [0, 1]
        mins = obj.min(axis=0)
        maxs = obj.max(axis=0)
        denom = maxs - mins
        denom[denom == 0] = 1.0
        return (obj - mins) / denom

    def _detect_knee_point(self, obj_norm: np.ndarray) -> int:
        """检测 Pareto 前沿的拐点 (knee point)

        使用 angle-based 方法: 拐点 = 与极值连线角度最大的点
        """
        n = obj_norm.shape[0]
        if n <= 2:
            return n - 1

        # 极值: 每维最大值 (归一化后)
        n_obj = obj_norm.shape[1]
        if n_obj == 2:
            # 2D: 到对角线的距离最大
            diag = np.array([1.0 / n_obj] * n_obj)
            dists = np.linalg.norm(obj_norm - diag, axis=1)
            return int(np.argmax(dists))
        else:
            # 3D+: 到 ideal 点 (1,1,...,1) 的 L2 距离最大
            ideal = np.ones(n_obj)
            dists = np.sum((obj_norm - ideal) ** 2, axis=1)
            return int(np.argmin(dists))  # 最近 → 最优折衷

    def _find_extremes(self, obj: np.ndarray, directions: List[str]) -> List[int]:
        extremes = []
        for i in range(obj.shape[1]):
            if directions[i] == "maximize":
                extremes.append(int(np.argmax(obj[:, i])))
            else:
                extremes.append(int(np.argmin(obj[:, i])))
        return extremes

    def _match_preferences(
        self,
        obj: np.ndarray,
        obj_norm: np.ndarray,
        preferences: Dict[str, float],
        names: List[str],
    ) -> Dict:
        """根据用户偏好找最佳折衷"""
        # 构建权重向量
        weights = np.array([
            preferences.get(name, 1.0 / len(names))
            for name in names
        ])
        weights = weights / weights.sum()

        # 加权得分
        scores = obj_norm @ weights
        best_idx = int(np.argmax(scores))

        return {
            "index": best_idx,
            "objectives": {
                names[i]: float(obj[best_idx, i])
                for i in range(len(names))
            },
            "match_score": float(scores[best_idx]),
            "explanation": (
                f"Solution {best_idx} best matches your preferences "
                f"(weights: {dict(zip(names, weights.round(2)))})."
            ),
        }

    def _compute_hypervolume(
        self, obj: np.ndarray, directions: List[str]
    ) -> float:
        """计算超体积 (简化版, 仅 2D/3D)"""
        obj = obj.copy()
        for i, d in enumerate(directions):
            if d == "minimize":
                obj[:, i] = -obj[:, i]

        # 简化: 使用 sum of dominance 近似
        ref = obj.min(axis=0) - 0.1
        hv = np.prod(np.maximum(obj - ref, 0).max(axis=0))
        return float(hv)

    def _generate_tradeoff_text(
        self,
        obj: np.ndarray,
        names: List[str],
        preferences: Dict[str, float],
        knee_idx: int,
        recommendation: Dict,
    ) -> str:
        """生成可读的权衡分析文本"""
        n_obj = len(names)
        if n_obj == 2:
            # 计算近似的 trade-off 斜率
            sorted_idx = np.argsort(obj[:, 0])
            if len(sorted_idx) >= 2:
                dx = obj[sorted_idx[-1], 0] - obj[sorted_idx[0], 0]
                dy = obj[sorted_idx[-1], 1] - obj[sorted_idx[0], 1]
                tradeoff_ratio = abs(dy / dx) if abs(dx) > 1e-6 else float("inf")
            else:
                tradeoff_ratio = 1.0
            return (
                f"{names[0]} vs {names[1]}: "
                f"Improving {names[0]} by 1 unit costs approximately "
                f"{tradeoff_ratio:.2f} units of {names[1]}.\n"
                f"Knee point (solution {knee_idx}) offers the best balance.\n"
                f"Your preferences: {preferences} → "
                f"Recommended solution: {recommendation['index']}"
            )
        else:
            return (
                f"Multi-objective trade-off across {n_obj} objectives.\n"
                f"Knee point: solution {knee_idx}.\n"
                f"Recommended based on preferences: solution {recommendation['index']}"
            )
