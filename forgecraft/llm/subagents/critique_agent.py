"""评审智能体 — 运行后分析与改进建议

职责:
  - 分析完整进化轨迹
  - 识别成功/失败模式
  - 生成可操作的改进建议
  - 将分析结果沉淀到知识库
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from forgecraft.llm.provider import BaseLLMProvider, create_provider
from forgecraft.logging import get_logger

_logger = get_logger(__name__)


class CritiqueAgent:
    """评审智能体 — 事后设计评审

    使用方式:
        agent = CritiqueAgent(kb, provider)
        report = agent.analyze(run_id)
        # report = {"overall_verdict": "good", "issues": [...], "suggestions": [...]}
    """

    def __init__(
        self,
        knowledge_base: "KnowledgeBase",  # noqa: F821
        provider: Optional[BaseLLMProvider] = None,
    ):
        self.kb = knowledge_base
        self.provider = provider or create_provider("mock")

    def analyze(self, run_id: str) -> Dict:
        """完整运行分析

        Returns:
            {
                "overall_verdict": "excellent" | "good" | "acceptable" | "poor",
                "scores": {
                    "convergence": 0-10,
                    "diversity": 0-10,
                    "efficiency": 0-10,
                    "morphology_quality": 0-10,
                },
                "strengths": [...],
                "weaknesses": [...],
                "failure_patterns": [...],
                "recommendations": [...],
                "improvement_potential": str,
            }
        """
        # 获取轨迹
        trajectory = self.kb.get_trajectory(run_id) if self.kb else []
        case = None
        try:
            # 尝试从关系库获取案例详情
            rows = self.kb.relational_kb._get_conn().execute(
                "SELECT * FROM design_cases WHERE run_id = ?", (run_id,)
            ).fetchall()
            if rows:
                case = dict(rows[0])
        except Exception:
            pass

        if not trajectory and not case:
            return {
                "overall_verdict": "unknown",
                "scores": {},
                "recommendations": ["No trajectory data available."],
            }

        # 评分
        scores = self._score_run(trajectory)

        # 识别模式
        strengths = self._identify_strengths(trajectory)
        weaknesses = self._identify_weaknesses(trajectory)
        failure_patterns = self._identify_failure_patterns(trajectory)

        # 生成建议
        recommendations = self._generate_recommendations(
            scores, strengths, weaknesses, failure_patterns, case,
        )

        # 判定
        avg_score = sum(scores.values()) / max(len(scores), 1)
        if avg_score >= 8:
            verdict = "excellent"
        elif avg_score >= 6:
            verdict = "good"
        elif avg_score >= 4:
            verdict = "acceptable"
        else:
            verdict = "poor"

        return {
            "overall_verdict": verdict,
            "scores": scores,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "failure_patterns": failure_patterns,
            "recommendations": recommendations,
            "improvement_potential": self._estimate_improvement(scores, case),
        }

    def compare_runs(self, run_ids: List[str]) -> Dict:
        """比较多次运行"""
        results = {}
        for rid in run_ids:
            results[rid] = self.analyze(rid)

        # 排名
        ranked = sorted(
            results.items(),
            key=lambda x: sum(x[1].get("scores", {}).values()),
            reverse=True,
        )

        return {
            "ranking": [{"run_id": r[0], "verdict": r[1]["overall_verdict"]}
                         for r in ranked],
            "best": ranked[0][0] if ranked else None,
            "details": results,
        }

    def extract_insights(self, run_id: str) -> List[str]:
        """从运行中提取可复用的设计insights"""
        analysis = self.analyze(run_id)

        insights = []

        # 成功因素
        if analysis["overall_verdict"] in ("excellent", "good"):
            insights.append(
                f"Task type with strategy {analysis.get('strategy', 'unknown')} "
                f"achieved {analysis['overall_verdict']} results."
            )

        # 失败教训
        for fp in analysis.get("failure_patterns", []):
            insights.append(
                f"Failure pattern '{fp.get('pattern', '?')}': {fp.get('evidence', '')}"
            )

        # 参数建议
        for rec in analysis.get("recommendations", [])[:3]:
            insights.append(f"Recommendation: {rec}")

        return insights

    # ── 评分 ─────────────────────────────────────────────────

    def _score_run(self, trajectory: List[Dict]) -> Dict[str, float]:
        """量化评分"""
        if not trajectory:
            return {"convergence": 0, "diversity": 0, "efficiency": 0, "morphology_quality": 0}

        scores = {}

        # 收敛性: QD-score 增长趋势
        qd_scores = [g.get("qd_score", 0) for g in trajectory if g.get("qd_score")]
        if len(qd_scores) >= 2:
            improvement = (qd_scores[-1] - qd_scores[0]) / max(abs(qd_scores[0]), 1e-6)
            scores["convergence"] = min(10, max(0, improvement * 5 + 5))
        else:
            scores["convergence"] = 5

        # 多样性: 最终覆盖率
        final_cov = trajectory[-1].get("coverage", 0) if trajectory else 0
        scores["diversity"] = final_cov * 10

        # 效率: 到达 90% 最大 QD 的代数 / 总代数
        if qd_scores:
            max_qd = max(qd_scores)
            target = max_qd * 0.9
            for i, q in enumerate(qd_scores):
                if q >= target:
                    scores["efficiency"] = max(1, 10 - (i / max(len(qd_scores), 1)) * 10)
                    break
            else:
                scores["efficiency"] = 3
        else:
            scores["efficiency"] = 5

        # 形态质量 (基于最终 QD-score, 需要更多信息)
        final_qd = qd_scores[-1] if qd_scores else 0
        # 这个维度在实际中需要结合 FEA 结果等
        scores["morphology_quality"] = min(10, final_qd * 5 + 3)

        return {k: round(v, 1) for k, v in scores.items()}

    # ── 识别 ─────────────────────────────────────────────────

    def _identify_strengths(self, trajectory: List[Dict]) -> List[str]:
        if not trajectory:
            return []
        strengths = []

        qd_scores = [g.get("qd_score", 0) for g in trajectory if g.get("qd_score")]
        if qd_scores and qd_scores[-1] > qd_scores[0] * 2:
            strengths.append("Strong convergence: QD-score more than doubled")
        if trajectory[-1].get("coverage", 0) > 0.5:
            strengths.append("High coverage: diverse set of solutions")
        if len(trajectory) >= 100:
            mid = len(trajectory) // 2
            if qd_scores and qd_scores[-1] - qd_scores[mid] > 0.2:
                strengths.append("Continued improvement in later generations")

        return strengths

    def _identify_weaknesses(self, trajectory: List[Dict]) -> List[str]:
        if not trajectory:
            return ["No data"]
        weaknesses = []

        if trajectory[-1].get("qd_score", 0) < 0.1:
            weaknesses.append("Poor final QD-score")
        if trajectory[-1].get("coverage", 0) < 0.1:
            weaknesses.append("Very low coverage")
        if trajectory[-1].get("stagnation_duration", 0) > 30:
            weaknesses.append("Extended stagnation without recovery")

        return weaknesses

    def _identify_failure_patterns(self, trajectory: List[Dict]) -> List[Dict]:
        patterns = []
        if not trajectory:
            return patterns

        qd_scores = [g.get("qd_score", 0) for g in trajectory if g.get("qd_score")]
        covs = [g.get("coverage", 0) for g in trajectory if g.get("coverage")]

        # 早期停滞
        if len(qd_scores) >= 50:
            mid = len(qd_scores) // 2
            if qd_scores[mid] > 0 and qd_scores[-1] < qd_scores[mid] * 1.05:
                patterns.append({
                    "pattern": "early_plateau",
                    "evidence": f"QD-score plateaued at gen ~{mid} at {qd_scores[mid]:.4f}",
                    "suggested_fix": "Switch strategy or increase exploration",
                })

        # 发散
        if len(qd_scores) >= 20:
            recent = qd_scores[-20:]
            if max(recent) - min(recent) > 0.5:
                patterns.append({
                    "pattern": "instability",
                    "evidence": "High variance in recent QD-scores",
                    "suggested_fix": "Reduce mutation rate or increase population size",
                })

        return patterns

    # ── 建议 ─────────────────────────────────────────────────

    def _generate_recommendations(
        self,
        scores: Dict,
        strengths: List[str],
        weaknesses: List[str],
        failure_patterns: List[Dict],
        case: Optional[Dict],
    ) -> List[str]:
        recs = []

        if scores.get("convergence", 5) < 4:
            recs.append("Consider increasing population size or switching strategy (e.g., CMA-ME for parameter refinement)")
        if scores.get("diversity", 5) < 4:
            recs.append("Increase exploration: use more random emitters or relax catalog constraints")
        if scores.get("efficiency", 5) < 4:
            recs.append("Consider early stopping or progressive budget reduction to save computation")

        for fp in failure_patterns:
            fix = fp.get("suggested_fix")
            if fix and fix not in recs:
                recs.append(fix)

        if not recs:
            recs.append("No specific recommendations — run appears healthy.")

        return recs

    def _estimate_improvement(self, scores: Dict, case: Optional[Dict]) -> str:
        avg = sum(scores.values()) / max(len(scores), 1)
        if avg >= 8:
            return "Limited — already near optimal. Focus on marginal gains (FEA validation, manufacturing cost)."
        elif avg >= 6:
            return "Moderate — 10-30% improvement possible with strategy optimization."
        elif avg >= 4:
            return "Significant — 30-60% improvement possible with catalog/strategy changes."
        else:
            return "High — fundamental redesign likely needed."
