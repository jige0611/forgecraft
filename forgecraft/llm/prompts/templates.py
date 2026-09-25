"""提示词模板 — 动态填充的上下文模板"""

from typing import Any, Dict, List

# ══════════════════════════════════════════════════════════════
#  任务分析模板
# ══════════════════════════════════════════════════════════════

TASK_ANALYSIS_TEMPLATE = """## 任务分析

### 任务描述
{task_description}

### 已有知识库统计
- 设计案例总数: {total_cases}
- 相似任务案例数: {similar_cases_count}
- 已知失败模式: {failure_patterns}
- 零件本体: {part_count} 类零件, {rule_count} 条组合约束

### 相关启发式规则
{heuristics}

### 推荐策略模板
{strategy_template}
"""

# ══════════════════════════════════════════════════════════════
#  进化状态上下文
# ══════════════════════════════════════════════════════════════

RUN_STATUS_TEMPLATE = """## 当前进化状态

- 代数: {generation}/{total_generations}
- QD-Score: {qd_score:.4f}
- 覆盖率: {coverage:.2%}
- 最佳适应度: {best_fitness:.4f}
- 精英数量: {num_elites}
- 种群多样性: {diversity:.4f}
- 停滞代数: {stagnation_duration}
- 活跃策略: {active_strategy}
"""

# ══════════════════════════════════════════════════════════════
#  设计案例上下文
# ══════════════════════════════════════════════════════════════

SIMILAR_CASES_TEMPLATE = """## 相似历史案例

{case_summaries}

### 统计
- 平均 QD-Score: {avg_qd_score:.4f}
- 最高 QD-Score: {max_qd_score:.4f}
- 平均覆盖率: {avg_coverage:.2%}
- 最常用目录: {most_used_catalog}
"""

# ══════════════════════════════════════════════════════════════
#  推荐方案模板
# ══════════════════════════════════════════════════════════════

RECOMMENDATION_TEMPLATE = """## 智能体推荐

### 推荐方案
{recommendation}

### 具体配置
```yaml
{config_yaml}
```

### 置信度
{confidence}

### 理由
{rationale}
"""

# ══════════════════════════════════════════════════════════════
#  运行时调整模板
# ══════════════════════════════════════════════════════════════

RUNTIME_ADJUSTMENT_TEMPLATE = """## 运行时调整

### 检测到的信号
{detected_signals}

### 执行的动作
{action_taken}

### 参数变化
{param_changes}

### 预期效果
{expected_effect}
"""


def render_task_analysis(
    task_description: str,
    kb_stats: Dict,
    heuristics: List[Dict],
    strategy_template: Dict,
    part_count: int = 56,
    rule_count: int = 6,
) -> str:
    return TASK_ANALYSIS_TEMPLATE.format(
        task_description=task_description,
        total_cases=kb_stats.get("semantic", {}).get("total_cases", 0),
        similar_cases_count=kb_stats.get("semantic", {}).get("total_cases", 0),
        failure_patterns=len(kb_stats.get("failure_patterns", [])),
        part_count=part_count,
        rule_count=rule_count,
        heuristics=_format_heuristics(heuristics),
        strategy_template=_format_strategy(strategy_template),
    )


def render_run_status(metrics: Dict) -> str:
    return RUN_STATUS_TEMPLATE.format(
        generation=metrics.get("generation", 0),
        total_generations=metrics.get("total_generations", 200),
        qd_score=metrics.get("qd_score", 0.0),
        coverage=metrics.get("coverage", 0.0),
        best_fitness=metrics.get("best_fitness", 0.0),
        num_elites=metrics.get("num_elites", 0),
        diversity=metrics.get("population_diversity", 0.0),
        stagnation_duration=metrics.get("stagnation_duration", 0),
        active_strategy=metrics.get("active_strategy", "map_elites"),
    )


def render_similar_cases(cases: List[Dict]) -> str:
    if not cases:
        return "## 相似历史案例\n(无相似案例)"

    summaries = []
    for i, c in enumerate(cases[:5]):
        obj_str = ", ".join(
            f"{k}={v:.2f}" for k, v in c.get("objectives", {}).items()
        )
        summaries.append(
            f"{i+1}. [{c.get('task_name', '?')}] "
            f"Catalog: {c.get('catalog_name', '?')}, "
            f"QD-Score: {c.get('qd_score_final', 0):.3f}, "
            f"Fitness: {c.get('best_fitness', 0):.3f}, "
            f"Objectives: {{{obj_str}}}"
        )

    qd_scores = [c.get("qd_score_final", 0) for c in cases if c.get("qd_score_final")]
    covs = [c.get("coverage_final", 0) for c in cases if c.get("coverage_final")]
    cats = [c.get("catalog_name", "") for c in cases]

    return SIMILAR_CASES_TEMPLATE.format(
        case_summaries="\n".join(summaries),
        avg_qd_score=sum(qd_scores) / len(qd_scores) if qd_scores else 0,
        max_qd_score=max(qd_scores) if qd_scores else 0,
        avg_coverage=sum(covs) / len(covs) if covs else 0,
        most_used_catalog=max(set(cats), key=cats.count) if cats else "N/A",
    )


def _format_heuristics(heuristics: List[Dict]) -> str:
    if not heuristics:
        return "(无相关规则)"
    lines = []
    for h in heuristics:
        lines.append(f"- [{h.get('id', '?')}] {h.get('name', '?')}: {h.get('description', '')}")
    return "\n".join(lines)


def _format_strategy(template: Dict) -> str:
    return (
        f"推荐策略: {template.get('recommended_strategy', 'map_elites')}\n"
        f"理由: {template.get('reason', 'N/A')}"
    )
