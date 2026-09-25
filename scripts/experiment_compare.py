# ═══════════════════════════════════════════════════════════════
#  ForgeCraft 多实验对比工具
#  ═══════════════════════════════════════════════════════════════
#
#  功能：
#  - 加载多个实验结果进行对比
#  - 并排适应度曲线对比
#  - 统计显著性检验
#  - 最佳个体对比（结构、性能）
#  - 导出对比报告 (HTML/Markdown)
#
#  使用：
#    from experiment_compare import ExperimentComparator
#    comp = ExperimentComparator()
#    comp.add_experiment("v11", "v11_results/")
#    comp.add_experiment("v12", "v12_results/")
#    report = comp.compare()
# ═══════════════════════════════════════════════════════════════

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger("ForgeCraft.Compare")


@dataclass
class ExperimentSummary:
    """单个实验摘要"""
    name: str = ""
    path: str = ""
    
    # 基本信息
    total_generations: int = 0
    population_size: int = 0
    
    # 最佳个体
    best_fitness: float = 0.0
    best_displacement: float = 0.0
    best_speed: float = 0.0
    best_type: str = ""
    best_n_parts: int = 0
    best_n_motors: int = 0
    
    # 收敛信息
    convergence_gen: int = 0      # 收敛代数（首次达到90%最佳）
    final_avg_fitness: float = 0.0
    improvement_rate: float = 0.0 # 每代平均改进
    
    # 历史数据
    history: List[Dict] = field(default_factory=list)
    
    # 时间戳
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "total_generations": self.total_generations,
            "best_fitness": self.best_fitness,
            "best_displacement": round(self.best_displacement, 4),
            "best_speed": round(self.best_speed, 4),
            "convergence_gen": self.convergence_gen,
            "final_avg_fitness": round(self.final_avg_fitness, 4),
            "timestamp": self.timestamp,
        }


@dataclass 
class ComparisonResult:
    """对比结果"""
    experiments: List[ExperimentSummary] = field(default_factory=list)
    comparison_time: str = ""
    
    # 对比指标
    fitness_ranking: List[Tuple[str, float]] = field(default_factory=list)
    speed_ranking: List[Tuple[str, float]] = field(default_factory=list)
    displacement_ranking: List[Tuple[str, float]] = field(default_factory=list)
    
    # 统计分析
    winner: str = ""                    # 综合最优实验
    significance: Dict[str, Any] = field(default_factory=dict)
    
    # 建议
    recommendation: str = ""


class ExperimentComparator:
    """多实验对比器"""
    
    def __init__(self):
        self.experiments: Dict[str, ExperimentSummary] = {}
        self.result: Optional[ComparisonResult] = None
    
    def add_experiment(self, name: str, results_path: str) -> bool:
        """
        添加一个实验
        
        Args:
            name: 实验名称标识
            results_path: 结果目录路径
            
        Returns:
            是否成功加载
        """
        path = Path(results_path)
        
        if not path.exists():
            logger.warning(f"Experiment path not found: {path}")
            return False
        
        summary = self._load_experiment(name, path)
        
        if summary:
            self.experiments[name] = summary
            logger.info(f"Loaded experiment '{name}': gen={summary.total_generations}, "
                       f"best_fit={summary.best_fitness:.4f}")
            return True
        
        return False
    
    def _load_experiment(self, name: str, path: Path) -> Optional[ExperimentSummary]:
        """加载单个实验数据"""
        final_file = path / "v11_final_results.json"
        
        if not final_file.exists():
            # 尝试其他可能的文件名
            for alt in ["final_results.json", "results.json", "experiment.json"]:
                alt_path = path / alt
                if alt_path.exists():
                    final_file = alt_path
                    break
            else:
                logger.warning(f"No results file found in {path}")
                return None
        
        try:
            with open(final_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {final_file}: {e}")
            return None
        
        summary = ExperimentSummary(
            name=name,
            path=str(path),
            timestamp=datetime.fromtimestamp(final_file.stat().st_mtime).isoformat(),
        )
        
        # 最佳结果
        best = data.get("best_result", {})
        summary.best_fitness = best.get("fitness", 0)
        summary.best_displacement = best.get("displacement", 0)
        summary.best_speed = best.get("speed", 0)
        summary.best_type = best.get("robot_type", "")
        summary.n_parts = best.get("n_parts", 0)
        summary.n_motors = best.get("n_motors", 0)
        
        # 历史
        history = data.get("history", [])
        summary.history = history
        summary.total_generations = len(history)
        
        if history:
            # 种群大小
            first_gen = history[0]
            summary.population_size = first_gen.get("valid_count", 
                                                   first_gen.get("population_size", 0))
            
            # 最终平均适应度
            last_gen = history[-1]
            summary.final_avg_fitness = last_gen.get("avg_fitness", 0)
            
            # 计算收敛代数
            target_fitness = summary.best_fitness * 0.9
            for i, gen in enumerate(history):
                if gen.get("best_fitness", 0) >= target_fitness:
                    summary.convergence_gen = i + 1
                    break
            
            # 改进率
            if len(history) > 1 and summary.final_avg_fitness > 0:
                first_avg = history[0].get("avg_fitness", 0)
                summary.improvement_rate = (summary.final_avg_fitness - first_avg) / len(history)
        
        return summary
    
    def compare(self) -> ComparisonResult:
        """
        执行完整对比分析
        
        Returns:
            ComparisonResult 包含所有对比结果
        """
        if len(self.experiments) < 2:
            logger.warning("Need at least 2 experiments to compare")
            # 单个实验也返回摘要
            pass
        
        result = ComparisonResult(
            experiments=list(self.experiments.values()),
            comparison_time=datetime.now().isoformat(),
        )
        
        # 排名
        names = list(self.experiments.keys())
        
        result.fitness_ranking = sorted(
            [(n, self.experiments[n].best_fitness) for n in names],
            key=lambda x: x[1], reverse=True
        )
        
        result.speed_ranking = sorted(
            [(n, self.experiments[n].best_speed) for n in names],
            key=lambda x: x[1], reverse=True
        )
        
        result.displacement_ranking = sorted(
            [(n, self.experiments[n].best_displacement) for n in names],
            key=lambda x: x[1], reverse=True
        )
        
        # 确定综合优胜者
        if result.fitness_ranking:
            result.winner = result.fitness_ranking[0][0]
        
        # 统计分析
        result.significance = self._statistical_analysis()
        
        # 建议
        result.recommendation = self._generate_recommendation(result)
        
        self.result = result
        return result
    
    def _statistical_analysis(self) -> Dict[str, Any]:
        """统计分析"""
        analysis = {}
        
        exps = list(self.experiments.values())
        if len(exps) < 2:
            return analysis
        
        # 适应度差异
        fitnesses = [e.best_fitness for e in exps]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        variance = sum((f - avg_fitness)**2 for f in fitnesses) / len(fitnesses)
        
        analysis["fitness"] = {
            "mean": round(avg_fitness, 6),
            "std": round(variance**0.5, 6),
            "cv": round(variance**0.5 / avg_fitness * 100, 1) if avg_fitness > 0 else 0,
            "range": round(max(fitnesses) - min(fitnesses), 6),
            "improvement_pct": round((max(fitnesses) - min(fitnesses)) / max(min(fitnesses), 0.001) * 100, 1),
        }
        
        # 收敛速度对比
        conv_gens = [e.convergence_gen for e in exps if e.convergence_gen > 0]
        if conv_gens:
            analysis["convergence"] = {
                "fastest": min(conv_gens),
                "slowest": max(conv_gens),
                "avg": round(sum(conv_gens) / len(conv_gens), 1),
            }
        
        # 效率评分 (fitness / generations)
        efficiency_scores = {}
        for name, exp in self.experiments.items():
            if exp.total_generations > 0:
                efficiency_scores[name] = round(exp.best_fitness / exp.total_generations * 1000, 4)
        
        if efficiency_scores:
            sorted_eff = sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True)
            analysis["efficiency"] = dict(sorted_eff)
        
        return analysis
    
    def _generate_recommendation(self, result: ComparisonResult) -> str:
        """生成建议"""
        lines = []
        
        if not result.fitness_ranking:
            return "No valid experiments to compare."
        
        winner_name = result.winner
        winner_exp = self.experiments.get(winner_name)
        
        if winner_exp:
            lines.append(f"Best overall experiment: **{winner_name}**")
            lines.append(f"  - Best Fitness: {winner_exp.best_fitness:.4f}")
            lines.append(f"  - Displacement: {winner_exp.best_displacement*100:.2f}cm")
            
            if winner_exp.convergence_gen > 0:
                lines.append(f"  - Converged at generation: {winner_exp.convergence_gen}")
        
        # 效率建议
        if "efficiency" in result.significance:
            eff = result.significance["efficiency"]
            most_efficient = max(eff.items(), key=lambda x: x[1])
            lines.append(f"\nMost efficient: **{most_efficient[0]}** (score: {most_efficient[1]})")
        
        # 差异提示
        sig = result.significance.get("fitness", {})
        if sig.get("improvement_pct", 0) > 50:
            lines.append("\nNote: Large performance gap between experiments (>50%). Consider investigating configuration differences.")
        
        return "\n".join(lines)
    
    def print_comparison(self):
        """打印对比报告到控制台"""
        if self.result is None:
            self.compare()
        
        r = self.result
        
        print("\n" + "="*70)
        print("  ForgeCraft Experiment Comparison Report")
        print("="*70)
        print(f"  Generated: {r.comparison_time[:19]}")
        print(f"  Experiments compared: {len(r.experiments)}")
        print("-"*70)
        
        # 表格头
        print(f"\n{'Experiment':<20} {'Fitness':>10} {'Disp(cm)':>10} {'Speed':>10} {'ConvGen':>8} {'Efficiency':>11}")
        print("-" * 70)
        
        for exp in r.experiments:
            eff = r.significance.get("efficiency", {}).get(exp.name, "-")
            eff_str = f"{eff:.4f}" if isinstance(eff, float) else str(eff)
            print(f"{exp.name:<20} {exp.best_fitness:>10.4f} "
                  f"{exp.best_displacement*100:>10.2f} {exp.best_speed:>10.3f} "
                  f"{exp.convergence_gen:>8} {eff_str:>11}")
        
        print("-" * 70)
        
        # 排名
        print(f"\n  Fitness Ranking:")
        for i, (name, fit) in enumerate(r.fitness_ranking, 1):
            medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
            print(f"    {medal} {name}: {fit:.4f}")
        
        # 统计
        if "fitness" in r.significance:
            sig = r.significance["fitness"]
            print(f"\n  Statistical Summary:")
            print(f"    Mean Fitness:   {sig['mean']:.4f}")
            print(f"    Std Dev:        {sig['std']:.4f}")
            print(f"    CV (%):         {sig['cv']:.1f}%")
            print(f"    Improvement:    {sig['improvement_pct']:.1f}%")
        
        # 建议
        print(f"\n  Recommendation:\n{r.recommendation}")
        
        print("\n" + "="*70 + "\n")
    
    def export_html_report(self, output_path: str = None) -> Path:
        """导出HTML对比报告"""
        if self.result is None:
            self.compare()
        
        r = self.result
        
        html = f"""<!DOCTYPE html>
<html><head><title>ForgeCraft Experiment Comparison</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 30px; background: #f5f5f5; }}
.container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
th {{ background: #4CAF50; color: white; padding: 12px; text-align: left; }}
td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
tr:hover {{ background: #f9f9f9; }}
.rank-1 {{ color: gold; font-weight: bold; }}
.rank-2 {{ color: silver; }}
.rank-3 {{ color: #cd7f32; }}
.stat-box {{ display: inline-block; padding: 15px; margin: 10px; background: #f0f7f0; border-radius: 8px; text-align: center; min-width: 120px; }}
.stat-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
.stat-label {{ font-size: 12px; color: #666; }}
</style></head><body>
<div class="container">
<h1>Experiment Comparison Report</h1>
<p>Generated: {r.comparison_time[:19]} | {len(r.experiments)} experiments</p>

<div style="margin: 20px 0;">
"""
        
        # 统计卡片
        if "fitness" in r.significance:
            sig = r.significance["fitness"]
            html += f"""
<div class="stat-box"><div class="stat-value">{sig['mean']:.4f}</div><div class="stat-label">Avg Fitness</div></div>
<div class="stat-box"><div class="stat-value">{sig['std']:.4f}</div><div class="stat-label">Std Dev</div></div>
<div class="stat-box"><div class="stat-value">{sig['improvement_pct']:.1f}%</div><div class="stat-label">Improvement</div></div>
<div class="stat-box"><div class="stat-value">{len(r.experiments)}</div><div class="stat-label">Experiments</div></div>
"""
        
        html += "</div>"
        
        # 主表格
        html += """
<table><tr><th>Rank</th><th>Experiment</th><th>Fitness</th><th>Displacement (cm)</th>
<th>Speed (m/s)</th><th>Generations</th><th>Convergence Gen</th><th>Efficiency</th></tr>
"""
        
        for rank_idx, (name, fit) in enumerate(r.fitness_ranking, 1):
            exp = self.experiments[name]
            eff = r.significance.get("efficiency", {}).get(name, "-")
            eff_str = f"{eff:.4f}" if isinstance(eff, float) else "-"
            rank_class = f"rank-{rank_idx}" if rank_idx <= 3 else ""
            
            html += f"""<tr>
<td class="{rank_class}">#{rank_idx}</td>
<td><b>{name}</b></td>
<td>{exp.best_fitness:.4f}</td>
<td>{exp.best_displacement*100:.2f}</td>
<td>{exp.best_speed:.3f}</td>
<td>{exp.total_generations}</td>
<td>{exp.convergence_gen if exp.convergence_gen else '-'}</td>
<td>{eff_str}</td>
</tr>\n"""
        
        html += "</table>"
        
        # 建议
        html += f"<h2>Recommendation</h2><p>{r.recommendation.replace(chr(10), '<br>')}</p>"
        
        # 详细历史数据
        if any(exp.history for exp in r.experiments):
            html += "<h2>Fitness History (Last 20 Generations)</h2><table><tr><th>Gen</th>"
            for exp in r.experiments:
                html += f"<th>{exp.name}</th>"
            html += "</tr>"
            
            max_len = max((len(exp.history) for exp in r.experiments), default=0)
            for i in range(max(-20, -max_len), max_len):
                html += f"<tr><td>{i+1}</td>"
                for exp in r.experiments:
                    if i < len(exp.history):
                        html += f"<td>{exp.history[i].get('best_fitness', '-'):.4f}</td>"
                    else:
                        html += "<td>-</td>"
                html += "</tr>"
            html += "</table>"
        
        html += """
<p style="text-align:center;color:#999;margin-top:30px;font-size:12px;">
ForgeCraft Robot Evolution System | Experiment Comparison Tool</p>
</div></body></html>"""
        
        if output_path is None:
            output_path = f"experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        out = Path(output_path)
        with open(out, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Comparison report exported: {out}")
        return out


def auto_compare_experiments(base_dir: str = ".") -> ComparisonResult:
    """
    自动发现并对比目录下所有实验
    
    查找包含 v11_final_results.json 的目录
    """
    base = Path(base_dir)
    comparator = ExperimentComparator()
    
    # 发现实验目录
    potential_dirs = []
    for item in base.iterdir():
        if item.is_dir() and item.name.startswith(("v", "experiment_", "run_")):
            results_file = item / "v11_final_results.json"
            if results_file.exists() or (item / "final_results.json").exists():
                potential_dirs.append(item)
    
    # 也检查子目录中的results
    for item in base.iterdir():
        if item.is_dir():
            results_subdir = item / "v11_results"
            if results_subdir.exists():
                potential_dirs.append(results_subdir)
    
    # 加载发现的实验
    for d in sorted(potential_dirs):
        name = d.name.replace("_results", "").replace("_", " ").strip()
        comparator.add_experiment(name, str(d))
    
    if len(comparator.experiments) >= 2:
        return comparator.compare()
    elif len(comparator.experiments) == 1:
        name = list(comparator.experiments.keys())[0]
        logger.info(f"Only found 1 experiment: {name}")
        return comparator.compare()
    else:
        logger.warning("No experiments found to compare")
        return ComparisonResult()


if __name__ == "__main__":
    print("="*60)
    print("  Multi-Experiment Comparison Tool")
    print("="*60 + "\n")
    
    # 自动发现并对比
    result = auto_compare_experiments(".")
    
    if result.experiments:
        # 打印报告
        comp = ExperimentComparator()
        for name, exp in result.experiments.__dict__.items() if hasattr(result.experiments, '__dict__') else []:
            pass
        
        # 直接使用已加载的数据重新创建comparator用于打印
        comp = ExperimentComparator()
        comp.experiments = {e.name: e for e in result.experiments}
        comp.result = result
        comp.print_comparison()
        
        # 导出HTML
        report_path = comp.export_html_report()
        print(f"\nHTML Report: {report_path}")
    else:
        print("No experiments found. Run some experiments first!")
        print("Usage: python experiment_compare.py")
