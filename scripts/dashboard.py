# ═══════════════════════════════════════════════════════════════
#  ForgeCraft 进化可视化仪表盘 (Plotly + 实时更新)
#  ═══════════════════════════════════════════════════════════════

import json
import numpy as np
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class EvolutionDashboard:
    """进化可视化仪表盘"""
    
    def __init__(self, results_dir: str = "v11_results"):
        self.results_dir = Path(results_dir)
        self.history_data = []
        self._load_history()
    
    def _load_history(self):
        """加载历史数据"""
        final_file = self.results_dir / "v11_final_results.json"
        
        if final_file.exists():
            with open(final_file) as f:
                data = json.load(f)
            self.history_data = data.get("history", [])
    
    def create_fitness_plot(self):
        """创建适应度曲线图"""
        if not PLOTLY_AVAILABLE or not self.history_data:
            return None
        
        generations = [h.get('generation', i+1) for i, h in enumerate(self.history_data)]
        best_fitness = [h.get('best_fitness', 0) for h in self.history_data]
        avg_fitness = [h.get('avg_fitness', 0) for h in self.history_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=generations,
            y=best_fitness,
            mode='lines',
            name='Best Fitness',
            line=dict(color='#2ecc71', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=generations,
            y=avg_fitness,
            mode='lines',
            name='Avg Fitness',
            line=dict(color='#3498db', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Evolution Fitness Progress",
            xaxis_title="Generation",
            yaxis_title="Fitness",
            template="plotly_white",
            hovermode="x unified"
        )
        
        return fig
    
    def create_displacement_plot(self):
        """创建位移曲线图"""
        if not PLOTLY_AVAILABLE or not self.history_data:
            return None
        
        generations = [h.get('generation', i+1) for i, h in enumerate(self.history_data)]
        avg_disp = [h.get('avg_displacement', 0)*100 for h in self.history_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=generations,
            y=avg_disp,
            fill='tozeroy',
            name='Avg Displacement (cm)',
            line=dict(color='#e74c3c', width=2)
        ))
        
        fig.update_layout(
            title="Average Displacement Over Generations",
            xaxis_title="Generation",
            yaxis_title="Displacement (cm)",
            template="plotly_white"
        )
        
        return fig
    
    def create_diversity_plot(self):
        """创建多样性指标图（基于有效个体数）"""
        if not PLOTLY_AVAILABLE or not self.history_data:
            return None
        
        generations = [h.get('generation', i+1) for i, h in enumerate(self.history_data)]
        valid_count = [h.get('valid_count', 0) for h in self.history_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=generations,
            y=valid_count,
            name='Valid Individuals',
            marker_color='#9b59b6'
        ))
        
        fig.update_layout(
            title="Population Validity",
            xaxis_title="Generation",
            yaxis_title="Valid Count",
            template="plotly_white"
        )
        
        return fig
    
    def create_combined_dashboard(self):
        """创建综合仪表盘"""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Fitness Progress', 'Displacement',
                'Population Validity', 'Type Distribution'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "pie"}]
            ]
        )
        
        # 适应度曲线
        if self.history_data:
            gens = [h['generation'] for h in self.history_data]
            
            fig.add_trace(go.Scatter(
                x=gens, y=[h['best_fitness'] for h in self.history_data],
                mode='lines', name='Best', line=dict(color='#2ecc71')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=gens, y=[h['avg_fitness'] for h in self.history_data],
                mode='lines', name='Avg', line=dict(color='#3498db', dash='dash')
            ), row=1, col=1)
            
            # 位移
            fig.add_trace(go.Scatter(
                x=gens, y=[h.get('avg_displacement', 0)*100 for h in self.history_data],
                fill='tozeroy', name='Disp (cm)', line=dict(color='#e74c3c')
            ), row=1, col=2)
            
            # 有效个体数
            fig.add_trace(go.Bar(
                x=gens, y=[h.get('valid_count', 0) for h in self.history_data],
                name='Valid', marker_color='#9b59b6'
            ), row=2, col=1)
        
        fig.update_layout(
            height=800,
            title_text="ForgeCraft Evolution Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def export_html(self, output_path: str = "v11_results/dashboard.html"):
        """导出为HTML交互式图表"""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return
        
        dashboard = self.create_combined_dashboard()
        if dashboard:
            dashboard.write_html(output_path)
            print(f"Dashboard saved: {output_path}")


def create_realtime_visualizer():
    """
    创建实时可视化器（用于运行中的进化实验）
    
    使用方法：
        viz = create_realtime_visualizer()
        viz.update(generation, fitness_dict)
    """
    
    class RealtimeVisualizer:
        def __init__(self):
            self.data = {
                'generations': [],
                'best_fitness': [],
                'avg_fitness': [],
                'displacement': [],
                'survival': []
            }
        
        def update(self, generation, metrics: dict):
            """添加新数据点"""
            self.data['generations'].append(generation)
            self.data['best_fitness'].append(metrics.get('best_fitness', 0))
            self.data['avg_fitness'].append(metrics.get('avg_fitness', 0))
            self.data['displacement'].append(metrics.get('displacement', 0))
            self.data['survival'].append(metrics.get('survival', 0))
        
        def get_summary(self):
            """获取当前摘要统计"""
            if not self.data['generations']:
                return {}
            
            return {
                'current_gen': self.data['generations'][-1],
                'best_fitness': max(self.data['best_fitness']),
                'avg_fitness_last': self.data['avg_fitness'][-1] if self.data['avg_fitness'] else 0,
                'improvement_rate': self._calc_improvement_rate(),
            }
        
        def _calc_improvement_rate(self):
            """计算改进率"""
            if len(self.data['best_fitness']) < 10:
                return 0.0
            
            recent = np.array(self.data['best_fitness'][-10:])
            older = np.array(self.data['best_fitness'][-20:-10]) if len(self.data['best_fitness']) >= 20 else recent
            
            return float(np.mean(recent) - np.mean(older))
        
        def to_json(self):
            """序列化为JSON"""
            return json.dumps(self.data, indent=2)
    
    return RealtimeVisualizer()


if __name__ == "__main__":
    # 测试仪表盘
    dash = EvolutionDashboard()
    
    if PLOTLY_AVAILABLE:
        fig = dash.create_combined_dashboard()
        if fig:
            fig.show()
            dash.export_html()
    else:
        print("Install plotly for visualization: pip install plotly")
