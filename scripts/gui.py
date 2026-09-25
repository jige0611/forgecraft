# ═══════════════════════════════════════════════════════════════
#  ForgeCraft GUI 控制面板 (Gradio)
#  ═══════════════════════════════════════════════════════════════
#
#  功能：
#  - 可视化配置进化参数
#  - 实时查看机器人3D模型
#  - 启动/监控进化实验
#  - 查看实验结果和报告
#
#  启动：python gui.py
# ═══════════════════════════════════════════════════════════════

import sys
import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from locomotion_templates import LocomotionTemplateGenerator
from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
from forgecraft.simulation.builder import build_mjcf_model
from direct_drive_evaluator import DirectDriveEvaluator, DirectDriveConfig


class ForgeCraftGUI:
    """ForgeCraft Gradio GUI"""
    
    def __init__(self):
        self.catalog = load_catalog()
        self.current_body = None
        self.evolution_thread = None
        self.evolution_running = False
        
    # ==================== Tab 1: 配置 ====================
    
    def create_config_tab(self):
        """创建配置面板"""
        with gr.Row():
            with gr.Column(scale=1):
                gens = gr.Slider(
                    minimum=10, maximum=500, value=100, step=10,
                    label="总代数 (Generations)"
                )
                pop = gr.Slider(
                    minimum=5, maximum=100, value=40, step=5,
                    label="种群大小 (Population)"
                )
                elite = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="精英数量 (Elite)"
                )
                mutation = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.8, step=0.05,
                    label="变异率 (Mutation Rate)"
                )
                
            with gr.Column(scale=1):
                sim_steps = gr.Slider(
                    minimum=500, maximum=10000, value=3000, step=100,
                    label="仿真步数 (Sim Steps)"
                )
                eval_eps = gr.Slider(
                    minimum=1, maximum=20, value=6, step=1,
                    label="评估次数 (Evaluations)"
                )
                seed = gr.Number(value=42, label="随机种子 (Seed)")
                output_dir = gr.Textbox(value="v11_results", label="输出目录")
        
        config_btn = gr.Button("保存配置", variant="primary")
        config_status = gr.Textbox(label="状态", interactive=False)
        
        config_btn.click(
            fn=self.save_config,
            inputs=[gens, pop, elite, mutation, sim_steps, eval_eps, seed, output_dir],
            outputs=config_status
        )
        
        return [gens, pop, elite, mutation, sim_steps, eval_eps, seed, output_dir]
    
    def save_config(self, gens, pop, elite, mutation, sim_steps, eval_eps, seed, output_dir):
        """保存配置到文件"""
        import yaml
        config = {
            "evolution": {
                "total_generations": int(gens),
                "population_size": int(pop),
                "elite_count": int(elite),
                "mutation_rate": float(mutation),
                "sim_steps": int(sim_steps),
                "eval_episodes": int(eval_eps)
            },
            "templates": {"seed": int(seed)},
            "output": {"results_dir": output_dir}
        }
        
        path = Path(output_dir) / "config.yaml"
        path.parent.mkdir(exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config, f)
        
        return f"✅ 配置已保存: {path}"
    
    # ==================== Tab 2: 生成器 ====================
    
    def create_generator_tab(self):
        """创建机器人生成器面板"""
        with gr.Row():
            template_type = gr.Dropdown(
                choices=["random", "differential_wheeled", "quad_wheeled", 
                        "bipedal", "quadruped", "crawler"],
                value="random",
                label="模板类型"
            )
            gen_count = gr.Slider(1, 10, value=3, step=1, label="生成数量")
            gen_seed = gr.Number(42, label="种子")
            
        gen_btn = gr.Button("🤖 生成机器人", variant="primary")
        
        with gr.Row():
            robot_gallery = gr.Gallery(label="生成的机器人", columns=3, height=300)
            
        with gr.Accordion("详细信息", open=False):
            robot_info = gr.JSON(label="机器人数据")
            robot_stats = gr.Dataframe(
                headers=["名称", "类型", "零件数", "关节数"],
                label="统计"
            )
        
        gen_btn.click(
            fn=self.generate_robots,
            inputs=[template_type, gen_count, gen_seed],
            outputs=[robot_gallery, robot_info, robot_stats]
        )
    
    def generate_robots(self, template_type, count, seed):
        """生成机器人并返回预览图"""
        from viewer import ViewerConfig, RealtimeViewer
        
        generator = LocomotionTemplateGenerator(seed=int(seed))
        
        if template_type != "random":
            for k in generator.template_weights:
                generator.template_weights[k] = 0.0
            if template_type in generator.template_weights:
                generator.template_weights[template_type] = 1.0
        
        population = generator.generate_population(size=count)
        images = []
        info_list = []
        stats_data = []
        
        for body in population:
            try:
                # 无头模式渲染
                cfg = ViewerConfig(auto_drive=False, show_hud=False)
                viewer = RealtimeViewer(body, cfg, self.catalog)
                
                if viewer.model is not None:
                    results = viewer.run_headless(duration=2.0, save_images=True)
                    
                    # 获取最后一帧图片
                    frame_path = Path(results.get('output_dir', 'viewer_output'))
                    frames = sorted(frame_path.glob('frame_*.png'))
                    if frames:
                        images.append(str(frames[-1]))
                    
                    info_list.append({
                        "name": body.name,
                        "type": getattr(body, 'robot_type', 'unknown'),
                        "parts": body.num_parts(),
                        "joints": len(body.joints())
                    })
                    
                    stats_data.append([
                        body.name,
                        getattr(body, 'robot_type', '?'),
                        str(body.num_parts()),
                        str(len(body.joints()))
                    ])
                    
            except Exception as e:
                print(f"Error rendering {body.name}: {e}")
        
        if not images:
            images = [None] * count
            for body in population:
                stats_data.append([body.name, '?', str(body.num_parts()), '?'])
        
        return images, info_list[:3], stats_data
    
    # ==================== Tab 3: 进化 ====================
    
    def create_evolution_tab(self):
        """创建进化实验面板"""
        with gr.Row():
            evo_gens = gr.Slider(10, 200, value=50, step=10, label="代数")
            evo_pop = gr.Slider(10, 50, value=20, step=5, label="种群")
            start_btn = gr.Button("▶️ 启动进化", variant="primary")
            stop_btn = gr.Button("⏹️ 停止", variant="stop")
        
        evo_progress = gr.Progress()
        evo_status = gr.Textbox(label="状态", interactive=False)
        
        with gr.Tabs():
            with gr.TabItem("实时曲线"):
                evo_plot = gr.LinePlot(
                    x="generation",
                    y="fitness",
                    title="适应度曲线",
                    height=350
                )
            
            with gr.TabItem("最佳个体"):
                best_info = gr.JSON(label="最佳个体信息")
                best_image = gr.Image(label="最佳机器人预览")
            
            with gr.TabItem("日志"):
                evo_log = gr.Textbox(label="运行日志", lines=15, max_lines=30)
        
        start_btn.click(
            fn=self.start_evolution,
            inputs=[evo_gens, evo_pop],
            outputs=[evo_status, evo_log, best_info, evo_plot]
        )
        
        stop_btn.click(fn=self.stop_evolution, outputs=evo_status)
    
    def start_evolution(self, generations, population_size):
        """启动进化实验（简化版，用于演示）"""
        if self.evolution_running:
            return ["⚠️ 进化已在运行中...", "", {}, None]
        
        self.evolution_running = True
        logs = []
        history = []
        
        # 简化版进化循环（用于演示）
        generator = LocomotionTemplateGenerator(seed=42)
        evaluator = DirectDriveEvaluator(DirectDriveConfig(
            sim_steps=500, n_frequencies=1, episodes_per_freq=1
        ))
        
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 开始进化...")
        logs.append(f"  代数: {generations}, 种群: {population_size}")
        
        best_fitness = 0
        best_result = {}
        
        for gen in range(int(generations)):
            if not self.evolution_running:
                break
            
            # 生成种群
            pop = generator.generate_population(size=int(population_size))
            
            # 评估
            gen_best_fit = 0
            valid_count = 0
            
            for i, body in enumerate(pop):
                try:
                    result = evaluator.evaluate_body(body, self.catalog, n_episodes=1)
                    if result and result.get('fitness', 0) > 0:
                        fitness = result['fitness']
                        gen_best_fit = max(gen_best_fit, fitness)
                        valid_count += 1
                        
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_result = result
                            best_result['name'] = body.name
                except Exception as e:
                    pass
            
            avg_fit = gen_best_fit / max(valid_count, 1)
            history.append({
                "generation": gen + 1,
                "fitness": round(best_fitness, 4),
                "avg_fitness": round(avg_fit, 4)
            })
            
            if (gen + 1) % 10 == 0 or gen == 0:
                logs.append(f"  Gen {gen+1}: best={best_fitness:.4f}, valid={valid_count}")
            
            time.sleep(0.01)  # 避免阻塞UI
        
        logs.append(f"\n✅ 进化完成! 最佳适应度: {best_fitness:.4f}")
        self.evolution_running = False
        
        plot_data = history if history else [{"generation": 0, "fitness": 0}]
        log_text = "\n".join(logs)
        
        return [
            "✅ 进化完成!",
            log_text,
            best_result if best_result else {},
            plot_data
        ]
    
    def stop_evolution(self):
        """停止进化"""
        self.evolution_running = False
        return "⏹️ 正在停止..."
    
    # ==================== Tab 4: 结果 ====================
    
    def create_results_tab(self):
        """创建结果查看面板"""
        with gr.Row():
            refresh_btn = gr.Button("🔄 刷新结果", variant="secondary")
            export_btn = gr.Button("📊 导出报告", variant="primary")
        
        with gr.Tabs():
            with gr.TabItem("实验摘要"):
                summary_text = gr.Markdown("""
                ## ForgeCraft 实验结果
                
                加载最新实验结果...
                """)
            
            with gr.TabItem("历史记录"):
                history_table = gr.Dataframe(
                    headers=["代数", "最佳适应度", "平均适应度", "有效个体", "位移(cm)"],
                    label="进化历史"
                )
            
            with gr.TabItem("排行榜"):
                leaderboard = gr.Dataframe(
                    headers=["排名", "名称", "类型", "适应度", "速度(m/s)", "存活率"],
                    label="Top 10 机器人"
                )
        
        refresh_btn.click(fn=self.load_results, outputs=[summary_text, history_table])
        export_btn.click(fn=self.export_report, outputs=summary_text)
        
        # 自动加载
        return self.load_results()
    
    def load_results(self):
        """加载实验结果"""
        results_dir = Path("v11_results")
        
        if not results_dir.exists():
            summary = """## ForgeCraft 实验结果
            
            ⚠️ 未找到实验结果。
            
            请先运行 `python forgecraft.py evolve` 启动进化实验。
            """
            return summary, [[]], [[]]
        
        final_file = results_dir / "v11_final_results.json"
        
        if not final_file.exists():
            summary = "## 结果目录存在，但未找到最终结果文件。"
            return summary, [[]], [[]]
        
        with open(final_file) as f:
            data = json.load(f)
        
        best = data.get("best_result", {})
        history = data.get("history", [])
        
        # 摘要
        summary = f"""## ForgeCraft 实验结果
        
### 最佳个体
| 指标 | 值 |
|------|-----|
| **适应度** | {best.get('fitness', 'N/A')} |
| **类型** | {best.get('robot_type', 'N/A')} |
| **位移** | {best.get('displacement', 0)*100:.2f} cm |
| **速度** | {best.get('speed', 0):.3f} m/s |
| **存活率** | {best.get('survival', 0)*100:.1f}% |
| **零件数** | {best.get('n_parts', 'N/A')} |
| **电机数** | {best.get('n_motors', 'N/A')} |

### 实验统计
- **总代数**: {len(history)}
- **种群规模**: {history[0].get('valid_count', 40) if history else 40}
- **评估方式**: Direct Drive + MuJoCo
- **完成时间**: {datetime.fromtimestamp(final_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}
"""
        
        # 历史表格
        hist_rows = [[
            h.get('generation', 0),
            f"{h.get('best_fitness', 0):.4f}",
            f"{h.get('avg_fitness', 0):.4f}",
            h.get('valid_count', 0),
            f"{h.get('avg_displacement', 0)*100:.3f}"
        ] for h in history[-20:]]
        
        return summary, hist_rows, [[]]
    
    def export_report(self):
        """导出HTML报告"""
        from forgecraft import cmd_report
        
        cmd_report(type('Args', (), {'output': None})())
        return "✅ 报告已导出到 v11_results/report.html"
    
    # ==================== 主界面 ====================
    
    def create_ui(self):
        """创建完整UI"""
        with gr.Blocks(
            title="ForgeCraft Robot Evolution System",
            theme=gr.themes.Soft(primary_hue="green"),
            css="""
            .main-title { text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px; }
            """
        ) as app:
            gr.HTML("""
            <div class="main-title">
                🤖 <b>ForgeCraft</b> Robot Evolution System
                <div style="font-size:14px;color:#666;margin-top:5px;">
                    Powered by MuJoCo + Numba + CasADi
                </div>
            </div>
            """)
            
            with gr.Tabs():
                with gr.TabItem("⚙️ 配置"):
                    self.create_config_tab()
                
                with gr.TabItem("🤖 生成器"):
                    self.create_generator_tab()
                
                with gr.TabItem("🧬 进化"):
                    self.create_evolution_tab()
                
                with gr.TabItem("📊 结果"):
                    self.create_results_tab()
            
            gr.HTML("""
            <div style="text-align:center;margin-top:20px;padding:10px;
                        background:#f5f5f5;border-radius:8px;font-size:12px;color:#666;">
                ForgeCraft v1.0 | CLI: python forgecraft.py --help | 
                <a href="#">Documentation</a>
            </div>
            """)
        
        return app


def launch_gui(host="127.0.0.1", port=7860, share=False):
    """启动GUI"""
    gui = ForgeCraftGUI()
    app = gui.create_ui()
    
    print("\n" + "="*60)
    print("  ForgeCraft GUI Starting...")
    print("="*60)
    print(f"  URL: http://{host}:{port}")
    if share:
        print("  Public URL: Generating...")
    print("="*60 + "\n")
    
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    
    launch_gui(args.host, args.port, args.share)
