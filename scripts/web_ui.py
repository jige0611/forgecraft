# ═══════════════════════════════════════════════════════════════
#  ForgeCraft Web远程界面 (Streamlit)
#  ═══════════════════════════════════════════════════════════════
#
#  功能：
#  - 远程监控进化实验状态
#  - 实时适应度曲线可视化
#  - 实验管理（启动/停止/查看结果）
#  - 多实验对比面板
#  - 系统资源监控
#  - 配置管理界面
#
#  启动：
#    streamlit run web_ui.py --server.port 8501
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import traceback

# 设置页面配置
st.set_page_config(
    page_title="ForgeCraft Robot Evolution",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
    }
    .status-running { color: #2ecc71; font-weight: bold; }
    .status-stopped { color: #e74c3c; font-weight: bold; }
    .status-idle { color: #f39c12; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent


def load_config():
    """加载项目配置"""
    try:
        from config import V11Config, load_config as cfg_load
        return cfg_load()
    except:
        return None


def get_experiments():
    """获取所有实验目录"""
    root = get_project_root()
    experiments = []
    
    for item in root.iterdir():
        if item.is_dir():
            results_file = item / "v11_final_results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    experiments.append({
                        "name": item.name,
                        "path": str(item),
                        "best_fitness": data.get("best_result", {}).get("fitness", 0),
                        "best_displacement": data.get("best_result", {}).get("displacement", 0),
                        "generations": len(data.get("history", [])),
                        "timestamp": datetime.fromtimestamp(results_file.stat().st_mtime),
                    })
                except:
                    pass
    
    experiments.sort(key=lambda x: x["timestamp"], reverse=True)
    return experiments


def get_system_info():
    """获取系统信息"""
    import platform
    
    info = {
        "Python": platform.python_version(),
        "OS": f"{platform.system()} {platform.release()}",
        "Processor": platform.processor(),
    }
    
    # 检查MuJoCo
    try:
        import mujoco
        info["MuJoCo"] = mujoco.__version__
    except:
        info["MuJoCo"] = "Not installed"
    
    # GPU信息
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            info["GPU"] = f"{gpu.name} ({gpu.total_memory // 1024}MB)"
        else:
            info["GPU"] = "No GPU detected"
    except:
        info["GPU"] = "N/A"
    
    return info


# ==================== 主界面 ====================
def main():
    st.markdown('<div class="main-header">🤖 ForgeCraft Robot Evolution System</div>', unsafe_allow_html=True)
    
    # 侧边栏导航
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "",
        ["Dashboard", "Experiments", "Compare", "Configuration", "System Info", "API Status"]
    )
    
    # 页面路由
    if page == "Dashboard":
        show_dashboard()
    elif page == "Experiments":
        show_experiments()
    elif page == "Compare":
        show_compare()
    elif page == "Configuration":
        show_configuration()
    elif page == "System Info":
        show_system_info()
    elif page == "API Status":
        show_api_status()


def show_dashboard():
    """主仪表盘"""
    st.subheader("📊 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    experiments = get_experiments()
    
    with col1:
        st.metric(
            label="Total Experiments",
            value=len(experiments),
            delta=None
        )
    
    with col2:
        total_gens = sum(e["generations"] for e in experiments)
        st.metric(
            label="Total Generations",
            value=total_gens,
            delta=None
        )
    
    with col3:
        best_fit = max((e["best_fitness"] for e in experiments), default=0)
        st.metric(
            label="Best Fitness",
            value=f"{best_fit:.4f}",
            delta=None
        )
    
    with col4:
        best_disp = max((e["best_displacement"] for e in experiments), default=0)
        st.metric(
            label="Best Displacement",
            value=f"{best_disp*100:.2f} cm",
            delta=None
        )
    
    # 最近实验
    st.subheader("📈 Recent Experiments")
    
    if experiments:
        # 显示最近5个实验的摘要数据
        recent = experiments[:5]
        
        exp_data = []
        for exp in recent:
            exp_data.append({
                "Name": exp["name"],
                "Fitness": f"{exp['best_fitness']:.4f}",
                "Displacement (cm)": f"{exp['best_displacement']*100:.2f}",
                "Generations": exp["generations"],
                "Last Modified": exp["timestamp"].strftime("%Y-%m-%d %H:%M"),
            })
        
        st.dataframe(exp_data, use_container_width=True)
        
        # 最佳个体详情
        st.subheader("🏆 Best Individual Details")
        if experiments:
            best_exp = max(experiments, key=lambda x: x["best_fitness"])
            best_path = Path(best_exp["path"]) / "v11_final_results.json"
            
            if best_path.exists():
                with open(best_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                best_result = data.get("best_result", {})
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.json({
                        "Experiment": best_exp["name"],
                        "Robot Type": best_result.get("robot_type", "-"),
                        "Parts Count": best_result.get("n_parts", 0),
                        "Motors": best_result.get("n_motors", 0),
                    })
                
                with col_b:
                    st.json({
                        "Fitness": round(best_result.get("fitness", 0), 4),
                        "Displacement (cm)": round(best_result.get("displacement", 0) * 100, 2),
                        "Speed (m/s)": round(best_result.get("speed", 0), 3),
                        "Survival Rate": f"{best_result.get('survival_rate', 0)*100:.1f}%",
                    })
    else:
        st.info("No experiments found. Run `python forgeraft.py` to start an evolution!")
    
    # 快速操作
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Run Quick Test", use_container_width=True):
            st.info("Quick test would run here...")
    
    with col2:
        if st.button("View Benchmark Results", use_container_width=True):
            bench_path = get_project_root() / "benchmark_results" / "latest_report.json"
            if bench_path.exists():
                with open(bench_path, 'r', encoding='utf-8') as f:
                    bench_data = json.load(f)
                st.json(bench_data)
            else:
                st.warning("No benchmark results found. Run `python benchmark.py` first.")
    
    with col3:
        if st.button("Export Report", use_container_width=True):
            from experiment_compare import auto_compare_experiment, ExperimentComparator
            result = auto_compare_experiment(".")
            if result and result.experiments:
                comp = ExperimentComparator()
                comp.experiments = {e.name: e for e in result.experiments}
                report_path = comp.export_html_report()
                st.success(f"Report exported to: {report_path}")
            else:
                st.warning("Not enough experiments to generate comparison.")


def show_experiments():
    """实验列表"""
    st.subheader("🧪 All Experiments")
    
    experiments = get_experiments()
    
    if not experiments:
        st.info("No experiments found.")
        return
    
    # 搜索/过滤
    search = st.text_input("Search experiments...", placeholder="Type experiment name...")
    
    filtered = [e for e in experiments if search.lower() in e["name"].lower()] if search else experiments
    
    for exp in filtered:
        with st.expander(f"📁 {exp['name']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Fitness", f"{exp['best_fitness']:.4f}")
            
            with col2:
                st.metric("Displacement", f"{exp['best_displacement']*100:.2f} cm")
            
            with col3:
                st.metric("Generations", exp['generations'])
            
            # 加载详细历史
            results_file = Path(exp["path"]) / "v11_final_results.json"
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                history = data.get("history", [])
                if history:
                    # 绘制简单折线图
                    gens = list(range(1, len(history)+1))
                    fits = [g.get("best_fitness", 0) for g in history]
                    avgs = [g.get("avg_fitness", 0) for g in history]
                    
                    st.line_chart(
                        {"Generation": gens, "Best Fitness": fits, "Avg Fitness": avgs},
                        x="Generation",
                        y=["Best Fitness", "Avg Fitness"]
                    )


def show_compare():
    """多实验对比"""
    st.subheader("🔬 Multi-Experiment Comparison")
    
    experiments = get_experiments()
    
    if len(experiments) < 2:
        st.warning("Need at least 2 experiments to compare.")
        return
    
    # 选择要对比的实验
    exp_names = [e["name"] for e in experiments]
    selected = st.multiselect(
        "Select experiments to compare:",
        exp_names,
        default=exp_names[:min(4, len(exp_names))]
    )
    
    if len(selected) < 2:
        st.error("Please select at least 2 experiments.")
        return
    
    # 对比表格
    compare_data = []
    for name in selected:
        exp = next(e for e in experiments if e["name"] == name)
        compare_data.append({
            "Experiment": name,
            "Fitness": exp["best_fitness"],
            "Disp (cm)": exp["best_displacement"] * 100,
            "Gens": exp["generations"],
        })
    
    st.dataframe(compare_data, use_container_width=True)
    
    # 条形图对比
    import pandas as pd
    df = pd.DataFrame(compare_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.bar_chart(df.set_index("Experiment")["Fitness"])
    
    with col2:
        st.bar_chart(df.set_index("Experiment")["Disp (cm)"])
    
    # 导出报告
    if st.button("Generate HTML Report"):
        from experiment_compare import ExperimentComparator
        
        comp = ExperimentComparator()
        for name in selected:
            exp = next(e for e in experiments if e["name"] == name)
            comp.add_experiment(name, exp["path"])
        
        report = comp.compare()
        path = comp.export_html_report()
        st.success(f"Report saved to: {path}")


def show_configuration():
    """配置管理"""
    st.subheader("⚙️ Configuration")
    
    config = load_config()
    
    if config is None:
        st.warning("Could not load configuration file.")
        return
    
    # 显示当前配置
    st.write("### Current Configuration")
    
    config_dict = {}
    for attr in dir(config):
        if not attr.startswith('_') and not callable(getattr(config, attr)):
            val = getattr(config, attr)
            if isinstance(val, (int, float, str, bool)):
                config_dict[attr] = val
    
    st.json(config_dict)
    
    # 编辑配置（简化版）
    st.write("### Quick Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_pop = st.number_input(
            "Population Size",
            min_value=4, max_value=256,
            value=getattr(config, 'population_size', 50)
        )
    
    with col2:
        new_gens = st.number_input(
            "Max Generations",
            min_value=1, max_value=500,
            value=getattr(config, 'max_generations', 50)
        )
    
    if st.button("Save Configuration"):
        st.info(f"Would save: pop={new_pop}, gens={new_gens}")
        st.warning("Configuration editing requires direct YAML modification for now.")


def show_system_info():
    """系统信息"""
    st.subheader("💻 System Information")
    
    info = get_system_info()
    
    cols = st.columns(len(info))
    for i, (key, value) in enumerate(info.items()):
        with cols[i % len(cols)]:
            st.metric(key, value)
    
    # 项目文件结构
    st.subheader("📂 Project Structure")
    
    root = get_project_root()
    files = []
    
    for item in sorted(root.iterdir()):
        if item.is_file() and item.suffix == '.py':
            size = item.stat().st_size
            mtime = datetime.fromtimestamp(item.stat().st_mtime)
            files.append({
                "File": item.name,
                "Size (KB)": round(size / 1024, 1),
                "Modified": mtime.strftime("%Y-%m-%d %H:%M"),
            })
    
    if files:
        st.dataframe(files, use_container_width=True)


def show_api_status():
    """API状态检查"""
    st.subheader("🌐 API Service Status")
    
    # 检查各模块是否可用
    modules = [
        ("Core Engine", "forgecraft_v11"),
        ("Part Catalog", "parameterized_parts_v2.yaml"),
        ("Template Generator", "locomotion_templates.py"),
        ("Model Builder", "model_builder.py"),
        ("Evaluator", "direct_drive_evaluator.py"),
        ("Viewer", "viewer.py"),
        ("CLI Entry", "forgecraft.py"),
        ("Config Manager", "config.py"),
        ("Checkpoint System", "checkpoint.py"),
        ("GUI Panel", "gui.py"),
        ("Dashboard", "dashboard.py"),
        ("URDF Exporter", "urdf_export.py"),
        ("Benchmark Suite", "benchmark.py"),
        ("Replay System", "replay.py"),
        ("Exp Comparator", "experiment_compare.py"),
        ("FastAPI Server", "api.py"),
        ("Docker Config", "Dockerfile"),
    ]
    
    root = get_project_root()
    
    status_data = []
    for name, filename in modules:
        filepath = root / filename
        exists = filepath.exists()
        
        status_data.append({
            "Module": name,
            "Status": "✅ Available" if exists else "❌ Missing",
            "Path": filename,
        })
    
    st.dataframe(status_data, use_container_width=True)
    
    # API端点说明
    st.subheader("📡 API Endpoints")
    
    endpoints = """
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | System status |
| GET | `/api/experiments` | List experiments |
| POST | `/api/evolve/start` | Start evolution |
| POST | `/api/evolve/stop` | Stop evolution |
| GET | `/api/results/{id}` | Get results |
| GET | `/api/config` | Get configuration |
| PUT | `/api/config` | Update configuration |
| GET | `/api/benchmark` | Benchmark results |
| GET | `/api/export/{id}` | Export robot URDF |
"""
    
    st.markdown(endpoints)


if __name__ == "__main__":
    main()
