# ══════════════════════════════════════════════════════════
# 🔍 最佳工具链安装验证与报告生成器
#
# 检测所有已安装的工具，生成完整的环境报告
#
# ══════════════════════════════════════════════════════════

import sys
import subprocess
import platform
from datetime import datetime


def check_package(package_name, import_name=None):
    """检查包是否已安装并获取版本"""
    try:
        module = __import__(import_name or package_name)
        version = getattr(module, '__version__', 'Unknown')
        return True, version
    except ImportError:
        return False, None


def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return True, f"{gpu_name} ({vram:.1f} GB VRAM)"
        return False, "CUDA not available"
    except:
        return False, "PyTorch not installed"


def main():
    print("=" * 70)
    print("🔍 V8 机器人项目 - 最佳工具链验证报告")
    print("=" * 70)
    print(f"\n📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 操作系统: {platform.system()} {platform.release()}")
    print(f"🐍 Python版本: {sys.version.split()[0]}")
    print(f"🖥️  处理器: {platform.processor()}")
    
    # 检查GPU
    print("\n" + "-" * 70)
    print("🎮 GPU硬件信息")
    print("-" * 70)
    
    cuda_available, cuda_info = check_cuda()
    status = "✅" if cuda_available else "⚠️"
    print(f"{status} CUDA/GPU: {cuda_info}")
    
    # 定义要检查的包
    packages = {
        '核心计算': {
            'numpy': ('numpy', 'NumPy'),
            'scipy': ('scipy', 'SciPy'),
            'pandas': ('pandas', 'Pandas'),
        },
        '机器学习': {
            'scikit-learn': ('sklearn', 'Scikit-Learn'),
            'torch': ('torch', 'PyTorch'),
            'torchvision': ('torchvision', 'TorchVision'),
        },
        '强化学习': {
            'gymnasium': ('gymnasium', 'Gymnasium'),
            'stable_baselines3': ('stable_baselines3', 'SB3'),
            'optuna': ('optuna', 'Optuna'),
        },
        '可视化': {
            'matplotlib': ('matplotlib', 'Matplotlib'),
            'seaborn': ('seaborn', 'Seaborn'),
            'plotly': ('plotly', 'Plotly'),
        },
        '3D处理': {
            'trimesh': ('trimesh', 'TriMesh'),
            'PIL': ('PIL', 'Pillow'),
            'cv2': ('cv2', 'OpenCV'),
        },
        '物理仿真': {
            'mujoco': ('mujoco', 'MuJoCo'),
        },
        '工具库': {
            'tqdm': ('tqdm', 'TQDM'),
            'yaml': ('yaml', 'PyYAML'),
            'pytest': ('pytest', 'PyTest'),
        }
    }
    
    total_checked = 0
    total_installed = 0
    
    for category, pkgs in packages.items():
        print(f"\n{'─' * 70}")
        print(f"📦 {category}")
        print(f"{'─' * 70}")
        
        for pkg_key, (pkg_import, display_name) in pkgs.items():
            installed, version = check_package(pkg_key, pkg_import)
            total_checked += 1
            
            if installed:
                total_installed += 1
                icon = "✅"
                version_str = f"v{version}" if version else ""
            else:
                icon = "❌"
                version_str = "未安装"
            
            print(f"  {icon} {display_name:<20s} {version_str}")
    
    # 统计摘要
    print("\n" + "=" * 70)
    print("📊 安装统计")
    print("=" * 70)
    
    install_rate = (total_installed / total_checked) * 100
    
    print(f"\n   总计检测: {total_checked} 个包")
    print(f"   已成功安装: {total_installed} 个")
    print(f"   安装成功率: {install_rate:.1f}%")
    
    if install_rate >= 90:
        grade = "A+"
        status = "🏆 完美！所有核心工具已就绪"
    elif install_rate >= 80:
        grade = "A"
        status = "✨ 优秀！环境配置完成"
    elif install_rate >= 70:
        grade = "B+"
        status = "👍 良好！大部分工具可用"
    elif install_rate >= 60:
        grade = "B"
        status = "⚠️ 基本可用，建议补充部分工具"
    else:
        grade = "C"
        status = "❌ 需要重新检查安装"
    
    print(f"\n   等级评定: {grade}")
    print(f"   状态: {status}")
    
    # 功能能力清单
    print("\n" + "=" * 70)
    print("🚀 已解锁的能力")
    print("=" * 70)
    
    capabilities = [
        ("深度学习训练", check_package('torch')[0]),
        ("GPU加速推理", cuda_available),
        ("强化学习 (PPO/SAC)", check_package('stable_baselines3')[0]),
        ("超参数优化", check_package('optuna')[0]),
        ("科学可视化", check_package('matplotlib')[0] and check_package('plotly')[0]),
        ("物理仿真", check_package('mujoco')[0]),
        ("图像处理", check_package('cv2')[0]),
        ("进化算法优化", True),  # 自定义实现
    ]
    
    for cap_name, available in capabilities:
        icon = "✅" if available else "❌"
        print(f"  {icon} {cap_name}")
    
    # 下一步建议
    print("\n" + "=" * 70)
    print("💡 后续可选安装（高级功能）")
    print("=" * 70)
    
    optional_tools = [
        ("Open3D", "点云处理/ICP配准", "pip install open3d"),
        ("pythonocc-core", "B-rep精确建模", "pip install pythonocc-core"),
        ("CadQuery", "参数化CAD设计", "pip install cadquery"),
        ("Isaac Lab", "NVIDIA GPU仿真加速", "从官网下载"),
        ("Blender", "专业3D渲染", "https://blender.org/download"),
        ("FreeCAD", "工程制图/CAD", "winget install FreeCAD.FreeCAD"),
    ]
    
    for tool_name, desc, install_cmd in optional_tools:
        print(f"\n  📌 {tool_name}: {desc}")
        print(f"     安装命令: {install_cmd}")
    
    print("\n" + "=" * 70)
    if install_rate >= 80:
        print("✅ 环境配置完成！可以开始使用最佳工具链进行开发")
    else:
        print("⚠️ 部分工具缺失，请根据上述提示补充安装")
    print("=" * 70)
    
    return 0 if install_rate >= 80 else 1


if __name__ == "__main__":
    exit(main())
