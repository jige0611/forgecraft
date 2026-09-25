# ══════════════════════════════════════════════════════════
# ForgeCraft — 基于强化学习的机械形态共进化系统
#
# 核心模块:
#   forgecraft.core      — 形态生成、材料库、碰撞检测、关节优化
#   forgecraft.evolution — 进化算法、MAP-Elites、分布式、超参优化
#   forgecraft.rl        — PPO 训练、形态编码器、迁移学习
#   forgecraft.simulation— MuJoCo 物理仿真
#   forgecraft.evaluation— 适应度评估、多目标优化
#   forgecraft.manufacturing — STL/STEP/URDF 制造导出
#   forgecraft.experiment— 实验管理、追踪与对比
#
# 快速开始:
#   python -m forgecraft --task speed --generations 100 --cuda
#   python -m forgecraft --hpo --hpo-trials 30
#   python -m forgecraft --list-catalogs
# ══════════════════════════════════════════════════════════

__version__ = "0.4.0"
__all__ = ["__version__"]
