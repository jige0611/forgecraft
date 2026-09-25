"""
课程学习深化 — 地形渐进模块

支持环境参数渐进变化:
  代 0-10:   平地 + 低摩擦
  代 11-20:  平地 + 正常摩擦
  代 21-30:  随机坡度地形
  代 31-50:  障碍物 + 复杂地形

环境参数通过 MuJoCo XML 动态修改实现。
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

__all__ = ["TerrainStage"]

_logger = logging.getLogger(__name__)


@dataclass
class TerrainStage:
    """地形课程 — 渐进式难度升级

    从平坦到崎岖的 5 阶段地形:
      flat → bumps → slopes → stairs → mixed (随机)
    每个阶段控制: 高度/频率/间距/碎片化程度
    难度自动升级: 连续 N 代高 fitness → 进入下一阶段
    """
    name: str
    friction: float = 0.6
    slope_range: Tuple[float, float] = (0.0, 0.0)  # 坡度范围 (rad)
    obstacle_count: int = 0
    obstacle_size: float = 0.05
    description: str = ""


# ── 预设地形课程 ──────────────────────────────────────────
TERRAIN_CURRICULUM = [
    TerrainStage("平原地形", 0.4, (0, 0), 0, 0.05, "低摩擦平地 — 学习基本运动"),
    TerrainStage("标准地面", 0.6, (0, 0), 0, 0.05, "标准摩擦 — 优化步态"),
    TerrainStage("微坡地形", 0.7, (0.05, 0.15), 0, 0.05, "轻微坡度 — 学习爬坡"),
    TerrainStage("中度坡度", 0.8, (0.1, 0.3), 2, 0.08, "中坡度 + 2个障碍物"),
    TerrainStage("复杂地形", 0.9, (0.15, 0.5), 5, 0.12, "高摩擦坡地 + 5个障碍物"),
]


def get_terrain_stage(progress: float) -> TerrainStage:
    """
    根据进化进度返回对应地形阶段。

    Args:
        progress: 0.0 ~ 1.0 的进度比例
    """
    n = len(TERRAIN_CURRICULUM)
    idx = min(int(progress * n), n - 1)
    return TERRAIN_CURRICULUM[idx]


def generate_terrain_mjcf(stage: TerrainStage, size: float = 10.0) -> str:
    """生成带地形特征的 MuJoCo worldbody 片段。"""
    lines = []
    friction = stage.friction

    # 基础地面
    lines.append(f'<geom name="floor" type="plane" pos="0 0 0" size="{size} {size} 0.1"'
                 f' rgba="0.2 0.3 0.4 1" friction="{friction} 0.01 0.01"/>')

    # 随机障碍物
    if stage.obstacle_count > 0:
        import random
        rng = random.Random(42)
        for i in range(stage.obstacle_count):
            x = rng.uniform(-size * 0.4, size * 0.4)
            y = rng.uniform(-size * 0.3, size * 0.3)
            s = stage.obstacle_size * rng.uniform(0.5, 1.5)
            h = s * rng.uniform(0.5, 2.0)
            lines.append(f'<body name="obstacle_{i}" pos="{x:.2f} {y:.2f} {h/2:.3f}">')
            lines.append(f'  <geom type="box" size="{s:.3f} {s:.3f} {h/2:.3f}" '
                         f'rgba="0.6 0.3 0.2 1" mass="0.01" friction="1.0 0.01 0.01"/>')
            lines.append(f'</body>')

    # 斜坡
    if stage.slope_range[1] > 0.01:
        slope = (stage.slope_range[0] + stage.slope_range[1]) / 2.0
        ramp_l = size * 0.3
        ramp_h = 0.0  # 简化，保持平坡
        lines.append(f'<body name="ramp" pos="{size * 0.3:.1f} 0 0" euler="0 {slope:.3f} 0">')
        lines.append(f'  <geom type="box" size="3 3 0.02" rgba="0.5 0.5 0.5 1" '
                     f'friction="{friction:.1f} 0.01 0.01"/>')
        lines.append('</body>')

    return "\n".join(lines)


def setup_terrain_curriculum(total_generations: int) -> Dict[int, TerrainStage]:
    """
    返回 代数 → 地形阶段 的映射。

    Args:
        total_generations: 总进化代数

    Returns:
        {generation_start: TerrainStage} 字典
    """
    curriculum = {}
    n = len(TERRAIN_CURRICULUM)
    for i, stage in enumerate(TERRAIN_CURRICULUM):
        gen_start = int(total_generations * i / n)
        curriculum[gen_start] = stage
    return curriculum
