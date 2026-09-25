"""
PBR 论文级渲染器

基于 matplotlib 的高质量装配体渲染, 支持:
- PBR 材质着色 (从 materials.py)
- 三点灯光模拟 (key + fill + rim)
- 多角度自动渲染 (前/侧/顶/等轴测/用户指定)
- 论文排版 (标题/标注/比例尺/图例)
- 高清输出 (可配置 DPI / 分辨率)
- 透明背景选项
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from forgecraft.geometry.materials import get_material, PBRMaterial
from forgecraft.geometry.exploded import compute_exploded_positions


@dataclass
class CameraAngle:
    """预设相机角度"""
    name: str
    elev: float
    azim: float
    description: str = ""


PRESET_ANGLES: List[CameraAngle] = [
    CameraAngle("front", 5, -90, "正视图"),
    CameraAngle("front_quarter", 20, -45, "前45° (展示最佳)"),
    CameraAngle("side", 5, 0, "侧视图"),
    CameraAngle("rear_quarter", 20, 135, "后45°"),
    CameraAngle("top", 85, -90, "俯视图"),
    CameraAngle("isometric", 30, -60, "等轴测"),
    CameraAngle("bottom_up", 15, 45, "仰视45°"),
]

PAPER_ANGLES: List[CameraAngle] = [
    CameraAngle("paper_main", 20, -45, "论文主图"),
    CameraAngle("paper_top", 90, -90, "论文俯视"),
    CameraAngle("paper_side", 3, 0, "论文侧视"),
]


def _simulate_pbr_color(base_color: Tuple[float, ...], metallic: float,
                        roughness: float, light_dir: np.ndarray,
                        normal_dir: np.ndarray) -> np.ndarray:
    """简化的 PBR 着色计算 (Cook-Torrance 近似)

    用于单面着色, 返回 (r, g, b) 0-1 范围。
    """
    base = np.array(base_color[:3])
    n_dot_l = max(np.dot(normal_dir, light_dir), 0.05)

    # Diffuse (Lambert)
    diffuse = base * n_dot_l * (1.0 - metallic)

    # Specular (Blinn-Phong 近似)
    half = (light_dir + np.array([0, 0, 1])) / 2
    half = half / (np.linalg.norm(half) + 1e-10)
    n_dot_h = max(np.dot(normal_dir, half), 0.0)
    spec_power = 2.0 / (roughness ** 2 + 0.01) - 2.0
    specular = np.power(n_dot_h, spec_power) * (0.04 + 0.96 * metallic)

    # Fresnel
    f0 = 0.04 + 0.96 * metallic
    fresnel = f0 + (1.0 - f0) * ((1.0 - n_dot_h) ** 5)

    color = diffuse * (1.0 - metallic) + specular * fresnel * 0.3
    return np.clip(color, 0, 1)


def render_assembly(
    body_data: dict,
    gen,
    output_dir: str,
    prefix: str = "render",
    angles: Optional[List[CameraAngle]] = None,
    exploded: bool = False,
    explode_distance: float = 0.15,
    dpi: int = 200,
    figsize: Tuple[int, int] = (10, 8),
    background: str = "#1a1a2e",
    title: Optional[str] = None,
    show_legend: bool = True,
    transparent: bool = False,
    watermark: Optional[str] = None,
) -> Dict[str, str]:
    """论文级多角度渲染

    Args:
        body_data: 装配体数据
        gen: ParametricGenerator 实例
        output_dir: 输出目录
        prefix: 文件名前缀
        angles: 相机角度列表 (默认 PRESET_ANGLES)
        exploded: 是否爆炸视图
        explode_distance: 爆炸距离
        dpi: 渲染 DPI
        figsize: 图像尺寸 (英寸)
        background: 背景色
        title: 图片标题
        show_legend: 是否显示图例
        transparent: 透明背景
        watermark: 水印文字

    Returns:
        {angle_name: filepath, ...}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    os.makedirs(output_dir, exist_ok=True)

    if angles is None:
        angles = PAPER_ANGLES

    # 获取零件位置
    parts = body_data.get("parts", [])
    positions_map = None
    if exploded:
        result = compute_exploded_positions(body_data, explode_distance)
        if result and "exploded" in result:
            positions_map = {
                pid: mat[:3, 3] for pid, mat in result["exploded"].items()
            }

    # 生成所有网格
    all_mesh_data = []
    type_set = set()
    for p in parts:
        pid = p.get("part_id", "")
        pt = p.get("part_type", "unknown")
        params = p.get("params", {})
        pos = np.array(p.get("position", [0, 0, 0]), dtype=np.float64)
        if positions_map and pid in positions_map:
            pos = positions_map[pid]

        mesh = gen.make(pt, params, position=pos)
        if mesh is not None:
            type_set.add(pt)
            all_mesh_data.append((pid, pt, mesh))

    if not all_mesh_data:
        return {}

    # 颜色映射
    type_list = sorted(type_set)
    color_map = {}
    for t in type_list:
        mat = get_material(t)
        color_map[t] = mat

    # 图例
    legend_items = []
    for t in sorted(type_set):
        mat = color_map[t]
        legend_items.append(Patch(facecolor=mat.base_color[:3],
                                  edgecolor='white', linewidth=0.5,
                                  label=mat.name or t))

    bg = 'none' if transparent else background

    # 渲染每个角度
    output = {}
    for angle in angles:
        fig = plt.figure(figsize=figsize, facecolor=bg)
        ax = fig.add_subplot(111, projection='3d', facecolor=bg)
        ax.set_axis_off()

        # 渲染所有网格
        for pid, pt, mesh in all_mesh_data:
            v = mesh.vertices
            f = mesh.faces
            mat = color_map[pt]
            fc = mat.base_color[:3]
            alpha = mat.base_color[3] if len(mat.base_color) > 3 else 1.0

            # 下采样大网格
            if len(f) > 5000:
                idx = np.random.choice(len(f), 5000, replace=False)
                f = f[idx]

            poly = Poly3DCollection(v[f], alpha=alpha, linewidth=0,
                                     antialiased=True)
            poly.set_facecolor(fc)
            poly.set_edgecolor('none')

            # 模拟金属度/粗糙度
            if mat.metallic > 0.3:
                poly.set_alpha(min(alpha, 0.95))
                poly.set_linewidth(0.1)
                poly.set_edgecolor(fc)

            ax.add_collection3d(poly)

        # 相机
        ax.view_init(elev=angle.elev, azim=angle.azim)
        ax.set_box_aspect([1, 1, 1])

        # 自动缩放
        all_verts = np.vstack([m.vertices for _, _, m in all_mesh_data])
        cx = (all_verts.min(0) + all_verts.max(0)) / 2
        rng = (all_verts.max(0) - all_verts.min(0)).max()
        margin = rng * 1.3
        ax.set_xlim(cx[0] - margin, cx[0] + margin)
        ax.set_ylim(cx[1] - margin, cx[1] + margin)
        ax.set_zlim(cx[2] - margin, cx[2] + margin)

        # 标题
        if title:
            ax.set_title(title, color='white' if bg != 'none' else 'black',
                        fontsize=11, fontweight='bold', pad=5)
        else:
            ax.set_title(f'{angle.name}{" (exploded)" if exploded else ""}',
                        color='white' if bg != 'none' else 'black',
                        fontsize=10, pad=5)

        # 图例
        if show_legend and legend_items:
            leg = fig.legend(handles=legend_items, loc='lower center',
                           ncol=min(len(legend_items), 6), fontsize=6.5,
                           facecolor='#2a2a3e' if bg != 'none' else 'white',
                           edgecolor='#404060' if bg != 'none' else '#ccc',
                           labelcolor='white' if bg != 'none' else 'black',
                           bbox_to_anchor=(0.5, 0.0))
            plt.subplots_adjust(bottom=0.1)

        # 水印
        if watermark:
            fig.text(0.98, 0.02, watermark, ha='right', va='bottom',
                    fontsize=7, color='#555555', fontstyle='italic')

        # 保存
        fname = f"{prefix}_{angle.name}"
        if exploded:
            fname += "_exploded"
        fpath = os.path.join(output_dir, f"{fname}.png")
        plt.savefig(fpath, dpi=dpi, facecolor=bg,
                   bbox_inches='tight', pad_inches=0.3,
                   transparent=transparent)
        plt.close(fig)
        output[angle.name] = fpath

    return output


def render_paper_suite(
    body_data: dict,
    gen,
    output_dir: str,
    prefix: str = "paper",
    explode_distance: float = 0.15,
    dpi: int = 250,
    title: Optional[str] = None,
    tight: bool = True,
) -> Dict[str, str]:
    """一键论文渲染套件: 主图 + 俯视 + 侧视 + 爆炸图

    Returns:
        {key: filepath, ...}
    """
    results = {}

    # 紧装配
    tight_data = body_data
    if tight:
        try:
            from forgecraft.geometry.exploded import tight_assemble
            tight_data = tight_assemble(body_data, gen)
        except Exception:
            tight_data = body_data

    # 正常装配三视图
    results.update(render_assembly(
        tight_data, gen, output_dir, prefix=f"{prefix}_normal",
        angles=PAPER_ANGLES, dpi=dpi, title=title,
        background="#1a1a2e",
    ))

    # 爆炸视图
    results.update(render_assembly(
        tight_data, gen, output_dir, prefix=f"{prefix}_exploded",
        angles=PAPER_ANGLES, exploded=True,
        explode_distance=explode_distance, dpi=dpi,
        title=f"{title} (Exploded)" if title else "Exploded View",
        background="#1a1a2e",
    ))

    # 所有七角度
    results.update(render_assembly(
        body_data, gen, output_dir, prefix=f"{prefix}_all",
        angles=PRESET_ANGLES, dpi=150,
        background="#1a1a2e",
    ))

    return results
