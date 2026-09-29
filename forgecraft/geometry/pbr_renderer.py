"""
离线装配体渲染器 (matplotlib 3D)

基于 matplotlib 的装配体离线渲染, 支持:
- 逐面 PBR 近似着色 (基色/金属度/粗糙度来自 materials.py)
- 三点灯光模拟 (key + fill + rim)
- 多角度自动渲染 (前/侧/顶/等轴测/用户指定)
- 排版元素: 标题 / 图例 / 可选水印
- 可配置 DPI 与图幅尺寸
- 透明背景选项

说明: 这是逐面着色 (Lambert 漫反射 + Blinn-Phong 高光 + Fresnel),
视线方向固定为 +Z, **不是**光线追踪或基于图像的光照, 不适用于材质对比。
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

# 三点光照 (key / fill / rim), 方向为世界坐标, 权重之和为 1.0
_LIGHTS: List[Tuple[np.ndarray, float]] = [
    (np.array([0.45, -0.70, 0.55]), 0.60),   # key  主光 (左前上方)
    (np.array([-0.65, -0.25, 0.30]), 0.26),  # fill 补光 (右前方, 压低阴影)
    (np.array([-0.10, 0.80, 0.45]), 0.14),   # rim  轮廓光 (后方, 勾勒边缘)
]


def _simulate_pbr_color(base_color: Tuple[float, ...], metallic: float,
                        roughness: float, light_dir: np.ndarray,
                        normal_dir: np.ndarray) -> np.ndarray:
    """简化的 PBR 着色计算 (单光源 Cook-Torrance 近似)

    支持逐面批量着色: ``normal_dir`` 传 (N, 3) 面法线时返回 (N, 3) 颜色,
    传单个 (3,) 法线时返回 (3,) 颜色。取值均在 0-1 范围。

    视线方向固定为 +Z (正交相机假设), 因此高光使用 Blinn-Phong 半程向量近似。
    """
    base = np.asarray(base_color, dtype=np.float64)[:3]
    ld = np.asarray(light_dir, dtype=np.float64)
    ld = ld / (np.linalg.norm(ld) + 1e-10)

    n = np.atleast_2d(np.asarray(normal_dir, dtype=np.float64))

    # Lambert 漫反射: 余弦项截断到 0.05, 避免背光面完全死黑
    n_dot_l = np.clip(n @ ld, 0.05, None)

    # 镜面高光 (Blinn-Phong 半程向量)
    half = ld + np.array([0.0, 0.0, 1.0])
    half = half / (np.linalg.norm(half) + 1e-10)
    n_dot_h = np.clip(n @ half, 0.0, None)
    spec_power = 2.0 / (roughness ** 2 + 0.01) - 2.0

    # Fresnel-Schlick
    f0 = 0.04 + 0.96 * metallic
    fresnel = f0 + (1.0 - f0) * np.power(1.0 - n_dot_h, 5.0)
    specular = np.power(n_dot_h, spec_power) * fresnel

    diffuse = n_dot_l[:, None] * base[None, :] * (1.0 - metallic)
    # 金属高光带基色 (金属反射着色), 非金属高光偏白
    spec = specular[:, None] * (base * metallic + (1.0 - metallic))[None, :] * 0.6

    color = np.clip(diffuse + spec, 0.0, 1.0)
    if np.ndim(normal_dir) == 1:
        return color[0]
    return color


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
    """多角度离线渲染

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
            fc = np.asarray(mat.base_color[:3], dtype=np.float64)
            alpha = mat.base_color[3] if len(mat.base_color) > 3 else 1.0

            # 下采样大网格 (均匀取样, 保证渲染结果可复现)
            if len(f) > 5000:
                idx = np.linspace(0, len(f) - 1, 5000).astype(np.int64)
                f = f[idx]

            # 逐面法线 -> 简化 PBR 着色 (三光源加权叠加)
            tri = v[f]
            fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            fl = np.linalg.norm(fn, axis=1)
            fl[fl < 1e-12] = 1.0
            fn = fn / fl[:, None]

            lit = np.tile(fc * (0.30 + 0.20 * (1.0 - mat.roughness)), (len(f), 1))
            for ldir, weight in _LIGHTS:
                lit += weight * _simulate_pbr_color(
                    fc, mat.metallic, mat.roughness, ldir, fn
                )
            lit = np.clip(lit, 0.0, 1.0)

            poly = Poly3DCollection(v[f], linewidth=0, antialiased=True)
            poly.set_facecolor(np.column_stack([lit, np.full(len(f), float(alpha))]))
            poly.set_edgecolor('none')

            # 模拟金属度/粗糙度
            if mat.metallic > 0.3:
                poly.set_alpha(min(alpha, 0.95))
                poly.set_linewidth(0.1)
                poly.set_edgecolor('none')

            ax.add_collection3d(poly)

        # 相机
        ax.view_init(elev=angle.elev, azim=angle.azim)
        ax.set_box_aspect([1, 1, 1])

        # 自动缩放: 以装配体包围盒的立方体中心取景, 仅留 12% 边距
        # (此前用 margin = rng * 1.3, 使包围盒达到模型的 2.6 倍, 模型只占画面 ~19%)
        all_verts = np.vstack([m.vertices for _, _, m in all_mesh_data])
        vmin = all_verts.min(0)
        vmax = all_verts.max(0)
        cx = (vmin + vmax) / 2
        rng = float((vmax - vmin).max())
        if not np.isfinite(rng) or rng <= 0:
            rng = 1.0
        half = rng * 0.56  # 半边长 -> 全边长 1.12 * rng, 模型占画面约 89%
        ax.set_xlim(cx[0] - half, cx[0] + half)
        ax.set_ylim(cx[1] - half, cx[1] + half)
        ax.set_zlim(cx[2] - half, cx[2] + half)

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
    """一键展示渲染套件: 正常三视图 + 爆炸三视图 + 七角度全视图

    注意: 三次 render_assembly 使用相同角度名 (paper_main 等), 若直接 update
    会互相覆盖, 返回的映射里只剩 10 项而磁盘上有 13 张图。
    这里为每批结果加上 normal_ / exploded_ / all_ 前缀, 保证返回项与文件一一对应。

    Returns:
        {key: filepath, ...}  # 共 13 项
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
    for key, path in render_assembly(
        tight_data, gen, output_dir, prefix=f"{prefix}_normal",
        angles=PAPER_ANGLES, dpi=dpi, title=title,
        background="#1a1a2e",
    ).items():
        results[f"normal_{key}"] = path

    # 爆炸视图
    for key, path in render_assembly(
        tight_data, gen, output_dir, prefix=f"{prefix}_exploded",
        angles=PAPER_ANGLES, exploded=True,
        explode_distance=explode_distance, dpi=dpi,
        title=f"{title} (Exploded)" if title else "Exploded View",
        background="#1a1a2e",
    ).items():
        results[f"exploded_{key}"] = path

    # 所有七角度
    for key, path in render_assembly(
        body_data, gen, output_dir, prefix=f"{prefix}_all",
        angles=PRESET_ANGLES, dpi=150,
        background="#1a1a2e",
    ).items():
        results[f"all_{key}"] = path

    return results
