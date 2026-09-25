"""
2D 工程图纸生成器

生成符合 GB/T 14689 标准的 2D 工程图纸:
- 三视图正交投影 (正视/俯视/侧视)
- 关键尺寸标注 (线性/径向)
- 公差带 (IT6-IT12)
- ISO 标准标题栏
- 输出格式: PNG (300 DPI) / PDF / DXF

加工厂可据此编制工艺卡片和 NC 程序。
"""

from __future__ import annotations

import datetime
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Rectangle
from matplotlib.lines import Line2D


# ================================================================
# 工程图纸预设
# ================================================================

# ISO A4 尺寸 (mm)
SHEET_SIZES = {
    "A4": (297, 210),
    "A3": (420, 297),
    "A4_L": (210, 297),  # 纵向
}

# 标准公差等级 (IT6-IT12 for 1-30mm)
TOLERANCE_TABLE = {
    "IT6": {10: 0.009, 18: 0.011, 30: 0.013},
    "IT7": {10: 0.015, 18: 0.018, 30: 0.021},
    "IT8": {10: 0.022, 18: 0.027, 30: 0.033},
    "IT9": {10: 0.036, 18: 0.043, 30: 0.052},
    "IT10": {10: 0.058, 18: 0.070, 30: 0.084},
    "IT11": {10: 0.090, 18: 0.110, 30: 0.130},
    "IT12": {10: 0.150, 18: 0.180, 30: 0.210},
}


def _get_tolerance(dim_mm: float, grade: str = "IT8") -> float:
    """获取标准公差值"""
    table = TOLERANCE_TABLE.get(grade, TOLERANCE_TABLE["IT8"])
    for upper, tol in table.items():
        if dim_mm <= upper:
            return tol
    return dim_mm * 0.001


def _orthographic_projection(vertices: np.ndarray, faces: np.ndarray,
                              view: str = "front") -> Tuple[np.ndarray, np.ndarray]:
    """正交投影: 提取 2D 轮廓边缘"""
    if view == "front":
        proj = np.column_stack([vertices[:, 0], vertices[:, 2]])
    elif view == "top":
        proj = np.column_stack([vertices[:, 0], vertices[:, 1]])
    elif view == "side":
        proj = np.column_stack([-vertices[:, 1], vertices[:, 2]])
    elif view == "bottom":
        proj = np.column_stack([vertices[:, 0], -vertices[:, 1]])
    else:
        proj = np.column_stack([vertices[:, 0], vertices[:, 2]])

    return proj, faces


def _compute_bbox_2d(pts: np.ndarray) -> Tuple[float, float, float, float]:
    """2D 边界框: (min_x, max_x, min_y, max_y)"""
    return (float(pts[:, 0].min()), float(pts[:, 0].max()),
            float(pts[:, 1].min()), float(pts[:, 1].max()))


def _extract_outline_edges(proj_pts: np.ndarray, faces: np.ndarray) -> List[Tuple]:
    """从投影后的网格提取轮廓边 (仅保留外轮廓)"""
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edge = (min(a, b), max(a, b))
            if edge in edges:
                edges.discard(edge)  # 移除共享边 (内边)
            else:
                edges.add(edge)
    return [(proj_pts[a], proj_pts[b]) for a, b in edges]


def _draw_dimension_line(ax, x1, y1, x2, y2, offset: float,
                          text: str, color='#1a1a1a'):
    """绘制尺寸标注线和箭头"""
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx * dx + dy * dy)

    if dist < 1e-6:
        return

    # 偏移方向 (垂直于尺寸线)
    nx, ny = -dy / dist, dx / dist

    # 延伸线
    ax.plot([x1, x1 + nx * offset], [y1, y1 + ny * offset],
            'k-', linewidth=0.4, alpha=0.7)
    ax.plot([x2, x2 + nx * offset], [y2, y2 + ny * offset],
            'k-', linewidth=0.4, alpha=0.7)

    # 尺寸线
    ax.plot([x1 + nx * offset, x2 + nx * offset],
            [y1 + ny * offset, y2 + ny * offset],
            'k-', linewidth=0.6)

    # 箭头
    arr_len = min(dist * 0.08, offset * 0.4)
    for ex, ey in [(x1 + nx * offset, y1 + ny * offset),
                    (x2 + nx * offset, y2 + ny * offset)]:
        ax.arrow(ex, ey, nx * arr_len * (1 if ex == x1 + nx * offset else -1),
                ny * arr_len * (1 if ey == y1 + ny * offset else -1),
                head_width=arr_len * 0.8, head_length=arr_len * 0.6,
                fc='k', ec='k', linewidth=0.3)

    # 标注文字
    mid_x = (x1 + x2) / 2 + nx * offset * 1.2
    mid_y = (y1 + y2) / 2 + ny * offset * 1.2
    ax.text(mid_x, mid_y, text, fontsize=5, ha='center', va='center',
            color=color, fontfamily='monospace')


def _draw_title_block(fig, sheet_w, sheet_h, info: dict):
    """绘制 ISO 标题栏"""
    # 标题栏位于右下角
    tb_w = 180  # mm
    tb_h = 56   # mm
    left = sheet_w - tb_w
    bottom = 0

    # 创建标题栏轴
    ax_tb = fig.add_axes([left / sheet_w, bottom / sheet_h,
                          tb_w / sheet_w, tb_h / sheet_h])
    ax_tb.set_xlim(0, tb_w)
    ax_tb.set_ylim(0, tb_h)
    ax_tb.set_aspect('equal')
    ax_tb.axis('off')

    # 分区线
    rows = [0, tb_h * 0.25, tb_h * 0.50, tb_h * 0.75, tb_h]
    cols = [0, tb_w * 0.35, tb_w * 0.55, tb_w * 0.75, tb_w]

    for y in rows:
        ax_tb.axhline(y, color='black', linewidth=0.5)
    for x in cols:
        ax_tb.axvline(x, color='black', linewidth=0.5)

    # 文字
    texts = [
        (cols[0] + 2, rows[2] + 5, "DRAWN", 4, 'left'),
        (cols[0] + 2, rows[1] + 5, info.get("designer", "ForgeCraft"), 5, 'left'),
        (cols[1] + 2, rows[2] + 5, "DATE", 4, 'left'),
        (cols[1] + 2, rows[1] + 5, info.get("date", ""), 5, 'left'),
        (cols[2] + 2, rows[2] + 5, "MATERIAL", 4, 'left'),
        (cols[2] + 2, rows[1] + 5, info.get("material", "Various"), 5, 'left'),
        (cols[3] + 2, rows[2] + 5, "SCALE", 4, 'left'),
        (cols[3] + 2, rows[1] + 5, info.get("scale", "1:1"), 5, 'left'),
        (tb_w / 2, rows[3] + tb_h * 0.12, info.get("title", "ASSEMBLY"), 7, 'center'),
        (tb_w / 2, tb_h * 0.05, info.get("part_id", ""), 5, 'center'),
    ]
    for x, y, txt, fs, ha in texts:
        ax_tb.text(x, y, txt, fontsize=fs, ha=ha, va='center',
                   fontfamily='monospace')


def generate_engineering_drawing(
    body_data: dict,
    gen,
    output_dir: str,
    prefix: str = "drawing",
    sheet_size: str = "A3",
    tolerance_grade: str = "IT8",
    dpi: int = 300,
) -> Dict[str, str]:
    """生成 2D 工程图纸

    Args:
        body_data: 装配体数据
        gen: ParametricGenerator
        output_dir: 输出目录
        prefix: 文件名前缀
        sheet_size: 图纸尺寸 (A3/A4)
        tolerance_grade: 公差等级 (IT6-IT12)
        dpi: 渲染 DPI

    Returns:
        {view_name: filepath, ...}
    """
    os.makedirs(output_dir, exist_ok=True)

    sheet_w, sheet_h = SHEET_SIZES.get(sheet_size, SHEET_SIZES["A3"])
    result = {}

    parts = body_data.get("parts", [])
    joints = body_data.get("joints", [])

    # ── 装配体三视图 ──
    # 收集所有投影点
    all_pts = {v: [] for v in ["front", "top", "side"]}
    all_edges = {v: [] for v in ["front", "top", "side"]}
    part_info = {}

    for p in parts:
        pt = p.get("part_type", "unknown")
        pid = p.get("part_id", "unknown")
        params = p.get("params", {})
        pos = np.array(p.get("position", [0, 0, 0]), dtype=np.float64)
        mesh = gen.make(pt, params, position=pos)
        if mesh is None:
            continue

        for view in ["front", "top", "side"]:
            proj, faces = _orthographic_projection(mesh.vertices, mesh.faces, view)
            edges = _extract_outline_edges(proj, faces)
            all_edges[view].extend(edges)
            all_pts[view].append(proj)
            part_info[pid] = {"type": pt, "bbox_3d": (
                mesh.vertices.min(0), mesh.vertices.max(0))}

    # ── 生成整体三视图 ──
    views_map = [
        ("front", "FRONT VIEW", 0.25, 0.55),
        ("top", "TOP VIEW", 0.25, 0.05),
        ("side", "RIGHT VIEW", 0.75, 0.55),
    ]

    fig = plt.figure(figsize=(sheet_w / 25.4, sheet_h / 25.4),
                     facecolor='white')
    fig.suptitle(prefix.upper(), fontsize=11, fontweight='bold',
                 fontfamily='monospace', y=0.98)

    for view_name, label, vx, vy in views_map:
        if not all_edges[view_name]:
            continue

        ax = fig.add_axes([vx, vy, 0.35, 0.35])
        ax.set_aspect('equal')
        ax.set_title(label, fontsize=7, fontfamily='monospace')

        for e in all_edges[view_name]:
            ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]],
                    'k-', linewidth=0.5, alpha=0.8)

        # 自动尺寸标注
        if all_pts[view_name]:
            pts = np.vstack(all_pts[view_name])
            bx = _compute_bbox_2d(pts)
            w = bx[1] - bx[0]
            h = bx[3] - bx[2]

            offset = max(w, h) * 0.18

            # 水平尺寸
            tol = _get_tolerance(w * 1000, tolerance_grade)
            _draw_dimension_line(ax, bx[0], bx[2] - offset * 0.1,
                                bx[1], bx[2] - offset * 0.1, -offset,
                                f"{w * 1000:.1f}", '#2255aa')

            # 垂直尺寸
            _draw_dimension_line(ax, bx[0] - offset * 0.1, bx[2],
                                bx[0] - offset * 0.1, bx[3], -offset,
                                f"{h * 1000:.1f}", '#2255aa')

            # 公差标注 (在右下角)
            ax.text(bx[1] + offset * 0.3, bx[2],
                    f"TOL: ±{tol:.3f} mm\nGRADE: {tolerance_grade}",
                    fontsize=4, color='#666', fontfamily='monospace')

        ax.axis('off')

    # 标题栏
    info = {
        "title": f"FORGECRAFT {prefix.upper()} ASSEMBLY",
        "part_id": prefix,
        "designer": "ForgeCraft AI",
        "date": datetime.date.today().strftime("%Y-%m-%d"),
        "material": "See BOM",
        "scale": "1:1",
    }
    _draw_title_block(fig, sheet_w / 25.4 * 100, sheet_h / 25.4 * 100, info)

    # 保存 PNG
    png_path = os.path.join(output_dir, f"{prefix}_assembly.png")
    fig.savefig(png_path, dpi=dpi, facecolor='white', bbox_inches='tight')
    result["assembly_png"] = png_path

    # 保存 PDF
    pdf_path = os.path.join(output_dir, f"{prefix}_assembly.pdf")
    fig.savefig(pdf_path, dpi=dpi, facecolor='white')
    result["assembly_pdf"] = pdf_path

    plt.close(fig)

    # ── 每个零件独立图纸 ──
    for p in parts:
        pt = p.get("part_type", "unknown")
        pid = p.get("part_id", "unknown")
        params = p.get("params", {})
        mesh = gen.make(pt, params)
        if mesh is None:
            continue

        fig2 = plt.figure(figsize=(297 / 25.4, 210 / 25.4), facecolor='white')
        fig2.suptitle(f"{pid} ({pt})", fontsize=10, fontweight='bold',
                      fontfamily='monospace')

        for vi, (view_name, label, vx, vy) in enumerate(views_map):
            proj, faces = _orthographic_projection(mesh.vertices, mesh.faces, view_name)
            edges = _extract_outline_edges(proj, faces)
            if not edges:
                continue

            ax = fig2.add_axes([vx, vy, 0.35, 0.35])
            ax.set_aspect('equal')
            ax.set_title(label, fontsize=6, fontfamily='monospace')
            for e in edges:
                ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]],
                        'k-', linewidth=0.6)
            ax.axis('off')

            bx = _compute_bbox_2d(proj)
            w = bx[1] - bx[0]
            h = bx[3] - bx[2]
            offset = max(w, h) * 0.18
            if w > h:
                _draw_dimension_line(ax, bx[0], bx[2] - offset * 0.5,
                                    bx[1], bx[2] - offset * 0.5, -offset,
                                    f"{w * 1000:.1f}")
            else:
                _draw_dimension_line(ax, bx[0] - offset * 0.5, bx[2],
                                    bx[0] - offset * 0.5, bx[3], -offset,
                                    f"{h * 1000:.1f}")

        info_p = {
            "title": f"{pid} ({pt})",
            "part_id": pid,
            "designer": "ForgeCraft AI",
            "date": datetime.date.today().strftime("%Y-%m-%d"),
            "material": pt,
            "scale": "1:1",
        }
        _draw_title_block(fig2, 297, 210, info_p)

        ppath = os.path.join(output_dir, f"{pid}_drawing.png")
        fig2.savefig(ppath, dpi=min(dpi, 200), facecolor='white', bbox_inches='tight')
        result[pid] = ppath
        plt.close(fig2)

    return result


def export_dxf(
    body_data: dict,
    gen,
    output_dir: str,
    prefix: str = "drawing",
) -> Dict[str, str]:
    """导出 DXF 格式 2D 图纸 (ezdxf)

    每个零件一个 DXF 文件, 含三视图图层。
    """
    import ezdxf

    os.makedirs(output_dir, exist_ok=True)
    result = {}

    parts = body_data.get("parts", [])

    for p in parts:
        pt = p.get("part_type", "unknown")
        pid = p.get("part_id", "unknown")
        params = p.get("params", {})
        pos = np.array(p.get("position", [0, 0, 0]), dtype=np.float64)
        mesh = gen.make(pt, params, position=pos)
        if mesh is None:
            continue

        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        # 图层
        for layer_name in ["FRONT", "TOP", "SIDE", "DIMENSIONS", "TITLE_BLOCK"]:
            doc.layers.add(name=layer_name,
                          color={"FRONT": 7, "TOP": 5, "SIDE": 3,
                                 "DIMENSIONS": 1, "TITLE_BLOCK": 7}[layer_name])

        views = [("front", "FRONT", 0, 0),
                 ("top", "TOP", 0, 150),
                 ("side", "SIDE", 200, 0)]

        scale = 1000  # m → mm

        for view_name, layer, dx, dy in views:
            proj_raw, faces = _orthographic_projection(mesh.vertices, mesh.faces, view_name)
            edges = _extract_outline_edges(proj_raw, faces)

            for (x1, y1), (x2, y2) in edges:
                msp.add_line(
                    (x1 * scale + dx, y1 * scale + dy),
                    (x2 * scale + dx, y2 * scale + dy),
                    dxfattribs={"layer": layer, "lineweight": 35}
                )

            # 尺寸标注
            bx = _compute_bbox_2d(proj_raw)
            if bx[1] - bx[0] > 1e-6:
                w_mm = (bx[1] - bx[0]) * scale
                msp.add_linear_dim(
                    base=(bx[0] * scale + dx, bx[2] * scale + dy - 10),
                    p1=(bx[0] * scale + dx, bx[2] * scale + dy - 15),
                    p2=(bx[1] * scale + dx, bx[2] * scale + dy - 15),
                    dimstyle="EZDXF",
                    dxfattribs={"layer": "DIMENSIONS"},
                ).render()

        fpath = os.path.join(output_dir, f"{pid}.dxf")
        doc.saveas(fpath)
        result[pid] = fpath

    return result
