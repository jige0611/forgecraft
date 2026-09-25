"""
装配体干涉/配合检查器

检测零件间:
- 硬干涉 (碰撞, 体积重叠)
- 软干涉 (间隙 < 最小安全值)
- 配合类型推断 (间隙/过渡/过盈)
- 临界配合面识别

输出: 分级报告 (PASS/CHECK/FAIL) + 修复建议
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh


@dataclass
class InterferenceResult:
    """单对零件的干涉结果"""
    part1_id: str
    part2_id: str
    part1_type: str = ""
    part2_type: str = ""
    min_distance: float = float("inf")  # 最小距离 (m), 负=碰撞
    contact_points: List = field(default_factory=list)
    status: str = "PASS"  # PASS / CHECK / FAIL
    recommendation: str = ""


@dataclass
class FitAnalysis:
    """配合分析"""
    part1_id: str
    part2_id: str
    joint_type: str = ""
    clearance: float = 0.0  # 间隙 (m)
    fit_type: str = "clearance"  # clearance / transition / interference
    nominal: float = 0.0
    tolerance_recommended: str = ""


# 配合类型阈值 (相对于名义尺寸)
FIT_THRESHOLDS = {
    "clearance": (0.0005, 0.05),     # 间隙配合: 0.05% - 5% 名义尺寸
    "transition": (-0.0002, 0.0005),  # 过渡配合: -0.02% - 0.05%
    "interference": (-0.05, -0.0002), # 过盈配合: -5% - -0.02%
}

# 最小安全间隙 (3D 打印常用, 米)
SAFETY_CLEARANCE = {
    "FDM": 0.0004,   # 0.4mm for FDM
    "SLA": 0.0001,   # 0.1mm for resin
    "SLS": 0.0003,   # 0.3mm for SLS
}


def _get_mesh(part: dict, gen, positions_map: dict) -> Optional[trimesh.Trimesh]:
    """获取零件的简化凸包网格"""
    pt = part.get("part_type", "unknown")
    params = part.get("params", {})
    pos = np.array(part.get("position", [0, 0, 0]), dtype=np.float64)
    mesh = gen.make(pt, params, position=pos)
    if mesh is None:
        return None
    return mesh


def _closest_distance(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> Tuple[float, List]:
    """计算两个 mesh 的最近距离 (负值=穿透)"""
    try:
        dist, p1, p2 = trimesh.proximity.closest_point(mesh_a, mesh_b)

        # 检查是否内部点 (穿透 → 负距离)
        contains_any = False
        samples = mesh_b.vertices[::max(1, len(mesh_b.vertices) // 100)]
        for sp in samples:
            try:
                if mesh_a.contains([sp])[0]:
                    contains_any = True
                    break
            except Exception:
                pass

        if contains_any:
            return float(-dist), [(p1, p2)]

        return float(dist), [(p1, p2)]
    except Exception:
        # Fallback: 用 AABB 中心距近似
        ca = mesh_a.bounding_box.centroid
        cb = mesh_b.bounding_box.centroid
        approx = float(np.linalg.norm(ca - cb))
        return approx, []


def check_interferences(
    body_data: dict,
    gen,
    safety_clearance_mm: float = 0.4,
    process: str = "FDM",
) -> Dict:
    """执行装配体干涉检查

    Args:
        body_data: 装配体数据
        gen: ParametricGenerator
        safety_clearance_mm: 最小安全间隙 (mm)
        process: 制造工艺 (FDM/SLA/SLS)

    Returns:
        {
            'status': 'PASS' | 'CHECK' | 'FAIL',
            'results': [InterferenceResult, ...],
            'fit_analyses': [FitAnalysis, ...],
            'summary': str,
            'recommendations': [str, ...],
        }
    """
    parts = body_data.get("parts", [])
    joints = body_data.get("joints", [])

    safety_m = safety_clearance_mm / 1000.0
    if process in SAFETY_CLEARANCE:
        safety_m = SAFETY_CLEARANCE[process]

    # 构建关节映射
    joint_map = {}
    for j in joints:
        p1 = j.get("part1", "")
        p2 = j.get("part2", "")
        if p1 and p2:
            key = tuple(sorted([p1, p2]))
            joint_map[key] = j

    # 生成所有零件的 mesh
    mesh_map = {}
    for p in parts:
        pid = p.get("part_id", "")
        mesh = _get_mesh(p, gen, {})
        if mesh is not None:
            mesh_map[pid] = mesh

    results: List[InterferenceResult] = []
    fit_analyses: List[FitAnalysis] = []
    recommendations = []

    part_ids = list(mesh_map.keys())

    # 全对检查
    for i in range(len(part_ids)):
        for j in range(i + 1, len(part_ids)):
            pid_a, pid_b = part_ids[i], part_ids[j]
            mesh_a = mesh_map[pid_a]
            mesh_b = mesh_map[pid_b]

            dist, contacts = _closest_distance(mesh_a, mesh_b)
            result = InterferenceResult(
                part1_id=pid_a,
                part2_id=pid_b,
                min_distance=dist,
                contact_points=contacts,
                status="PASS",
            )

            # 判定
            joint_key = tuple(sorted([pid_a, pid_b]))
            is_connected = joint_key in joint_map

            if is_connected:
                joint_info = joint_map[joint_key]
                jtype = joint_info.get("type", "fixed")
                # 连接的零件: 允许接触但不允许穿透
                if dist < -1e-6:
                    result.status = "CHECK"
                    result.recommendation = f"Connected joint ({jtype}) shows slight penetration ({dist*1e6:.0f}um). Check joint anchor alignment."
                elif dist < 1e-7:
                    result.status = "PASS"
                    result.recommendation = "Surface contact — acceptable for {jtype}."

                # 配合分析
                dim_a = np.linalg.norm(mesh_a.bounding_box.extents)
                dim_b = np.linalg.norm(mesh_b.bounding_box.extents)
                nominal = (dim_a + dim_b) / 2

                fit = FitAnalysis(
                    part1_id=pid_a,
                    part2_id=pid_b,
                    joint_type=jtype,
                    clearance=dist,
                    nominal=nominal,
                )

                ratio = dist / nominal if nominal > 1e-10 else 0
                if ratio > 0.002:
                    fit.fit_type = "clearance"
                    fit.tolerance_recommended = "H7/g6" if jtype == "hinge" else "H8/f7"
                elif ratio > -0.001:
                    fit.fit_type = "transition"
                    fit.tolerance_recommended = "H7/k6"
                else:
                    fit.fit_type = "interference"
                    fit.tolerance_recommended = "H7/p6" if jtype == "fixed" else "H7/r6"

                fit_analyses.append(fit)
            else:
                # 非连接零件: 需要安全间隙
                if dist < -1e-4:
                    result.status = "FAIL"
                    result.recommendation = f"HARD INTERFERENCE: {abs(dist)*1000:.2f}mm penetration. Redesign required."
                    recommendations.append(f"CRITICAL: {pid_a} <-> {pid_b}: {abs(dist)*1000:.2f}mm hard interference")
                elif dist < safety_m:
                    result.status = "CHECK"
                    result.recommendation = f"Clearance {dist*1000:.2f}mm < safety ({safety_clearance_mm}mm for {process}). Check if acceptable."
                else:
                    result.status = "PASS"
                    result.recommendation = f"Clearance {dist*1000:.2f}mm >= safety ({safety_clearance_mm}mm). OK."

            results.append(result)

    # 汇总
    fails = sum(1 for r in results if r.status == "FAIL")
    checks = sum(1 for r in results if r.status == "CHECK")
    passes = sum(1 for r in results if r.status == "PASS")

    if fails > 0:
        overall = "FAIL"
        summary = f"INTERFERENCE DETECTED: {fails} hard collision(s), {checks} marginal, {passes} OK"
    elif checks > 0:
        overall = "CHECK"
        summary = f"NEEDS REVIEW: {checks} clearance checks below safety, {passes} OK"
    else:
        overall = "PASS"
        summary = f"ALL CLEAR: {passes} pairs checked, no interference"

    # 配合汇总
    fit_summary = {
        "clearance": sum(1 for f in fit_analyses if f.fit_type == "clearance"),
        "transition": sum(1 for f in fit_analyses if f.fit_type == "transition"),
        "interference": sum(1 for f in fit_analyses if f.fit_type == "interference"),
    }

    return {
        "status": overall,
        "results": [r.__dict__ for r in results],
        "fit_analyses": [f.__dict__ for f in fit_analyses],
        "fit_summary": fit_summary,
        "summary": summary,
        "recommendations": recommendations,
        "check_count": len(results),
        "fails": fails,
        "checks": checks,
        "passes": passes,
        "safety_clearance_mm": safety_clearance_mm,
        "process": process,
    }


def generate_interference_report(
    body_data: dict,
    gen,
    output_path: str,
    process: str = "FDM",
    safety_clearance_mm: float = 0.4,
) -> str:
    """生成干涉检查报告 (JSON + TXT)

    Returns:
        报告文件路径
    """
    result = check_interferences(body_data, gen, safety_clearance_mm, process)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # JSON 报告
    json_path = output_path.replace(".json", "") + "_interference.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False,
                 default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    # TXT 报告 (人工可读)
    txt_path = output_path.replace(".json", "") + "_interference.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  FORGECRAFT INTERFERENCE CHECK REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Status: {result['status']}\n")
        f.write(f"Process: {process}\n")
        f.write(f"Safety Clearance: {safety_clearance_mm} mm\n")
        f.write(f"Pairs Checked: {result['check_count']}\n")
        f.write(f"  PASS:  {result['passes']}\n")
        f.write(f"  CHECK: {result['checks']}\n")
        f.write(f"  FAIL:  {result['fails']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("  FIT ANALYSIS\n")
        f.write("-" * 60 + "\n")
        fs = result["fit_summary"]
        f.write(f"  Clearance:    {fs['clearance']}\n")
        f.write(f"  Transition:   {fs['transition']}\n")
        f.write(f"  Interference: {fs['interference']}\n\n")
        for fa in result["fit_analyses"]:
            f.write(f"  {fa['part1_id']} <-> {fa['part2_id']} ({fa['joint_type']})\n")
            f.write(f"    Clearance: {fa['clearance']*1000:.3f} mm\n")
            f.write(f"    Fit Type:  {fa['fit_type']}\n")
            f.write(f"    Tolerance: {fa.get('tolerance_recommended', 'N/A')}\n\n")

        f.write("-" * 60 + "\n")
        f.write("  PAIR DETAILS\n")
        f.write("-" * 60 + "\n\n")
        for r in result["results"]:
            f.write(f"  [{r['status']}] {r['part1_id']} <-> {r['part2_id']}\n")
            f.write(f"    Distance: {r['min_distance']*1000:.4f} mm\n")
            f.write(f"    {r['recommendation']}\n\n")

        if result["recommendations"]:
            f.write("-" * 60 + "\n")
            f.write("  RECOMMENDATIONS\n")
            f.write("-" * 60 + "\n\n")
            for rec in result["recommendations"]:
                f.write(f"  - {rec}\n")

    return json_path
