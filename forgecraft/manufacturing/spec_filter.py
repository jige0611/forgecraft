import json
import os
from typing import Any, Dict, List, Optional, Tuple

from forgecraft.core.morphology import MechanicalBody, Part
__all__ = [
    "spec_body",
    "export_bom",
]




# ── 已知标准件数据库 ──
# servo: {name, torque(N·m), velocity(rad/s), mass_kg, voltage}
# rod:   {name, radius_m, length_range, density(kg/m³), material}
# box:   {name, thickness, density, material}
# foot:  {name, radius, material}

_KNOWN_SERVOS = [
    {"name": "SG90",    "torque": 0.18, "velocity": 10.0, "mass_kg": 0.009, "voltage": 5.0},
    {"name": "MG90S",   "torque": 0.22, "velocity": 10.0, "mass_kg": 0.013, "voltage": 5.0},
    {"name": "MG996R",  "torque": 1.00, "velocity": 7.0,  "mass_kg": 0.055, "voltage": 6.0},
    {"name": "DS3225",  "torque": 2.10, "velocity": 5.0,  "mass_kg": 0.060, "voltage": 6.8},
    {"name": "M2006",   "torque": 1.00, "velocity": 40.0, "mass_kg": 0.090, "voltage": 24.0},
    {"name": "AK80-9",  "torque": 9.00, "velocity": 30.0, "mass_kg": 0.485, "voltage": 48.0},
    {"name": "AK80-6",  "torque": 6.00, "velocity": 30.0, "mass_kg": 0.360, "voltage": 48.0},
]

_KNOWN_RODS = [
    {"name": "CF tube 4mm",     "radius_m": 0.002,  "length_range": [0.02, 0.30], "density": 1600, "material": "carbon fiber"},
    {"name": "CF tube 6mm",     "radius_m": 0.003,  "length_range": [0.03, 0.50], "density": 1600, "material": "carbon fiber"},
    {"name": "CF tube 10mm",    "radius_m": 0.005,  "length_range": [0.05, 1.00], "density": 1600, "material": "carbon fiber"},
    {"name": "CF tube 16mm",    "radius_m": 0.008,  "length_range": [0.08, 1.50], "density": 1600, "material": "carbon fiber"},
    {"name": "Al tube 8mm",     "radius_m": 0.004,  "length_range": [0.03, 0.60], "density": 2700, "material": "aluminum"},
    {"name": "Al tube 12mm",    "radius_m": 0.006,  "length_range": [0.05, 1.00], "density": 2700, "material": "aluminum"},
    {"name": "PLA rod 5mm",     "radius_m": 0.0025, "length_range": [0.01, 0.25], "density": 1240, "material": "PLA"},
    {"name": "PLA rod 10mm",    "radius_m": 0.005,  "length_range": [0.02, 0.25], "density": 1240, "material": "PLA"},
]

_KNOWN_BOXES = [
    {"name": "PLA plate 3mm",   "thickness": 0.003, "density": 1240, "material": "PLA"},
    {"name": "PLA plate 5mm",   "thickness": 0.005, "density": 1240, "material": "PLA"},
    {"name": "CF sheet 2mm",    "thickness": 0.002, "density": 1600, "material": "carbon fiber"},
    {"name": "Al sheet 3mm",    "thickness": 0.003, "density": 2700, "material": "aluminum"},
]

_KNOWN_FEET = [
    {"name": "rubber dome 20mm", "radius": 0.010, "material": "rubber"},
    {"name": "rubber dome 30mm", "radius": 0.015, "material": "rubber"},
    {"name": "rubber dome 40mm", "radius": 0.020, "material": "rubber"},
    {"name": "3D-print spike",   "radius": 0.008, "material": "PLA"},
    {"name": "3D-print foot pad","radius": 0.025, "material": "TPU"},
]


def _nearest(items: List[dict], key: str, value: float) -> dict:
    """最近邻匹配 — 在标准件库中找到最接近目标参数的条目"""
    best = items[0]
    best_diff = abs(items[0][key] - value)
    for item in items[1:]:
        diff = abs(item[key] - value)
        if diff < best_diff:
            best_diff = diff
            best = item
    return best


def _recommend_servo(torque: float, velocity: float) -> dict:
    best_t = _nearest(_KNOWN_SERVOS, "torque", torque)
    best_v = _nearest(_KNOWN_SERVOS, "velocity", velocity)

    candidates = [best_t, best_v]
    for s in _KNOWN_SERVOS:
        if s["torque"] >= torque * 0.8 and s["velocity"] >= velocity * 0.7:
            candidates.append(s)

    scored = []
    for s in candidates:
        t_ok = s["torque"] >= torque
        v_ok = s["velocity"] >= velocity
        t_diff = abs(s["torque"] - torque) / max(torque, 0.01)
        v_diff = abs(s["velocity"] - velocity) / max(velocity, 0.01)
        score = (0 if t_ok else t_diff * 2) + (0 if v_ok else v_diff * 1.5)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0])
    best_match = scored[0][1]
    note = ""
    if best_match["torque"] < torque:
        note = f"力矩不足{torque - best_match['torque']:.2f}N·m，建议串联/定制"
    if best_match["velocity"] < velocity:
        note = f"速度不足{velocity - best_match['velocity']:.0f}rad/s"

    return {
        "recommendation": best_match["name"],
        "matched_torque": best_match["torque"],
        "matched_velocity": best_match["velocity"],
        "mass_kg": best_match["mass_kg"],
        "note": note if note else "匹配",
    }


def _recommend_rod(radius: float, length: float, density: float) -> dict:
    best_r = _nearest(_KNOWN_RODS, "radius_m", radius)
    best_d = _nearest(_KNOWN_RODS, "density", density)

    candidates = [best_r, best_d]
    for r in _KNOWN_RODS:
        if abs(r["radius_m"] - radius) < radius * 0.5 and abs(r["density"] - density) < density * 0.3:
            candidates.append(r)

    scored = []
    for r in candidates:
        r_diff = abs(r["radius_m"] - radius) / max(radius, 0.001)
        d_diff = abs(r["density"] - density) / max(density, 1.0)
        in_range = r["length_range"][0] <= length <= r["length_range"][1]
        score = r_diff + d_diff * 0.3 + (0 if in_range else 5.0)
        scored.append((score, r, in_range))

    scored.sort(key=lambda x: x[0])
    best = scored[0][1]
    in_range = scored[0][2]
    note = ""
    if not in_range:
        note = f"长度{length:.2f}m超出{best['name']}范围, 需拼接或定制"

    return {
        "recommendation": best["name"],
        "material": best["material"],
        "matched_radius": best["radius_m"],
        "note": note if note else "匹配",
    }


def _recommend_foot(radius: float) -> dict:
    """足端推荐 — 按接触半径最近邻"""
    best = _nearest(_KNOWN_FEET, "radius", radius)
    return {
        "recommendation": best["name"],
        "material": best["material"],
        "matched_radius": best["radius"],
        "note": f"最优匹配, 偏差{abs(best['radius']-radius)*1000:.1f}mm",
    }


def spec_body(body: MechanicalBody) -> Dict[str, Any]:
    """对每个零件推荐标准件型号: 根据扭矩/尺寸匹配已知伺服/电机"""
    boms = []
    notes = []

    for part in body.parts():
        params = part.params
        entry = {
            "part_id": part.part_id,
            "part_type": part.part_type,
            "current_params": dict(params),
            "recommendations": {},
        }

        if part.part_type in ("rod", "segment", "link", "limb_segment", "upper_leg", "lower_leg"):
            radius = params.get("radius", 0.01)
            length = params.get("length", 0.1)
            mass = params.get("mass", 0.05)
            volume = np.pi * radius**2 * length
            density = mass / max(volume, 1e-8) if volume > 1e-8 else 1200.0
            entry["recommendations"]["rod"] = _recommend_rod(radius, length, density)

        elif part.part_type in ("hinge_motor", "motor", "hinge_joint", "hip_joint", "knee_motor",
                                 "hip_motor", "hinge", "knee", "ankle"):
            torque = params.get("max_torque", 1.0)
            velocity = params.get("max_velocity", 10.0)
            entry["recommendations"]["servo"] = _recommend_servo(torque, velocity)

        elif part.part_type in ("foot_contact", "foot", "foot_pad", "foot_spike"):
            radius = params.get("radius", 0.02)
            entry["recommendations"]["foot"] = _recommend_foot(radius)

        boms.append(entry)

    summary = {
        "total_parts": len(boms),
        "servo_count": sum(1 for b in boms if "servo" in b.get("recommendations", {})),
        "rod_count": sum(1 for b in boms if "rod" in b.get("recommendations", {})),
        "foot_count": sum(1 for b in boms if "foot" in b.get("recommendations", {})),
        "warnings": notes,
    }

    return {
        "summary": summary,
        "bill_of_materials": boms,
    }


def export_bom(body: MechanicalBody, output_path: str):
    result = spec_body(body)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"BOM已导出: {output_path}")
    return result


import numpy as np
