"""
自动 BOM 生成 + 采购链接推荐

集成标准件推荐引擎, 输出包含采购信息的物料清单。

使用:
  from forgecraft.manufacturing.bom_generator import generate_full_bom
import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "generate_full_bom",
]


  bom = generate_full_bom(best_body)
  # bom 包含零件名、数量、推荐型号、参考价格、采购链接
"""

import json
import os
from typing import Any, Dict, List, Optional

from forgecraft.core.morphology import MechanicalBody
from forgecraft.manufacturing.spec_filter import (
    _KNOWN_SERVOS, _KNOWN_RODS, _KNOWN_BOXES, _KNOWN_FEET,
    _recommend_servo, _recommend_rod,
)


# ── 采购链接 ────────────────────────────────────────────
_PURCHASE_LINKS = {
    "SG90": "https://www.aliexpress.com/wholesale?SearchText=SG90+servo",
    "MG90S": "https://www.aliexpress.com/wholesale?SearchText=MG90S+servo",
    "MG996R": "https://www.aliexpress.com/wholesale?SearchText=MG996R+servo",
    "DS3225": "https://www.aliexpress.com/wholesale?SearchText=DS3225+servo",
    "CF tube": "https://www.aliexpress.com/wholesale?SearchText=carbon+fiber+tube",
    "Al tube": "https://www.aliexpress.com/wholesale?SearchText=aluminum+tube",
    "PLA rod": "https://www.aliexpress.com/wholesale?SearchText=PLA+filament",
    "rubber dome": "https://www.aliexpress.com/wholesale?SearchText=rubber+furniture+feet",
    "3D-print": None,  # 本地打印
}


def _estimate_cost(part_type: str, qty: int, params: dict) -> float:
    """粗略成本估算 (USD)。"""
    if "servo" in part_type.lower() or "motor" in part_type.lower():
        torque = params.get("max_torque", 0.5)
        if torque < 0.3:
            return qty * 2.0
        elif torque < 1.5:
            return qty * 6.0
        elif torque < 5.0:
            return qty * 25.0
        else:
            return qty * 80.0
    elif "tube" in part_type.lower() or "rod" in part_type.lower():
        return qty * 3.0
    elif "foot" in part_type.lower():
        return qty * 1.5
    elif "spring" in part_type.lower():
        return qty * 0.5
    else:
        return qty * 1.0


def generate_full_bom(
    body: MechanicalBody,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    生成完整物料清单(BOM) 含采购链接。

    Returns:
        {
            "summary": {"total_parts": int, "total_cost_estimate": float},
            "items": [
                {"name": str, "qty": int, "recommendation": str,
                 "unit_cost": float, "purchase_link": str}
            ],
            "print_parts": [...],  # 需3D打印的零件
        }
    """
    items = []
    part_counts: Dict[str, int] = {}
    part_params: Dict[str, dict] = {}

    for part in body.parts():
        pt = part.part_type
        part_counts[pt] = part_counts.get(pt, 0) + 1
        if pt not in part_params:
            part_params[pt] = part.params

    for pt, qty in part_counts.items():
        params = part_params.get(pt, {})
        recommendation = ""
        purchase_link = ""

        # 舵机/电机推荐
        if pt in ("motor", "hinge_motor", "hinge_joint") or "motor" in pt:
            torque = params.get("max_torque", 1.0)
            velocity = params.get("max_velocity", 10.0)
            rec = _recommend_servo(torque, velocity)
            recommendation = rec["recommendation"]
            purchase_link = _PURCHASE_LINKS.get(rec["recommendation"], "")

        # 管材推荐
        elif "rod" in pt or "tube" in pt or "segment" in pt:
            radius = params.get("radius", 0.003)
            length = params.get("length", 0.1)
            density = params.get("density", 1600)
            rec = _recommend_rod(radius, length, density)
            recommendation = rec["recommendation"]
            purchase_link = _PURCHASE_LINKS.get(
                "CF tube" if "CF" in rec["recommendation"] else
                "Al tube" if "Al" in rec["recommendation"] else "PLA rod", ""
            )

        # 脚垫推荐
        elif "foot" in pt:
            for ft in _KNOWN_FEET:
                r = params.get("radius", 0.015)
                if abs(ft.get("radius", 0) - r) < 0.01:
                    recommendation = ft["name"]
                    break
            if not recommendation:
                recommendation = "rubber dome 30mm"
            purchase_link = _PURCHASE_LINKS["rubber dome"]

        cost = _estimate_cost(pt, qty, params)
        items.append({
            "name": pt,
            "qty": qty,
            "params_snapshot": {k: round(v, 4) for k, v in list(params.items())[:4]},
            "recommendation": recommendation,
            "unit_cost": round(cost / qty, 2),
            "total_cost": round(cost, 2),
            "purchase_link": purchase_link or "",
        })

    total_cost = sum(it["total_cost"] for it in items)
    total_parts = sum(it["qty"] for it in items)

    bom = {
        "summary": {
            "total_unique_types": len(items),
            "total_parts": total_parts,
            "total_cost_estimate": round(total_cost, 2),
            "currency": "USD",
        },
        "items": items,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(bom, f, indent=2, ensure_ascii=False)
        print(f"BOM 已保存: {output_path}")
        print(f"  总零件数: {total_parts}, 总成本估计: ${total_cost:.2f}")

    return bom
