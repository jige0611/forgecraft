#!/usr/bin/env python3
"""
从 NopSCADlib 源码提取精确尺寸，生成 Three.js 零件数据
结合 v8_3d_model_library_v2 的零件分类，用真实工程尺寸
"""

import json, re
from pathlib import Path

BASE = Path(r"c:\Users\刘振鑫\Desktop\第三个")
NOPSCAD = BASE / "NopSCADlib-master"

# ============================================================
# 1. 从 NopSCADlib 提取步进电机精确尺寸
# ============================================================
# 格式: ["NAME", side, length, body_radius, boss_radius, boss_height, ...]
STEPPER_DATA = {
    "NEMA8_30":   {"side":20, "length":30, "body_r":15.0, "boss_r":7.5, "boss_h":1.6, "shaft_d":4,  "shaft_l":6,  "hole_pitch":8, "holes":2},
    "NEMA8_30BH": {"side":20, "length":30, "body_r":15.0, "boss_r":7.5, "boss_h":1.6, "shaft_d":5,  "shaft_l":12, "hole_pitch":8, "holes":2},
    "NEMA14_36":  {"side":35.2,"length":36, "body_r":23.2, "boss_r":10.5,"boss_h":2,   "shaft_d":5,  "shaft_l":21, "hole_pitch":8, "holes":3},
    "NEMA16_19":  {"side":39.5,"length":19.2,"body_r":25.3,"boss_r":11,  "boss_h":2,   "shaft_d":5,  "shaft_l":12, "hole_pitch":8, "holes":3},
    "NEMA17_27":  {"side":42.3,"length":26.5,"body_r":26.8,"boss_r":11,  "boss_h":2,   "shaft_d":5,  "shaft_l":26.5,"hole_pitch":10,"holes":3},
    "NEMA17_34":  {"side":42.3,"length":34,  "body_r":26.8,"boss_r":11,  "boss_h":2,   "shaft_d":5,  "shaft_l":24, "hole_pitch":8, "holes":3},
    "NEMA17_40":  {"side":42.3,"length":40,  "body_r":26.8,"boss_r":11,  "boss_h":2,   "shaft_d":5,  "shaft_l":20, "hole_pitch":12.5,"holes":3},
    "NEMA17_47":  {"side":42.3,"length":47,  "body_r":26.8,"boss_r":11,  "boss_h":2,   "shaft_d":5,  "shaft_l":24, "hole_pitch":11.5,"holes":3},
    "NEMA17_47L80":{"side":42.3,"length":47, "body_r":26.8,"boss_r":11,  "boss_h":2,   "shaft_d":5,  "shaft_l":80, "hole_pitch":11.5,"holes":3},
    "NEMA23_51":  {"side":56.4,"length":51.2,"body_r":37.85,"boss_r":19.05,"boss_h":1.6,"shaft_d":6.35,"shaft_l":24,"hole_pitch":8,"holes":3},
}

# ============================================================
# 2. 从 NopSCADlib 提取轴承精确尺寸
# ============================================================
# 格式: ["name", bore, od, width, color, rim, hub, flange_d, flange_w]
BEARING_DATA = {
    "BBSMR95": {"bore":5,  "od":9,  "w":2.5, "seal":"silver"},
    "BB624":   {"bore":4,  "od":13, "w":5,   "seal":"blue"},
    "BB686":   {"bore":6,  "od":13, "w":5,   "seal":"silver"},
    "BB696":   {"bore":6,  "od":16, "w":5,   "seal":"silver"},
    "BB608":   {"bore":8,  "od":22, "w":7,   "seal":"black"},
    "BB6200":  {"bore":10, "od":30, "w":9,   "seal":"black"},
    "BB6201":  {"bore":12, "od":32, "w":10,  "seal":"black"},
    "BB6808":  {"bore":40, "od":52, "w":7,   "seal":"black"},
    "BBMR63":  {"bore":3,  "od":6,  "w":2.5, "seal":"silver"},
    "BBMR83":  {"bore":3,  "od":8,  "w":3,   "seal":"silver"},
    "BBMR85":  {"bore":5,  "od":8,  "w":2.5, "seal":"silver"},
    "BBMR93":  {"bore":3,  "od":9,  "w":4,   "seal":"silver"},
    "BBMR95":  {"bore":5,  "od":9,  "w":3,   "seal":"silver"},
    "BBF623":  {"bore":3,  "od":10, "w":4,   "seal":"black", "flange_d":11.5, "flange_w":1},
    "BBF693":  {"bore":3,  "od":8,  "w":3,   "seal":"silver","flange_d":9.5,  "flange_w":0.7},
    "BBF625":  {"bore":5,  "od":16, "w":5,   "seal":"silver","flange_d":18,   "flange_w":1},
    "BBF695":  {"bore":5,  "od":13, "w":4,   "seal":"silver","flange_d":15,   "flange_w":1},
}

# ============================================================
# 3. 同步带轮精确尺寸 (GT2/HTD/T5 系列)
# ============================================================
PULLEY_DATA = {
    "GT2_16":   {"pitch":"GT2", "teeth":16, "bore":5, "od":9.75, "flange":13, "w":5},
    "GT2_20":   {"pitch":"GT2", "teeth":20, "bore":5, "od":12.22,"flange":16, "w":7},
    "GT2_80":   {"pitch":"GT2", "teeth":80, "bore":5, "od":50.42,"flange":55, "w":9},
    "HTD3M_15": {"pitch":"HTD3M","teeth":15,"bore":5, "od":14,   "flange":18, "w":6},
    "HTD5M_12": {"pitch":"HTD5M","teeth":12,"bore":8, "od":19,   "flange":25, "w":9},
    "HTD5M_16": {"pitch":"HTD5M","teeth":16,"bore":8, "od":25.5, "flange":32, "w":9},
    "T5_10":    {"pitch":"T5",   "teeth":10,"bore":5, "od":15,   "flange":19.3,"w":7},
    "T2p5_16":  {"pitch":"T2.5", "teeth":16,"bore":5, "od":12.16,"flange":16, "w":5.7},
}

# ============================================================
# 4. 铝型材精确尺寸 (欧标)
# ============================================================
EXTRUSION_DATA = {
    "E1515": {"size":15,  "slot":3.2, "center_hole":3.0},
    "E2020": {"size":20,  "slot":5.0, "center_hole":4.2},
    "E2040": {"size_x":20,"size_y":40,"slot":5.0, "center_hole":4.2},
    "E2060": {"size_x":20,"size_y":60,"slot":5.0, "center_hole":4.2},
    "E3030": {"size":30,  "slot":7.0, "center_hole":6.2},
    "E3060": {"size_x":30,"size_y":60,"slot":7.0, "center_hole":6.2},
    "E4040": {"size":40,  "slot":10.0,"center_hole":8.0},
}

# ============================================================
# 5. 直线轴承/光轴
# ============================================================
LINEAR_BEARING_DATA = {
    "LM6UU":  {"bore":6,  "od":12,  "l":19},
    "LM8UU":  {"bore":8,  "od":15,  "l":24},
    "LM10UU": {"bore":10, "od":19,  "l":29},
    "LM12UU": {"bore":12, "od":21,  "l":30},
    "LM12LUU":{"bore":12, "od":21,  "l":57},
    "LM16UU": {"bore":16, "od":28,  "l":37},
    "LM20UU": {"bore":20, "od":32,  "l":42},
    "LM25UU": {"bore":25, "od":40,  "l":59},
}

# ============================================================
# 6. 精确螺丝规格 (公制+英制)
# ============================================================
SCREW_DATA = {
    "M2_cap_screw":     {"d":2.0,  "head_d":3.8,  "head_h":2.0,  "type":"cap"},
    "M3_cap_screw":     {"d":3.0,  "head_d":5.5,  "head_h":3.0,  "type":"cap"},
    "M3_cs_cap_screw":  {"d":3.0,  "head_d":6.0,  "head_h":1.7,  "type":"cs"},
    "M3_pan_screw":     {"d":3.0,  "head_d":6.0,  "head_h":2.5,  "type":"pan"},
    "M4_cap_screw":     {"d":4.0,  "head_d":7.0,  "head_h":4.0,  "type":"cap"},
    "M4_pan_screw":     {"d":4.0,  "head_d":8.0,  "head_h":3.2,  "type":"pan"},
    "M5_cap_screw":     {"d":5.0,  "head_d":8.5,  "head_h":5.0,  "type":"cap"},
    "M6_cap_screw":     {"d":6.0,  "head_d":10.0, "head_h":6.0,  "type":"cap"},
    "M8_cap_screw":     {"d":8.0,  "head_d":13.0, "head_h":8.0,  "type":"cap"},
}

BOLT_HEX_DATA = {
    "M2": {"d":2.0, "head_w":4.0, "head_h":1.4},
    "M3": {"d":3.0, "head_w":5.5, "head_h":2.0},
    "M4": {"d":4.0, "head_w":7.0, "head_h":2.8},
    "M5": {"d":5.0, "head_w":8.0, "head_h":3.5},
    "M6": {"d":6.0, "head_w":10.0,"head_h":4.0},
    "M8": {"d":8.0, "head_w":13.0,"head_h":5.3},
}

NUT_DATA = {
    "M2": {"d":2.0, "w":4.0,  "h":1.6},
    "M3": {"d":3.0, "w":5.5,  "h":2.4},
    "M4": {"d":4.0, "w":7.0,  "h":3.2},
    "M5": {"d":5.0, "w":8.0,  "h":4.0},
    "M6": {"d":6.0, "w":10.0, "h":5.0},
    "M8": {"d":8.0, "w":13.0, "h":6.5},
}

WASHER_DATA = {
    "M2": {"d":2.2, "od":5.0,  "h":0.3},
    "M3": {"d":3.2, "od":7.0,  "h":0.5},
    "M4": {"d":4.3, "od":9.0,  "h":0.8},
    "M5": {"d":5.3, "od":10.0, "h":1.0},
    "M6": {"d":6.4, "od":12.0, "h":1.6},
    "M8": {"d":8.4, "od":16.0, "h":1.6},
}

# ============================================================
# 7. 生成零件库 JSON 供 Three.js 使用
# ============================================================
def generate_parts():
    """生成完整的零件数据 JSON"""
    parts = []
    
    # 步进电机
    for name, d in STEPPER_DATA.items():
        parts.append({
            "id": name.replace("_","-").lower(),
            "name": f"NEMA{name.split('_')[0][4:]} {d['length']}mm",
            "category": "actuator",
            "type": "stepper_motor",
            "side": d["side"], "length": d["length"],
            "body_r": d["body_r"], "boss_r": d["boss_r"], "boss_h": d["boss_h"],
            "shaft_d": d["shaft_d"], "shaft_l": d["shaft_l"],
            "hole_pitch": d.get("hole_pitch",8), "holes": d.get("holes",4),
            "color": "#2c3e50",
        })
    
    # 轴承
    for name, d in BEARING_DATA.items():
        flange = {}
        if "flange_d" in d:
            flange = {"flange_d": d["flange_d"], "flange_w": d["flange_w"]}
        parts.append({
            "id": name.replace("_","-").lower(),
            "name": f"轴承 {name[2:] if name.startswith('BB') else name} {d['bore']}x{d['od']}x{d['w']}",
            "category": "transmission",
            "type": "ball_bearing",
            "bore": d["bore"], "od": d["od"], "width": d["w"],
            "seal_color": "#808080" if d["seal"]=="silver" else "#1a1a2e" if d["seal"]=="black" else "#2196F3",
            **flange,
            "color": "#bdc3c7",
        })
    
    # 同步带轮
    for name, d in PULLEY_DATA.items():
        parts.append({
            "id": name.lower().replace("_","-"),
            "name": f"{d['pitch']} 同步带轮 {d['teeth']}T",
            "category": "transmission",
            "type": "pulley",
            "teeth": d["teeth"], "bore": d["bore"],
            "od": d["od"], "flange_d": d["flange"],
            "width": d["w"],
            "color": "#d4a574",
        })
    
    # 铝型材
    for name, d in EXTRUSION_DATA.items():
        sx = d.get("size_x", d.get("size",20))
        sy = d.get("size_y", d.get("size",20))
        parts.append({
            "id": name.lower(),
            "name": f"欧标铝型材 {sx}x{sy}",
            "category": "structure",
            "type": "extrusion",
            "size_x": sx, "size_y": sy,
            "slot": d["slot"],
            "center_hole": d["center_hole"],
            "color": "#c0c0c0",
        })
    
    # 直线轴承
    for name, d in LINEAR_BEARING_DATA.items():
        parts.append({
            "id": name.lower(),
            "name": f"直线轴承 {name}",
            "category": "transmission",
            "type": "linear_bearing",
            "bore": d["bore"], "od": d["od"], "length": d["l"],
            "color": "#8e9eab",
        })
    
    # 螺丝
    for name, d in SCREW_DATA.items():
        parts.append({
            "id": name.replace("_","-").lower(),
            "name": f"{'沉头' if d['type']=='cs' else '圆柱头' if d['type']=='cap' else '盘头'}螺丝 {name.split('_')[0].upper()}",
            "category": "fastener",
            "type": "screw",
            "diameter": d["d"], "head_d": d["head_d"],
            "head_h": d["head_h"], "head_type": d["type"],
            "color": "#8e9eab",
        })
    
    # 螺母
    for name, d in NUT_DATA.items():
        parts.append({
            "id": f"{name.lower()}-nut",
            "name": f"{name} 螺母",
            "category": "fastener",
            "type": "nut",
            "diameter": d["d"], "width": d["w"], "height": d["h"],
            "color": "#8e9eab",
        })
    
    # 垫圈
    for name, d in WASHER_DATA.items():
        parts.append({
            "id": f"{name.lower()}-washer",
            "name": f"{name} 垫圈",
            "category": "fastener",
            "type": "washer",
            "inner_d": d["d"], "outer_d": d["od"], "thickness": d["h"],
            "color": "#c0c0c0",
        })
    
    return parts


if __name__ == "__main__":
    parts = generate_parts()
    print(f"生成 {len(parts)} 个精确尺寸零件")
    
    # 分类统计
    from collections import Counter
    cats = Counter(p["category"] for p in parts)
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")
    
    # 保存 JSON
    out_path = BASE / "stl_parts" / "nopscad_parts.json"
    out_path.write_text(json.dumps(parts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON saved to: {out_path}")
