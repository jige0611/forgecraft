#!/usr/bin/env python3
"""
NopSCADlib 批量高精度 STL 导出脚本
$fn=300 极致曲面质量
"""

import subprocess
from pathlib import Path

BASE = Path(r"c:\Users\刘振鑫\Desktop\第三个")
NOPSCAD = BASE / "NopSCADlib-master"
OPENSCAD_EXE = str(BASE / "OpenSCAD/openscad-2021.01/openscad.exe")
OUTPUT = BASE / "stl_parts"
OUTPUT.mkdir(exist_ok=True)

FN = 300

# ============================================================
# 各类别零件模板 (变量名精确匹配 NopSCADlib 源码)
# ============================================================

CATEGORIES = {
    "stepper_motors": {
        "variants": [
            "NEMA8_30","NEMA8_30BH","NEMA14_36","NEMA16_19",
            "NEMA17_27","NEMA17_34","NEMA17_40","NEMA17_47",
            "NEMA17_47L80","NEMA23_51",
        ],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/stepper_motors.scad> NEMA({name});',
    },

    "ball_bearings": {
        "variants": [
            "BBSMR95","BB624","BB686","BB696","BB608","BB6200",
            "BB6201","BB6808","BBMR63","BBMR83","BBMR85",
            "BBMR93","BBMR95","BBF623","BBF693","BBF625","BBF695",
        ],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/ball_bearings.scad> ball_bearing({name});',
    },

    "pulleys": {
        "variants": [
            "T5x10_pulley","T2p5x16_pulley",
            "GT2x20_pulley_9","GT2x20um_pulley","GT2x20ob_pulley",
            "GT2x16_pulley_9","GT2x16_pulley","GT2x12_pulley",
            "GT2x16_plain_idler","GT2x16x7_plain_idler","GT2x20_plain_idler",
            "GT2x16_toothed_idler","GT2x16_toothed_idler_9",
            "GT2x20_toothed_idler","GT2x20_toothed_idler_9",
            "GT2x80_pulley",
        ],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/pulleys.scad> pulley({name});',
    },

    "shaft_couplings": {
        "variants": ["SC5_5","SC5_8","SC8_8"],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/shaft_couplings.scad> shaft_coupling({name});',
    },

    "extrusions": {
        "variants": ["E1515","E2020","E2040","E2060","E3030","E3060","E4040"],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/extrusions.scad> color("silver") extrusion({name}, 60);',
    },

    "linear_bearings": {
        "variants": ["LM6UU","LM8UU","LM10UU","LM12UU","LM12LUU","LM16UU","LM20UU","LM25UU"],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/linear_bearings.scad> linear_bearing({name});',
    },

    "rod": {
        "variants": ["R4","R6","R8","R10","R12","R16"],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/rod.scad> rod({name}, 100);',
    },

    "leadnuts": {
        "variants": ["T8L4","T10L2","T12L3"],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/leadnuts.scad> leadnut({name});',
    },

    "gears": {
        "variants": ["12","15","20","25","30","40","50","60"],
        "scad": '$fn={fn}; include <core.scad> use <utils/gears.scad> spur_gear(2, {name}, 5, 5);',
    },

    "screws": {
        "variants": [
            "M3_cap_screw","M3_cs_cap_screw","M3_pan_screw",
            "M4_cap_screw","M4_pan_screw","M5_cap_screw",
            "M6_cap_screw","M8_cap_screw","M2_cap_screw",
            "No2_screw","No4_screw","No6_screw","No8_screw",
        ],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/screws.scad> screw({name}, 20);',
    },

    "nuts": {
        "variants": ["M2_nut","M3_nut","M4_nut","M5_nut","M6_nut","M8_nut"],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/nuts.scad> nut({name});',
    },

    "washers": {
        "variants": ["M2_washer","M3_washer","M4_washer","M5_washer","M6_washer","M8_washer"],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/washers.scad> washer({name});',
    },

    "bldc_motors": {
        "variants": [
            "BLDC0603","BLDC0802","BLDC1105","BLDC1306",
            "BLDC1804","BLDC2205","BLDC2212","BLDC3548","BLDC4250",
        ],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/BLDC_motors.scad> BLDC_motor({name});',
    },

    "gear_motors": {
        "variants": ["FIT0492_A","GMAG_404327"],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/gear_motors.scad> gear_motor({name});',
    },

    "geared_steppers": {
        "variants": ["28BYJ_48","35BYGHJ75"],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/geared_steppers.scad> geared_stepper({name});',
    },

    "servo_motors": {
        "variants": [
            "Lichuan_M01330_80ST","Lichuan_M02430_80ST",
            "Lichuan_M03530_80ST","Lichuan_M04030_80ST",
        ],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/servo_motors.scad> servo_motor({name});',
    },

    "fans": {
        "variants": [
            "fan120x25","fan80x38","fan80x25","fan70x15",
            "fan60x25","fan60x15","fan50x15","fan40x11",
            "fan30x10","fan25x10","fan17x8",
        ],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/fans.scad> fan({name});',
    },

    "batteries": {
        "variants": ["AACELL","AAACELL"],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/batteries.scad> battery({name});',
    },

    "kp_pillow_blocks": {
        "variants": ["KP08_15","KP08_18","KP000","KP001"],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/pillow_blocks.scad> pillow_block({name});',
    },

    "scs_bearing_blocks": {
        "variants": [
            "SCS6UU","SCS8UU","SCS10UU","SCS12UU","SCS16UU",
            "SCS8LUU","SCS10LUU","SCS12LUU","SCS16LUU",
        ],
        "scad": '$fn={fn}; include <core.scad> use <vitamins/bearing_blocks.scad> scs_bearing_block({name});',
    },
}


def export_stl(variant, template, count, total):
    """导出单个高精度 STL"""
    name = variant.replace("_", "-").lower()
    stl_path = OUTPUT / f"{name}.stl"

    if stl_path.exists():
        size_kb = stl_path.stat().st_size / 1024
        print(f"  [{count}/{total}] SKIP {name} ({size_kb:.0f}KB)")
        return True

    scad_path = NOPSCAD / f"_tmp_{name}.scad"
    scad_code = template.format(fn=FN, name=variant)
    scad_path.write_text(scad_code, encoding="ascii")

    cmd = [OPENSCAD_EXE, "-o", str(stl_path), str(scad_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        scad_path.unlink(missing_ok=True)
        if result.returncode != 0:
            err = result.stderr.strip()[-200:] if result.stderr else "unknown"
            print(f"  [{count}/{total}] FAIL {name}: {err}")
            return False
        size_kb = stl_path.stat().st_size / 1024
        print(f"  [{count}/{total}] OK   {name} ({size_kb:.0f}KB)")
        return True
    except subprocess.TimeoutExpired:
        scad_path.unlink(missing_ok=True)
        print(f"  [{count}/{total}] TIMEOUT {name}")
        return False
    except Exception as e:
        scad_path.unlink(missing_ok=True)
        print(f"  [{count}/{total}] ERROR {name}: {e}")
        return False


def main():
    all_ok, all_fail, all_skip = 0, 0, 0
    total_variants = sum(len(c["variants"]) for c in CATEGORIES.values())
    global_count = 0

    print(f"\n{'='*60}")
    print(f"  NopSCADlib 高精 STL 批量导出  ($fn={FN})")
    print(f"  共 {len(CATEGORIES)} 类 / {total_variants} 个零件")
    print(f"  STL 目录: {OUTPUT}")
    print(f"{'='*60}")

    for cat_name, cat_info in CATEGORIES.items():
        variants = cat_info["variants"]
        print(f"\n--- {cat_name} ({len(variants)} 种) ---")
        for variant in variants:
            global_count += 1
            ok = export_stl(variant, cat_info["scad"], global_count, total_variants)
            if ok:
                all_ok += 1
            else:
                all_fail += 1

    print(f"\n{'='*60}")
    print(f"  完成! 成功:{all_ok}  失败:{all_fail}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
