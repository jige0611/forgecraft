# ═══════════════════════════════════════════════════════════════
#  V8 零件库 v2 — 高精3D预览渲染引擎 (1920×1080, 全几何类型)
#  ═══════════════════════════════════════════════════════════════

import mujoco
import numpy as np
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent))
from v8_3d_model_library_v2 import get_all_part_specs, PartSpec

IMG_W = 1920
IMG_H = 1080
SCALE = 2.2  # 全局放大因子
OUTPUT_DIR = "part_library_previews_v2"

VIEWS = [
    {"name": "iso",      "lookat": [0, 0, 0.06], "distance": 0.45, "azimuth": 45, "elevation": -25},
    {"name": "front",    "lookat": [0, 0, 0.06], "distance": 0.40, "azimuth": 90, "elevation": -8},
    {"name": "top",      "lookat": [0, 0, 0.04], "distance": 0.55, "azimuth": 0,  "elevation": 89},
    {"name": "close_up", "lookat": [0, 0, 0.03], "distance": 0.18, "azimuth": 28, "elevation": -18},
]


def make_xml(spec: PartSpec) -> str:
    d = spec.dimensions; c = spec.color; gt = spec.geometry_type; s = SCALE
    geoms = []

    def g(**kw):
        """便捷添加geom"""
        geoms.append(" ".join(f'{k}="{v}"' for k, v in kw.items()))

    # ═══════════════════════════════════════════════════════════════
    #  核心形状路由
    # ═══════════════════════════════════════════════════════════════

    # ─── 圆柱类 (电机主体) ───
    if gt in ('cylinder_detail','cylinder_simple','cylinder_complex',
              'cylinder_with_hub','cylinder_frc','cylinder_cim',
              'cylinder_re30','cylinder_medium','cylinder_tall',
              'cylinder_step','cylinder_small','cylinder_small_long'):
        od = d.get('outer_diameter', d.get('body_diameter', 0.03))
        r = od / 2 * s
        h = d.get('total_height', d.get('body_length', 0.05)) * s
        g(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")
        # 轴
        sr = d.get('shaft_diameter', r*0.25) / 2 * s
        sl = d.get('shaft_length', h*0.3) * s
        g(type="cylinder", size=f"{sr:.5f} {sl/2:.5f}", rgba="0.65 0.65 0.68 1",
          pos=f"0 0 {h+sl/2:.5f}")
        # 法兰
        if 'flange_diameter' in d:
            fr = d['flange_diameter'] / 2 * s
            ft = d.get('flange_thickness', 0.003) * s
            g(type="cylinder", size=f"{fr:.5f} {ft:.5f}", rgba="0.42 0.40 0.45 1",
              pos=f"0 0 {ft:.5f}")
        # 安装孔
        if 'mounting_holes' in d:
            for i in range(min(d['mounting_holes'], 6)):
                ang = 2*np.pi*i/d['mounting_holes']
                bx = d.get('hole_circle', fr*0.8)/2*np.cos(ang)*s
                by = d.get('hole_circle', fr*0.8)/2*np.sin(ang)*s
                br = d.get('bolt_diameter', 0.003)/2 * s
                g(type="cylinder", size=f"{br:.5f} {h*0.15:.5f}", rgba="0.12 0.12 0.14 1",
                  pos=f"{bx:.5f} {by:.5f} {ft:.5f}")

    elif gt == 'pancake_motor':
        r = d.get('outer_diameter', 0.045) / 2 * s
        h = d.get('total_height', 0.03) * s
        g(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")
        sr = d.get('shaft_diameter', 0.008) / 2 * s
        sl = d.get('shaft_length', 0.014) * s
        g(type="cylinder", size=f"{sr:.5f} {sl/2:.5f}", rgba="0.65 0.65 0.68 1", pos=f"0 0 {h+sl/2:.5f}")
        fr = d.get('flange_diameter', 0.050) / 2 * s
        ft = d.get('flange_thickness', 0.003) * s
        g(type="cylinder", size=f"{fr:.5f} {ft:.5f}", rgba="0.42 0.40 0.45 1", pos=f"0 0 {ft:.5f}")

    elif gt == 'square_cylinder_stepper':
        sz = d.get('square_size', 0.042) / 2 * s
        bl = d.get('body_length', 0.048) * s
        g(type="box", size=f"{sz:.5f} {sz:.5f} {bl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {bl/2:.5f}")
        sr = d.get('shaft_diameter', 0.005) / 2 * s
        sl = d.get('shaft_length', 0.022) * s
        g(type="cylinder", size=f"{sr:.5f} {sl/2:.5f}", rgba="0.70 0.70 0.73 1", pos=f"0 0 {bl+sl/2:.5f}")
        # 前盖
        fh = 0.005 * s
        g(type="cylinder", size=f"{sz*1.15:.5f} {fh:.5f}", rgba="0.15 0.15 0.17 1", pos=f"0 0 {fh:.5f}")

    elif gt in ('square_cylinder_large','square_cylinder_xlarge'):
        sz = d.get('square_size', 0.057) / 2 * s
        bl = d.get('body_length', 0.076) * s
        g(type="box", size=f"{sz:.5f} {sz:.5f} {bl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {bl/2:.5f}")
        sr = d.get('shaft_diameter', 0.008) / 2 * s
        sl = d.get('shaft_length', 0.024) * s
        g(type="cylinder", size=f"{sr:.5f} {sl/2:.5f}", rgba="0.70 0.70 0.73 1", pos=f"0 0 {bl+sl/2:.5f}")
        fh = 0.008 * s
        g(type="cylinder", size=f"{sz*1.15:.5f} {fh:.5f}", rgba="0.15 0.15 0.17 1", pos=f"0 0 {fh:.5f}")

    elif gt == 'disc_motor':
        r = d.get('outer_diameter', 0.085) / 2 * s
        h = d.get('total_height', 0.037) * s
        g(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")
        sr = d.get('shaft_diameter', 0.010) / 2 * s
        sl = d.get('shaft_length', 0.015) * s
        g(type="cylinder", size=f"{sr:.5f} {sl/2:.5f}", rgba="0.70 0.70 0.73 1", pos=f"0 0 {h+sl/2:.5f}")

    elif gt == 'joint_motor':
        r = d.get('outer_diameter', 0.060) / 2 * s
        h = d.get('total_height', 0.068) * s
        g(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")
        fr = d.get('flange_diameter', 0.070) / 2 * s
        ft = d.get('flange_thickness', 0.004) * s
        g(type="cylinder", size=f"{fr:.5f} {ft:.5f}", rgba="0.45 0.42 0.48 1", pos=f"0 0 {h+ft:.5f}")

    elif gt == 'multirotor_motor':
        r = d.get('outer_diameter', 0.050) / 2 * s
        h = d.get('total_height', 0.028) * s
        g(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")
        br = d.get('base_diameter', 0.035) / 2 * s
        bh = d.get('base_height', 0.008) * s
        g(type="cylinder", size=f"{br:.5f} {bh:.5f}", rgba="0.50 0.47 0.55 1", pos=f"0 0 {h+bh:.5f}")

    elif gt == 'disc_motor_thin':
        r = d.get('outer_diameter', 0.056) / 2 * s
        h = d.get('total_height', 0.028) * s
        g(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    # ─── Dynamixel 外壳 ───
    elif gt in ('dynamixel_shell','dynamixel_mini'):
        l = d.get('length', 0.043) / 2 * s
        w = d.get('width', 0.043) / 2 * s
        h = d.get('height', 0.050) / 2 * s
        g(type="box", size=f"{l:.5f} {w:.5f} {h:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h:.5f}")
        hub_r = d.get('hub_diameter', 0.028) / 2 * s
        hub_h = d.get('hub_height', 0.004) * s
        g(type="cylinder", size=f"{hub_r:.5f} {hub_h:.5f}", rgba="0.55 0.52 0.58 1", pos=f"0 0 {h+hub_h:.5f}")
        # 螺丝孔
        bc = d.get('horn_bolt_circle', 0.020) / 2 * s
        for i in range(4):
            ang = 2*np.pi*i/4
            g(type="cylinder", size=f"0.001 0.003", rgba="0.1 0.1 0.1 1",
              pos=f"{bc*np.cos(ang):.5f} {bc*np.sin(ang):.5f} {h+hub_h:.5f}")

    # ─── 舵机 ───
    elif gt in ('servo_standard','servo_metal'):
        l = d.get('length', 0.040) / 2 * s
        w = d.get('width', 0.020) / 2 * s
        h = d.get('height', 0.038) / 2 * s
        g(type="box", size=f"{l:.5f} {w:.5f} {h:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h:.5f}")
        horn_l = d.get('horn_length', 0.028) / 2 * s
        horn_w = d.get('horn_width', 0.004) / 2 * s
        g(type="box", size=f"{horn_l:.5f} {horn_w:.5f} 0.002", rgba="0.82 0.82 0.84 1",
          pos=f"{horn_l:.5f} 0 {h:.5f}")

    # ─── 减速箱 ───
    elif gt == 'gearbox_dc':
        gb_r = d.get('gearbox_od', 0.025) / 2 * s
        gb_l = d.get('gearbox_len', 0.025) * s
        motor_r = d.get('motor_od', 0.025) / 2 * s
        motor_l = d.get('motor_len', 0.035) * s
        g(type="cylinder", size=f"{gb_r:.5f} {gb_l/2:.5f}", rgba="0.45 0.47 0.49 1", pos=f"0 0 {gb_l/2:.5f}")
        g(type="cylinder", size=f"{motor_r:.5f} {motor_l/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {gb_l+motor_l/2:.5f}")
        sr = d.get('shaft_diameter', 0.008) / 2 * s
        sl = d.get('shaft_length', 0.015) * s
        g(type="cylinder", size=f"{sr:.5f} {sl/2:.5f}", rgba="0.70 0.70 0.73 1", pos=f"0 0 {sl/2:.5f}")

    elif gt == 'gearbox_dc_small':
        gb_r = d.get('gearbox_od', 0.010) / 2 * s
        gb_l = d.get('gearbox_len', 0.012) * s
        motor_r = d.get('motor_od', 0.025) / 2 * s
        motor_l = d.get('motor_len', 0.025) * s
        g(type="cylinder", size=f"{gb_r:.5f} {gb_l/2:.5f}", rgba="0.50 0.52 0.54 1", pos=f"0 0 {gb_l/2:.5f}")
        g(type="cylinder", size=f"{motor_r:.5f} {motor_l/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {gb_l+motor_l/2:.5f}")

    elif gt == 'micro_dc':
        r = d.get('body_diameter', 0.028) / 2 * s
        l = d.get('body_length', 0.050) * s
        g(type="cylinder", size=f"{r:.5f} {l/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {l/2:.5f}")
        sr = d.get('shaft_diameter', 0.003) / 2 * s
        sl = d.get('shaft_length', 0.010) * s
        g(type="cylinder", size=f"{sr:.5f} {sl/2:.5f}", rgba="0.70 0.70 0.73 1", pos=f"0 0 {l+sl/2:.5f}")

    # ─── 线性执行器 ───
    elif gt == 'linear_actuator':
        bl = d.get('body_length', 0.180) * s
        bw = d.get('body_width', 0.040) / 2 * s
        g(type="box", size=f"{bl/2:.5f} {bw:.5f} {bw:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {bw:.5f}")
        rod_r = d.get('rod_diameter', 0.010) / 2 * s
        stroke = d.get('stroke', 0.100) * s
        g(type="cylinder", size=f"{rod_r:.5f} {stroke/2:.5f}", rgba="0.75 0.75 0.77 1", pos=f"{bl/2+stroke/2:.5f} 0 {bw:.5f}")

    # ─── 谐波减速器 ───
    elif gt in ('harmonic_assembly','harmonic_compact','harmonic_mini'):
        cr = d.get('casing_od', 0.085) / 2 * s
        cl = d.get('casing_len', 0.070) * s
        g(type="cylinder", size=f"{cr:.5f} {cl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {cl/2:.5f}")
        fr = d.get('output_flange_dia', 0.060) / 2 * s
        g(type="cylinder", size=f"{fr:.5f} 0.005", rgba="0.45 0.47 0.49 1", pos=f"0 0 {cl+0.005:.5f}")

    # ─── 行星减速箱 ───
    elif gt in ('planetary_gearbox','planetary_gearbox_large',
                 'planetary_neugart','planetary_apex'):
        ir = d.get('input_diameter', 0.020) / 2 * s
        ol = d.get('length', 0.055) * s
        fr = d.get('flange_dia', 0.030) / 2 * s
        g(type="cylinder", size=f"{ir:.5f} {ol/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {ol/2:.5f}")
        g(type="cylinder", size=f"{fr:.5f} 0.004", rgba="0.40 0.42 0.44 1", pos=f"0 0 {ol+0.004:.5f}")

    # ─── 摆线减速机 ───
    elif gt in ('cycloidal_gearbox','cycloidal_gearbox_large'):
        fr = d.get('flange_input_dia', 0.070) / 2 * s
        ol = d.get('length', 0.095) * s
        g(type="cylinder", size=f"{fr:.5f} {ol/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {ol/2:.5f}")

    # ─── 音圈电机 ───
    elif gt == 'voice_coil':
        r = d.get('body_diameter', 0.030) / 2 * s
        stroke = d.get('stroke', 0.012) * s
        g(type="cylinder", size=f"{r:.5f} {stroke/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {stroke/2:.5f}")

    # ─── 直线电机 ───
    elif gt == 'linear_motor':
        w = d.get('width', 0.100) / 2 * s
        h = d.get('height', 0.070) / 2 * s
        l = 0.15 * s
        g(type="box", size=f"{l/2:.5f} {w:.5f} {h:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h:.5f}")

    # ═══════════════════════════════════════════════════════════════
    #  传动系统
    # ═══════════════════════════════════════════════════════════════
    elif gt in ('timing_pulley','timing_pulley_large','timing_pulley_htd'):
        pd = d.get('pitch_diameter', 0.01277) * s
        w = d.get('width', 0.015) * s
        bore = d.get('bore', 0.005) * s
        g(type="cylinder", size=f"{pd/2:.5f} {w/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {w/2:.5f}")
        g(type="cylinder", size=f"{bore/2:.5f} {w*0.75:.5f}", rgba="0.06 0.06 0.06 1", pos=f"0 0 {w/2:.5f}")

    elif gt in ('sprocket','sprocket_precision'):
        pd = d.get('pitch_diameter', 0.06511) * s
        w = d.get('width', 0.017) * s
        bore = d.get('bore', 0.012) * s
        g(type="cylinder", size=f"{pd/2:.5f} {w/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {w/2:.5f}")
        g(type="cylinder", size=f"{bore/2:.5f} {w*0.75:.5f}", rgba="0.06 0.06 0.06 1", pos=f"0 0 {w/2:.5f}")

    elif gt in ('spur_gear','spur_gear_large','spur_gear_heavy'):
        pd = d.get('pitch_diameter', 0.020) * s
        fw = d.get('face_width', 0.012) * s
        bore = d.get('bore', 0.005) * s
        g(type="cylinder", size=f"{pd/2:.5f} {fw/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {fw/2:.5f}")
        g(type="cylinder", size=f"{bore/2:.5f} {fw*0.8:.5f}", rgba="0.06 0.06 0.06 1", pos=f"0 0 {fw/2:.5f}")

    elif gt == 'helical_gear':
        pd = d.get('pitch_diameter', 0.025) * s
        fw = d.get('face_width', 0.015) * s
        bore = d.get('bore', 0.006) * s
        g(type="cylinder", size=f"{pd/2:.5f} {fw/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {fw/2:.5f}")
        g(type="cylinder", size=f"{bore/2:.5f} {fw*0.8:.5f}", rgba="0.06 0.06 0.06 1", pos=f"0 0 {fw/2:.5f}")

    elif gt == 'bevel_gear':
        pd = d.get('pitch_diameter', 0.040) * s
        cd = d.get('cone_distance', 0.028) * s
        g(type="cylinder", size=f"{pd/2:.5f} {cd/3:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {cd/3:.5f}")

    elif gt == 'worm_wheel':
        pd = d.get('pitch_diameter', 0.080) * s
        fw = d.get('face_width', 0.020) * s
        g(type="cylinder", size=f"{pd/2:.5f} {fw/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {fw/2:.5f}")

    elif gt == 'worm_shaft':
        md = d.get('major_diameter', 0.012) * s
        ml = d.get('length', 0.040) * s
        g(type="cylinder", size=f"{md/2:.5f} {ml/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {ml/2:.5f}")

    elif gt in ('gear_rack','gear_rack_heavy'):
        rl = d.get('length', 0.100) * s
        rh = d.get('height', 0.015) * s
        rw = d.get('width', 0.015) / 2 * s
        g(type="box", size=f"{rl/2:.5f} {rw:.5f} {rh/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {rh/2:.5f}")

    # ─── 联轴器 ───
    elif gt in ('coupling_elastic','coupling_jaw','coupling_bellows',
                 'coupling_oldham','coupling_magnetic','coupling_rigid'):
        od = d.get('outer_diameter', 0.020) * s
        tl = d.get('total_length', 0.030) * s
        bore = d.get('bore_a', d.get('bore', 0.006)) * s
        g(type="cylinder", size=f"{od/2:.5f} {tl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {tl/2:.5f}")
        g(type="cylinder", size=f"{bore/2:.5f} {tl*0.9:.5f}", rgba="0.06 0.06 0.06 1", pos=f"0 0 {tl/2:.5f}")

    elif gt in ('timing_belt_loop','v_belt'):
        bl = d.get('length', d.get('inner_length', 0.200)) * s
        bw = d.get('width', d.get('top_width', 0.015)) * s
        bt = d.get('thickness', 0.006) * s
        g(type="cylinder", size=f"{bl/(2*np.pi):.5f} {bw/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('roller_chain','roller_chain_precision'):
        pitch = d.get('pitch', 0.0127) * s
        lc = d.get('link_count', 100)
        tl = pitch * lc
        rw = max(d.get('inner_width', 0.008), 0.004) * s
        g(type="cylinder", size=f"{tl/(2*np.pi):.5f} {rw/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── 离合器 / 张紧器 ───
    elif gt == 'one_way_clutch':
        od = d.get('outer_diameter', 0.034) * s
        wid = d.get('width', 0.016) * s
        g(type="cylinder", size=f"{od/2:.5f} {wid/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {wid/2:.5f}")
        id_ = d.get('inner_diameter', 0.015) * s
        g(type="cylinder", size=f"{id_/2:.5f} {wid*0.9:.5f}", rgba="0.06 0.06 0.06 1", pos=f"0 0 {wid/2:.5f}")

    elif gt == 'torque_limiter':
        od = d.get('outer_diameter', 0.055) * s
        tl = d.get('length', 0.050) * s
        g(type="cylinder", size=f"{od/2:.5f} {tl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {tl/2:.5f}")

    elif gt in ('belt_tensioner','chain_tensioner'):
        pd = d.get('pulley_diameter', d.get('sprocket_teeth', 12)*0.002) * s
        al = d.get('arm_length', 0.040) * s
        g(type="cylinder", size=f"{pd/2:.5f} 0.006", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 0.006")
        g(type="box", size=f"{al/2:.5f} 0.003 0.003", rgba="0.30 0.32 0.34 1", pos=f"{-al/2:.5f} 0 0.003")

    # ═══════════════════════════════════════════════════════════════
    #  能源系统
    # ═══════════════════════════════════════════════════════════════
    elif gt == 'battery_pack':
        l = d.get('length', 0.138) * s
        w = d.get('width', 0.046) * s
        h = d.get('height', 0.050) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")
        # 标签线
        g(type="cylinder", size=f"{l*0.05:.5f} {h/2:.5f}", rgba="0.50 0.45 0.10 1", pos=f"{-l*0.25:.5f} {w*0.55:.5f} {h/2:.5f}")

    elif gt == 'battery_pack_large':
        l = d.get('length', 0.165) * s
        w = d.get('width', 0.056) * s
        h = d.get('height', 0.136) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    elif gt == 'battery_pack_small':
        l = d.get('length', 0.070) * s
        w = d.get('width', 0.034) * s
        h = d.get('height', 0.025) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    elif gt == 'battery_prismatic':
        l = d.get('length', 0.148) * s
        w = d.get('width', 0.066) * s
        h = d.get('height', 0.095) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    elif gt in ('cylindrical_cell','cylindrical_cell_large'):
        dia = d.get('diameter', 0.0186) * s
        h = d.get('height', 0.065) * s
        g(type="cylinder", size=f"{dia/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    elif gt in ('esc_heatsink','esc_compact'):
        l = d.get('length', 0.065) * s
        w = d.get('width', 0.055) * s
        h = d.get('height', 0.030) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")
        fins = min(d.get('heatsink_fins', 8), 10)
        fw = (w/2)/(fins*2)
        for i in range(fins):
            px = -w/2 + fw*(i*2+1)
            g(type="box", size=f"{fw*0.3:.5f} {w*0.4:.5f} {h*0.3:.5f}", rgba="0.78 0.80 0.82 1",
              pos=f"{px:.5f} 0 {h+h*0.15:.5f}")

    elif gt == 'esc_micro':
        l = d.get('length', 0.045) * s
        w = d.get('width', 0.030) * s
        h = d.get('height', 0.012) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    elif gt in ('bms_board','bms_board_large'):
        l = d.get('length', 0.060) * s
        w = d.get('width', 0.045) * s
        h = d.get('height', 0.008) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    elif gt == 'power_module':
        l = d.get('length', 0.050) * s
        w = d.get('width', 0.040) * s
        h = d.get('height', 0.025) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    elif gt == 'pdb':
        l = d.get('length', 0.055) * s
        w = d.get('width', 0.040) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} 0.002", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('dc_dc_converter','dc_dc_enclosed'):
        l = d.get('length', 0.043) * s
        w = d.get('width', 0.021) * s
        h = d.get('height', 0.014) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    elif gt in ('supercapacitor_can','supercapacitor_large'):
        dia = d.get('diameter', 0.016) * s
        h = d.get('height', 0.026) * s
        g(type="cylinder", size=f"{dia/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    # ─── 连接器 ───
    elif gt in ('connector_power','connector_power_small','connector_deans'):
        sl = d.get('shell_length', 0.030) * s
        sw = d.get('shell_width', 0.015) * s
        g(type="box", size=f"{sl/2:.5f} {sw/2:.5f} {sw/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {sw/2:.5f}")

    elif gt == 'connector_power_high_volt':
        sd = d.get('shell_diameter', 0.025) * s
        sl = d.get('shell_length', 0.045) * s
        g(type="cylinder", size=f"{sd/2:.5f} {sl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {sl/2:.5f}")

    elif gt == 'connector_anderson':
        hl = d.get('housing_length', 0.030) * s
        hw = d.get('housing_width', 0.025) * s
        g(type="box", size=f"{hl/2:.5f} {hw/2:.5f} {hw/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {hw/2:.5f}")

    elif gt in ('fuse_holder','fuse_blade_large'):
        l = d.get('length', 0.035) * s
        w = d.get('width', 0.020) * s
        h = d.get('height', d.get('blade_length', 0.040)) / 2 * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    elif gt == 'current_shunt':
        l = d.get('length', 0.030) * s
        w = d.get('width', 0.015) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} 0.001", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ═══════════════════════════════════════════════════════════════
    #  传感系统
    # ═══════════════════════════════════════════════════════════════
    elif gt in ('qfn_package','lga28_package','lga14_package','lga8_package',
                 'so8_package','so16_package','ssop24_package',
                 'dfn_package','lga12_package','sot23_8','to220_package'):
        l = d.get('length', 0.005) * 10 * s
        w = d.get('width', 0.004) * 10 * s
        h = max(d.get('height', 0.001), 0.001) * 15 * s
        g(type="box", size=f"{l:.5f} {w:.5f} {h:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h:.5f}")

    elif gt == 'small_pcb_module':
        l = d.get('length', 0.01) * 5 * s
        w = d.get('width', 0.008) * 5 * s
        g(type="box", size=f"{l:.5f} {w:.5f} 0.002", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 0.002")
        lr = d.get('lens_dia', 0.003) * 2 * s
        g(type="cylinder", size=f"{lr/2:.5f} 0.003", rgba="0.80 0.84 0.89 0.80", pos=f"0 0 0.005")

    elif gt in ('lidar_module','lidar_module_long'):
        l = d.get('length', 0.029) * s
        w = d.get('width', 0.021) * s
        h = d.get('height', 0.016) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    elif gt in ('lidar_cylindrical','lidar_disk'):
        dia = d.get('diameter', 0.075) * s
        h = d.get('height', 0.040) * s
        g(type="cylinder", size=f"{dia/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    elif gt in ('encoder_incremental','encoder_absolute','encoder_hollow'):
        bd = d.get('body_diameter', d.get('od', 0.040)) * s
        bt = d.get('body_thickness', d.get('thickness', 0.025)) * s
        sd = d.get('shaft_dia', 0.006) * s
        g(type="cylinder", size=f"{bd/2:.5f} {bt/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {bt/2:.5f}")
        g(type="cylinder", size=f"{sd/2:.5f} {bt*0.6:.5f}", rgba="0.70 0.70 0.73 1", pos=f"0 0 {bt:.5f}")

    elif gt in ('load_cell_aluminum','load_cell_mini'):
        l = d.get('length', 0.125) * s
        w = d.get('width', 0.013) * s
        h = d.get('height', 0.025) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    elif gt == 'ultrasonic_sensor':
        l = d.get('length', 0.045) * s
        w = d.get('width', 0.020) * s
        h = d.get('height', 0.015) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    elif gt == 'pressure_sensor_can':
        dia = d.get('diameter', 0.0065) * s
        h = d.get('height', 0.0049) * s
        g(type="cylinder", size=f"{dia/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    # ═══════════════════════════════════════════════════════════════
    #  控制系统
    # ═══════════════════════════════════════════════════════════════
    elif gt in ('pcb_board','pcb_board_large'):
        l = d.get('length', 0.085) * s
        w = d.get('width', 0.056) * s
        th = 0.003 * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {th:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {th}")
        # USB端口
        usb_h = d.get('usb_height', 0.012) * s
        g(type="box", size=f"{l*0.06:.5f} {w*0.08:.5f} {usb_h/2:.5f}", rgba="0.66 0.66 0.68 1",
          pos=f"{-l*0.4:.5f} {w*0.25:.5f} {th+usb_h/2:.5f}")
        gpio_h = d.get('gpio_header_height', 0.010) * s
        g(type="box", size=f"{l*0.04:.5f} {w*0.18:.5f} {gpio_h/2:.5f}", rgba="0.10 0.10 0.10 1",
          pos=f"{l*0.1:.5f} {w*0.45:.5f} {th+gpio_h/2:.5f}")

    elif gt == 'button_mushroom':
        md = d.get('mushroom_dia', 0.036) * s
        bd = d.get('body_diameter', 0.022) * s
        bdep = d.get('body_depth', 0.040) * s
        g(type="cylinder", size=f"{md/2:.5f} {md*0.12:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {md*0.12:.5f}")
        g(type="cylinder", size=f"{bd/2:.5f} {bdep/2:.5f}", rgba="0.22 0.20 0.22 1", pos=f"0 0 {-bdep/2:.5f}")

    elif gt == 'micro_switch':
        l = d.get('length', 0.0125) * s
        w = d.get('width', 0.0063) * s
        h = d.get('height', 0.0125) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")
        g(type="cylinder", size=f"{w*0.2:.5f} {h*0.15:.5f}", rgba="0.85 0.82 0.80 1", pos=f"{-l*0.3:.5f} 0 {h+h*0.1:.5f}")

    elif gt == 'joystick_module':
        l = d.get('length', 0.040) * s
        w = d.get('width', 0.030) * s
        h = d.get('height', 0.035) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h*0.3:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h*0.15:.5f}")
        g(type="cylinder", size=f"{w*0.1:.5f} {h*0.5:.5f}", rgba="0.30 0.32 0.34 1", pos=f"0 0 {h*0.3+h*0.25:.5f}")

    elif gt == 'potentiometer_rotary':
        bd = d.get('body_diameter', 0.016) * s
        bdep = d.get('body_depth', 0.025) * s
        g(type="cylinder", size=f"{bd/2:.5f} {bdep/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {bdep/2:.5f}")
        sd = d.get('shaft_dia', 0.006) * s
        sl = d.get('shaft_length', 0.020) * s
        g(type="cylinder", size=f"{sd/2:.5f} {sl/2:.5f}", rgba="0.70 0.70 0.73 1", pos=f"0 0 {bdep+sl/2:.5f}")

    elif gt == 'slider_pot':
        l = d.get('length', 0.060) * s
        w = d.get('width', 0.010) * s
        h = d.get('height', 0.017) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    # ═══════════════════════════════════════════════════════════════
    #  连接件/结构件
    # ═══════════════════════════════════════════════════════════════
    elif gt in ('ring_with_holes','ring_split'):
        od = d.get('outer_diameter', 0.060) * s
        id_ = d.get('inner_diameter', 0.020) * s
        th = d.get('thickness', 0.005) * s
        g(type="cylinder", size=f"{od/2:.5f} {th/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {th/2:.5f}")
        g(type="cylinder", size=f"{id_/2:.5f} {th*1.5:.5f}", rgba="0.06 0.06 0.06 1", pos=f"0 0 {th/2:.5f}")

    elif gt in ('bearing_assembly','bearing_thrust','bearing_angular'):
        od = d.get('outer_diameter', 0.026) * s
        id_ = d.get('inner_diameter', 0.010) * s
        width = d.get('width', 0.008) * s
        g(type="cylinder", size=f"{od/2:.5f} {width/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {width/2:.5f}")
        g(type="cylinder", size=f"{id_/2:.5f} {width*0.9:.5f}", rgba="0.06 0.06 0.06 1", pos=f"0 0 {width/2:.5f}")

    elif gt == 'pin_with_ball':
        dia = d.get('diameter', 0.006) * s
        l = d.get('length', 0.030) * s
        ball_d = d.get('ball_diameter', 0.004) * s
        g(type="cylinder", size=f"{dia/2:.5f} {l/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {l/2:.5f}")
        g(type="sphere", size=f"{ball_d/2:.5f}", rgba="0.82 0.82 0.84 1", pos=f"0 0 {l+ball_d/2:.5f}")

    elif gt == 'extrusion_profile':
        sz = d.get('size', 0.020) * s
        l = d.get('length', 0.100) * s
        g(type="box", size=f"{sz/2:.5f} {sz/2:.5f} {l/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {l/2:.5f}")
        # T型槽
        slot_size = sz * 0.3
        g(type="box", size=f"{slot_size:.5f} {sz/2:.5f} {l/2:.5f}", rgba="0.15 0.15 0.17 1", pos=f"0 {sz/2:.5f} {l/2:.5f}")

    elif gt == 'tube_hollow':
        od = d.get('outer_diameter', 0.016) * s
        hl = d.get('length', 0.200) * s
        id_ = d.get('inner_diameter', od*0.85) * s
        g(type="cylinder", size=f"{od/2:.5f} {hl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {hl/2:.5f}")
        g(type="cylinder", size=f"{id_/2:.5f} {hl/2:.5f}", rgba="0.06 0.06 0.06 1", pos=f"0 0 {hl/2:.5f}")

    elif gt == 'flat_plate':
        l = d.get('length', 0.100) * s
        w = d.get('width', 0.100) * s
        th = d.get('thickness', 0.003) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {th/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {th/2:.5f}")

    elif gt == 'helix_spring':
        od = d.get('outer_diameter', 0.010) * s
        fl = d.get('free_length', 0.030) * s
        g(type="cylinder", size=f"{od/2:.5f} {fl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {fl/2:.5f}")
        id_ = (od * 0.6) / 2
        g(type="cylinder", size=f"{id_:.5f} {fl/2:.5f}", rgba="0.12 0.12 0.14 1", pos=f"0 0 {fl/2:.5f}")

    elif gt == 'sphere_cap':
        dia = d.get('diameter', 0.030) * s
        g(type="sphere", size=f"{dia/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'aluminum_block':
        l = d.get('length', 0.008) * s
        w = d.get('width', 0.006) * s
        h = d.get('height', 0.004) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h/2:.5f}")

    elif gt == 'l_bracket':
        al = d.get('arm_length', 0.030) * s
        aw = d.get('arm_width', 0.020) * s
        th = d.get('thickness', 0.004) * s
        g(type="box", size=f"{al/2:.5f} {aw/2:.5f} {th:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"{al/2:.5f} 0 {th:.5f}")
        g(type="box", size=f"{th:.5f} {aw/2:.5f} {al/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {al/2:.5f}")

    else:
        # 默认：通用box
        l, w, h = 0.03*s, 0.025*s, 0.02*s
        g(type="box", size=f"{l:.5f} {w:.5f} {h:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {h:.5f}")

    geom_xml = '\n      '.join(f'<geom {g_str}/>' for g_str in geoms)

    xml = f"""<mujoco model="{spec.part_id}">
  <visual><global offwidth="{IMG_W}" offheight="{IMG_H}"/></visual>
  <option timestep="0.001" gravity="0 0 0"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.18 0.18" rgb2="0.28 0.28 0.28" width="512" height="512"/>
    <material name="ground" texture="grid" texrepeat="6 6"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.95 0.95 0.92"/>
    <light pos="-1.5 -1.5 1.5" dir="1 1 -1" diffuse="0.35 0.35 0.37"/>
    <light pos="1 1 1.5" dir="-1 -1 -1" diffuse="0.25 0.28 0.32"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.05" material="ground"/>
    <body name="part" pos="0 0 0.12">
      {geom_xml}
    </body>
  </worldbody>
</mujoco>"""
    return xml


def render_one(spec, out_dir, idx, total):
    results = []
    label = f"[{idx+1:3d}/{total}] {spec.display_name[:35]:<35s}"
    try:
        xml_str = make_xml(spec)
        tmp = out_dir / "_tmp.xml"
        tmp.write_text(xml_str, encoding='utf-8')
        model = mujoco.MjModel.from_xml_path(str(tmp))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
        mujoco.mj_resetData(model, data)
        for _ in range(40):
            mujoco.mj_step(model, data)
        for view in VIEWS:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, cam)
            cam.lookat[:] = np.array(view['lookat'])
            cam.distance = view['distance']
            cam.azimuth = view['azimuth']
            cam.elevation = view['elevation']
            mujoco.mj_resetData(model, data)
            for _ in range(15):
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            img = Image.fromarray(rgb)
            fname = f"{idx+1:03d}_{spec.part_id}_{view['name']}.png"
            fpath = out_dir / fname
            img.save(fpath)
            results.append({'file': fpath, 'view': view['name']})
        renderer.close()
        tmp.unlink(missing_ok=True)
        print(f"  {label} OK ({len(results)} views)")
    except Exception as e:
        print(f"  {label} FAIL: {str(e)[:80]}")
    return results


def build_html(out_dir, all_res):
    html = '''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>V8 零件库 v2 — 3D预览 (1920x1080)</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Segoe UI",Arial,sans-serif;background:#0d1117;color:#e6edf3;padding:20px}
h1{text-align:center;font-size:1.4em;color:#fff;margin-bottom:4px}
.sub{text-align:center;color:#8b949e;margin-bottom:20px;font-size:.85em}
.cat{margin-bottom:32px}
.cat-hd{background:linear-gradient(90deg,#161b22,#21262d);padding:8px 16px;border-radius:6px;font-size:1.05em;color:#58a6ff;margin-bottom:12px;border-left:3px solid #58a6ff}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.card{background:#161b22;border-radius:8px;overflow:hidden;border:1px solid #30363d;transition:transform .12s}
.card:hover{transform:translateY(-2px);border-color:#58a6ff}
.card img{width:100%;height:180px;object-fit:contain;background:#0d1117;padding:6px}
.info{padding:8px 10px}
.name{font-weight:600;color:#fff;font-size:.82em;margin-bottom:2px}
.detail{color:#8b949e;font-size:.7em}.ok{color:#3fb950}.fail{color:#f85149}
</style></head><body>
<h1>V8 零件库 v2 — 3D预览图集</h1>
<p class="sub"><strong>''' + str(len(all_res)) + '''</strong> 种零件 | 1920x1080 高清渲染 | 4视角 | 6大类</p>
'''
    cat_order = [('actuators', '执行器'), ('transmission', '传动系统'),
                 ('energy', '能源系统'), ('sensors', '传感系统'),
                 ('controllers', '控制系统'), ('connectors', '连接件')]
    ok_count = 0
    for cat_id, cat_name in cat_order:
        items = [(pid, info) for pid, info in all_res.items()
                 if info['spec'].category == cat_id]
        if not items:
            continue
        html += f'<div class="cat"><div class="cat-hd">{cat_name} ({len(items)}种)</div><div class="grid">\n'
        for pid, info in items:
            spec = info['spec']
            iso_file = next((r['file'].name for r in info['results'] if r['view'] == 'iso'), '')
            has_img = len(info['results']) > 0
            if has_img:
                ok_count += 1
            status = '<span class="ok">OK</span>' if has_img else '<span class="fail">FAIL</span>'
            html += f'''<div class="card">
<img src="{iso_file}" alt="{spec.display_name}" onerror="this.style.display='none'">
<div class="info"><div class="name">{spec.display_name} {status}</div>
<div class="detail">{spec.manufacturer} {spec.part_number}</div></div></div>\n'''
        html += '</div></div>\n'
    html += f'<p style="text-align:center;color:#8b949e;margin-top:16px">成功渲染 {ok_count}/{len(all_res)} 个零件</p></body></html>'
    p = out_dir / 'index.html'
    p.write_text(html, encoding='utf-8')
    return p


def main():
    print("=" * 68)
    print("  V8 零件库 v2 — 高精3D预览渲染引擎 (1920x1080)")
    print("=" * 68)
    out = Path(OUTPUT_DIR)
    out.mkdir(exist_ok=True)
    print(f"\n  加载零件库...")
    parts = get_all_part_specs()
    ids = list(parts.keys())
    n = len(ids)
    print(f"  共 {n} 个零件\n")
    all_results = {}
    for i, pid in enumerate(ids):
        spec = parts[pid]
        res = render_one(spec, out, i, n)
        all_results[pid] = {'spec': spec, 'results': res}
    print(f"\n  生成HTML...")
    build_html(out, all_results)
    total_imgs = sum(len(r['results']) for r in all_results.values())
    ok = sum(1 for r in all_results.values() if len(r['results']) > 0)
    print(f"\n{'='*68}")
    print(f"  完成! 成功 {ok}/{n}, {total_imgs}张图片")
    print(f"  {out.absolute()}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()