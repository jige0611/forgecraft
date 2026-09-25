# ═══════════════════════════════════════════════════════════
#  V8 零件库 v2 交互式3D浏览器 (168种零件)
#  支持：拖拽旋转 | 滚轮缩放 | 中键平移 | 键盘切换
# ═══════════════════════════════════════════════════════════

import mujoco
try:
    import mujoco_viewer
except ImportError:
    import mujoco_python_viewer as mujoco_viewer
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from v8_3d_model_library_v2 import get_all_part_specs, PartSpec

SCALE = 2.2
PARTS_PER_XML = 12  # 每个XML文件最多包含的零件数


def make_xml_for_spec(spec: PartSpec) -> str:
    """为单个零件生成MJCF XML"""
    d = spec.dimensions; c = spec.color; gt = spec.geometry_type; s = SCALE
    geoms = []

    def g(**kw):
        geoms.append(" ".join(f'{k}="{v}"' for k, v in kw.items()))

    # ─── 圆柱类 ───
    if gt in ('cylinder_detail','cylinder_simple','cylinder_complex',
              'cylinder_with_hub','cylinder_frc','cylinder_cim',
              'cylinder_re30','cylinder_medium','cylinder_tall',
              'cylinder_step','cylinder_small','cylinder_small_long'):
        od = d.get('outer_diameter', d.get('body_diameter', 0.03))
        r = od / 2 * s
        h = d.get('total_height', d.get('body_length', 0.05)) * s
        g(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        sr = d.get('shaft_diameter', r*0.25) / 2 * s
        sl = d.get('shaft_length', h*0.3) * s
        g(type="cylinder", size=f"{sr:.5f} {sl/2:.5f}", rgba="0.65 0.65 0.68 1", pos=f"0 0 {h+sl/2:.5f}")
        if 'flange_diameter' in d:
            fr = d['flange_diameter'] / 2 * s
            ft = d.get('flange_thickness', 0.003) * s
            g(type="cylinder", size=f"{fr:.5f} {ft:.5f}", rgba="0.42 0.40 0.45 1", pos=f"0 0 {ft:.5f}")

    elif gt == 'pancake_motor':
        r = d.get('outer_diameter', 0.045) / 2 * s
        h = d.get('total_height', 0.03) * s
        g(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        sr = d.get('shaft_diameter', 0.008) / 2 * s
        sl = d.get('shaft_length', 0.014) * s
        g(type="cylinder", size=f"{sr:.5f} {sl/2:.5f}", rgba="0.65 0.65 0.68 1", pos=f"0 0 {h+sl/2:.5f}")

    elif gt in ('square_cylinder_stepper','square_cylinder_large','square_cylinder_xlarge'):
        sz = d.get('square_size', 0.042) / 2 * s
        bl = d.get('body_length', 0.048) * s
        g(type="box", size=f"{sz:.5f} {sz:.5f} {bl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        sr = d.get('shaft_diameter', 0.005) / 2 * s
        sl = d.get('shaft_length', 0.022) * s
        g(type="cylinder", size=f"{sr:.5f} {sl/2:.5f}", rgba="0.70 0.70 0.73 1", pos=f"0 0 {bl+sl/2:.5f}")

    elif gt == 'disc_motor':
        r = d.get('outer_diameter', 0.085) / 2 * s
        h = d.get('total_height', 0.037) * s
        g(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'joint_motor':
        r = d.get('outer_diameter', 0.060) / 2 * s
        h = d.get('total_height', 0.068) * s
        g(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'multirotor_motor':
        r = d.get('outer_diameter', 0.050) / 2 * s
        h = d.get('total_height', 0.028) * s
        g(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'disc_motor_thin':
        r = d.get('outer_diameter', 0.056) / 2 * s
        h = d.get('total_height', 0.028) * s
        g(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── Dynamixel外壳 ───
    elif gt in ('dynamixel_shell','dynamixel_mini'):
        l = d.get('length', 0.043) / 2 * s
        w = d.get('width', 0.043) / 2 * s
        h = d.get('height', 0.050) / 2 * s
        g(type="box", size=f"{l:.5f} {w:.5f} {h:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        hub_r = d.get('hub_diameter', 0.028) / 2 * s
        hub_h = d.get('hub_height', 0.004) * s
        g(type="cylinder", size=f"{hub_r:.5f} {hub_h:.5f}", rgba="0.55 0.52 0.58 1", pos=f"0 0 {h+hub_h:.5f}")

    # ─── 舵机 ───
    elif gt in ('servo_standard','servo_metal'):
        l = d.get('length', 0.040) / 2 * s
        w = d.get('width', 0.020) / 2 * s
        h = d.get('height', 0.038) / 2 * s
        g(type="box", size=f"{l:.5f} {w:.5f} {h:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

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
        g(type="cylinder", size=f"{r:.5f} {l/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── 线性执行器 ───
    elif gt == 'linear_actuator':
        bl = d.get('body_length', 0.180) * s
        bw = d.get('body_width', 0.040) / 2 * s
        g(type="box", size=f"{bl/2:.5f} {bw:.5f} {bw:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        rod_r = d.get('rod_diameter', 0.010) / 2 * s
        stroke = d.get('stroke', 0.100) * s
        g(type="cylinder", size=f"{rod_r:.5f} {stroke/2:.5f}", rgba="0.75 0.75 0.77 1", pos=f"{bl/2+stroke/2:.5f} 0 {bw:.5f}")

    # ─── 谐波减速器 ───
    elif gt in ('harmonic_assembly','harmonic_compact','harmonic_mini'):
        cr = d.get('casing_od', 0.085) / 2 * s
        cl = d.get('casing_len', 0.070) * s
        g(type="cylinder", size=f"{cr:.5f} {cl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── 行星减速箱 ───
    elif gt in ('planetary_gearbox','planetary_gearbox_large',
                 'planetary_neugart','planetary_apex'):
        ir = d.get('input_diameter', 0.020) / 2 * s
        ol = d.get('length', 0.055) * s
        g(type="cylinder", size=f"{ir:.5f} {ol/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── 摆线减速机 ───
    elif gt in ('cycloidal_gearbox','cycloidal_gearbox_large'):
        fr = d.get('flange_input_dia', 0.070) / 2 * s
        ol = d.get('length', 0.095) * s
        g(type="cylinder", size=f"{fr:.5f} {ol/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── 音圈电机 ───
    elif gt == 'voice_coil':
        r = d.get('body_diameter', 0.030) / 2 * s
        stroke = d.get('stroke', 0.012) * s
        g(type="cylinder", size=f"{r:.5f} {stroke/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── 直线电机 ───
    elif gt == 'linear_motor':
        w = d.get('width', 0.100) / 2 * s
        h = d.get('height', 0.070) / 2 * s
        l = 0.15 * s
        g(type="box", size=f"{l/2:.5f} {w:.5f} {h:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── 传动系统：带轮/链轮/齿轮 ───
    elif gt in ('timing_pulley','timing_pulley_large','timing_pulley_htd',
                 'sprocket','sprocket_precision',
                 'spur_gear','spur_gear_large','spur_gear_heavy'):
        pd = d.get('pitch_diameter', 0.01277) * s
        w = d.get('width', d.get('face_width', 0.015)) * s
        bore = d.get('bore', 0.005) * s
        g(type="cylinder", size=f"{pd/2:.5f} {w/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        g(type="cylinder", size=f"{bore/2:.5f} {w*0.6:.5f}", rgba="0.06 0.06 0.06 1")

    elif gt == 'helical_gear':
        pd = d.get('pitch_diameter', 0.025) * s
        fw = d.get('face_width', 0.015) * s
        bore = d.get('bore', 0.006) * s
        g(type="cylinder", size=f"{pd/2:.5f} {fw/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        g(type="cylinder", size=f"{bore/2:.5f} {fw*0.6:.5f}", rgba="0.06 0.06 0.06 1")

    elif gt == 'bevel_gear':
        pd = d.get('pitch_diameter', 0.040) * s
        cd = d.get('cone_distance', 0.028) * s
        g(type="cylinder", size=f"{pd/2:.5f} {cd/3:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'worm_wheel':
        pd = d.get('pitch_diameter', 0.080) * s
        fw = d.get('face_width', 0.020) * s
        g(type="cylinder", size=f"{pd/2:.5f} {fw/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'worm_shaft':
        md = d.get('major_diameter', 0.012) * s
        ml = d.get('length', 0.040) * s
        g(type="cylinder", size=f"{md/2:.5f} {ml/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('gear_rack','gear_rack_heavy'):
        rl = d.get('length', 0.100) * s
        rh = d.get('height', 0.015) * s
        rw = d.get('width', 0.015) / 2 * s
        g(type="box", size=f"{rl/2:.5f} {rw:.5f} {rh/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── 联轴器 ───
    elif gt in ('coupling_elastic','coupling_jaw','coupling_bellows',
                 'coupling_oldham','coupling_magnetic','coupling_rigid'):
        od = d.get('outer_diameter', 0.020) * s
        tl = d.get('total_length', 0.030) * s
        bore = d.get('bore_a', d.get('bore', 0.006)) * s
        g(type="cylinder", size=f"{od/2:.5f} {tl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        g(type="cylinder", size=f"{bore/2:.5f} {tl*0.7:.5f}", rgba="0.06 0.06 0.06 1")

    # ─── 带/链 ───
    elif gt in ('timing_belt_loop','v_belt'):
        bl = d.get('length', d.get('inner_length', 0.200)) * s
        bw = d.get('width', d.get('top_width', 0.015)) * s
        g(type="cylinder", size=f"{bl/(2*np.pi):.5f} {bw/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('roller_chain','roller_chain_precision'):
        pitch = d.get('pitch', 0.0127) * s
        lc = d.get('link_count', 100)
        tl = pitch * lc
        rw = max(d.get('inner_width', 0.008), 0.004) * s
        g(type="cylinder", size=f"{tl/(2*np.pi):.5f} {rw/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── 离合器/张紧器 ───
    elif gt == 'one_way_clutch':
        od = d.get('outer_diameter', 0.034) * s
        wid = d.get('width', 0.016) * s
        id_ = d.get('inner_diameter', 0.015) * s
        g(type="cylinder", size=f"{od/2:.5f} {wid/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        g(type="cylinder", size=f"{id_/2:.5f} {wid*0.7:.5f}", rgba="0.06 0.06 0.06 1")

    elif gt == 'torque_limiter':
        od = d.get('outer_diameter', 0.055) * s
        tl = d.get('length', 0.050) * s
        g(type="cylinder", size=f"{od/2:.5f} {tl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('belt_tensioner','chain_tensioner'):
        pd = d.get('pulley_diameter', d.get('sprocket_teeth', 12)*0.002) * s
        g(type="cylinder", size=f"{pd/2:.5f} 0.006", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── 能源系统 ───
    elif gt == 'battery_pack':
        l = d.get('length', 0.138) * s
        w = d.get('width', 0.046) * s
        h = d.get('height', 0.050) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'battery_pack_large':
        l = d.get('length', 0.165) * s
        w = d.get('width', 0.056) * s
        h = d.get('height', 0.136) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'battery_pack_small':
        l = d.get('length', 0.070) * s
        w = d.get('width', 0.034) * s
        h = d.get('height', 0.025) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'battery_prismatic':
        l = d.get('length', 0.148) * s
        w = d.get('width', 0.066) * s
        h = d.get('height', 0.095) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('cylindrical_cell','cylindrical_cell_large'):
        dia = d.get('diameter', 0.0186) * s
        h = d.get('height', 0.065) * s
        g(type="cylinder", size=f"{dia/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('esc_heatsink','esc_compact'):
        l = d.get('length', 0.065) * s
        w = d.get('width', 0.055) * s
        h = d.get('height', 0.030) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'esc_micro':
        l = d.get('length', 0.045) * s
        w = d.get('width', 0.030) * s
        h = d.get('height', 0.012) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('bms_board','bms_board_large'):
        l = d.get('length', 0.060) * s
        w = d.get('width', 0.045) * s
        h = d.get('height', 0.008) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'power_module':
        l = d.get('length', 0.050) * s
        w = d.get('width', 0.040) * s
        h = d.get('height', 0.025) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'pdb':
        l = d.get('length', 0.055) * s
        w = d.get('width', 0.040) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} 0.002", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('dc_dc_converter','dc_dc_enclosed'):
        l = d.get('length', 0.043) * s
        w = d.get('width', 0.021) * s
        h = d.get('height', 0.014) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('supercapacitor_can','supercapacitor_large'):
        dia = d.get('diameter', 0.016) * s
        h = d.get('height', 0.026) * s
        g(type="cylinder", size=f"{dia/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── 连接器 ───
    elif gt in ('connector_power','connector_power_small','connector_deans',
                 'connector_anderson'):
        sl = d.get('shell_length', d.get('housing_length', d.get('total_length', 0.030))) * s
        sw = d.get('shell_width', d.get('housing_width', d.get('width', 0.015))) * s
        g(type="box", size=f"{sl/2:.5f} {sw/2:.5f} {sw/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'connector_power_high_volt':
        sd = d.get('shell_diameter', 0.025) * s
        sl = d.get('shell_length', 0.045) * s
        g(type="cylinder", size=f"{sd/2:.5f} {sl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('fuse_holder','fuse_blade_large'):
        l = d.get('length', 0.035) * s
        w = d.get('width', 0.020) * s
        h = d.get('height', d.get('blade_length', 0.040)) / 2 * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'current_shunt':
        l = d.get('length', 0.030) * s
        w = d.get('width', 0.015) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} 0.001", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── 传感系统 ───
    elif gt in ('qfn_package','lga28_package','lga14_package','lga8_package',
                 'so8_package','so16_package','ssop24_package',
                 'dfn_package','lga12_package','sot23_8','to220_package'):
        l = d.get('length', 0.005) * 10 * s
        w = d.get('width', 0.004) * 10 * s
        h = max(d.get('height', 0.001), 0.001) * 15 * s
        g(type="box", size=f"{l:.5f} {w:.5f} {h:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'small_pcb_module':
        l = d.get('length', 0.01) * 5 * s
        w = d.get('width', 0.008) * 5 * s
        g(type="box", size=f"{l:.5f} {w:.5f} 0.002", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('lidar_module','lidar_module_long'):
        l = d.get('length', 0.029) * s
        w = d.get('width', 0.021) * s
        h = d.get('height', 0.016) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('lidar_cylindrical','lidar_disk'):
        dia = d.get('diameter', 0.075) * s
        h = d.get('height', 0.040) * s
        g(type="cylinder", size=f"{dia/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('encoder_incremental','encoder_absolute','encoder_hollow'):
        bd = d.get('body_diameter', d.get('od', 0.040)) * s
        bt = d.get('body_thickness', d.get('thickness', 0.025)) * s
        g(type="cylinder", size=f"{bd/2:.5f} {bt/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt in ('load_cell_aluminum','load_cell_mini'):
        l = d.get('length', 0.125) * s
        w = d.get('width', 0.013) * s
        h = d.get('height', 0.025) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'ultrasonic_sensor':
        l = d.get('length', 0.045) * s
        w = d.get('width', 0.020) * s
        h = d.get('height', 0.015) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'pressure_sensor_can':
        dia = d.get('diameter', 0.0065) * s
        h = d.get('height', 0.0049) * s
        g(type="cylinder", size=f"{dia/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── 控制系统 ───
    elif gt in ('pcb_board','pcb_board_large'):
        l = d.get('length', 0.085) * s
        w = d.get('width', 0.056) * s
        th = 0.003 * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {th:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        usb_h = d.get('usb_height', 0.012) * s
        g(type="box", size=f"{l*0.06:.5f} {w*0.08:.5f} {usb_h/2:.5f}", rgba="0.66 0.66 0.68 1",
          pos=f"{-l*0.4:.5f} {w*0.25:.5f} {th+usb_h/2:.5f}")
        gpio_h = d.get('gpio_header_height', 0.010) * s
        g(type="box", size=f"{l*0.04:.5f} {w*0.18:.5f} {gpio_h/2:.5f}", rgba="0.10 0.10 0.10 1",
          pos=f"{l*0.1:.5f} {w*0.45:.5f} {th+gpio_h/2:.5f}")

    elif gt == 'button_mushroom':
        md = d.get('mushroom_dia', 0.036) * s
        g(type="cylinder", size=f"{md/2:.5f} {md*0.12:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'micro_switch':
        l = d.get('length', 0.0125) * s
        w = d.get('width', 0.0063) * s
        h = d.get('height', 0.0125) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'joystick_module':
        l = d.get('length', 0.040) * s
        w = d.get('width', 0.030) * s
        h = d.get('height', 0.035) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h*0.3:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        g(type="cylinder", size=f"{w*0.1:.5f} {h*0.5:.5f}", rgba="0.30 0.32 0.34 1", pos=f"0 0 {h*0.3+h*0.25:.5f}")

    elif gt == 'potentiometer_rotary':
        bd = d.get('body_diameter', 0.016) * s
        bdep = d.get('body_depth', 0.025) * s
        g(type="cylinder", size=f"{bd/2:.5f} {bdep/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'slider_pot':
        l = d.get('length', 0.060) * s
        w = d.get('width', 0.010) * s
        h = d.get('height', 0.017) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    # ─── 连接件/结构件 ───
    elif gt in ('ring_with_holes','ring_split'):
        od = d.get('outer_diameter', 0.060) * s
        id_ = d.get('inner_diameter', 0.020) * s
        th = d.get('thickness', 0.005) * s
        g(type="cylinder", size=f"{od/2:.5f} {th/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        g(type="cylinder", size=f"{id_/2:.5f} {th*1.2:.5f}", rgba="0.06 0.06 0.06 1")

    elif gt in ('bearing_assembly','bearing_thrust','bearing_angular'):
        od = d.get('outer_diameter', 0.026) * s
        id_ = d.get('inner_diameter', 0.010) * s
        width = d.get('width', 0.008) * s
        g(type="cylinder", size=f"{od/2:.5f} {width/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        g(type="cylinder", size=f"{id_/2:.5f} {width*0.7:.5f}", rgba="0.06 0.06 0.06 1")

    elif gt == 'pin_with_ball':
        dia = d.get('diameter', 0.006) * s
        l = d.get('length', 0.030) * s
        ball_d = d.get('ball_diameter', 0.004) * s
        g(type="cylinder", size=f"{dia/2:.5f} {l/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        g(type="sphere", size=f"{ball_d/2:.5f}", rgba="0.82 0.82 0.84 1", pos=f"0 0 {l+ball_d/2:.5f}")

    elif gt == 'extrusion_profile':
        sz = d.get('size', 0.020) * s
        l = d.get('length', 0.100) * s
        g(type="box", size=f"{sz/2:.5f} {sz/2:.5f} {l/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'tube_hollow':
        od = d.get('outer_diameter', 0.016) * s
        hl = d.get('length', 0.200) * s
        id_ = d.get('inner_diameter', od*0.85) * s
        g(type="cylinder", size=f"{od/2:.5f} {hl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        g(type="cylinder", size=f"{id_/2:.5f} {hl/2:.5f}", rgba="0.06 0.06 0.06 1")

    elif gt == 'flat_plate':
        l = d.get('length', 0.100) * s
        w = d.get('width', 0.100) * s
        th = d.get('thickness', 0.003) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {th/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'helix_spring':
        od = d.get('outer_diameter', 0.010) * s
        fl = d.get('free_length', 0.030) * s
        g(type="cylinder", size=f"{od/2:.5f} {fl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'sphere_cap':
        dia = d.get('diameter', 0.030) * s
        g(type="sphere", size=f"{dia/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'aluminum_block':
        l = d.get('length', 0.008) * s
        w = d.get('width', 0.006) * s
        h = d.get('height', 0.004) * s
        g(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    elif gt == 'l_bracket':
        al = d.get('arm_length', 0.030) * s
        aw = d.get('arm_width', 0.020) * s
        th = d.get('thickness', 0.004) * s
        g(type="box", size=f"{al/2:.5f} {aw/2:.5f} {th:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"{al/2:.5f} 0 {th:.5f}")
        g(type="box", size=f"{th:.5f} {aw/2:.5f} {al/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}", pos=f"0 0 {al/2:.5f}")

    else:
        l, w, h = 0.03*s, 0.025*s, 0.02*s
        g(type="box", size=f"{l:.5f} {w:.5f} {h:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

    geom_xml = '\n        '.join(f'<geom {g_str}/>' for g_str in geoms)

    return f"""<mujoco model="{spec.part_id}">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.15 0.15 0.15" rgb2="0.22 0.22 0.22" width="512" height="512"/>
    <material name="ground" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.9 0.9 0.85"/>
    <light pos="-1 -1 1.5" dir="0.5 0.5 -1" diffuse="0.3 0.3 0.32"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="ground"/>
    <body name="part" pos="0 0 0.12">
        {geom_xml}
    </body>
  </worldbody>
</mujoco>"""


def build_multi_xml(parts_dict: dict) -> tuple:
    """将所有零件放到一个XML的body中，每个隔开一段距离"""
    content = []
    parts_list = list(parts_dict.items())
    n = len(parts_list)
    cols = 4
    spacing = 0.25 * SCALE
    z_offset = 0.12

    for i, (pid, spec) in enumerate(parts_list):
        row = i // cols
        col = i % cols
        x = col * spacing - (cols-1)*spacing/2
        y = row * spacing

        d = spec.dimensions; c = spec.color; gt = spec.geometry_type; s = SCALE

        # 简化渲染逻辑（只取核心几何体）
        body_geoms = []
        def bg(**kw):
            body_geoms.append(" ".join(f'{k}="{v}"' for k, v in kw.items()))

        if 'cylinder' in gt[:8] or gt in ('pancake_motor','gearbox_dc','gearbox_dc_small',
                'micro_dc','joint_motor','multirotor_motor','disc_motor_thin'):
            od = d.get('outer_diameter', d.get('body_diameter', d.get('casing_od', d.get('flange_input_dia',0.03))))
            r = od / 2 * s
            h = d.get('total_height', d.get('body_length', d.get('casing_len', d.get('length', d.get('gearbox_len',d.get('motor_len',0.05)))))) * s
            bg(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('square_cylinder_stepper','square_cylinder_large','square_cylinder_xlarge'):
            sz = d.get('square_size', 0.042) / 2 * s
            bl = d.get('body_length', 0.048) * s
            bg(type="box", size=f"{sz:.5f} {sz:.5f} {bl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif 'box' in gt[:3] or gt in ('esc_heatsink','esc_compact','esc_micro','bms_board',
                'bms_board_large','power_module','dc_dc_converter','dc_dc_enclosed','pdb',
                'battery_pack','battery_pack_large','battery_pack_small','battery_prismatic'):
            l = d.get('length', 0.04) / 2 * s
            w = d.get('width', 0.03) / 2 * s
            h = d.get('height', 0.02) / 2 * s
            bg(type="box", size=f"{l:.5f} {w:.5f} {h:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('dynamixel_shell','dynamixel_mini','servo_standard','servo_metal'):
            l = d.get('length', 0.043) / 2 * s
            w = d.get('width', 0.043) / 2 * s
            h = d.get('height', 0.050) / 2 * s
            bg(type="box", size=f"{l:.5f} {w:.5f} {h:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('linear_actuator','linear_motor'):
            bl = d.get('body_length', d.get('width', 0.180)) * s
            bw = d.get('body_width', d.get('height', 0.040)) / 2 * s
            bg(type="box", size=f"{bl/2:.5f} {bw:.5f} {bw:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('disc_motor',):
            r = d.get('outer_diameter', 0.085) / 2 * s
            h = d.get('total_height', 0.037) * s
            bg(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('voice_coil',):
            r = d.get('body_diameter', 0.030) / 2 * s
            h = d.get('stroke', 0.012) * s
            bg(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('timing_pulley','timing_pulley_large','timing_pulley_htd',
                     'sprocket','sprocket_precision',
                     'spur_gear','spur_gear_large','spur_gear_heavy',
                     'helical_gear','bevel_gear','worm_wheel','worm_shaft',
                     'harmonic_assembly','harmonic_compact','harmonic_mini',
                     'planetary_gearbox','planetary_gearbox_large',
                     'planetary_neugart','planetary_apex',
                     'cycloidal_gearbox','cycloidal_gearbox_large'):
            pd = d.get('pitch_diameter', d.get('casing_od', d.get('input_diameter', d.get('flange_input_dia', d.get('major_diameter',0.02)))))
            r = pd / 2 * s
            h = d.get('width', d.get('face_width', d.get('casing_len', d.get('length', d.get('cone_distance',0.03))))) * s
            bg(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('coupling_elastic','coupling_jaw','coupling_bellows',
                     'coupling_oldham','coupling_magnetic','coupling_rigid',
                     'one_way_clutch','torque_limiter','belt_tensioner','chain_tensioner'):
            od = d.get('outer_diameter',d.get('pulley_diameter',0.020)) * s
            tl = d.get('total_length',d.get('length',d.get('width',0.030))) * s
            bg(type="cylinder", size=f"{od/2:.5f} {tl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('timing_belt_loop','v_belt','roller_chain','roller_chain_precision'):
            bl = d.get('length',d.get('inner_length',0.200)) * s
            bw = d.get('width',d.get('top_width',0.015)) * s
            bg(type="cylinder", size=f"{bl/(2*np.pi):.5f} {bw/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('cylindrical_cell','cylindrical_cell_large',
                     'supercapacitor_can','supercapacitor_large',
                     'lidar_cylindrical','lidar_disk',
                     'pressure_sensor_can','encoder_incremental','encoder_absolute','encoder_hollow'):
            r = d.get('diameter', d.get('od',0.020)) / 2 * s
            h = d.get('height', d.get('body_thickness', d.get('thickness',0.030))) * s
            bg(type="cylinder", size=f"{r:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('qfn_package','lga28_package','lga14_package','lga8_package',
                     'so8_package','so16_package','ssop24_package',
                     'dfn_package','lga12_package','sot23_8','to220_package'):
            l = d.get('length', 0.005) * 8 * s
            w = d.get('width', 0.004) * 8 * s
            bh = max(d.get('height', 0.001), 0.001) * 12 * s
            bg(type="box", size=f"{l:.5f} {w:.5f} {bh:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt == 'small_pcb_module':
            l = d.get('length', 0.01) * 4 * s
            w = d.get('width', 0.008) * 4 * s
            bg(type="box", size=f"{l:.5f} {w:.5f} 0.002", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('lidar_module','lidar_module_long','ultrasonic_sensor'):
            l = d.get('length', 0.029) * s
            w = d.get('width', 0.021) * s
            h = d.get('height', 0.016) * s
            bg(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('load_cell_aluminum','load_cell_mini'):
            l = d.get('length', 0.125) * s
            w = d.get('width', 0.013) * s
            h = d.get('height', 0.025) * s
            bg(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('pcb_board','pcb_board_large'):
            l = d.get('length', 0.085) * s
            w = d.get('width', 0.056) * s
            bg(type="box", size=f"{l/2:.5f} {w/2:.5f} 0.003", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt == 'button_mushroom':
            md = d.get('mushroom_dia', 0.036) * s
            bg(type="cylinder", size=f"{md/2:.5f} {md*0.12:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt == 'micro_switch':
            l = d.get('length', 0.0125) * s
            w = d.get('width', 0.0063) * s
            h = d.get('height', 0.0125) * s
            bg(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt == 'slider_pot':
            l = d.get('length', 0.060) * s
            w = d.get('width', 0.010) * s
            h = d.get('height', 0.017) * s
            bg(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('ring_with_holes','ring_split'):
            od = d.get('outer_diameter', 0.060) * s
            th = d.get('thickness', 0.005) * s
            bg(type="cylinder", size=f"{od/2:.5f} {th/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt in ('bearing_assembly','bearing_thrust','bearing_angular'):
            od = d.get('outer_diameter', 0.026) * s
            width = d.get('width', 0.008) * s
            bg(type="cylinder", size=f"{od/2:.5f} {width/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt == 'extrusion_profile':
            sz = d.get('size', 0.020) * s
            l = d.get('length', 0.100) * s
            bg(type="box", size=f"{sz/2:.5f} {sz/2:.5f} {l/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt == 'tube_hollow':
            od = d.get('outer_diameter', 0.016) * s
            hl = d.get('length', 0.200) * s
            bg(type="cylinder", size=f"{od/2:.5f} {hl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt == 'flat_plate':
            l = d.get('length', 0.100) * s
            w = d.get('width', 0.100) * s
            th = d.get('thickness', 0.003) * s
            bg(type="box", size=f"{l/2:.5f} {w/2:.5f} {th/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt == 'helix_spring':
            od = d.get('outer_diameter', 0.010) * s
            fl = d.get('free_length', 0.030) * s
            bg(type="cylinder", size=f"{od/2:.5f} {fl/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt == 'sphere_cap':
            dia = d.get('diameter', 0.030) * s
            bg(type="sphere", size=f"{dia/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt == 'aluminum_block':
            l = d.get('length', 0.008) * s
            w = d.get('width', 0.006) * s
            h = d.get('height', 0.004) * s
            bg(type="box", size=f"{l/2:.5f} {w/2:.5f} {h/2:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        elif gt == 'l_bracket':
            al = d.get('arm_length', 0.030) * s
            aw = d.get('arm_width', 0.020) * s
            th = d.get('thickness', 0.004) * s
            bg(type="box", size=f"{al/2:.5f} {aw/2:.5f} {th:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")
        else:
            l, w, h = 0.03*s, 0.025*s, 0.02*s
            bg(type="box", size=f"{l:.5f} {w:.5f} {h:.5f}", rgba=f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}")

        bgs = '\n            '.join(f'<geom {g_str}/>' for g_str in body_geoms)
        name_short = spec.display_name[:18]
        content.append(f'''
    <body name="{pid}" pos="{x:.4f} {y:.4f} {z_offset:.4f}">
        {bgs}
    </body>''')

    full_xml = f"""<mujoco model="parts_grid">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.08 0.08 0.08" rgb2="0.18 0.18 0.18" width="512" height="512"/>
    <material name="ground" texture="grid" texrepeat="20 20"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.95 0.95 0.90"/>
    <light pos="2 2 2" dir="-1 -1 -1" diffuse="0.4 0.4 0.42"/>
    <geom name="floor" type="plane" size="5 5 0.05" material="ground"/>
    {''.join(content)}
  </worldbody>
</mujoco>"""
    return full_xml, parts_list


def main():
    print("=" * 64)
    print("  V8 零件库 v2 — 交互式3D浏览器 (168种零件)")
    print("=" * 64)
    print()
    print("  🖱️  操作说明:")
    print("      左键拖拽 → 旋转")
    print("      右键拖拽 → 平移")
    print("      滚轮     → 缩放")
    print("      Tab       → 切换自由相机模式")
    print()
    print("  ⌨️  快捷键:")
    print("      Space → 暂停/播放物理仿真")
    print("      1-6   → 按编号跳转零件视图")
    print("      Esc   → 退出")
    print()

    parts = get_all_part_specs()
    print(f"  加载 {len(parts)} 个零件...")
    print(f"  生成场景XML...")

    xml, parts_list = build_multi_xml(parts)
    tmp_path = Path("_part_viewer_temp.xml")
    tmp_path.write_text(xml, encoding='utf-8')

    print(f"  启动MuJoCo Viewer...\n")

    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_path))
        data = mujoco.MjData(model)

        # 使用 mujoco_viewer 启动交互窗口
        viewer = mujoco_viewer.MujocoViewer(model, data)

        # 设置相机初始位置（俯瞰所有零件）
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -35
        viewer.cam.lookat[:] = [0, 1.0, 0.1]

        print("  ✅ 窗口已打开！可以用鼠标拖拽旋转、滚轮缩放、右键平移。")
        print("     按 Escape 键退出。\n")

        # 运行渲染循环
        while viewer.is_alive:
            mujoco.mj_step(model, data)
            viewer.render()

        viewer.close()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    main()