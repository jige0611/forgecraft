# ═══════════════════════════════════════════════════════════════
#  零件库3D预览生成器 v2 — 全兼容版本 (确保全部47个零件成功)
#  ═══════════════════════════════════════════════════════════════

import mujoco
import numpy as np
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent))
from v8_3d_model_library import get_all_part_specs, PartSpec

# ============ 配置 ============
IMG_W = 640
IMG_H = 480
OUTPUT_DIR = "part_library_previews"

VIEWS = [
    {"name": "iso",      "lookat": [0, 0, 0.08], "distance": 0.5,  "azimuth": 45,  "elevation": -25},
    {"name": "front",    "lookat": [0, 0, 0.08], "distance": 0.45, "azimuth": 90,  "elevation": -10},
    {"name": "top",      "lookat": [0, 0, 0.08], "distance": 0.6,  "azimuth": 0,   "elevation": 89},
    {"name": "close_up", "lookat": [0, 0, 0.06], "distance": 0.22, "azimuth": 30,  "elevation": -20},
]


def make_xml(spec: PartSpec) -> str:
    """生成最大兼容性的MJCF XML — 只用最基础的元素"""
    d = spec.dimensions
    c = spec.color
    gt = spec.geometry_type
    s = 2.5  # 统一放大因子

    # ===== 根据类型选择主形状 =====
    if gt in ('cylinder_simple', 'cylinder_complex', 'cylinder_detail',
              'cylinder_with_hub', 'cylinder_step'):
        r = d.get('outer_diameter', d.get('body_diameter', d.get('body_diameter', 0.03))) / 2 * s
        h = d.get('total_height', d.get('total_length', 0.05)) * s
        main = f'<geom type="cylinder" size="{r:.4f} {h/2:.4f}" rgba="{c[0]:.2f} {c[1]:.2f} {c[2]:.2f} {c[3]:.2f}"/>'
        # 轴
        sr = d.get('shaft_diameter', r*0.25) / 2 * s
        sl = d.get('shaft_length', h*0.3) * s
        extras = f'<geom pos="0 0 {h/2:.4f}" type="cylinder" size="{sr:.4f} {sl/2:.4f}" rgba="0.65 0.65 0.68 1"/>'
        # 法兰（如果有）
        if 'flange' in d:
            fr = d['flange_diameter'] / 2 * s
            ft = d.get('flange_thickness', 0.003) * s
            extras += f'\n<geom pos="0 0 {-h/2+ft:.4f}" type="cylinder" size="{fr:.4f} {ft:.4f}" rgba="0.42 0.40 0.45 1"/>'

    elif gt == 'square_cylinder':
        sz = d.get('square_size', 0.03) / 2 * s
        h = d.get('body_length', 0.03) * s
        main = f'<geom type="box" size="{sz:.4f} {sz:.4f} {h/2:.4f}" rgba="{c[0]:.2f} {c[1]:.2f} {c[2]:.2f} {c[3]:.2f}"/>'
        sr = d.get('shaft_diameter', 0.005) / 2 * s
        sl = d.get('shaft_length', 0.02) * s
        extras = f'<geom pos="0 0 {h/2+sl/2:.4f}" type="cylinder" size="{sr:.4f} {sl/2:.4f}" rgba="0.70 0.70 0.73 1"/>'

    elif gt in ('box_simple', 'box_small'):
        l = d.get('length', 0.02) / 2 * s
        w = d.get('width', 0.015) / 2 * s
        h = d.get('height', 0.01) / 2 * s
        main = f'<geom type="box" size="{l:.4f} {w:.4f} {h:.4f}" rgba="{c[0]:.2f} {c[1]:.2f} {c[2]:.2f} {c[3]:.2f}"/>'
        extras = ''

    elif gt in ('box_complex', 'box_heatsink', 'box_detail'):
        l = d.get('length', 0.04) / 2 * s
        w = d.get('width', 0.025) / 2 * s
        h = d.get('height', 0.02) / 2 * s
        main = f'<geom type="box" size="{l:.4f} {w:.4f} {h:.4f}" rgba="{c[0]:.2f} {c[1]:.2f} {c[2]:.2f} {c[3]:.2f}"/>'
        hub_r = d.get('hub_diameter', 0.02) / 2 * s
        extras = f'<geom pos="0 0 {h+0.005*s:.4f}" type="cylinder" size="{hub_r:.4f} 0.004" rgba="0.65 0.60 0.55 1"/>'
        # 散热片
        if 'heatsink_fins' in d:
            fins = min(d['heatsink_fins'], 6)
            fw = (w*2)/(fins*2)
            for i in range(fins):
                px = -w + fw*(i*2+1)
                extras += f'\n<geom pos="{px:.4f} 0 {h+0.005*s:.4f}" type="box" size="{fw*0.35:.4f} {w*0.85:.4f} 0.004" rgba="0.80 0.72 0.66 1"/>'

    elif gt in ('bearing_assembly',):
        ro = d.get('outer_diameter', 0.02) / 2 * s
        ri = d.get('inner_diameter', 0.008) / 2 * s
        hw = d.get('width', 0.006) / 2 * s
        main = f'<geom type="cylinder" size="{ro:.4f} {hw:.4f}" rgba="0.50 0.52 0.56 1"/>'
        extras = f'''<geom type="cylinder" size="{ri:.4f} {hw*0.9:.4f}" rgba="0.56 0.58 0.62 1"/>
<geom type="cylinder" size="{(ri+ro)/2:.4f} {hw*0.4:.4f}" rgba="0.63 0.65 0.68 1"/>'''

    elif gt in ('ring_with_holes', 'ring_split'):
        ro = d.get('outer_diameter', 0.04) / 2 * s
        ri = d.get('inner_diameter', 0.015) / 2 * s
        th = d.get('thickness', 0.003) / 2 * s
        main = f'<geom type="cylinder" size="{ro:.4f} {th:.4f}" rgba="{c[0]:.2f} {c[1]:.2f} {c[2]:.2f} {c[3]:.2f}"/>'
        extras = f'<geom type="cylinder" size="{ri:.4f} {th*2:.4f}" rgba="0.06 0.06 0.06 1"/>'
        # 螺栓孔
        bc = d.get('bolt_circle_diameter', ro*0.85)
        nb = d.get('bolt_count', 4)
        br = d.get('bolt_diameter', 0.003) / 2 * s
        for i in range(nb):
            ang = 2*np.pi*i/nb
            bx = bc/2*np.cos(ang)*s
            by = bc/2*np.sin(ang)*s
            extras += f'\n<geom pos="{bx:.4f} {by:.4f} 0" type="cylinder" size="{br:.4f} {th*1.5:.4f}" rgba="0.26 0.26 0.28 1"/>'

    elif gt in ('pcb_board',):
        l = d.get('length', 0.06) / 2 * s
        w = d.get('width', 0.04) / 2 * s
        th = 0.003
        main = f'<geom type="box" size="{l:.4f} {w:.4f} {th:.4f}" rgba="0.0 0.30 0.10 1"/>'
        usb_h = d.get('usb_height', 0.012) * s
        gpio_h = d.get('gpio_header_height', 0.01) * s
        eth_h = d.get('ethernet_height', 0.01) * s
        extras = f'''<geom pos="-{l*0.8:.4f} {w*0.2:.4f} {th+usb_h/2:.4f}" type="box" size="{l*0.13:.4f} {w*0.10:.4f} {usb_h/2:.4f}" rgba="0.66 0.66 0.68 1"/>
<geom pos="{l*0.3:.4f} {w*0.5:.4f} {th+gpio_h/2:.4f}" type="box" size="{l*0.06:.4f} {w*0.22:.4f} {gpio_h/2:.4f}" rgba="0.10 0.10 0.10 1"/>
<geom pos="{l*0.8:.4f} {-w*0.1:.4f} {th+eth_h/2:.4f}" type="box" size="{l*0.13:.4f} {w*0.20:.4f} {eth_h/2:.4f}" rgba="0.12 0.17 0.22 1"/>
<geom pos="-{l*0.2:.4f} {-w*0.2:.4f} {th+0.002:.4f}" type="box" size="{l*0.14:.4f} {w*0.14:.4f} 0.002" rgba="0.04 0.04 0.05 1"/>'''

    elif gt in ('qfn_package', 'so8_package', 'ssop24_package', 'lga28_package'):
        l = d.get('length', 0.005) * 10 * s
        w = d.get('width', 0.004) * 10 * s
        h = max(d.get('height', 0.001), 0.001) * 18 * s
        main = f'<geom type="box" size="{l:.4f} {w:.4f} {h:.4f}" rgba="0.03 0.03 0.03 1"/>'
        # 引脚
        if 'so8' in gt: npins = 8
        elif 'ssop' in gt: npins = 24
        else: npins = 16
        pps = npins // 2
        ps = (l*2)/(pps+1)
        extras = ''
        for side in [-1, 1]:
            for p in range(pps):
                px = -l + ps*(p+1)
                extras += f'\n<geom pos="{px:.4f} {side*(w+0.003):.4f} 0" type="box" size="{l*0.04:.4f} {w*0.10:.4f} {h*0.6:.4f}" rgba="0.70 0.66 0.60 1"/>'
        extras += f'\n<geom pos="-{l*0.6:.4f} {w*0.6:.4f} {h:.4f}" type="sphere" size="{min(l,w)*0.07:.4f}" rgba="0.90 0.10 0.10 1"/>'

    elif gt == 'aluminum_block':
        l = d.get('length', 0.03) / 2 * s
        w = d.get('width', 0.015) / 2 * s
        h = d.get('height', 0.015) / 2 * s
        hr = d.get('hole_dia', 0.003) / 2 * s
        main = f'<geom type="box" size="{l:.4f} {w:.4f} {h:.4f}" rgba="{c[0]:.2f} {c[1]:.2f} {c[2]:.2f} {c[3]:.2f}"/>'
        extras = f'''<geom pos="-{l*0.7:.4f} 0 0" type="cylinder" size="{hr:.4f} {h*1.2:.4f}" rgba="0.06 0.06 0.06 1"/>
<geom pos="{l*0.7:.4f} 0 0" type="cylinder" size="{hr:.4f} {h*1.2:.4f}" rgba="0.06 0.06 0.06 1"/>'''

    elif gt == 'small_pcb_module':
        l = d.get('length', 0.01) * 5 * s
        w = d.get('width', 0.008) * 5 * s
        lr = d.get('lens_dia', 0.003) * 3 * s
        main = f'<geom type="box" size="{l:.4f} {w:.4f} 0.001" rgba="0.04 0.04 0.04 1"/>'
        extras = f'<geom pos="0 0 0.005" type="cylinder" size="{lr:.4f} 0.002" rgba="0.80 0.84 0.89 0.80"/>'

    elif gt == 'complex':
        pd = d.get('pitch_diameter', 0.02) / 2 * s
        bore = d.get('bore', 0.005) / 2 * s
        width = d.get('width', 0.01) / 2 * s
        main = f'<geom type="cylinder" size="{pd:.4f} {width:.4f}" rgba="{c[0]:.2f} {c[1]:.2f} {c[2]:.2f} {c[3]:.2f}"/>'
        extras = f'<geom type="cylinder" size="{bore:.4f} {width*1.5:.4f}" rgba="0.06 0.06 0.06 1"/>'

    elif gt in ('tube_hollow', 'extrusion_profile', 'flat_plate', 'helix_spring',
                 'pin_with_ball', 'sphere_cap'):
        # 结构类零件
        if gt == 'tube_hollow':
            ro = d.get('outer_diameter', 0.02) / 2 * s
            ri = ro * 0.75
            hl = d.get('length', d.get('total_length', 0.1)) * s
            main = f'<geom type="cylinder" size="{ro:.4f} {hl/2:.4f}" rgba="{c[0]:.2f} {c[1]:.2f} {c[2]:.2f} {c[3]:.2f}"/>'
            extras = f'<geom type="cylinder" size="{ri:.4f} {hl/2:.4f}" rgba="0.06 0.06 0.06 1"/>'
        elif gt == 'extrusion_profile':
            sz = d.get('size', 0.02) / 2 * s
            hl = d.get('length', 0.1) * s
            main = f'<geom type="box" size="{sz:.4f} {sz:.4f} {hl/2:.4f}" rgba="0.72 0.72 0.74 1"/>'
            # 内部空心
            isz = sz * 0.7
            extras = f'<geom type="box" size="{isz:.4f} {isz:.4f} {hl/2:.4f}" rgba="0.15 0.15 0.17 1"/>'
        elif gt == 'flat_plate':
            l = d.get('length', 0.1) / 2 * s
            w = d.get('width', 0.1) / 2 * s
            th = d.get('thickness', 0.003) / 2 * s
            main = f'<geom type="box" size="{l:.4f} {w:.4f} {th:.4f}" rgba="0.60 0.62 0.64 1"/>'
            extras = ''
        elif gt == 'helix_spring':
            sr = d.get('outer_diameter', 0.01) / 2 * s
            sl = d.get('free_length', 0.03) * s
            main = f'<geom type="cylinder" size="{sr:.4f} {sl/2:.4f}" rgba="0.70 0.72 0.74 1"/>'
            ir = sr * 0.6
            extras = f'<geom type="cylinder" size="{ir:.4f} {sl/2:.4f}" rgba="0.12 0.12 0.14 1"/>'
        elif gt == 'pin_with_ball':
            pr = d.get('diameter', 0.006) / 2 * s
            pl = d.get('length', 0.03) * s
            br = d.get('ball_diameter', pr*1.5)
            main = f'<geom type="cylinder" size="{pr:.4f} {pl/2:.4f}" rgba="0.75 0.72 0.68 1"/>'
            extras = f'<geom pos="0 0 {pl/2:.4f}" type="sphere" size="{br:.4f}" rgba="0.82 0.82 0.84 1"/>'
        else:  # sphere_cap
            sr = d.get('diameter', 0.03) / 2 * s
            main = f'<geom type="sphere" size="{sr:.4f}" rgba="0.25 0.22 0.18 1"/>'
            extras = ''
    else:
        # 默认：通用盒子
        l, w, h = 0.03*s, 0.025*s, 0.02*s
        main = f'<geom type="box" size="{l:.4f} {w:.4f} {h:.4f}" rgba="{c[0]:.2f} {c[1]:.2f} {c[2]:.2f} {c[3]:.2f}"/>'
        extras = ''

    xml = f"""<mujoco model="{spec.part_id}">
  <option timestep="0.001" gravity="0 0 0"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.18 0.18" rgb2="0.28 0.28 0.28" width="256" height="256"/>
    <material name="ground" texture="grid" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" diffuse="0.95 0.95 0.92"/>
    <light pos="-1 -1 2" dir="1 1 -1" diffuse="0.3 0.3 0.32"/>
    <geom name="floor" type="plane" size="1.2 1.2 0.05" material="ground"/>
    <body name="part" pos="0 0 0.1">
      {main}
{extras}
    </body>
  </worldbody>
</mujoco>"""
    return xml


def render_one(spec: PartSpec, out_dir: Path, idx: int, total: int) -> list:
    """渲染一个零件的4个视角"""
    results = []
    print(f"  [{idx+1:2d}/{total}] {spec.display_name}", end='')

    try:
        xml_str = make_xml(spec)
        tmp = out_dir / "_tmp.xml"
        tmp.write_text(xml_str, encoding='utf-8')

        model = mujoco.MjModel.from_xml_path(str(tmp))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

        mujoco.mj_resetData(model, data)
        for _ in range(30): mujoco.mj_step(model, data)

        for view in VIEWS:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, cam)
            cam.lookat[:] = np.array(view['lookat'])
            cam.distance = view['distance']
            cam.azimuth = view['azimuth']
            cam.elevation = view['elevation']

            mujoco.mj_resetData(model, data)
            for _ in range(20): mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()

            img = Image.fromarray(rgb)
            fname = f"{idx+1:02d}_{spec.part_id}_{view['name']}.png"
            fpath = out_dir / fname
            img.save(fpath)
            results.append({'file': fpath, 'view': view['name']})

        renderer.close()
        tmp.unlink(missing_ok=True)
        print(f" ✓ ({len(results)}图)")

    except Exception as e:
        print(f" ✗ {str(e)[:60]}")

    return results


def build_html(out_dir: Path, all_res: dict):
    """构建HTML汇总页"""
    cats = [
        ('actuators', '执行器'),
        ('transmission', '传动系统'),
        ('energy', '能源系统'),
        ('sensors', '传感系统'),
        ('controllers', '控制系统'),
        ('connectors', '连接件'),
    ]

    html = '''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>V8 零件库 3D预览</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,"Segoe UI",Arial,sans-serif;background:#0d1117;color:#e6edf3;padding:24px}
h1{text-align:center;font-size:1.6em;color:#fff;margin-bottom:8px}
.sub{text-align:center;color:#8b949e;margin-bottom:24px;font-size:0.92em}
.cat{margin-bottom:36px}
.cat-hd{background:linear-gradient(90deg,#161b22,#21262d);padding:10px 18px;border-radius:8px;font-size:1.15em;color:#58a6ff;margin-bottom:14px;border-left:4px solid #58a6ff}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}
.card{background:#161b22;border-radius:10px;overflow:hidden;border:1px solid #30363d;transition:transform .15s,border-color .15s}
.card:hover{transform:translateY(-2px);border-color:#58a6ff}
.card img{width:100%;height:190px;object-fit:contain;display:block;background:#0d1117;padding:8px}
.info{padding:10px 12px}
.name{font-weight:600;color:#fff;font-size:0.9em;margin-bottom:3px}
.detail{color:#8b949e;font-size:0.78em;line-height:1.5}
.ok{color:#3fb950}.warn{color:#d29922}
.stat{text-align:center;color:#8b949e;margin:16px 0}
</style></head><body>
<h1>🔧 V8 零件库 — 3D预览图集</h1>
<p class="sub">共 <strong>''' + str(len(all_res)) + '''</strong> 种工业级零件 × 4视角 = <strong>''' + str(sum(len(r['results']) for r in all_res.values())) + '''</strong> 张图片</p>
'''

    ok_count = 0
    for cat_id, cat_name in cats:
        items = [(pid, info) for pid, info in all_res.items() if info['spec'].category == cat_id]
        if not items: continue

        html += f'<div class="cat"><div class="cat-hd">{cat_name} ({len(items)}种)</div><div class="grid">\n'

        for pid, info in items:
            spec = info['spec']
            iso_file = next((r['file'].name for r in info['results'] if r['view'] == 'iso'), '')
            has_img = len(info['results']) > 0
            if has_img: ok_count += 1

            dims_str = ', '.join(f'{k}:{v*100:.1f}cm' for k, v in list(spec.dimensions.items())[:3])
            specs_list = []
            if spec.voltage_v > 0: specs_list.append(f"{spec.voltage_v}V")
            if spec.torque_nm > 0: specs_list.append(f"{spec.torque_nm}N·m")
            if spec.mass_kg > 0: specs_list.append(f"{spec.mass_kg:.3f}kg")
            if spec.speed_rpm > 0: specs_list.append(f"{spec.speed_rpm:.0f}rpm")

            status_cls = 'ok' if has_img else 'warn'
            status_txt = '✓ 已渲染' if has_img else '✗ 缺失'

            html += f'''<div class="card">
<img src="{iso_file}" alt="{spec.display_name}" onerror="this.style.display='none'">
<div class="info">
<div class="name">{spec.display_name} <span class="{status_cls}">{status_txt}</span></div>
<div class="detail">
{spec.manufacturer} · {spec.part_number}<br>
{' | '.join(specs_list) if specs_list else ''}<br>
{dims_str}
</div>
</div>
</div>\n'''

        html += '</div></div>\n'

    html += f'<p class="stat">成功渲染 {ok_count}/{len(all_res)} 个零件</p></body></html>'

    p = out_dir / 'index.html'
    p.write_text(html, encoding='utf-8')
    return p


def main():
    print("=" * 62)
    print("  🔧 V8 零件库 — 3D预览生成器 v2 (全兼容版)")
    print("=" * 62)

    out = Path(OUTPUT_DIR)
    out.mkdir(exist_ok=True)

    print("\n📦 加载零件库...")
    parts = get_all_part_specs()
    ids = list(parts.keys())
    n = len(ids)
    print(f"   共 {n} 个零件\n")

    all_results = {}
    for i, pid in enumerate(ids):
        spec = parts[pid]
        res = render_one(spec, out, i, n)
        all_results[pid] = {'spec': spec, 'results': res}

    print("\n📄 生成HTML...")
    html_path = build_html(out, all_results)
    print(f"   → {html_path.name}")

    total_imgs = sum(len(r['results']) for r in all_results.values())
    total_sz = sum(r['file'].stat().st_size for info in all_results.values() for r in info['results'] if r['file'].exists())

    print(f"\n{'='*62}")
    print(f"  完成! {total_imgs}张图片, {total_sz/1024/1024:.1f}MB")
    print(f"  📁 {out.absolute()}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
