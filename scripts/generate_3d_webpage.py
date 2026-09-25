# ═══════════════════════════════════════════════════════════
#  生成交互式3D零件库网页 (Three.js + 拖拽旋转缩放)
# ═══════════════════════════════════════════════════════════

import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from v8_3d_model_library_v2 import get_all_part_specs

parts = get_all_part_specs()

# 将零件数据转为 JS 友好的格式
parts_js = []
for pid, spec in sorted(parts.items()):
    d = spec.dimensions
    item = {
        "id": pid,
        "name": spec.display_name,
        "category": spec.category,
        "manufacturer": spec.manufacturer,
        "part_number": spec.part_number,
        "color": spec.color[:3],
        "geo_type": spec.geometry_type,
        "dims": {k: v for k, v in d.items()},
        "voltage": spec.voltage_v,
        "torque": spec.torque_nm,
        "speed": spec.speed_rpm,
        "mass": spec.mass_kg,
        "current": spec.current_a,
    }
    parts_js.append(item)

parts_json = json.dumps(parts_js, ensure_ascii=False)

cat_names = {
    "actuators": "执行器", "transmission": "传动系统",
    "energy": "能源系统", "sensors": "传感系统",
    "controllers": "控制系统", "connectors": "连接件",
}

# ═══ 生成 HTML ═══
html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V8 零件库 — 交互式3D浏览器 (168种)</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Segoe UI",Arial,sans-serif;background:#0d1117;color:#e6edf3;overflow-x:hidden}}
h1{{text-align:center;padding:20px 0 4px;font-size:1.3em;color:#fff}}
.sub{{text-align:center;color:#8b949e;font-size:.82em;margin-bottom:16px}}
.toolbar{{display:flex;justify-content:center;gap:6px;flex-wrap:wrap;padding:0 16px 16px;position:sticky;top:0;z-index:10;background:#0d1117;padding-top:8px}}
.toolbar button{{padding:6px 14px;border-radius:6px;border:1px solid #30363d;background:#161b22;color:#c9d1d9;cursor:pointer;font-size:.8em;transition:all .15s}}
.toolbar button:hover,.toolbar button.active{{background:#1f6feb;border-color:#1f6feb;color:#fff}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px;padding:0 16px 24px;max-width:1400px;margin:0 auto}}
.card{{background:#161b22;border-radius:8px;overflow:hidden;border:1px solid #30363d;cursor:pointer;transition:transform .12s,border-color .12s}}
.card:hover{{transform:translateY(-3px);border-color:#58a6ff}}
.card canvas{{width:100%;height:150px;display:block;background:#0d1117}}
.info{{padding:8px 10px}}
.nm{{font-weight:600;color:#fff;font-size:.78em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.dt{{color:#8b949e;font-size:.68em;margin-top:2px}}

/* 全屏模态 */
.modal{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.92);z-index:100;flex-direction:column}}
.modal.show{{display:flex}}
.modal-head{{display:flex;justify-content:space-between;align-items:center;padding:12px 20px;background:#161b22;border-bottom:1px solid #30363d;flex-shrink:0}}
.modal-head h2{{font-size:1.1em;color:#fff}}
.modal-head button{{padding:8px 20px;border-radius:6px;border:1px solid #30363d;background:#21262d;color:#c9d1d9;cursor:pointer;font-size:.85em}}
.modal-head button:hover{{background:#da3633;border-color:#da3633;color:#fff}}
.modal-body{{flex:1;position:relative;overflow:hidden}}
.modal-body canvas{{display:block}}
.modal-info{{position:absolute;bottom:16px;left:16px;background:rgba(22,27,34,.9);padding:10px 14px;border-radius:8px;border:1px solid #30363d;font-size:.78em;color:#8b949e;pointer-events:none}}
.modal-info strong{{color:#fff;font-size:.9em}}
.hint{{text-align:center;color:#484f58;padding:8px 0 0;font-size:.75em}}
</style>
</head>
<body>

<h1>🔧 V8 零件库 — 交互式3D浏览器</h1>
<p class="sub">共 <b>168</b> 种工业级零件 · <b>点击任意卡片</b>进入全屏3D查看 · 拖拽旋转 | 滚轮缩放 | 右键平移</p>

<div class="toolbar">
  <button class="active" data-cat="all">全部</button>
  <button data-cat="actuators">执行器</button>
  <button data-cat="transmission">传动系统</button>
  <button data-cat="energy">能源系统</button>
  <button data-cat="sensors">传感系统</button>
  <button data-cat="controllers">控制系统</button>
  <button data-cat="connectors">连接件</button>
</div>

<div class="grid" id="grid"></div>

<!-- 全屏模态 -->
<div class="modal" id="modal">
  <div class="modal-head">
    <h2 id="modalTitle"></h2>
    <button onclick="closeModal()">✕ 关闭</button>
  </div>
  <div class="modal-body" id="modalBody">
    <div class="modal-info" id="modalInfo"></div>
  </div>
</div>

<script type="importmap">
{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const PARTS = {parts_json};
const SCALE = 2.2;

// 颜色映射
const CAT_COLORS = {{
  actuators: '#d9741a', transmission: '#8b949e', energy: '#3fb950',
  sensors: '#a371f7', controllers: '#58a6ff', connectors: '#f778ba'
}};
const CAT_NAMES = {json.dumps(cat_names, ensure_ascii=False)};

// ═══ 根据零件规格创建 Three.js 几何体 ═══
function createPartMesh(spec) {{
  const group = new THREE.Group();
  const d = spec.dims;
  const c = spec.color;
  const gt = spec.geo_type;
  const s = SCALE;
  const color = new THREE.Color(c[0], c[1], c[2]);

  const metalMat = (clr) => new THREE.MeshStandardMaterial({{ color:clr||color, roughness:0.35, metalness:0.7 }});
  const mat = metalMat();

  // 各种几何类型
  if (gt.includes('cylinder') || gt.startsWith('cylinder') || ['pancake_motor','disc_motor','disc_motor_thin','joint_motor','multirotor_motor'].includes(gt)) {{
    let od = d.outer_diameter || d.body_diameter || d.casing_od || d.flange_input_dia || 0.03;
    let r = od/2 * s;
    let h = (d.total_height || d.body_length || d.casing_len || d.total_length || 0.05) * s;
    let geo = new THREE.CylinderGeometry(r, r, h, 32);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h/2;
    group.add(mesh);
    // 轴
    if (d.shaft_diameter) {{
      let sr = d.shaft_diameter/2 * s;
      let sl = (d.shaft_length || h*0.3) * s;
      let sgeo = new THREE.CylinderGeometry(sr, sr, sl, 16);
      let smat = new THREE.MeshStandardMaterial({{color:0xa6a6ad, roughness:0.3, metalness:0.8}});
      let smesh = new THREE.Mesh(sgeo, smat);
      smesh.position.y = h + sl/2;
      group.add(smesh);
    }}
  }}

  else if (['square_cylinder_stepper','square_cylinder_large','square_cylinder_xlarge'].includes(gt)) {{
    let sz = (d.square_size || 0.042)/2 * s;
    let bl = (d.body_length || 0.048) * s;
    let geo = new THREE.BoxGeometry(sz*2, bl, sz*2);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = bl/2;
    group.add(mesh);
    if (d.shaft_diameter) {{
      let sr = d.shaft_diameter/2 * s;
      let sl = (d.shaft_length || 0.022) * s;
      let sgeo = new THREE.CylinderGeometry(sr, sr, sl, 16);
      let smesh = new THREE.Mesh(sgeo, new THREE.MeshStandardMaterial({{color:0xa6a6ad, roughness:0.3, metalness:0.8}}));
      smesh.position.y = bl + sl/2;
      group.add(smesh);
    }}
  }}

  else if (gt.startsWith('box') || ['esc_heatsink','esc_compact','esc_micro','bms_board','bms_board_large','power_module','dc_dc_converter','dc_dc_enclosed','pdb','battery_pack','battery_pack_large','battery_pack_small','battery_prismatic','slider_pot'].includes(gt)) {{
    let l = (d.length || 0.04)/2 * s;
    let w = (d.width || 0.03)/2 * s;
    let h = (d.height || 0.02)/2 * s;
    let geo = new THREE.BoxGeometry(l*2, h*2, w*2);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h;
    group.add(mesh);
  }}

  else if (['dynamixel_shell','dynamixel_mini','servo_standard','servo_metal'].includes(gt)) {{
    let l = (d.length || 0.043)/2 * s;
    let w = (d.width || 0.043)/2 * s;
    let h = (d.height || 0.050)/2 * s;
    let geo = new THREE.BoxGeometry(l*2, h*2, w*2);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h;
    group.add(mesh);
    if (d.hub_diameter) {{
      let hr = d.hub_diameter/2 * s;
      let hh = (d.hub_height || 0.004) * s;
      let hgeo = new THREE.CylinderGeometry(hr, hr, hh, 24);
      let hmesh = new THREE.Mesh(hgeo, new THREE.MeshStandardMaterial({{color:0x8c8a94, roughness:0.3, metalness:0.7}}));
      hmesh.position.y = h + hh/2;
      group.add(hmesh);
    }}
  }}

  else if (['gearbox_dc','gearbox_dc_small','micro_dc'].includes(gt)) {{
    let gbr = (d.gearbox_od || d.body_diameter || 0.025)/2 * s;
    let gbl = (d.gearbox_len || d.body_length || 0.025) * s;
    let mr = (d.motor_od || d.body_diameter || 0.025)/2 * s;
    let ml = (d.motor_len || d.body_length || 0.035) * s;
    let ggeo = new THREE.CylinderGeometry(gbr, gbr, gbl, 24);
    group.add(new THREE.Mesh(ggeo, new THREE.MeshStandardMaterial({{color:0x737578, roughness:0.3, metalness:0.7}})));
    let mgeo = new THREE.CylinderGeometry(mr, mr, ml, 24);
    let mmesh = new THREE.Mesh(mgeo, mat);
    mmesh.position.y = gbl + ml/2;
    group.add(mmesh);
  }}

  else if (['linear_actuator','linear_motor'].includes(gt)) {{
    let bl = (d.body_length || d.width || 0.180) * s;
    let bw = (d.body_width || d.height || 0.040)/2 * s;
    let geo = new THREE.BoxGeometry(bl, bw*2, bw*2);
    group.add(new THREE.Mesh(geo, mat));
    if (d.rod_diameter) {{
      let rr = d.rod_diameter/2 * s;
      let stroke = (d.stroke || 0.100) * s;
      let rgeo = new THREE.CylinderGeometry(rr, rr, stroke, 16);
      let rmesh = new THREE.Mesh(rgeo, new THREE.MeshStandardMaterial({{color:0xbfbfc2, roughness:0.25, metalness:0.85}}));
      rmesh.position.x = bl/2 + stroke/2;
      rmesh.position.y = bw;
      rmesh.rotation.z = Math.PI/2;
      group.add(rmesh);
    }}
  }}

  else if (['harmonic_assembly','harmonic_compact','harmonic_mini','planetary_gearbox','planetary_gearbox_large','planetary_neugart','planetary_apex','cycloidal_gearbox','cycloidal_gearbox_large','voice_coil'].includes(gt)) {{
    let od = d.casing_od || d.input_diameter || d.flange_input_dia || d.body_diameter || 0.04;
    let r = od/2 * s;
    let h = (d.casing_len || d.length || d.stroke || 0.05) * s;
    let geo = new THREE.CylinderGeometry(r, r, h, 32);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h/2;
    group.add(mesh);
  }}

  else if (['timing_pulley','timing_pulley_large','timing_pulley_htd','sprocket','sprocket_precision','spur_gear','spur_gear_large','spur_gear_heavy','helical_gear','bevel_gear','worm_wheel','worm_shaft'].includes(gt)) {{
    let pd = d.pitch_diameter || d.major_diameter || 0.02;
    let r = pd/2 * s;
    let h = (d.width || d.face_width || d.length || 0.015) * s;
    let geo = new THREE.CylinderGeometry(r, r, h, 32);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h/2;
    group.add(mesh);
    if (d.bore) {{
      let br = d.bore/2 * s;
      let bgeo = new THREE.CylinderGeometry(br, br, h*0.8, 16);
      let bmesh = new THREE.Mesh(bgeo, new THREE.MeshStandardMaterial({{color:0x0f0f0f, roughness:0.4, metalness:0.5}}));
      bmesh.position.y = h/2;
      group.add(bmesh);
    }}
  }}

  else if (['gear_rack','gear_rack_heavy'].includes(gt)) {{
    let rl = (d.length || 0.100) * s;
    let rh = (d.height || 0.015) * s;
    let rw = (d.width || 0.015)/2 * s;
    let geo = new THREE.BoxGeometry(rl, rh, rw*2);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = rh/2;
    group.add(mesh);
  }}

  else if (['coupling_elastic','coupling_jaw','coupling_bellows','coupling_oldham','coupling_magnetic','coupling_rigid','one_way_clutch','torque_limiter','belt_tensioner','chain_tensioner'].includes(gt)) {{
    let od = d.outer_diameter || d.pulley_diameter || 0.020;
    let r = od/2 * s;
    let h = (d.total_length || d.length || d.width || 0.030) * s;
    let geo = new THREE.CylinderGeometry(r, r, h, 24);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h/2;
    group.add(mesh);
  }}

  else if (['timing_belt_loop','v_belt','roller_chain','roller_chain_precision'].includes(gt)) {{
    let bl = d.length || d.inner_length || 0.200;
    let bw = d.width || d.top_width || 0.015;
    let torusR = bl/(2*Math.PI) * s;
    let tubeR = bw/2 * s;
    let geo = new THREE.TorusGeometry(torusR, tubeR, 16, 64);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = Math.PI/2;
    group.add(mesh);
  }}

  else if (['cylindrical_cell','cylindrical_cell_large','supercapacitor_can','supercapacitor_large','pressure_sensor_can'].includes(gt)) {{
    let dia = d.diameter || 0.0186;
    let r = dia/2 * s;
    let h = (d.height || 0.065) * s;
    let geo = new THREE.CylinderGeometry(r, r, h, 24);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h/2;
    group.add(mesh);
  }}

  else if (['encoder_incremental','encoder_absolute','encoder_hollow'].includes(gt)) {{
    let od = d.body_diameter || d.od || 0.040;
    let r = od/2 * s;
    let h = (d.body_thickness || d.thickness || 0.025) * s;
    let geo = new THREE.CylinderGeometry(r, r, h, 24);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h/2;
    group.add(mesh);
  }}

  else if (['load_cell_aluminum','load_cell_mini'].includes(gt)) {{
    let l = (d.length || 0.125)/2 * s;
    let w = (d.width || 0.013)/2 * s;
    let h = (d.height || 0.025)/2 * s;
    let geo = new THREE.BoxGeometry(l*2, h*2, w*2);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h;
    group.add(mesh);
  }}

  else if (['lidar_module','lidar_module_long','ultrasonic_sensor','joystick_module'].includes(gt)) {{
    let l = (d.length || 0.029)/2 * s;
    let w = (d.width || 0.021)/2 * s;
    let h = (d.height || 0.016)/2 * s;
    let geo = new THREE.BoxGeometry(l*2, h*2, w*2);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h;
    group.add(mesh);
  }}

  else if (['lidar_cylindrical','lidar_disk'].includes(gt)) {{
    let dia = d.diameter || 0.075;
    let r = dia/2 * s;
    let h = (d.height || 0.040) * s;
    let geo = new THREE.CylinderGeometry(r, r, h, 32);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h/2;
    group.add(mesh);
  }}

  else if (['pcb_board','pcb_board_large'].includes(gt)) {{
    let l = (d.length || 0.085)/2 * s;
    let w = (d.width || 0.056)/2 * s;
    let th = 0.003 * s;
    let geo = new THREE.BoxGeometry(l*2, th, w*2);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = th/2;
    group.add(mesh);
    // USB
    if (d.usb_height) {{
      let uh = d.usb_height * s;
      let ugeo = new THREE.BoxGeometry(l*0.12, uh, w*0.16);
      let umesh = new THREE.Mesh(ugeo, new THREE.MeshStandardMaterial({{color:0xa8a8ad, roughness:0.25, metalness:0.8}}));
      umesh.position.set(-l*0.8, th+uh/2, w*0.25);
      group.add(umesh);
    }}
    // GPIO
    if (d.gpio_header_height) {{
      let gh = d.gpio_header_height * s;
      let ggeo = new THREE.BoxGeometry(l*0.08, gh, w*0.36);
      let gmesh = new THREE.Mesh(ggeo, new THREE.MeshStandardMaterial({{color:0x1a1a1a, roughness:0.5, metalness:0.3}}));
      gmesh.position.set(l*0.1, th+gh/2, w*0.45);
      group.add(gmesh);
    }}
  }}

  else if (['qfn_package','lga28_package','lga14_package','lga8_package','so8_package','so16_package','ssop24_package','dfn_package','lga12_package','sot23_8','to220_package','small_pcb_module'].includes(gt)) {{
    let l = (d.length || 0.005) * 8 * s;
    let w = (d.width || 0.004) * 8 * s;
    let h = Math.max(d.height || 0.001, 0.001) * 12 * s;
    let geo = new THREE.BoxGeometry(l, h, w);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h/2;
    group.add(mesh);
  }}

  else if (['button_mushroom'].includes(gt)) {{
    let md = (d.mushroom_dia || 0.036) * s;
    let r = md/2;
    let geo = new THREE.CylinderGeometry(r, r, md*0.12, 32);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = md*0.06;
    group.add(mesh);
    let br = (d.body_diameter || 0.022)/2 * s;
    let bgeo = new THREE.CylinderGeometry(br, br, (d.body_depth||0.040)*s, 24);
    let bmesh = new THREE.Mesh(bgeo, new THREE.MeshStandardMaterial({{color:0x383338, roughness:0.4, metalness:0.6}}));
    bmesh.position.y = -(d.body_depth||0.040)*s/2;
    group.add(bmesh);
  }}

  else if (['micro_switch','potentiometer_rotary','micro_dc'].includes(gt)) {{
    let l = (d.length || d.body_diameter || 0.0125)/2 * s;
    let w = (d.width || d.body_diameter || 0.0063)/2 * s;
    let h = (d.height || d.body_diameter || 0.0125)/2 * s;
    let geo = new THREE.BoxGeometry(l*2, h*2, w*2);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h;
    group.add(mesh);
  }}

  else if (['connector_power','connector_power_small','connector_deans','connector_anderson','connector_power_high_volt'].includes(gt)) {{
    let sl = (d.shell_length || d.housing_length || d.total_length || d.shell_diameter || 0.030) * s;
    let sw = (d.shell_width || d.housing_width || d.width || d.shell_diameter || 0.015) * s;
    let hgeo, mesh;
    if (gt == 'connector_power_high_volt') {{
      hgeo = new THREE.CylinderGeometry(sl/2, sl/2, sw, 24);
      mesh = new THREE.Mesh(hgeo, mat);
      mesh.position.y = sw/2;
    }} else {{
      hgeo = new THREE.BoxGeometry(sl, sw, sw);
      mesh = new THREE.Mesh(hgeo, mat);
      mesh.position.y = sw/2;
    }}
    group.add(mesh);
  }}

  else if (['fuse_holder','fuse_blade_large'].includes(gt)) {{
    let l = (d.length || 0.035)/2 * s;
    let w = (d.width || 0.020)/2 * s;
    let h = (d.height || d.blade_length || 0.040)/2 * s;
    let geo = new THREE.BoxGeometry(l*2, h*2, w*2);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h;
    group.add(mesh);
  }}

  else if (['current_shunt'].includes(gt)) {{
    let l = (d.length || 0.030)/2 * s;
    let w = (d.width || 0.015)/2 * s;
    let geo = new THREE.BoxGeometry(l*2, 0.001*s, w*2);
    group.add(new THREE.Mesh(geo, mat));
  }}

  else if (['ring_with_holes','ring_split'].includes(gt)) {{
    let od = (d.outer_diameter || 0.060) * s;
    let id = (d.inner_diameter || 0.020) * s;
    let th = (d.thickness || 0.005) * s;
    let geo = new THREE.RingGeometry(id/2, od/2, 48);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI/2;
    mesh.position.y = th/2;
    group.add(mesh);
  }}

  else if (['bearing_assembly','bearing_thrust','bearing_angular'].includes(gt)) {{
    let od = (d.outer_diameter || 0.026) * s;
    let id = (d.inner_diameter || 0.010) * s;
    let width = (d.width || 0.008) * s;
    let geo = new THREE.RingGeometry(id/2, od/2, 48);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI/2;
    mesh.position.y = width/2;
    group.add(mesh);
    // 厚度
    let tgeo = new THREE.CylinderGeometry(od/2, od/2, width, 48, 1, true);
    let tmesh = new THREE.Mesh(tgeo, mat);
    tmesh.position.y = width/2;
    group.add(tmesh);
  }}

  else if (['pin_with_ball'].includes(gt)) {{
    let dia = (d.diameter || 0.006) * s;
    let l = (d.length || 0.030) * s;
    let pgeo = new THREE.CylinderGeometry(dia/2, dia/2, l, 16);
    let pmesh = new THREE.Mesh(pgeo, mat);
    pmesh.position.y = l/2;
    group.add(pmesh);
    let bd = (d.ball_diameter || 0.004) * s;
    let bgeo = new THREE.SphereGeometry(bd/2, 16, 16);
    let bmesh = new THREE.Mesh(bgeo, new THREE.MeshStandardMaterial({{color:0xd1d1d6, roughness:0.2, metalness:0.9}}));
    bmesh.position.y = l + bd/2;
    group.add(bmesh);
  }}

  else if (['extrusion_profile','tube_hollow'].includes(gt)) {{
    let sz = (d.size || d.outer_diameter || 0.020) * s;
    let l = (d.length || 0.100) * s;
    let geo = new THREE.CylinderGeometry(sz/2, sz/2, l, gt=='tube_hollow'?24:4);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = l/2;
    group.add(mesh);
  }}

  else if (['flat_plate'].includes(gt)) {{
    let l = (d.length || 0.100)/2 * s;
    let w = (d.width || 0.100)/2 * s;
    let th = (d.thickness || 0.003)/2 * s;
    let geo = new THREE.BoxGeometry(l*2, th*2, w*2);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = th;
    group.add(mesh);
  }}

  else if (['helix_spring'].includes(gt)) {{
    let od = (d.outer_diameter || 0.010) * s;
    let fl = (d.free_length || 0.030) * s;
    let geo = new THREE.CylinderGeometry(od/2, od/2, fl, 16, 1, true);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = fl/2;
    group.add(mesh);
  }}

  else if (['sphere_cap'].includes(gt)) {{
    let dia = (d.diameter || 0.030) * s;
    let geo = new THREE.SphereGeometry(dia/2, 32, 16, 0, Math.PI*2, 0, Math.PI/2);
    let mesh = new THREE.Mesh(geo, mat);
    group.add(mesh);
  }}

  else if (['aluminum_block','l_bracket'].includes(gt)) {{
    let l = (d.length || d.arm_length || d.arm_width || 0.008)/2 * s;
    let w = (d.width || d.arm_width || d.slot_spacing || 0.006)/2 * s;
    let h = (d.height || d.thickness || d.slot_count || 0.004)/2 * s;
    let geo = new THREE.BoxGeometry(l*2, h*2, w*2);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = h;
    group.add(mesh);
  }}

  else {{
    // 默认
    let geo = new THREE.BoxGeometry(0.06, 0.04, 0.05);
    let mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = 0.02;
    group.add(mesh);
  }}

  return group;
}}

// ═══ 缩略图渲染器 (所有卡片共用一个渲染器) ═══
class ThumbRenderer {{
  constructor() {{
    this.renderer = new THREE.WebGLRenderer({{antialias:true, alpha:true}});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0d1117);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.01, 10);
    this.camera.position.set(0.25, 0.2, 0.35);
    this.camera.lookAt(0, 0.04, 0);
    this.light1 = new THREE.DirectionalLight(0xffffff, 2.5);
    this.light1.position.set(1, 2, 1);
    this.scene.add(this.light1);
    this.light2 = new THREE.AmbientLight(0x404060, 1.5);
    this.scene.add(this.light2);
    this.floorGeo = new THREE.PlaneGeometry(1, 1);
    this.floorMat = new THREE.MeshStandardMaterial({{color:0x1a1a1a, roughness:0.8, metalness:0.1}});
    this.floor = new THREE.Mesh(this.floorGeo, this.floorMat);
    this.floor.rotation.x = -Math.PI/2;
    this.floor.position.y = -0.02;
    this.scene.add(this.floor);
  }}

  renderPart(spec) {{
    let canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 300;
    this.renderer.setSize(400, 300);

    let group = createPartMesh(spec);
    if (!group) return canvas;

    // 计算包围盒并居中
    let box = new THREE.Box3().setFromObject(group);
    let center = box.getCenter(new THREE.Vector3());
    group.position.sub(center);
    let size = box.getSize(new THREE.Vector3());
    let maxDim = Math.max(size.x, size.y, size.z);
    let cameraDist = maxDim * 3.5;
    this.camera.position.set(cameraDist*0.7, cameraDist*0.5, cameraDist*0.7);
    this.camera.lookAt(0, 0, 0);

    this.scene.add(group);
    this.renderer.render(this.scene, this.camera);
    this.renderer.domElement.getContext('2d');
    // 将渲染结果画到 canvas
    let ctx = canvas.getContext('2d');
    ctx.drawImage(this.renderer.domElement, 0, 0, 400, 300);
    this.scene.remove(group);
    return canvas;
  }}
}}

const thumbRenderer = new ThumbRenderer();

// ═══ 模态3D查看器 ═══
let modalScene, modalCamera, modalRenderer, modalControls, modalPart, modalAnimId;
let currentPartIdx = -1;

function initModal() {{
  let body = document.getElementById('modalBody');
  modalScene = new THREE.Scene();

  // 棋盘格地面
  let gridTex = (() => {{
    let c = document.createElement('canvas'); c.width=256; c.height=256;
    let ctx=c.getContext('2d');
    for(let y=0;y<8;y++) for(let x=0;x<8;x++){{
      ctx.fillStyle=(x+y)%2==0?'#1a1a1a':'#2a2a2a';
      ctx.fillRect(x*32,y*32,32,32);
    }}
    return new THREE.CanvasTexture(c);
  }})();
  gridTex.wrapS = gridTex.wrapT = THREE.RepeatWrapping;
  gridTex.repeat.set(4,4);
  let floor = new THREE.Mesh(new THREE.PlaneGeometry(4,4), new THREE.MeshStandardMaterial({{map:gridTex, roughness:0.9, metalness:0.1}}));
  floor.rotation.x = -Math.PI/2;
  floor.position.y = -0.05;
  modalScene.add(floor);

  modalScene.add(new THREE.AmbientLight(0x404060, 2));
  let dl = new THREE.DirectionalLight(0xffffff, 3);
  dl.position.set(2,3,2);
  modalScene.add(dl);
  let dl2 = new THREE.DirectionalLight(0x8888cc, 1.2);
  dl2.position.set(-1,1,-1);
  modalScene.add(dl2);

  modalCamera = new THREE.PerspectiveCamera(40, body.clientWidth/body.clientHeight, 0.005, 20);
  modalCamera.position.set(0.4, 0.25, 0.4);
  modalCamera.lookAt(0, 0.04, 0);

  modalRenderer = new THREE.WebGLRenderer({{antialias:true, alpha:false}});
  modalRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  modalRenderer.setSize(body.clientWidth, body.clientHeight);
  modalRenderer.shadowMap.enabled = true;
  modalRenderer.setClearColor(0x0d1117);
  body.appendChild(modalRenderer.domElement);

  modalControls = new OrbitControls(modalCamera, modalRenderer.domElement);
  modalControls.enableDamping = true;
  modalControls.dampingFactor = 0.08;
  modalControls.target.set(0, 0.04, 0);
  modalControls.update();

  window.addEventListener('resize', () => {{
    if (document.getElementById('modal').classList.contains('show')) {{
      let b = document.getElementById('modalBody');
      modalCamera.aspect = b.clientWidth/b.clientHeight;
      modalCamera.updateProjectionMatrix();
      modalRenderer.setSize(b.clientWidth, b.clientHeight);
    }}
  }});

  let animate = () => {{
    modalAnimId = requestAnimationFrame(animate);
    modalControls.update();
    modalRenderer.render(modalScene, modalCamera);
  }};
  animate();
}}

function openModal(idx) {{
  if (modalPart) {{ modalScene.remove(modalPart); modalPart = null; }}
  currentPartIdx = idx;
  let spec = PARTS[idx];
  modalPart = createPartMesh(spec);
  if (!modalPart) return;

  let box = new THREE.Box3().setFromObject(modalPart);
  let center = box.getCenter(new THREE.Vector3());
  modalPart.position.sub(center);
  modalScene.add(modalPart);

  modalControls.target.set(0, 0, 0);
  modalControls.update();
  document.getElementById('modalTitle').textContent = `${{spec.name}}`;
  let info = `${{spec.manufacturer}} ${{spec.part_number}} | ${{CAT_NAMES[spec.category]||spec.category}}`;
  if (spec.voltage) info += ` | ${{spec.voltage}}V`;
  if (spec.torque) info += ` | ${{spec.torque}}N·m`;
  document.getElementById('modalInfo').innerHTML = `<strong>${{spec.name}}</strong><br>${{info}}`;
  document.getElementById('modal').classList.add('show');

  let b = document.getElementById('modalBody');
  modalCamera.aspect = b.clientWidth/b.clientHeight;
  modalCamera.updateProjectionMatrix();
  modalRenderer.setSize(b.clientWidth, b.clientHeight);

  document.getElementById('modalBody').addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowRight') {{ e.preventDefault(); openModal(Math.min(currentPartIdx+1, PARTS.length-1)); }}
    if (e.key === 'ArrowLeft') {{ e.preventDefault(); openModal(Math.max(currentPartIdx-1, 0)); }}
  }});
}}

function closeModal() {{
  document.getElementById('modal').classList.remove('show');
  if (modalPart) {{ modalScene.remove(modalPart); modalPart = null; }}
  currentPartIdx = -1;
}}

// ═══ 构建卡片网格 ═══
function buildGrid(filter) {{
  let grid = document.getElementById('grid');
  grid.innerHTML = '';
  let filtered = filter === 'all' ? PARTS : PARTS.filter(p => p.category === filter);

  for (let i = 0; i < filtered.length; i++) {{
    let p = filtered[i];
    let originalIdx = PARTS.indexOf(p);
    let card = document.createElement('div');
    card.className = 'card';
    card.onclick = () => openModal(originalIdx);

    let canvas = thumbRenderer.renderPart(p);
    card.appendChild(canvas);

    let info = document.createElement('div');
    info.className = 'info';
    info.innerHTML = `<div class="nm" style="color:${{CAT_COLORS[p.category]||'#fff'}}">${{p.name}}</div><div class="dt">${{p.manufacturer}} ${{p.part_number}}</div>`;
    card.appendChild(info);
    grid.appendChild(card);
  }}
}}

// 分类按钮
document.querySelectorAll('.toolbar button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.toolbar button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    buildGrid(btn.dataset.cat);
  }});
}});

// 键盘关闭模态
document.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape' && document.getElementById('modal').classList.contains('show')) {{
    closeModal();
  }}
}});

// 初始化
initModal();
buildGrid('all');
</script>

</body>
</html>'''

# 写入文件
out_path = Path("part_library_previews_v2") / "index_3d.html"
out_path.write_text(html, encoding='utf-8')
print(f"  生成完成: {out_path}")
print(f"  零件数: {len(parts_js)}")
print(f"  文件大小: {out_path.stat().st_size/1024:.0f} KB")