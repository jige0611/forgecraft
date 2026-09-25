#!/usr/bin/env python3
"""
生成终极 Three.js 零件库 — PBR + 高精度几何体
- NopSCADlib 精确尺寸
- STLLoader 加载已导出的 STL
- PBR 材质 (metalness/roughness)
- OrbitControls 交互
"""

import json
from pathlib import Path

BASE = Path(r"c:\Users\刘振鑫\Desktop\第三个")

# 读取零件数据
parts_json_path = BASE / "stl_parts" / "nopscad_parts.json"
with open(parts_json_path, "r", encoding="utf-8") as f:
    nopscad_parts = json.load(f)

# 也读取 v2 零件库
sys_path = str(BASE)
import sys
sys.path.insert(0, sys_path)
from v8_3d_model_library_v2 import get_all_part_specs, PartSpec

v2_specs = get_all_part_specs()

# 合并零件库：v2 零件 + NopSCADlib 精确零件
all_parts = []

# NopSCADlib 精确零件
for p in nopscad_parts:
    all_parts.append({
        "id": p["id"],
        "name": p["name"],
        "category": p["category"],
        "type": p["type"],
        "params": {k: v for k, v in p.items() if k not in ("id","name","category","type")},
    })

# v2 零件（去重）
nop_ids = {p["id"] for p in nopscad_parts}
for pid, spec in sorted(v2_specs.items()):
    if pid not in nop_ids:
        all_parts.append({
            "id": pid,
            "name": spec.display_name,
            "category": spec.category if hasattr(spec, 'category') else "misc",
            "type": spec.geometry_type,
            "params": {
                "color": spec.color[:3] if hasattr(spec, 'color') else [0.5,0.5,0.5],
                "dims": {k: v for k, v in spec.dimensions.items()} if hasattr(spec, 'dimensions') else {},
            },
        })

parts_json = json.dumps(all_parts, ensure_ascii=False)
print(f"零件总数: {len(all_parts)} (NopSCADlib精确: {len(nopscad_parts)}, v2补充: {len(all_parts)-len(nopscad_parts)})")

# ============================================================
# 生成 HTML
# ============================================================
html = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>工程零件库 — PBR 交互式3D</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Segoe UI","Microsoft YaHei",Arial,sans-serif;background:#0d1117;color:#e6edf3;overflow-x:hidden}
.header{background:#161b22;border-bottom:1px solid #30363d;padding:12px 24px;position:sticky;top:0;z-index:100;display:flex;align-items:center;gap:16px}
.header h1{font-size:18px;background:linear-gradient(135deg,#58a6ff,#bc8cff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header .stats{font-size:12px;color:#8b949e}
.filter-bar{display:flex;gap:6px;flex-wrap:wrap;padding:12px 24px;background:#161b22;border-bottom:1px solid #30363d;position:sticky;top:49px;z-index:99}
.filter-btn{padding:6px 14px;border:1px solid #30363d;border-radius:20px;background:transparent;color:#8b949e;cursor:pointer;font-size:12px;transition:all .2s}
.filter-btn:hover,.filter-btn.active{background:#1f6feb33;border-color:#58a6ff;color:#58a6ff}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;padding:16px 24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden;cursor:pointer;transition:all .25s}
.card:hover{border-color:#58a6ff;transform:translateY(-2px);box-shadow:0 8px 24px rgba(88,166,255,.15)}
.card canvas{width:100%;aspect-ratio:1;display:block;background:radial-gradient(ellipse at center,#1a2332 0%,#0d1117 100%)}
.card .info{padding:8px 10px;border-top:1px solid #30363d}
.card .info .name{font-size:11px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .info .cat{font-size:10px;color:#8b949e;margin-top:2px}
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.9);z-index:200;display:flex;align-items:center;justify-content:center;opacity:0;pointer-events:none;transition:opacity .3s}
.modal-overlay.active{opacity:1;pointer-events:all}
.modal-content{background:#161b22;border:1px solid #30363d;border-radius:12px;width:90vw;height:85vh;display:flex;flex-direction:column;overflow:hidden}
.modal-header{display:flex;align-items:center;justify-content:space-between;padding:12px 20px;border-bottom:1px solid #30363d}
.modal-header .part-name{font-size:16px;font-weight:600}
.modal-header .part-specs{font-size:11px;color:#8b949e}
.modal-close{background:none;border:none;color:#8b949e;font-size:24px;cursor:pointer;padding:4px 12px;border-radius:6px;transition:all .2s}
.modal-close:hover{color:#f85149;background:#f8514922}
.modal-viewer{flex:1;position:relative}
.modal-viewer canvas{width:100%;height:100%}
.modal-nav{display:flex;align-items:center;justify-content:center;gap:16px;padding:10px;border-top:1px solid #30363d}
.modal-nav button{padding:6px 18px;border:1px solid #30363d;border-radius:6px;background:transparent;color:#c9d1d9;cursor:pointer;font-size:13px;transition:all .2s}
.modal-nav button:hover{background:#1f6feb33;border-color:#58a6ff}
.tooltip{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);background:#161b22cc;backdrop-filter:blur(8px);border:1px solid #30363d;border-radius:8px;padding:6px 16px;font-size:11px;color:#8b949e;pointer-events:none}
</style>
</head>
<body>

<div class="header">
  <h1>工程零件库 · PBR 3D</h1>
  <span class="stats" id="stats">加载中...</span>
</div>

<div class="filter-bar" id="filterBar"></div>
<div class="grid" id="grid"></div>

<div class="modal-overlay" id="modal">
  <div class="modal-content">
    <div class="modal-header">
      <div>
        <div class="part-name" id="modalName">-</div>
        <div class="part-specs" id="modalSpecs">-</div>
      </div>
      <button class="modal-close" id="modalClose">&times;</button>
    </div>
    <div class="modal-viewer" id="modalViewer"></div>
    <div class="tooltip">拖拽旋转 · 滚轮缩放 · 右键平移</div>
    <div class="modal-nav">
      <button id="btnPrev">&larr; 上一个</button>
      <span style="font-size:12px;color:#8b949e" id="navLabel">-</span>
      <button id="btnNext">下一个 &rarr;</button>
    </div>
  </div>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

// ============================================================
// 零件数据
// ============================================================
const ALL_PARTS = ''' + parts_json + r''';

// ============================================================
// 高精度几何体工厂 (segments~200 = $fn=200级别)
// ============================================================
const SEG = 256;  // 圆的段数 ($fn=256 级别，256段肉眼完全不可见棱角)
const TORUS_RADIAL = 32;  // 圆环截面段数（之前16太低，肉眼可见多边形棱角）

// ============================================================
// V2 零件库完整几何体工厂 (100+ geometry_type)
// ============================================================
function v2Color(part) {
  const p = part.params;
  return p.color ? new THREE.Color(p.color[0],p.color[1],p.color[2]) : 0x4a6fa5;
}
function v2Rough(part) { return part.params.roughness || 0.3; }
function v2Metal(part) { return part.params.metallic !== undefined ? part.params.metallic : 0.5; }
function v2Mat(part, col) {
  return new THREE.MeshStandardMaterial({color:col||v2Color(part), roughness:v2Rough(part), metalness:v2Metal(part)});
}

// 提取尺寸 (v2 的 dims 对象，单位是米，转换为显示单位)
function D(part, key, fallback) {
  const d = part.params.dims || {};
  if (d[key] !== undefined) return d[key] * 10;  // m → display units
  return fallback;
}

// 创建带齿齿轮几何体
function createGearTeeth(pitchD, od, teethCount, faceW, g, mat) {
  const outerR = od / 2;
  const pitchR = pitchD / 2;
  const shape = new THREE.Shape();
  const toothH = (outerR - pitchR) * 1.2;
  for (let i = 0; i < teethCount; i++) {
    const a = (i / teethCount) * Math.PI * 2;
    const r = outerR;
    const r2 = outerR - toothH * 0.6;
    if (i === 0) shape.moveTo(Math.cos(a)*outerR, Math.sin(a)*outerR);
    else shape.lineTo(Math.cos(a)*outerR, Math.sin(a)*outerR);
    const an = ((i+0.5)/teethCount)*Math.PI*2;
    shape.lineTo(Math.cos(an)*r2, Math.sin(an)*r2);
  }
  shape.closePath();
  const hole = new THREE.Path();
  for (let i = 0; i <= SEG; i++) {
    const a = (i/SEG)*Math.PI*2;
    if (i===0) hole.moveTo(Math.cos(a)*5, Math.sin(a)*5);
    else hole.lineTo(Math.cos(a)*5, Math.sin(a)*5);
  }
  shape.holes.push(hole);
  const geo = new THREE.ExtrudeGeometry(shape, {depth:faceW, bevelEnabled:true, bevelThickness:0.1, bevelSize:0.1, bevelSegments:3});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.z = -faceW/2;
  g.add(mesh);
}

function buildV2Geometry(part, targetGroup) {
  const t = part.type;
  const d = part.params.dims || {};
  const col = v2Color(part);
  const mat = v2Mat(part, col);
  const g = targetGroup;

  // ── 辅助添加材质 ──
  const addCyl = (r, h, yoff, m) => {
    const geo = new THREE.CylinderGeometry(r, r, h, SEG, SEG);
    const mesh = new THREE.Mesh(geo, m || mat);
    if (yoff !== 0) mesh.position.y = yoff;
    g.add(mesh);
  };
  const addBox = (w, h, d2, m) => {
    const geo = new THREE.BoxGeometry(w, h, d2);
    const mesh = new THREE.Mesh(geo, m || mat);
    g.add(mesh);
  };

  // ── 1. 圆柱电机族 (cylinder_*) ──
  if (t.startsWith('cylinder') || t === 'pancake_motor' || t === 'disc_motor' || t === 'disc_motor_thin' ||
      t === 'joint_motor' || t === 'multirotor_motor' || t === 'square_cylinder_stepper' ||
      t === 'square_cylinder_large' || t === 'square_cylinder_xlarge' ||
      t === 'gearbox_dc' || t === 'gearbox_dc_small' || t === 'micro_dc') {

    const od = D(part,'outer_diameter', D(part,'body_diameter',30));
    const bl = D(part,'total_height', D(part,'body_length', D(part,'body_height',50)));
    const r = od / 2;
    const shaftR = D(part,'shaft_diameter', od*0.2) / 2;
    const shaftL = D(part,'shaft_length', bl*0.3);
    const flD = D(part,'flange_diameter', od*1.15);
    const flT = D(part,'flange_thickness', 3);
    const isFlat = t === 'pancake_motor' || t === 'disc_motor' || t === 'disc_motor_thin';
    const bodyLen = isFlat ? bl * 0.3 : bl;
    const endCapMat = new THREE.MeshStandardMaterial({color:0xd0d0d8, roughness:0.25, metalness:0.85});

    // 主体
    addCyl(r, bodyLen, 0, mat);
    // 端盖
    if (flD > od) addCyl(flD/2, flT, bodyLen/2, endCapMat);
    else addCyl(r, 1.5, bodyLen/2, endCapMat);
    // 轴
    const shaftMat = new THREE.MeshStandardMaterial({color:0xe8e8e8, roughness:0.2, metalness:0.9});
    addCyl(shaftR, shaftL, bodyLen/2 + shaftL/2, shaftMat);
    // 法兰孔（简化：4个沉头孔标记）
    if (flD > od) {
      for (let i = 0; i < 4; i++) {
        const a = (i/4)*Math.PI*2 + Math.PI/8;
        const holeR = (flD/2 + r)/2;
        const holeGeo = new THREE.CylinderGeometry(0.3, 0.3, 2, 8);
        const holeMesh = new THREE.Mesh(holeGeo, new THREE.MeshStandardMaterial({color:0x1a1a2e, roughness:0.8, metalness:0}));
        holeMesh.position.set(Math.cos(a)*holeR, bodyLen/2 + flT/2, Math.sin(a)*holeR);
        g.add(holeMesh);
      }
    }
  }

  // ── 2. Dynamixel / 伺服外壳 (矩形) ──
  else if (t.startsWith('dynamixel') || t.startsWith('servo')) {
    const w = D(part,'width', D(part,'body_width',40));
    const h = D(part,'height', D(part,'body_height',30));
    const depth = D(part,'depth', D(part,'body_depth',25));
    const servoMat = v2Mat(part, 0x3a3a3e);
    addBox(w, h, depth, servoMat);
    // 输出轴
    const sr = D(part,'shaft_diameter',6)/2;
    addCyl(sr, 15, h/2 + 7, mat);
    // 安装耳
    const earMat = new THREE.MeshStandardMaterial({color:0xcdcdd4, roughness:0.15, metalness:0.9});
    const earGeo = new THREE.BoxGeometry(w*0.25, 2, depth*0.6);
    const ear1 = new THREE.Mesh(earGeo, earMat);
    ear1.position.set(-w*0.35, -h/2-4, 0);
    g.add(ear1);
    const ear2 = new THREE.Mesh(earGeo, earMat);
    ear2.position.set(w*0.35, -h/2-4, 0);
    g.add(ear2);
  }

  // ── 3. 齿轮族 (spur_gear, helical_gear, bevel_gear, worm_wheel, worm_shaft, gear_rack) ──
  else if (t.includes('gear') || t === 'worm_wheel' || t === 'worm_shaft' || t.includes('rack')) {
    const pd = D(part,'pitch_diameter', 40);
    const od = D(part,'outer_diameter', pd*1.05);
    const tc = Math.round(d.teeth_count || 20);
    const fw = D(part,'face_width', 15);
    const gearMat = new THREE.MeshStandardMaterial({color:0x2c3038, roughness:0.5, metalness:0.9});

    if (t === 'worm_shaft') {
      const md = D(part,'major_diameter',12);
      const len = D(part,'length',40);
      addCyl(md/2, len, 0, gearMat);
      // 螺纹螺旋
      for (let i = 0; i < len; i+=1.5) {
        const a = i * 0.8;
        const ringGeo = new THREE.TorusGeometry(md/2+0.5, 0.3, 8, SEG);
        const ring = new THREE.Mesh(ringGeo, gearMat);
        ring.position.y = i - len/2;
        ring.rotation.y = a;
        g.add(ring);
      }
    } else if (t.includes('rack')) {
      const len = D(part,'length',100);
      const h = D(part,'height',15);
      const w = D(part,'width',15);
      addBox(w, h, len, gearMat);
    } else if (t === 'bevel_gear') {
      const cr = od/2;
      const coneGeo = new THREE.ConeGeometry(cr*0.8, fw, SEG);
      const cone = new THREE.Mesh(coneGeo, gearMat);
      cone.rotation.x = Math.PI/2;
      g.add(cone);
      const baseGeo = new THREE.CylinderGeometry(cr, cr, fw*0.3, SEG);
      const base = new THREE.Mesh(baseGeo, gearMat);
      base.position.y = -fw*0.4;
      g.add(base);
    } else if (t === 'helical_gear') {
      createGearTeeth(pd, od, tc, fw, g, gearMat);
    } else {
      // spur_gear, worm_wheel
      createGearTeeth(pd, od, tc, fw, g, gearMat);
      // 轮毂
      const hubR = od*0.25;
      addCyl(hubR, fw*1.5, 0, gearMat);
    }
  }

  // ── 4. 同步带轮 / 链轮 ──
  else if (t.includes('pulley') || t.includes('sprocket')) {
    const od = D(part,'pitch_diameter', D(part,'outer_diameter',25));
    const bw = D(part,'width', D(part,'face_width',15));
    const bore = D(part,'bore', 5);
    const tc = Math.round(d.teeth_count || 16);
    const pulleyMat = new THREE.MeshStandardMaterial({color:0xd4a574, roughness:0.4, metalness:0.6});

    // 主体带齿
    const shape = new THREE.Shape();
    for (let i = 0; i < tc; i++) {
      const a = (i/tc)*Math.PI*2;
      const r1 = od/2, r2 = od/2 - 1.2;
      if (i===0) shape.moveTo(Math.cos(a)*r1, Math.sin(a)*r1);
      else shape.lineTo(Math.cos(a)*r1, Math.sin(a)*r1);
      const an = ((i+0.5)/tc)*Math.PI*2;
      shape.lineTo(Math.cos(an)*r2, Math.sin(an)*r2);
    }
    shape.closePath();
    const hole = new THREE.Path();
    for (let i = 0; i <= SEG; i++) {
      const a = (i/SEG)*Math.PI*2;
      if (i===0) hole.moveTo(Math.cos(a)*bore/2, Math.sin(a)*bore/2);
      else hole.lineTo(Math.cos(a)*bore/2, Math.sin(a)*bore/2);
    }
    shape.holes.push(hole);
    const geo = new THREE.ExtrudeGeometry(shape, {depth:bw, bevelEnabled:true, bevelThickness:0.1, bevelSize:0.1, bevelSegments:4});
    const mesh = new THREE.Mesh(geo, pulleyMat);
    mesh.position.z = -bw/2;
    g.add(mesh);

    // 法兰
    const flMat = new THREE.MeshStandardMaterial({color:0xc0c0c0, roughness:0.3, metalness:0.85});
    const flR = od/2 + 3;
    addCyl(flR, 1.5, bw/2, flMat);
    addCyl(flR, 1.5, -bw/2, flMat);
  }

  // ── 5. 联轴器 ──
  else if (t.includes('coupling')) {
    const od = D(part,'outer_diameter', 25);
    const len = D(part,'total_length', D(part,'length',40));
    const couplingMat = new THREE.MeshStandardMaterial({color:0xc0c0c0, roughness:0.3, metalness:0.85});
    addCyl(od/2, len, 0, couplingMat);
    // 夹紧槽
    for (let s = 0; s < 2; s++) {
      const grooveGeo = new THREE.BoxGeometry(od*0.8, 1.2, 1.2);
      const grv = new THREE.Mesh(grooveGeo, new THREE.MeshStandardMaterial({color:0x1a1a2e, roughness:0.8, metalness:0}));
      grv.position.y = (s-0.5)*len*0.4;
      g.add(grv);
    }
  }

  // ── 6. 减速器 (harmonic, planetary, cycloidal) ──
  else if (t.includes('harmonic') || t.includes('planetary') || t.includes('cycloidal')) {
    const od = D(part,'outer_diameter', D(part,'body_diameter',60));
    const len = D(part,'total_length', D(part,'body_length',40));
    const inputR = D(part,'input_diameter', od*0.5)/2;
    const outputR = D(part,'output_diameter', od*0.6)/2;
    const gbMat = new THREE.MeshStandardMaterial({color:0x3a3a40, roughness:0.5, metalness:0.8});
    addCyl(od/2, len, 0, gbMat);
    // 输入轴
    const shaftMat = new THREE.MeshStandardMaterial({color:0xe8e8e8, roughness:0.2, metalness:0.9});
    addCyl(inputR, len*0.3, len/2 + len*0.15, shaftMat);
    // 输出法兰
    const flMat = new THREE.MeshStandardMaterial({color:0xd0d0d8, roughness:0.25, metalness:0.85});
    addCyl(outputR, len*0.15, -len/2, flMat);
  }

  // ── 7. 电池 ──
  else if (t.includes('battery') || t.includes('cylindrical_cell')) {
    if (t.includes('cylindrical_cell')) {
      const od = D(part,'diameter', D(part,'outer_diameter',18));
      const h = D(part,'length', D(part,'total_length',65));
      const batMat = new THREE.MeshStandardMaterial({color:0x3a8c3a, roughness:0.5, metalness:0.05});
      addCyl(od/2, h*0.85, 0, batMat);
      // 正极
      const posMat = new THREE.MeshStandardMaterial({color:0xd4d4d0, roughness:0.2, metalness:0.9});
      addCyl(od*0.2, 3, h/2 + 1.5, posMat);
    } else {
      const w = D(part,'length', D(part,'width',100));
      const h = D(part,'height', 30);
      const dep = D(part,'width', 50);
      const batMat = new THREE.MeshStandardMaterial({color:0x2d6a2d, roughness:0.5, metalness:0.05});
      addBox(w, h, dep, batMat);
      // 标签
      const labelMat = new THREE.MeshStandardMaterial({color:0xff6600, roughness:0.6, metalness:0});
      const labelGeo = new THREE.BoxGeometry(w*0.6, h*0.4, dep*0.05);
      const label = new THREE.Mesh(labelGeo, labelMat);
      g.add(label);
    }
  }

  // ── 8. ESC / 电源模块 ──
  else if (t.includes('esc_') || t.includes('bms_') || t.includes('power_') ||
           t === 'pdb' || t.includes('dc_dc')) {
    const w = D(part,'length', D(part,'width',50));
    const h = D(part,'height', 12);
    const dep = D(part,'width', 40);
    const escMat = new THREE.MeshStandardMaterial({color:0x1a3a1a, roughness:0.4, metalness:0.1});
    addBox(w, h, dep, escMat);
    // 散热片纹理 (ESC)
    if (t === 'esc_heatsink' || t.includes('heatsink')) {
      const finMat = new THREE.MeshStandardMaterial({color:0xb0b0b8, roughness:0.3, metalness:0.85});
      for (let i = 0; i < 8; i++) {
        const finGeo = new THREE.BoxGeometry(w*0.9, 0.5, dep*0.15);
        const fin = new THREE.Mesh(finGeo, finMat);
        fin.position.y = h/2 + 1.5 + i*0.8;
        fin.position.x = (i-3.5)*2;
        g.add(fin);
      }
    }
  }

  // ── 9. 传感器芯片 ──
  else if (t.includes('qfn') || t.includes('lga') || t.includes('pcb_module') || t.includes('package')) {
    const w = D(part,'length', D(part,'width',15));
    const h = D(part,'height', 4);
    const dep = D(part,'width', 15);
    const pcbMat = new THREE.MeshStandardMaterial({color:0x0a3a0a, roughness:0.4, metalness:0.05});
    addBox(w, h, dep, pcbMat);
    // IC 芯片块
    const icMat = new THREE.MeshStandardMaterial({color:0x1a1a1a, roughness:0.3, metalness:0.1});
    const icGeo = new THREE.BoxGeometry(w*0.5, h*0.6, dep*0.5);
    const ic = new THREE.Mesh(icGeo, icMat);
    ic.position.y = h/2 + h*0.3;
    g.add(ic);
    // 引脚标记
    for (let i = 0; i < 4; i++) {
      const side = (i<2) ? -w/2 : w/2;
      const pinGeo = new THREE.BoxGeometry(0.6, 0.3, 1.8);
      const pinMat = new THREE.MeshStandardMaterial({color:0xc0c0c0, roughness:0.2, metalness:0.9});
      for (let j = 0; j < 3; j++) {
        const pin = new THREE.Mesh(pinGeo, pinMat);
        pin.position.set(side, -h/2-0.2, (j-1)*dep*0.25);
        g.add(pin);
      }
    }
  }

  // ── 10. 同步带/链条/皮带 ──
  else if (t.includes('belt') || t.includes('chain') || t === 'v_belt') {
    const len = D(part,'length', D(part,'inner_length',200));
    const w = D(part,'width', 15);
    const thk = D(part,'thickness', 6);
    if (t.includes('v_belt')) {
      const beltMat = new THREE.MeshStandardMaterial({color:0x1a1a1a, roughness:0.9, metalness:0});
      addCyl(thk/2, w, 0, beltMat);
      addCyl(thk*0.6, w, -0.01, new THREE.MeshStandardMaterial({color:0x2a2a2a, roughness:0.85, metalness:0}));
    } else if (t.includes('chain')) {
      const chainMat = new THREE.MeshStandardMaterial({color:0x5a5a5e, roughness:0.5, metalness:0.9});
      const segLen = D(part,'pitch', 12);
      const segs = Math.floor(len/segLen);
      for (let i = 0; i < segs; i++) {
        const seg = new THREE.Mesh(new THREE.BoxGeometry(segLen*0.7, segLen*0.4, w*0.6), chainMat);
        seg.position.y = -len/2 + i*segLen;
        g.add(seg);
        // 滚子
        addCyl(segLen*0.15, w*0.6, seg.position.y, new THREE.MeshStandardMaterial({color:0x808084, roughness:0.2, metalness:0.95}));
      }
    } else {
      const beltMat = new THREE.MeshStandardMaterial({color:0x2a2a3e, roughness:0.85, metalness:0});
      addBox(w, thk, len, beltMat);
    }
  }

  // ── 11. 连接器 ──
  else if (t.includes('connector')) {
    const w = D(part,'length', D(part,'width',25));
    const h = D(part,'height', 15);
    const dep = D(part,'width', 20);
    const connMat = new THREE.MeshStandardMaterial({color:0xffcc00, roughness:0.4, metalness:0.1});
    addBox(w, h, dep, connMat);
    // 端子
    for (let i = 0; i < 4; i++) {
      const termGeo = new THREE.BoxGeometry(2, 1.5, dep*0.4);
      const termMat = new THREE.MeshStandardMaterial({color:0xc8a800, roughness:0.2, metalness:0.95});
      const term = new THREE.Mesh(termGeo, termMat);
      term.position.set((i-1.5)*w*0.2, -h/2-1, 0);
      g.add(term);
    }
  }

  // ── 12. 保险丝 / 分流器 ──
  else if (t.includes('fuse') || t.includes('current_shunt')) {
    if (t.includes('fuse')) {
      const w = D(part,'length', D(part,'width',20));
      const h = D(part,'height', 30);
      const dep = D(part,'width', 12);
      const fuseMat = new THREE.MeshStandardMaterial({color:0xcc3333, roughness:0.3, metalness:0.1});
      addBox(w, h, dep, fuseMat);
      // 刀片端子
      const bladeMat = new THREE.MeshStandardMaterial({color:0xc0c0c0, roughness:0.2, metalness:0.9});
      for (let i = 0; i < 2; i++) {
        const bladeGeo = new THREE.BoxGeometry(w*0.3, 10, 2);
        const blade = new THREE.Mesh(bladeGeo, bladeMat);
        blade.position.set((i-0.5)*w*0.3, -h/2-5, 0);
        g.add(blade);
      }
    } else {
      const w = D(part,'length', 30);
      const h = D(part,'height', 10);
      const dep = D(part,'width', 25);
      const shuntMat = new THREE.MeshStandardMaterial({color:0xb87333, roughness:0.3, metalness:0.85});
      addBox(w, h, dep, shuntMat);
    }
  }

  // ── 13. 线执行器 / 音圈电机 ──
  else if (t.includes('linear_actuator') || t === 'voice_coil' || t === 'linear_motor') {
    const od = D(part,'body_diameter', D(part,'outer_diameter',25));
    const len = D(part,'stroke', D(part,'total_length',60));
    const actMat = new THREE.MeshStandardMaterial({color:0x3a3a40, roughness:0.4, metalness:0.75});
    addCyl(od/2, len*0.6, 0, actMat);
    // 推杆
    const shaftMat = new THREE.MeshStandardMaterial({color:0xe8e8e8, roughness:0.15, metalness:0.9});
    const shaftR = od*0.2;
    addCyl(shaftR, len, len*0.6 + len*0.3, shaftMat);
  }

  // ── 14. 型材 / 管子 ──
  else if (t.includes('extrusion') || t.includes('tube_hollow') || t.includes('profile')) {
    const size = D(part,'size', 20);
    const len = D(part,'length', 100);
    const slot = D(part,'slot', 6);
    const extMat = new THREE.MeshStandardMaterial({color:0xc0c0c0, roughness:0.25, metalness:0.85});
    const shape = new THREE.Shape();
    const hs = size/2;
    shape.moveTo(-hs,-hs); shape.lineTo(hs,-hs); shape.lineTo(hs,hs); shape.lineTo(-hs,hs); shape.closePath();
    const geo = new THREE.ExtrudeGeometry(shape, {steps:1, depth:len, bevelEnabled:true, bevelThickness:0.3, bevelSize:0.3, bevelSegments:6});
    const mesh = new THREE.Mesh(geo, extMat);
    mesh.position.z = -len/2;
    g.add(mesh);
  }

  // ── 15. 其他 (弹性元件、离合器、张紧器等) ──
  else if (t.includes('tensioner') || t.includes('clutch') || t.includes('torque_limiter')) {
    const od = D(part,'outer_diameter', 30);
    const h = D(part,'total_length', D(part,'length',20));
    const miscMat = new THREE.MeshStandardMaterial({color:0x5a5a60, roughness:0.5, metalness:0.7});
    addCyl(od/2, h, 0, miscMat);
  }

  // ── 16. 超级电容 ──
  else if (t.includes('supercapacitor')) {
    const od = D(part,'diameter', D(part,'outer_diameter',35));
    const h = D(part,'length', D(part,'total_length',60));
    const capMat = new THREE.MeshStandardMaterial({color:0x2255aa, roughness:0.35, metalness:0.1});
    addCyl(od/2, h*0.8, 0, capMat);
    // 极性标记
    const polarMat = new THREE.MeshStandardMaterial({color:0xcc3333, roughness:0.4, metalness:0});
    const polarGeo = new THREE.CylinderGeometry(od*0.4, od*0.4, 2, SEG);
    const polar = new THREE.Mesh(polarGeo, polarMat);
    polar.position.y = h/2;
    g.add(polar);
  }

  // ── 17. 超级通用fallback ──
  else {
    const od = D(part,'outer_diameter', D(part,'diameter', D(part,'body_diameter',30)));
    const h = D(part,'total_length', D(part,'length', D(part,'height',40)));
    const r = Math.max(od/2, 2);
    const len = Math.min(Math.max(h, 5), 80);
    addCyl(r, len, 0, mat);
  }
}
function createPartGeometry(part) {
  const t = part.type;
  const p = part.params;
  const g = new THREE.Group();

  switch(t) {
    // ---- 步进电机 ----
    case 'stepper_motor': {
      const side = p.side || 42.3;
      const length = p.length || 40;
      const bodyR = p.body_r || 26.8;
      const bossR = p.boss_r || 11;
      const bossH = p.boss_h || 2;
      const shaftD = p.shaft_d || 5;
      const shaftL = p.shaft_l || 20;

      // 机身 (黑色叠片)
      const bodyGeo = new THREE.CylinderGeometry(bodyR, bodyR, length - 4, SEG);
      const bodyMat = new THREE.MeshStandardMaterial({color:0x1a1a2e, roughness:0.7, metalness:0.1});
      const body = new THREE.Mesh(bodyGeo, bodyMat);
      g.add(body);

      // 端盖 (银色铝)
      const capGeo = new THREE.CylinderGeometry(side/2, side/2, 2, SEG);
      const capMat = new THREE.MeshStandardMaterial({color:0xc0c0c0, roughness:0.3, metalness:0.8});
      const capTop = new THREE.Mesh(capGeo, capMat);
      capTop.position.y = length/2 - 1;
      g.add(capTop);
      const capBot = new THREE.Mesh(capGeo, capMat);
      capBot.position.y = -length/2 + 1;
      g.add(capBot);

      // 凸台
      const bossGeo = new THREE.CylinderGeometry(bossR, bossR, bossH*2, SEG);
      const bossTop = new THREE.Mesh(bossGeo, capMat);
      bossTop.position.y = length/2 + bossH - 0.5;
      g.add(bossTop);

      // 轴
      const shaftGeo = new THREE.CylinderGeometry(shaftD/2, shaftD/2, shaftL + 1, SEG);
      const shaftMat = new THREE.MeshStandardMaterial({color:0xe8e8e8, roughness:0.2, metalness:0.9});
      const shaft = new THREE.Mesh(shaftGeo, shaftMat);
      shaft.position.y = length/2 + shaftL/2;
      g.add(shaft);
      break;
    }

    // ---- 球轴承 ----
    case 'ball_bearing': {
      const bore = p.bore/2 || 4;
      const od = p.od/2 || 11;
      const w = p.width || 7;
      const flangeD = p.flange_d;
      const flangeW = p.flange_w;

      // 外圈
      const outerGeo = new THREE.TorusGeometry(od - 1.2, 1.2, TORUS_RADIAL, SEG);
      outerGeo.rotateX(Math.PI/2);
      const outerMat = new THREE.MeshStandardMaterial({color:0xbdc3c7, roughness:0.2, metalness:0.9});
      g.add(new THREE.Mesh(outerGeo, outerMat));

      // 外圈圆柱体
      const outerCyl = new THREE.CylinderGeometry(od, od, w, SEG, SEG, true);
      g.add(new THREE.Mesh(outerCyl, outerMat));

      // 内圈
      const innerGeo = new THREE.TorusGeometry(bore + 1.2, 1.2, TORUS_RADIAL, SEG);
      innerGeo.rotateX(Math.PI/2);
      const innerMat = new THREE.MeshStandardMaterial({color:0xd5d8dc, roughness:0.15, metalness:0.95});
      g.add(new THREE.Mesh(innerGeo, innerMat));
      const innerCyl = new THREE.CylinderGeometry(bore, bore, w, SEG, SEG, true);
      g.add(new THREE.Mesh(innerCyl, innerMat));

      // 密封圈
      const sealGeo = new THREE.RingGeometry(bore+1.5, od-0.5, SEG);
      const sealColor = p.seal_color ? parseInt(p.seal_color.slice(1),16) : 0x1a1a2e;
      const sealMat = new THREE.MeshStandardMaterial({color:sealColor, roughness:0.6, metalness:0, side:THREE.DoubleSide});
      const sealTop = new THREE.Mesh(sealGeo, sealMat);
      sealTop.position.y = w/2;
      g.add(sealTop);
      const sealBot = new THREE.Mesh(sealGeo, sealMat);
      sealBot.position.y = -w/2;
      sealBot.rotation.y = Math.PI;
      g.add(sealBot);

      // 滚珠环
      for (let i = 0; i < 8; i++) {
        const angle = (i/8) * Math.PI*2;
        const r = (bore + od) / 2;
        const ball = new THREE.Mesh(
          new THREE.SphereGeometry(1.0, SEG, SEG),
          new THREE.MeshStandardMaterial({color:0xecf0f1, roughness:0.1, metalness:1})
        );
        ball.position.set(Math.cos(angle)*r, 0, Math.sin(angle)*r);
        g.add(ball);
      }

      if (flangeD) {
        const flangeGeo = new THREE.CylinderGeometry(flangeD/2, flangeD/2, flangeW, SEG);
        g.add(new THREE.Mesh(flangeGeo, outerMat));
      }
      break;
    }

    // ---- 同步带轮 ----
    case 'pulley': {
      const od = p.od/2 || 6;
      const flange = p.flange_d/2 || 8;
      const width = p.width || 5;
      const bore = p.bore/2 || 2.5;

      const mat = new THREE.MeshStandardMaterial({color:0xd4a574, roughness:0.4, metalness:0.6});

      // 主体
      const bodyGeo = new THREE.CylinderGeometry(od, od, width, SEG);
      g.add(new THREE.Mesh(bodyGeo, mat));

      // 法兰
      const flangeGeo = new THREE.CylinderGeometry(flange, flange, width*0.15, SEG);
      const flangeTop = new THREE.Mesh(flangeGeo, mat);
      flangeTop.position.y = width/2;
      g.add(flangeTop);
      const flangeBot = new THREE.Mesh(flangeGeo, mat);
      flangeBot.position.y = -width/2;
      g.add(flangeBot);

      // 中心孔
      const holeGeo = new THREE.CylinderGeometry(bore, bore, width+0.5, SEG);
      const hole = new THREE.Mesh(holeGeo,
        new THREE.MeshStandardMaterial({color:0x2c3e50, roughness:0.5, metalness:0.3}));
      g.add(hole);
      break;
    }

    // ---- 铝型材 ----
    case 'extrusion': {
      const sx = p.size_x || 20;
      const sy = p.size_y || 20;
      const slot = p.slot || 5;
      const mat = new THREE.MeshStandardMaterial({color:0xc0c0c0, roughness:0.25, metalness:0.85});

      // 主体
      const len = sx > sy ? sx*2 : sy*2;
      const shape = new THREE.Shape();
      const hsx = sx/2, hsy = sy/2;
      shape.moveTo(-hsx, -hsy);
      shape.lineTo( hsx, -hsy);
      shape.lineTo( hsx,  hsy);
      shape.lineTo(-hsx,  hsy);
      shape.closePath();

      const extrudeSettings = {steps:1, depth: len, bevelEnabled:true, bevelThickness:0.3, bevelSize:0.3, bevelSegments:6};
      const geo = new THREE.ExtrudeGeometry(shape, extrudeSettings);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.z = -len/2;
      g.add(mesh);
      break;
    }

    // ---- 直线轴承 ----
    case 'linear_bearing': {
      const bore = p.bore/2 || 4;
      const od = p.od/2 || 7.5;
      const l = p.length || 24;
      const mat = new THREE.MeshStandardMaterial({color:0x8e9eab, roughness:0.2, metalness:0.85});

      const outerGeo = new THREE.CylinderGeometry(od, od, l, SEG);
      g.add(new THREE.Mesh(outerGeo, mat));

      const innerGeo = new THREE.CylinderGeometry(bore, bore, l+0.1, SEG);
      g.add(new THREE.Mesh(innerGeo,
        new THREE.MeshStandardMaterial({color:0x2c3e50, roughness:0.5, metalness:0.3})));

      // 滚珠槽
      for (let i = 0; i < 4; i++) {
        const angle = (i/4)*Math.PI*2;
        const r = (bore + od) / 2;
        const row = new THREE.Mesh(
          new THREE.CylinderGeometry(0.6, 0.6, l*0.8, SEG),
          new THREE.MeshStandardMaterial({color:0xecf0f1, roughness:0.1, metalness:1})
        );
        row.position.set(Math.cos(angle)*r, 0, Math.sin(angle)*r);
        g.add(row);
      }
      break;
    }

    // ---- 螺丝 ----
    case 'screw': {
      const d = p.diameter || 3;
      const headD = p.head_d || 5.5;
      const headH = p.head_h || 3;
      const headType = p.head_type || 'cap';
      const shaftMat = new THREE.MeshStandardMaterial({color:0x8e9eab, roughness:0.3, metalness:0.85});

      // 螺杆 (带简化螺纹)
      const shaftLen = 20;
      const shaftGeo = new THREE.CylinderGeometry(d/2, d/2, shaftLen, SEG);
      const shaft = new THREE.Mesh(shaftGeo, shaftMat);
      shaft.position.y = -shaftLen/2;
      g.add(shaft);

      // 螺纹环 (简化表现)
      for (let i = 0; i < shaftLen/2; i+=1.5) {
        const ringGeo = new THREE.TorusGeometry(d/2+0.1, 0.08, 16, SEG);
        const ring = new THREE.Mesh(ringGeo, shaftMat);
        ring.position.y = -i;
        g.add(ring);
      }

      // 头部
      if (headType === 'cs') {
        const headGeo = new THREE.ConeGeometry(headD/2, headH, SEG);
        const head = new THREE.Mesh(headGeo, shaftMat);
        head.position.y = headH/2;
        g.add(head);
      } else {
        const headGeo = new THREE.CylinderGeometry(headD/2, headD/2, headH, SEG);
        const head = new THREE.Mesh(headGeo, shaftMat);
        head.position.y = headH/2;
        g.add(head);
      }
      break;
    }

    // ---- 螺母 ----
    case 'nut': {
      const w = p.width || 5.5;
      const h = p.height || 2.4;
      const d = p.diameter || 3;
      const mat = new THREE.MeshStandardMaterial({color:0x8e9eab, roughness:0.3, metalness:0.85});

      // 六角形
      const shape = new THREE.Shape();
      const r = w/2;
      for (let i = 0; i < 6; i++) {
        const angle = (i/6)*Math.PI*2 - Math.PI/6;
        const x = Math.cos(angle)*r;
        const y = Math.sin(angle)*r;
        if (i===0) shape.moveTo(x,y);
        else shape.lineTo(x,y);
      }
      shape.closePath();
      const extrudeSettings = {steps:1, depth:h, bevelEnabled:true, bevelThickness:0.15, bevelSize:0.15, bevelSegments:6};
      const geo = new THREE.ExtrudeGeometry(shape, extrudeSettings);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.z = -h/2;
      g.add(mesh);

      // 内孔
      const holeGeo = new THREE.CylinderGeometry(d/2, d/2, h+0.1, SEG);
      g.add(new THREE.Mesh(holeGeo,
        new THREE.MeshStandardMaterial({color:0x1a1a2e, roughness:0.5, metalness:0.2})));
      break;
    }

    // ---- 垫圈 ----
    case 'washer': {
      const id = p.inner_d/2 || 1.6;
      const od = p.outer_d/2 || 3.5;
      const h = p.thickness || 0.5;
      const mat = new THREE.MeshStandardMaterial({color:0xc0c0c0, roughness:0.2, metalness:0.9});

      const shape = new THREE.Shape();
      const segs = SEG;
      for (let i = 0; i <= segs; i++) {
        const angle = (i/segs)*Math.PI*2;
        const x = Math.cos(angle);
        const y = Math.sin(angle);
        if (i===0) shape.moveTo(x*id, y*id);
        else shape.lineTo(x*id, y*id);
      }
      const hole = new THREE.Path();
      for (let i = 0; i <= segs; i++) {
        const angle = (i/segs)*Math.PI*2;
        if (i===0) hole.moveTo(Math.cos(angle)*od, Math.sin(angle)*od);
        else hole.lineTo(Math.cos(angle)*od, Math.sin(angle)*od);
      }
      shape.holes.push(hole);
      const geo = new THREE.ExtrudeGeometry(shape, {depth:h, bevelEnabled:true, bevelThickness:0.05, bevelSize:0.05, bevelSegments:6});
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.z = -h/2;
      g.add(mesh);
      break;
    }

    // ---- V2 零件库完整几何体路由 ----
    default: {
      buildV2Geometry(part, g);
    }
  }

  return g;
}

// ============================================================
// 缩略图渲染系统 — 单一共享 WebGLRenderer (避免多上下文崩溃)
// ============================================================
const TW = 180, TH = 180;

// 全局唯一的缩略图渲染器
const thumbRenderer = new THREE.WebGLRenderer({antialias:true, alpha:true, preserveDrawingBuffer:true});
thumbRenderer.setSize(TW, TH);
thumbRenderer.setPixelRatio(1);  // 缩略图无需高分
thumbRenderer.outputColorSpace = THREE.SRGBColorSpace;
thumbRenderer.toneMapping = THREE.ACESFilmicToneMapping;
thumbRenderer.toneMappingExposure = 1.2;

// 共享缩略图场景 + 灯光 + 相机 (只创建一次，复用)
const thumbScene = new THREE.Scene();
thumbScene.add(new THREE.AmbientLight(0x404060, 2.5));
const thumbKey = new THREE.DirectionalLight(0xffffff, 3);
thumbKey.position.set(5,8,5);
thumbScene.add(thumbKey);
const thumbFill = new THREE.DirectionalLight(0x4488ff, 1.5);
thumbFill.position.set(-3,2,-3);
thumbScene.add(thumbFill);
const thumbRim = new THREE.DirectionalLight(0xff8844, 1);
thumbRim.position.set(0,-1,5);
thumbScene.add(thumbRim);
const thumbCamera = new THREE.PerspectiveCamera(35, 1, 0.1, 200);
thumbCamera.position.set(20, 15, 25);
thumbCamera.lookAt(0,0,0);

// 已渲染缩略图缓存 (canvas 复用，避免创建额外 renderer)
const thumbDone = new Set();
const RENDER_QUEUE = [];
let rendering = false;
const MAX_CONCURRENT = 3;  // 串行渲染，一次一个

// 模态窗口全屏渲染器
function createModalRenderer(container) {
  const w = container.clientWidth;
  const h = container.clientHeight;
  const renderer = new THREE.WebGLRenderer({antialias:true, alpha:false});
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  renderer.shadowMap.enabled = true;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1117);

  // HDR 模拟环境光
  const ambient = new THREE.AmbientLight(0x404060, 3);
  scene.add(ambient);
  const hemi = new THREE.HemisphereLight(0xddeeff, 0x0d1117, 1.5);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 5);
  key.position.set(8,12,8);
  key.castShadow = true;
  key.shadow.mapSize.set(1024,1024);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x4488ff, 2);
  fill.position.set(-5,3,-5);
  scene.add(fill);
  const rim = new THREE.DirectionalLight(0xff8844, 1.5);
  rim.position.set(0,-2,8);
  scene.add(rim);

  const camera = new THREE.PerspectiveCamera(40, w/h, 0.1, 500);
  camera.position.set(30, 20, 35);
  camera.lookAt(0,0,0);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.0;
  controls.minDistance = 0.5;
  controls.maxDistance = 500;

  return {renderer, scene, camera, controls};
}

// ============================================================
// 构建 UI
// ============================================================
const categories = [...new Set(ALL_PARTS.map(p=>p.category))];
const catNames = {
  actuator:'执行器', transmission:'传动系统', fastener:'紧固件',
  structure:'结构件', energy:'能源系统', sensors:'传感系统',
  controllers:'控制系统', connectors:'连接件', misc:'其他',
};

// 过滤按钮
const filterBar = document.getElementById('filterBar');
let activeFilter = 'all';

function renderFilters() {
  filterBar.innerHTML = '<button class="filter-btn active" data-cat="all">全部 ('+ALL_PARTS.length+')</button>';
  categories.forEach(cat => {
    const count = ALL_PARTS.filter(p=>p.category===cat).length;
    const btn = document.createElement('button');
    btn.className = 'filter-btn';
    btn.dataset.cat = cat;
    btn.textContent = catNames[cat] || cat + ' ('+count+')';
    btn.onclick = () => {
      document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      activeFilter = cat;
      renderGrid();
    };
    filterBar.appendChild(btn);
  });
  document.querySelector('.filter-btn[data-cat="all"]').onclick = () => {
    document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
    document.querySelector('.filter-btn[data-cat="all"]').classList.add('active');
    activeFilter = 'all';
    renderGrid();
  };
}

// ============================================================
// 缩略图渲染 — 单一渲染器 + IntersectionObserver 懒加载
// ============================================================
function disposeGroup(group) {
  group.traverse(child => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) {
        child.material.forEach(m => m.dispose());
      } else {
        child.material.dispose();
      }
    }
  });
}

function renderOneThumbnail(part, targetCanvas) {
  if (thumbDone.has(part.id)) return;
  thumbDone.add(part.id);

  targetCanvas.width = TW; targetCanvas.height = TH;

  const model = createPartGeometry(part);
  thumbScene.add(model);

  // 居中到原点，相机用固定距离（保持相对大小对比）
  const box = new THREE.Box3().setFromObject(model);
  const center = box.getCenter(new THREE.Vector3());
  model.position.sub(center);
  thumbCamera.position.set(45, 30, 50);
  thumbCamera.lookAt(0, 0, 0);

  thumbRenderer.render(thumbScene, thumbCamera);

  // 从渲染器 canvas 复制到目标 canvas
  const ctx = targetCanvas.getContext('2d');
  ctx.drawImage(thumbRenderer.domElement, 0, 0);

  // 清理模型
  thumbScene.remove(model);
  disposeGroup(model);
}

// 任务队列串行执行
function processQueue() {
  if (rendering || RENDER_QUEUE.length === 0) return;
  rendering = true;
  const batch = RENDER_QUEUE.splice(0, MAX_CONCURRENT);
  batch.forEach(({part, canvas}) => {
    try { renderOneThumbnail(part, canvas); } catch(e) {}
  });
  rendering = false;
  if (RENDER_QUEUE.length > 0) {
    requestAnimationFrame(processQueue);
  }
}

function scheduleRender(part, canvas) {
  RENDER_QUEUE.push({part, canvas});
  if (!rendering) requestAnimationFrame(processQueue);
}

// IntersectionObserver 懒加载
let observer;
function setupObserver() {
  if (observer) observer.disconnect();
  observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const canvas = entry.target.querySelector('canvas');
        const partId = entry.target.dataset.pid;
        const part = ALL_PARTS.find(p => p.id === partId);
        if (part && canvas && !thumbDone.has(partId)) {
          scheduleRender(part, canvas);
          observer.unobserve(entry.target);
        }
      }
    });
  }, { rootMargin: '200px' });
}

// 网格渲染
function renderGrid() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';

  const filtered = activeFilter === 'all' ? ALL_PARTS : ALL_PARTS.filter(p=>p.category===activeFilter);
  document.getElementById('stats').textContent = filtered.length + ' / ' + ALL_PARTS.length + ' 个零件';

  setupObserver();

  filtered.forEach((part) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.dataset.pid = part.id;

    const canvas = document.createElement('canvas');
    card.appendChild(canvas);

    const info = document.createElement('div');
    info.className = 'info';
    info.innerHTML = `<div class="name">${part.name}</div><div class="cat">${catNames[part.category]||part.category} · ${part.type}</div>`;
    card.appendChild(info);

    card.onclick = () => openModal(part, filtered);
    grid.appendChild(card);

    observer.observe(card);
  });
}

// 模态窗口
let modalPartIndex = -1;
let modalParts = [];
let modalRenderer = null;
let modalScene = null;
let modalCamera = null;
let modalControls = null;
let modalModel = null;
let animId = null;

function openModal(part, partsList) {
  modalPartIndex = partsList.indexOf(part);
  modalParts = partsList;

  document.getElementById('modal').classList.add('active');
  document.getElementById('modalName').textContent = part.name;
  document.getElementById('modalSpecs').textContent = (catNames[part.category]||part.category) + ' · ' + part.type;
  document.getElementById('navLabel').textContent = (modalPartIndex+1) + ' / ' + modalParts.length;

  const viewer = document.getElementById('modalViewer');
  viewer.innerHTML = '';

  const mr = createModalRenderer(viewer);
  modalRenderer = mr.renderer;
  modalScene = mr.scene;
  modalCamera = mr.camera;
  modalControls = mr.controls;

  loadModalPart(part);

  function animate() {
    animId = requestAnimationFrame(animate);
    modalControls.update();
    modalRenderer.render(modalScene, modalCamera);
  }
  animate();
}

function loadModalPart(part) {
  if (modalModel) modalScene.remove(modalModel);
  modalModel = createPartGeometry(part);
  modalScene.add(modalModel);

  const box = new THREE.Box3().setFromObject(modalModel);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  modalCamera.position.set(maxDim*3, maxDim*2.5, maxDim*3.6);
  modalControls.target.copy(center);
  modalModel.position.sub(center);
  modalControls.update();

  document.getElementById('modalName').textContent = part.name;
  document.getElementById('navLabel').textContent = (modalPartIndex+1) + ' / ' + modalParts.length;
}

function closeModal() {
  document.getElementById('modal').classList.remove('active');
  if (animId) cancelAnimationFrame(animId);
  if (modalModel) { modalScene.remove(modalModel); disposeGroup(modalModel); }
  if (modalRenderer) { modalRenderer.dispose(); modalRenderer.forceContextLoss(); }
  if (modalControls) modalControls.dispose();
  modalRenderer = modalScene = modalCamera = modalControls = modalModel = null;
  document.getElementById('modalViewer').innerHTML = '';
}

document.getElementById('modalClose').onclick = closeModal;
document.getElementById('modal').onclick = (e) => {
  if (e.target === document.getElementById('modal')) closeModal();
};

document.getElementById('btnPrev').onclick = () => {
  if (modalPartIndex > 0) {
    modalPartIndex--;
    loadModalPart(modalParts[modalPartIndex]);
  }
};
document.getElementById('btnNext').onclick = () => {
  if (modalPartIndex < modalParts.length-1) {
    modalPartIndex++;
    loadModalPart(modalParts[modalPartIndex]);
  }
};

// 键盘导航
document.addEventListener('keydown', (e) => {
  if (!document.getElementById('modal').classList.contains('active')) return;
  if (e.key === 'Escape') closeModal();
  if (e.key === 'ArrowLeft' && modalPartIndex > 0) {
    modalPartIndex--;
    loadModalPart(modalParts[modalPartIndex]);
  }
  if (e.key === 'ArrowRight' && modalPartIndex < modalParts.length-1) {
    modalPartIndex++;
    loadModalPart(modalParts[modalPartIndex]);
  }
});

// 窗口大小变化
window.addEventListener('resize', () => {
  if (!modalRenderer) return;
  const viewer = document.getElementById('modalViewer');
  modalRenderer.setSize(viewer.clientWidth, viewer.clientHeight);
  modalCamera.aspect = viewer.clientWidth / viewer.clientHeight;
  modalCamera.updateProjectionMatrix();
});

// ============================================================
// 启动
// ============================================================
renderFilters();
renderGrid();
</script>
</body>
</html>
'''

out_path = BASE / "part_library_previews_v2" / "index_pbr.html"
out_path.write_text(html, encoding="utf-8")
print(f"HTML saved: {out_path}")
print(f"Size: {len(html)/1024:.0f} KB")
