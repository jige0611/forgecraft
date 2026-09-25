"""
ForgeCraft Web 仪表盘

用法：
  python -m forgecraft.dashboard
  python -m forgecraft.dashboard --port 8080
  python -m forgecraft.dashboard --evolution-file evolution_history.json --body-file best_body.json
"""

import json

__all__ = ["app", "run_dashboard"]
import os
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent
_TEMPLATE_DIR = _THIS_DIR / "templates"

# ── Dashboard Data Pipeline ──
# 1. generate_dashboard_data() → JSON 聚合 evolution+bodi json
# 2. render_dashboard_html()  → HTML 模板渲染
# 3. run_dashboard()          → http.server 静态服务


def generate_dashboard_data(
    evolution_file: Optional[str] = None,
    body_file: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evolution": None,
        "best_body": None,
        "exported_files": [],
    }

    if evolution_file and os.path.exists(evolution_file):
        with open(evolution_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        history = []
        for entry in raw:
            history.append({
                "generation": entry.get("generation", 0),
                "best_fitness": entry.get("best_fitness", 0),
                "mean_fitness": entry.get("mean_fitness", 0),
                "median_fitness": entry.get("median_fitness", 0),
                "worst_fitness": entry.get("worst_fitness", 0),
                "std_fitness": entry.get("std_fitness", 0),
            })
        data["evolution"] = history

    if body_file and os.path.exists(body_file):
        with open(body_file, "r", encoding="utf-8") as f:
            body_data = json.load(f)
        data["best_body"] = {
            "name": body_data.get("name", "?"),
            "fitness": body_data.get("fitness", 0),
            "num_parts": body_data.get("num_parts", 0),
            "num_joints": body_data.get("num_joints", 0),
            "fitness_components": body_data.get("fitness_components", {}),
            "parts": body_data.get("parts", []),
            "joints": body_data.get("joints", []),
        }

    if output_dir and os.path.isdir(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, output_dir)
                size = os.path.getsize(fpath)
                ext = os.path.splitext(fname)[1]
                ftype = ""
                if ext == ".stl":
                    ftype = "3D模型"
                elif ext == ".urdf":
                    ftype = "机器人描述"
                elif ext == ".onnx":
                    ftype = "AI策略"
                elif ext == ".cpp":
                    ftype = "C++代码"
                elif ext == ".json":
                    ftype = "数据文件"
                elif ext == ".md":
                    ftype = "文档"
                data["exported_files"].append({
                    "name": fname,
                    "path": rel,
                    "size": size,
                    "type": ftype,
                })

    return data


def render_dashboard_html(data: dict, standalone: bool = True) -> str:
    data_json = json.dumps(data, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ForgeCraft · 机械形态进化仪表盘</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script type="importmap">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.157.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.157.0/examples/jsm/"
  }}
}}
</script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', 'PingFang SC', Microsoft YaHei, sans-serif; background: #0a0e17; color: #c9d1d9; }}
.header {{ background: linear-gradient(135deg, #1a1f2e 0%, #161b22 100%); padding: 24px 32px; border-bottom: 1px solid #21262d; }}
.header h1 {{ font-size: 28px; font-weight: 700; background: linear-gradient(90deg, #58a6ff, #3fb950); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.header .subtitle {{ color: #8b949e; font-size: 14px; margin-top: 4px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; padding: 20px 32px; }}
.stat-card {{ background: #161b22; border: 1px solid #21262d; border-radius: 12px; padding: 20px; }}
.stat-card .label {{ color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }}
.stat-card .value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
.stat-card .value.green {{ color: #3fb950; }}
.stat-card .value.blue {{ color: #58a6ff; }}
.stat-card .value.orange {{ color: #d2991d; }}
.stat-card .value.purple {{ color: #a371f7; }}
.main-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 0 32px 20px; }}
@media (max-width: 900px) {{ .main-grid {{ grid-template-columns: 1fr; }} }}
.panel {{ background: #161b22; border: 1px solid #21262d; border-radius: 12px; overflow: hidden; }}
.panel-header {{ padding: 16px 20px; border-bottom: 1px solid #21262d; font-size: 16px; font-weight: 600; display: flex; align-items: center; gap: 8px; }}
.panel-body {{ padding: 20px; }}
.chart-container {{ position: relative; height: 300px; }}
.chart-container canvas {{ width: 100% !important; }}
.viewer-container {{ width: 100%; height: 400px; background: #0d1117; border-radius: 8px; overflow: hidden; }}
.full-width {{ grid-column: 1 / -1; }}
.file-list {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.file-tag {{ background: #21262d; border: 1px solid #30363d; border-radius: 8px; padding: 8px 14px; font-size: 13px; display: flex; align-items: center; gap: 6px; }}
.file-tag .ftype {{ font-size: 11px; color: #8b949e; background: #0d1117; padding: 2px 6px; border-radius: 4px; }}
.tree {{ font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 13px; line-height: 1.8; white-space: pre; overflow-x: auto; }}
.tree .root {{ color: #58a6ff; }}
.tree .joint {{ color: #d2991d; }}
.tree .part {{ color: #3fb950; }}
.tree .actuator {{ color: #f778ba; }}
.progress-bar {{ height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; margin-top: 6px; }}
.progress-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
.comp-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }}
.comp-item {{ background: #21262d; border-radius: 8px; padding: 12px; text-align: center; }}
.comp-item .comp-label {{ font-size: 11px; color: #8b949e; }}
.comp-item .comp-value {{ font-size: 18px; font-weight: 600; margin-top: 4px; }}
.part-list {{ max-height: 300px; overflow-y: auto; }}
.part-row {{ display: grid; grid-template-columns: 24px 1fr 80px 80px; gap: 8px; padding: 8px 0; border-bottom: 1px solid #21262d; align-items: center; font-size: 13px; }}
.part-row .part-icon {{ width: 20px; height: 20px; border-radius: 4px; }}
.footer {{ text-align: center; padding: 20px; color: #484f58; font-size: 12px; border-top: 1px solid #21262d; margin: 0 32px; }}
</style>
</head>
<body>

<div class="header">
  <h1>🔧 ForgeCraft 机械形态进化仪表盘</h1>
  <div class="subtitle">基于强化学习 + 进化算法的机械形态共进化系统 · 生成于 {data["generated_at"]}</div>
</div>

<div class="grid" id="stats-cards"></div>

<div class="main-grid">
  <div class="panel">
    <div class="panel-header">📈 适应度进化曲线</div>
    <div class="panel-body"><div class="chart-container"><canvas id="fitnessChart"></canvas></div></div>
  </div>
  <div class="panel">
    <div class="panel-header">📊 适应度分布</div>
    <div class="panel-body"><div class="chart-container"><canvas id="distributionChart"></canvas></div></div>
  </div>
  <div class="panel">
    <div class="panel-header">🔬 3D 形态预览</div>
    <div class="panel-body"><div class="viewer-container" id="viewer3d"></div></div>
  </div>
  <div class="panel">
    <div class="panel-header">🌳 机械结构树</div>
    <div class="panel-body"><div class="tree" id="bodyTree"></div></div>
  </div>
  <div class="panel full-width">
    <div class="panel-header">📦 制造导出文件</div>
    <div class="panel-body"><div class="file-list" id="fileList"></div></div>
  </div>
</div>

<div class="footer">ForgeCraft v0.3.0 · Reinforcement Learning + Evolutionary Morphology Co-Evolution</div>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const DATA = {data_json};

function renderStats() {{
    const body = DATA.best_body;
    const evo = DATA.evolution;
    const lastGen = evo ? evo[evo.length - 1] : null;

    const cards = [
        {{ label: '最佳适应度', value: body ? body.fitness.toFixed(4) : '-', cls: 'green' }},
        {{ label: '零件数', value: body ? body.num_parts : '-', cls: 'blue' }},
        {{ label: '驱动关节', value: body ? body.num_joints : '-', cls: 'orange' }},
        {{ label: '进化代数', value: evo ? evo.length : '-', cls: 'purple' }},
        {{ label: '形态名称', value: body ? body.name : '-', cls: 'blue' }},
        {{ label: '最终平均适应度', value: lastGen ? lastGen.mean_fitness.toFixed(4) : '-', cls: 'orange' }},
        {{ label: '导出文件数', value: DATA.exported_files.length, cls: 'purple' }},
        {{ label: '适应度标准差', value: lastGen ? lastGen.std_fitness ? lastGen.std_fitness.toFixed(4) : '-' : '-', cls: 'green' }},
    ];

    document.getElementById('stats-cards').innerHTML = cards.map(c =>
        `<div class="stat-card"><div class="label">${{c.label}}</div><div class="value ${{c.cls}}">${{c.value}}</div></div>`
    ).join('');
}}

function renderCharts() {{
    const evo = DATA.evolution;
    if (!evo || evo.length === 0) return;

    const labels = evo.map(e => 'Gen ' + e.generation);
    const best = evo.map(e => e.best_fitness);
    const mean = evo.map(e => e.mean_fitness);
    const median = evo.map(e => e.median_fitness);
    const worst = evo.map(e => e.worst_fitness);

    new Chart(document.getElementById('fitnessChart'), {{
        type: 'line',
        data: {{
            labels: labels,
            datasets: [
                {{ label: '最佳', data: best, borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.1)', fill: true, tension: 0.3, borderWidth: 2 }},
                {{ label: '中位数', data: median, borderColor: '#58a6ff', borderDash: [5,5], tension: 0.3, borderWidth: 1.5 }},
                {{ label: '平均', data: mean, borderColor: '#d2991d', borderDash: [2,2], tension: 0.3, borderWidth: 1.5 }},
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }},
            scales: {{
                x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
                y: {{ ticks: {{ color: '#8b949e', callback: v => v.toFixed(3) }}, grid: {{ color: '#21262d' }} }},
            }}
        }}
    }});

    const lastData = evo[evo.length - 1];
    new Chart(document.getElementById('distributionChart'), {{
        type: 'bar',
        data: {{
            labels: ['最佳', '平均', '中位数', '标准差'],
            datasets: [{{
                data: [lastData.best_fitness, lastData.mean_fitness, lastData.median_fitness, lastData.std_fitness || 0],
                backgroundColor: ['#3fb950', '#d2991d', '#58a6ff', '#a371f7'],
                borderRadius: 4,
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
                x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
                y: {{ ticks: {{ color: '#8b949e', callback: v => v.toFixed(3) }}, grid: {{ color: '#21262d' }} }},
            }}
        }}
    }});
}}

function render3D() {{
    try {{
        const container = document.getElementById('viewer3d');
        const w = container.clientWidth, h = container.clientHeight;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0d1117);

        const camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 100);
        camera.position.set(0.3, 0.25, 0.4);
        camera.lookAt(0, 0.1, 0);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(w, h);
        renderer.shadowMap.enabled = true;
        container.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.target.set(0, 0.05, 0);

        scene.add(new THREE.AmbientLight(0x404060, 2));
        const dirLight = new THREE.DirectionalLight(0xffffff, 3);
        dirLight.position.set(1, 1, 0.5);
        scene.add(dirLight);

        const grid = new THREE.GridHelper(0.4, 20, 0x30363d, 0x21262d);
        grid.position.y = -0.02;
        scene.add(grid);

        const body = DATA.best_body;
        if (body && body.parts) {{
            const colors = {{
                'main_body': 0x4488cc, 'base': 0x4466aa, 'chassis': 0x5566aa,
                'hip_motor': 0xcc8844, 'knee_motor': 0xcc8844, 'hinge_joint': 0xcc8844,
                'upper_leg': 0x44aa66, 'lower_leg': 0x44aa66, 'link': 0x44aa66,
                'foot_pad': 0xaa6644, 'foot': 0xaa6644, 'wheel_drive': 0x888888,
                'wheel': 0x888888, 'brick': 0x777799,
                'structure': 0x666a70, 'actuator': 0xcc5533, 'contact': 0x333333,
            }};

            body.parts.forEach(part => {{
                const pos = part.position || [0,0,0];
                const params = part.params || {{}};
                const length = params.length || 0.08;
                const radius = params.radius || 0.03;
                const thickness = params.thickness || 0.02;
                const color = colors[part.part_type] || 0x888888;

                let geometry;
                let mesh;

                if (part.part_type === 'structure') {{
                    geometry = new THREE.CylinderGeometry(radius, radius, length, 24);
                    mesh = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({{ 
                        color: color,
                        shininess: 30
                    }}));
                    mesh.position.set(pos[0], pos[1], pos[2]);
                    scene.add(mesh);
                }} else if (part.part_type === 'actuator') {{
                    geometry = new THREE.CylinderGeometry(radius, radius, length, 24);
                    mesh = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({{ 
                        color: color,
                        emissive: 0x331100,
                        shininess: 50
                    }}));
                    mesh.position.set(pos[0], pos[1], pos[2]);
                    scene.add(mesh);
                }} else if (part.part_type === 'contact') {{
                    geometry = new THREE.SphereGeometry(radius, 16, 16);
                    mesh = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({{ 
                        color: color,
                        shininess: 80
                    }}));
                    mesh.position.set(pos[0], pos[1], pos[2]);
                    scene.add(mesh);
                }} else if (part.part_type.includes('wheel')) {{
                    geometry = new THREE.CylinderGeometry(radius, radius, thickness, 24);
                    mesh = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({{ color }}));
                    mesh.rotation.x = Math.PI / 2;
                    mesh.position.set(pos[0], pos[1], pos[2]);
                    scene.add(mesh);
                }} else {{
                    const w = length * (part.part_type.includes('body') ? 1.2 : 1.0);
                    const h = thickness;
                    const d = length;
                    geometry = new THREE.BoxGeometry(w, h, d);
                    mesh = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({{ color }}));
                    mesh.position.set(pos[0], pos[1], pos[2]);
                    scene.add(mesh);
                }}
            }});

            if (body.joints) {{
                body.joints.forEach(joint => {{
                    const pParts = body.parts || [];
                    const parent = pParts.find(p => p.part_id === joint.parent_id);
                    const child = pParts.find(p => p.part_id === joint.child_id);
                    if (parent && child) {{
                        const pp = parent.position || [0,0,0];
                        const cp = child.position || [0,0,0];
                        const distance = Math.sqrt(
                            Math.pow(cp[0] - pp[0], 2) +
                            Math.pow(cp[1] - pp[1], 2) +
                            Math.pow(cp[2] - pp[2], 2)
                        );
                        const safeDistance = Math.max(distance, 0.001);
                        const geo = new THREE.CylinderGeometry(0.002, 0.002, safeDistance, 8);
                        const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({{ 
                            color: 0xffaa00,
                            transparent: true,
                            opacity: 0.6
                        }}));
                        const mid = [(pp[0]+cp[0])/2, (pp[1]+cp[1])/2, (pp[2]+cp[2])/2];
                        mesh.position.set(mid[0], mid[1], mid[2]);
                        const dir = new THREE.Vector3(cp[0]-pp[0], cp[1]-pp[1], cp[2]-pp[2]);
                        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
                        scene.add(mesh);
                    }}
                }});
            }}
        }}

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            const w2 = container.clientWidth, h2 = container.clientHeight;
            camera.aspect = w2 / Math.max(h2, 1);
            camera.updateProjectionMatrix();
            renderer.setSize(w2, h2);
        }});
    }} catch(e) {{ document.getElementById('viewer3d').innerHTML = '<div style="padding:40px;color:#8b949e">3D 渲染加载中...</div>'; }}
}}

function renderBodyTree() {{
    const body = DATA.best_body;
    if (!body || !body.parts) return;

    const parts = body.parts;
    const joints = body.joints || [];

    const partMap = {{}};
    parts.forEach(p => {{ partMap[p.part_id] = p; }});

    const children = {{}};
    parts.forEach(p => {{ children[p.part_id] = []; }});
    joints.forEach(j => {{
        if (children[j.parent_id]) children[j.parent_id].push(j.child_id);
    }});

    const root = parts.find(p => !joints.some(j => j.child_id === p.part_id));
    if (!root) return;

    let html = '';
    function renderNode(pid, depth) {{
        const p = partMap[pid];
        if (!p) return;
        const prefix = '  '.repeat(depth) + (depth > 0 ? '├─ ' : '');
        const actuated = (p.params || {{}}).actuated;
        const cls = depth === 0 ? 'root' : (actuated && actuated > 0.5 ? 'actuator' : 'part');
        const icon = depth === 0 ? '🏠' : (actuated && actuated > 0.5 ? '⚡' : '🟩');
        const pos = (p.position || [0,0,0]).map(v => v.toFixed(3)).join(', ');
        html += `<span class="${{cls}}">${{prefix}}${{icon}} ${{p.part_type}} <span style="color:#484f58">[${{pid.substring(0,6)}}]</span> <span style="color:#8b949e;font-size:11px">(${{pos}})</span></span>\n`;

        const kids = children[pid] || [];
        kids.forEach(cid => {{
            const j = joints.find(j => j.parent_id === pid && j.child_id === cid);
            if (j) {{
                html += `<span class="joint">${{'  '.repeat(depth+1)}}🔗 ${{j.joint_type}}</span>\n`;
            }}
            renderNode(cid, depth + 1);
        }});
    }}
    renderNode(root.part_id, 0);
    document.getElementById('bodyTree').innerHTML = html;
}}

function renderFiles() {{
    const files = DATA.exported_files;
    const typeIcons = {{ '3D模型': '🔷', '机器人描述': '🤖', 'AI策略': '🧠', 'C++代码': '⚙️', '数据文件': '📊', '文档': '📝' }};
    document.getElementById('fileList').innerHTML = files.map(f =>
        `<div class="file-tag"><span>${{typeIcons[f.type] || '📄'}}</span> <span>${{f.name}}</span> <span class="ftype">${{f.type}}</span> <span style="color:#484f58;font-size:11px">${{(f.size/1024).toFixed(1)}}KB</span></div>`
    ).join('');
}}

renderStats();
renderCharts();
render3D();
renderBodyTree();
renderFiles();
</script>
</body>
</html>"""


# ---- HTTP Server ----

WEB_DIR = _THIS_DIR.parent / "forgecraft_web"
os.makedirs(WEB_DIR, exist_ok=True)


class ForgeCraftHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = WEB_DIR / "dashboard.html"
            if html_path.exists():
                self.wfile.write(html_path.read_bytes())
            else:
                self.wfile.write(b"<h1>Dashboard not found. Run: python -m forgecraft.dashboard --generate</h1>")
            return
        super().do_GET()

    def log_message(self, format, *args):
        pass  # suppress logs


def serve(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), ForgeCraftHandler)
    print(f"\n  🔧 ForgeCraft 仪表盘已启动")
    print(f"  📡 http://localhost:{port}")
    print(f"  📡 http://127.0.0.1:{port}")
    print(f"\n  按 Ctrl+C 停止\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务器已停止")
        server.server_close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ForgeCraft Web 仪表盘")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--generate", action="store_true", help="生成仪表盘 HTML")
    parser.add_argument("--evolution-file", type=str, default="evolution_history.json")
    parser.add_argument("--body-file", type=str, default="best_body.json")
    parser.add_argument("--export-dir", type=str, default="design_output")

    args = parser.parse_args()

    # Try to run evolution first if no data exists
    if not os.path.exists(args.evolution_file) and not os.path.exists(args.body_file):
        print("未找到进化数据，运行快速进化...")
        old_args = sys.argv
        run_data = {
            "evolution_file": args.evolution_file,
            "body_file": args.body_file,
        }
        # Generate data
        try:
            import subprocess
            subprocess.run([
                sys.executable, "-m", "forgecraft.main",
                "--generations", "8", "--population", "6", "--workers", "6",
                "--quick", "--seed", "123",
                "--export", "--export-dir", args.export_dir,
            ], check=True, cwd=str(_THIS_DIR.parent))
        except Exception as e:
            print(f"进化运行失败: {e}，使用已有数据继续")

    data = generate_dashboard_data(
        evolution_file=args.evolution_file,
        body_file=args.body_file,
        output_dir=args.export_dir,
    )

    html = render_dashboard_html(data)
    html_path = WEB_DIR / "dashboard.html"
    html_path.write_text(html, encoding="utf-8")

    print(f"仪表盘已生成: {html_path}")
    print(f"数据来源: {args.evolution_file}, {args.body_file}")

    if args.generate:
        return

    serve(port=args.port)


if __name__ == "__main__":
    main()
