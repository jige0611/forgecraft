"""渲染进化产生的最佳形态 3D 视图"""
import trimesh
import numpy as np
import json
import os
from collections import Counter

# 加载 body JSON
with open('design_output/best_body.json') as f:
    body = json.load(f)

print('=== 进化最佳形态 ===')
print(f"名称: {body['name']}")
print(f"适应度: {body['fitness']:.4f}")
print(f"零件数: {body['num_parts']}")
print(f"关节数: {body['num_joints']}")
print()

# 零件类型统计
type_counts = Counter(p['part_type'] for p in body['parts'])
print('零件类型分布:')
for t, c in type_counts.most_common():
    print(f'  {t}: {c}')

print()
print('关节类型分布:')
joint_counts = Counter(j['joint_type'] for j in body['joints'])
for t, c in joint_counts.most_common():
    print(f'  {t}: {c}')

# 加载 STL 文件
stl_dir = 'design_output/stl'
stl_files = sorted(os.listdir(stl_dir))
print(f'\nSTL 文件: {len(stl_files)} 个')

# 渲染 3D: 按零件类型着色
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

colors = {
    'carbon_tube': '#444444',
    'micro_bearing': '#888888',
    'compact_lipo': '#ff6600',
    'high_power_lipo': '#cc0000',
    'launch_spring': '#3399ff',
    'alloy_chassis': '#aaaaaa',
    'brushless_motor': '#00cc66',
    'brushless_motor_compact': '#009944',
    'sprint_foot': '#ffcc00',
}

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#1a1a2e')

for p in body['parts']:
    pid = p['part_id']
    ptype = p['part_type']
    stl_path = os.path.join(stl_dir, f'{pid}.stl')
    if not os.path.exists(stl_path):
        print(f'  [跳过] {stl_path} 不存在')
        continue
    
    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    color = colors.get(ptype, '#666666')
    
    print(f'  渲染 {pid} ({ptype}): {len(verts)} 顶点, {len(faces)} 面')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                     triangles=faces, color=color, alpha=0.85,
                     edgecolor='none', linewidth=0, antialiased=True)

ax.set_xlabel('X (m)', color='white', fontsize=12)
ax.set_ylabel('Y (m)', color='white', fontsize=12)
ax.set_zlabel('Z (m)', color='white', fontsize=12)
ax.tick_params(colors='white')
ax.set_title(
    f"ForgeCraft Best: {body['name']}  |  "
    f"Fitness={body['fitness']:.3f}  |  "
    f"{body['num_parts']} parts / {body['num_joints']} joints",
    color='white', fontsize=14)

# 图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=t.replace('_', ' ').title())
                   for t, c in sorted(colors.items()) if t in type_counts]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
          facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

ax.view_init(elev=25, azim=-60)
plt.tight_layout()
outpath = 'design_output/best_body_render.png'
plt.savefig(outpath, dpi=150, facecolor='#1a1a2e')
print(f'\n3D 渲染已保存: {outpath}')
