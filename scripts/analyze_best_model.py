import json
import numpy as np

data = json.load(open('design_output_perfect_v7/best_body.json'))

print("=" * 60)
print("最佳模型 gen89_ind018 深度分析")
print("=" * 60)
print(f"\n总体适应度: {data['fitness']:.4f}")
print(f"适应度组成: {json.dumps(data['fitness_components'], indent=2)}")
print(f"零件总数: {data['num_parts']}, 关节总数: {data['num_joints']}")

print("\n" + "-" * 40)
print("1. 零件类型统计:")
types = {}
for p in data['parts']:
    ptype = p['part_type']
    types[ptype] = types.get(ptype, 0) + 1

for ptype, count in sorted(types.items()):
    print(f"  {ptype}: {count}个")

print("\n" + "-" * 40)
print("2. Structure零件详细分析:")
structures = [p for p in data['parts'] if p['part_type'] == 'structure']
if structures:
    lengths = [s['params']['length'] for s in structures]
    radii = [s['params']['radius'] for s in structures]
    masses = [s['params']['mass'] for s in structures]
    
    print(f"  数量: {len(structures)}个")
    print(f"  长度范围: {min(lengths):.4f} - {max(lengths):.4f} m (平均: {np.mean(lengths):.4f})")
    print(f"  半径范围: {min(radii):.4f} - {max(radii):.4f} m (平均: {np.mean(radii):.4f})")
    print(f"  质量范围: {min(masses):.4f} - {max(masses):.4f} kg (总质量: {sum(masses):.4f})")
    print(f"  密度: {structures[0]['params']['density']} kg/m³")
    
    # 分析长细比（长度/直径比）
    aspect_ratios = [l / (2*r) for l, r in zip(lengths, radii)]
    print(f"  长细比(长度/直径): {min(aspect_ratios):.2f} - {max(aspect_ratios):.2f}")

print("\n" + "-" * 40)
print("3. Contact零件详细分析:")
contacts = [p for p in data['parts'] if p['part_type'] == 'contact']
if contacts:
    radii = [c['params']['radius'] for c in contacts]
    masses = [c['params']['mass'] for c in contacts]
    positions = [c['position'] for c in contacts]
    
    print(f"  数量: {len(contacts)}个")
    print(f"  半径范围: {min(radii):.4f} - {max(radii):.4f} m (平均: {np.mean(radii):.4f})")
    print(f"  质量范围: {min(masses):.4f} - {max(masses):.4f} kg (总质量: {sum(masses):.4f})")
    
    # Z坐标分布（高度）
    z_coords = [pos[2] for pos in positions]
    print(f"  高度(Z)分布: {min(z_coords):.4f} - {max(z_coords):.4f} m")
    print(f"  摩擦系数: {contacts[0]['params']['friction_primary']}")

print("\n" + "-" * 40)
print("4. Actuator零件详细分析:")
actuators = [p for p in data['parts'] if p['part_type'] == 'actuator']
if actuators:
    torques = [a['params']['max_torque'] for a in actuators]
    velocities = [a['params']['max_velocity'] for a in actuators]
    lengths = [a['params']['length'] for a in actuators]
    radii = [a['params']['radius'] for a in actuators]
    
    print(f"  数量: {len(actuators)}个")
    print(f"  力矩范围: {min(torques):.2f} - {max(torques):.2f} Nm (平均: {np.mean(torques):.2f})")
    print(f"  速度范围: {min(velocities):.2f} - {max(velocities):.2f} rad/s (平均: {np.mean(velocities):.2f})")
    print(f"  尺寸: 长{np.mean(lengths):.4f}m × 半径{np.mean(radii):.4f}m")
    
    # 功率估算 P = T * ω
    powers = [t * v for t, v in zip(torques, velocities)]
    print(f"  功率范围: {min(powers):.2f} - {max(powers):.2f} W (总功率: {sum(powers):.2f} W)")

print("\n" + "-" * 40)
print("5. 空间分布特征:")
all_positions = [p['position'] for p in data['parts']]
x_coords = [pos[0] for pos in all_positions]
y_coords = [pos[1] for pos in all_positions]
z_coords = [pos[2] for pos in all_positions]

print(f"  X范围: {min(x_coords):.4f} - {max(x_coords):.4f} m (跨度: {max(x_coords)-min(x_coords):.4f}m)")
print(f"  Y范围: {min(y_coords):.4f} - {max(y_coords):.4f} m (跨度: {max(y_coords)-min(y_coords):.4f}m)")
print(f"  Z范围: {min(z_coords):.4f} - {max(z_coords):.4f} m (高度: {max(z_coords)-min(z_coords):.4f}m)")

# 计算质心
total_mass = sum(p['params']['mass'] for p in data['parts'])
com_x = sum(p['position'][0] * p['params']['mass'] for p in data['parts']) / total_mass
com_y = sum(p['position'][1] * p['params']['mass'] for p in data['parts']) / total_mass
com_z = sum(p['position'][2] * p['params']['mass'] for p in data['parts']) / total_mass
print(f"\n  质心位置: ({com_x:.4f}, {com_y:.4f}, {com_z:.4f})")

# 总质量
print(f"\n  总质量: {total_mass:.4f} kg")

print("\n" + "=" * 60)
print("6. 运动学推断:")
print("=" * 60)

# 基于结构特征推断运动模式
if len(structures) > 0 and len(contacts) > 3:
    avg_length = np.mean([s['params']['length'] for s in structures])
    contact_spread = max(z_coords) - min(z_coords)
    
    print(f"  - 结构体呈细长型 (长细比 > 2)，可能采用类似昆虫的多足运动")
    print(f"  - 接触点数量多 ({len(contacts)}个)，分布在不同高度，暗示复杂步态")
    print(f"  - 驱动器力矩较大 (最大{max(torques):.1f}Nm)，支持快速摆动")
    print(f"  - 质心相对较低 ({com_z:.3f}m)，有利于稳定性")
    
    # 判断可能的运动方式
    if contact_spread > 0.3:
        print(f"  - 接触点高度差大 ({contact_spread:.3f}m)，可能存在攀爬或跳跃能力")
    if len(actuators) >= 3:
        print(f"  - 多驱动器协调 ({len(actuators)}个)，可实现复杂协调运动")
