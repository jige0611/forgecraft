import json
import numpy as np

data = json.load(open('design_output_perfect_v7/best_body.json', 'r', encoding='utf-8'))

print("=" * 70)
print("  Spatial Analysis - Finding Ground Penetration")
print("=" * 70)

def get_part_bounds(part):
    """Calculate actual bounding box of a part considering its geometry"""
    pos = np.array(part['position'])
    params = part['params']
    ptype = part['part_type']

    if ptype == 'structure':
        length = params.get('length', 0)
        radius = params.get('radius', 0)
        # Cylinder along X axis (after rotation)
        dx = length / 2
        dy = radius
        dz = radius
    elif ptype == 'contact':
        radius = params.get('radius', 0)
        dx = dy = dz = radius
    elif ptype == 'actuator':
        length = params.get('length', 0)
        radius = params.get('radius', 0) * 1.8  # Housing is larger
        dx = length / 2
        dy = radius
        dz = radius
    else:
        dx = dy = dz = 0.05

    min_pt = pos - np.array([dx, dy, dz])
    max_pt = pos + np.array([dx, dy, dz])

    return min_pt, max_pt

print("\n[PART BOUNDING BOXES]")
global_min = np.array([999, 999, 999])
global_max = np.array([-999, -999, -999])
ground_penetrating_parts = []

for i, part in enumerate(data['parts']):
    min_pt, max_pt = get_part_bounds(part)
    global_min = np.minimum(global_min, min_pt)
    global_max = np.maximum(global_max, max_pt)

    if min_pt[2] < 0.01:  # Near or below ground
        ground_penetrating_parts.append({
            'id': part['part_id'],
            'type': part['part_type'],
            'position': part['position'],
            'min_z': min_pt[2],
            'max_z': max_pt[2],
            'penetration': abs(min(0, min_pt[2]))
        })

    print(f"  #{i+1} [{part['part_id'][:8]}] {part['part_type']:10s} | "
          f"pos=({part['position'][0]:6.3f}, {part['position'][1]:6.3f}, {part['position'][2]:6.3f}) | "
          f"bounds Z:[{min_pt[2]:6.3f}, {max_pt[2]:6.3f}]")

print(f"\n[GLOBAL BOUNDS]")
print(f"  X: [{global_min[0]:.3f}, {global_max[0]:.3f}]  size={global_max[0]-global_min[0]:.3f}m")
print(f"  Y: [{global_min[1]:.3f}, {global_max[1]:.3f}]  size={global_max[1]-global_min[1]:.3f}m")
print(f"  Z: [{global_min[2]:.3f}, {global_max[2]:.3f}]  size={global_max[2]-global_min[2]:.3f}m")

center = (global_min + global_max) / 2
print(f"\n[CENTER OF MASS] ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")

if ground_penetrating_parts:
    print(f"\n⚠️  GROUND PENETRATION DETECTED! ({len(ground_penetrating_parts)} parts)")
    for p in ground_penetrating_parts:
        print(f"   [{p['id'][:8]}] {p['type']:10s} penetration={p['penetration']:.3f}m | bottom at Z={p['min_z']:.3f}")

    offset_z = -global_min[2] + 0.005  # Small margin above ground
    print(f"\n[SOLUTION] Shift entire robot UP by {offset_z:.3f}m")

    print("\n[CORRECTED POSITIONS]")
    for part in data['parts']:
        new_pos = [round(part['position'][0], 4),
                   round(part['position'][1], 4),
                   round(part['position'][2] + offset_z, 4)]
        print(f"  {part['part_id'][:8]}: ({part['position'][0]:6.3f}, {part['position'][1]:6.3f}, {part['position'][2]:6.3f}) -> ({new_pos[0]:6.3f}, {new_pos[1]:6.3f}, {new_pos[2]:6.3f})")

    # Also fix joints
    print("\n[CORRECTED JOINT ANCHORS]")
    for joint in data['joints']:
        old_anchor = joint['anchor']
        new_anchor = [round(old_anchor[0] + 0, 4),  # X unchanged
                      round(old_anchor[1] + 0, 4),  # Y unchanged
                      round(old_anchor[2] + offset_z, 4)]  # Z shifted
        print(f"  {joint['parent_id'][:6]}->{joint['child_id'][:6]}: anchor Z {old_anchor[2]:.3f} -> {new_anchor[2]:.3f}")

else:
    print(f"\n✅ No ground penetration detected (lowest point at Z={global_min[2]:.3f}m)")

# Find contacts closest to ground
contacts = [(p, get_part_bounds(p)) for p in data['parts'] if p['part_type'] == 'contact']
contacts.sort(key=lambda x: x[1][0][2])  # Sort by minimum Z

print(f"\n[CONTACT POINTS - Sorted by height]")
for p, (mn, mx) in contacts[:5]:
    print(f"  {p['part_id'][:8]} R={p['params']['radius']:.3f}m | bottom Z={mn[2]:.3f}m | top Z={mx[2]:.3f}m")
