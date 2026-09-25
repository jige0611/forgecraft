import json
import numpy as np

data = json.load(open('design_output_perfect_v7/best_body.json', 'r', encoding='utf-8'))

print('=' * 70)
print('  gen89_ind018 Model Deep Analysis')
print('=' * 70)

print('\n[BASIC INFO]')
print(f'   Name: {data["name"]}')
print(f'   Total Fitness: {data["fitness"]:.4f}')
print(f'   Parts: {data["num_parts"]}')
print(f'   Joints: {data["num_joints"]}')

print('\n[FITNESS BREAKDOWN]')
for k, v in data['fitness_components'].items():
    print(f'   {k}: {v:.4f}')

parts_by_type = {'structure': [], 'contact': [], 'actuator': []}
for p in data['parts']:
    parts_by_type[p['part_type']].append(p)

total_mass = 0
type_names = {'structure': 'Structure', 'contact': 'Contact', 'actuator': 'Actuator'}

print('\n[PARTS INVENTORY]')
for ptype, plist in parts_by_type.items():
    print(f'\n  {type_names[ptype]} ({len(plist)} items):')
    for i, p in enumerate(plist):
        pos = p['position']
        params = p['params']
        mass = params.get('mass', 0)
        total_mass += mass
        if ptype == 'structure':
            print(f'     #{i+1} [{p["part_id"]}] L={params.get("length",0):.3f}m R={params.get("radius",0):.3f}m M={mass:.3f}kg pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})')
        elif ptype == 'contact':
            print(f'     #{i+1} [{p["part_id"]}] R={params.get("radius",0):.3f}m M={mass:.3f}kg f={params.get("friction_primary",0)} pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})')
        else:
            print(f'     #{i+1} [{p["part_id"]}] L={params.get("length",0):.3f}m T={params.get("max_torque",0):.1f}Nm M={mass:.3f}kg pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})')

print(f'\n[TOTAL MASS] {total_mass:.3f} kg')

hinge_joints = [j for j in data['joints'] if j['joint_type'] == 'hinge']
fixed_joints = [j for j in data['joints'] if j['joint_type'] == 'fixed']

print('\n[JOINT ANALYSIS]')
print(f'   Hinge (movable): {len(hinge_joints)}')
print(f'   Fixed: {len(fixed_joints)}')

print('\n  Hinge Details:')
for j in hinge_joints:
    params = j['params']
    range_deg = (params['range_max'] - params['range_min']) * 180 / 3.14159
    print(f'   {j["parent_id"][:6]} -> {j["child_id"][:6]} | range:{params["range_min"]:.2f}~{params["range_max"]:.2f}rad ({range_deg:.0f}deg) | damping:{params["damping"]}')

positions = np.array([p['position'] for p in data['parts']])
bbox_min = positions.min(axis=0)
bbox_max = positions.max(axis=0)
center = positions.mean(axis=0)
extent = bbox_max - bbox_min
volume_estimate = extent[0] * extent[1] * extent[2]

print('\n[SPATIAL ANALYSIS]')
print(f'   Bounding Box:')
print(f'     X: [{bbox_min[0]:.2f}, {bbox_max[0]:.2f}] m')
print(f'     Y: [{bbox_min[1]:.2f}, {bbox_max[1]:.2f}] m')
print(f'     Z: [{bbox_min[2]:.2f}, {bbox_max[2]:.2f}] m')
print(f'   Dimensions: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m')
print(f'   Center of Mass: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m')
print(f'   Estimated Volume: {volume_estimate:.4f} m^3')

contact_pos = np.array([p['position'] for p in parts_by_type['contact']])
if len(contact_pos) > 0:
    contact_z_min = contact_pos[:, 2].min()
    contact_z_max = contact_pos[:, 2].max()
    front_contacts = sum(1 for c in contact_pos if c[0] > 0)
    back_contacts = len(contact_pos) - front_contacts
    left_contacts = sum(1 for c in contact_pos if c[1] < 0)
    right_contacts = len(contact_pos) - left_contacts
    
    print('\n[MORPHOLOGY FEATURES]')
    print(f'   Contact height range: Z=[{contact_z_min:.3f}, {contact_z_max:.3f}] m')
    print(f'   Contact distribution: Front={front_contacts}/Back={back_contacts}, Left={left_contacts}/Right={right_contacts}')
    
    # Check asymmetry
    front_x_avg = np.mean([c[0] for c in contact_pos if c[0] > 0]) if front_contacts > 0 else 0
    back_x_avg = np.mean([c[0] for c in contact_pos if c[0] <= 0]) if back_contacts > 0 else 0
    print(f'   Avg X position - Front contacts: {front_x_avg:.3f}m, Back contacts: {back_x_avg:.3f}m')

print('\n[MOVEMENT MECHANISM HYPOTHESIS]')
print('   Based on morphology analysis, this robot likely uses:')
print('   1. Main actuator e986fb08 at rear provides primary propulsion')
print('   2. Multiple front contacts form stable support surface')
print('   3. Asymmetric design may exploit inertia/swinging motion')
print('   4. Only 2 actuators for high efficiency (minimal energy waste)')
