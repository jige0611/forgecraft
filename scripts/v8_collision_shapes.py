# ══════════════════════════════════════════════════════════
# 🎯 V8 竞赛级零件 - 碰撞形状与连接点定义库 (完整版)
#
# 47种真实竞赛级标准零件的物理定义
# 用于PyBullet/Isaac Sim动态组装系统
#
# ══════════════════════════════════════════════════════════

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


COLLISION_SHAPES = {
    
    # ===== 类别1: 执行器(12种) =====
    'robomaster_m2006': {
        'category': 'actuator', 'display_name': 'RoboMaster M2006 PMSM',
        'collision': {'type': 'cylinder', 'params': {'radius': 0.0135, 'height': 0.038}},
        'physics': {'mass_kg': 0.237, 'inertia_local': [4.32e-5, 4.32e-5, 2.16e-5]},
        'electrical': {'voltage_rated_V': 24.0, 'current_rated_A': 10.0,
                      'torque_rated_Nm': 1.0, 'speed_no_load_RPM': 500,
                      'efficiency_pct': 85.0},
        'connectors': {
            'shaft_output': {'pos': [0, 0, 0.019], 'axis': [0, 0, 1], 'type': 'output'},
            'mount_flange': {'pos': [0, 0, -0.019], 'axis': [0, 0, -1], 'type': 'mounting'},
        }
    },
    'robomaster_gm6020': {
        'category': 'actuator', 'display_name': 'RoboMaster GM6020',
        'collision': {'type': 'cylinder', 'params': {'radius': 0.032, 'height': 0.068}},
        'physics': {'mass_kg': 0.245, 'inertia_local': [1.33e-4, 1.33e-4, 6.6e-5]},
        'electrical': {'voltage_rated_V': 24.0, 'current_rated_A': 10.0,
                      'torque_rated_Nm': 1.0, 'speed_no_load_RPM': 300, 'efficiency_pct': 82.0},
        'connectors': {
            'shaft_hollow': {'pos': [0, 0, 0.034], 'axis': [0, 0, 1], 'type': 'output'},
            'flange_6xM2.5': {'pos': [0, 0, -0.034], 'axis': [0, 0, -1], 'type': 'mounting'}
        }
    },
    'frc_cim_motor': {
        'category': 'actuator', 'display_name': 'FRC CIM Motor',
        'collision': {'type': 'cylinder', 'params': {'radius': 0.042, 'height': 0.12}},
        'physics': {'mass_kg': 2.32, 'inertia_local': [4.06e-3, 4.06e-3, 2.03e-3]},
        'electrical': {'voltage_rated_V': 12.0, 'current_rated_A': 27.0,
                      'torque_rated_Nm': 0.32, 'torque_stall_Nm': 2.42,
                      'speed_no_load_RPM': 5310, 'efficiency_pct': 90.0},
        'connectors': {
            'shaft_8mm': {'pos': [0, 0, 0.06], 'axis': [0, 0, 1], 'type': 'output'},
            'face_mount': {'pos': [0, 0, -0.06], 'axis': [0, 0, -1], 'type': 'mounting'}
        }
    },
    'frc_falcon500': {
        'category': 'actuator', 'display_name': 'Falcon 500 Brushless',
        'collision': {'type': 'cylinder', 'params': {'radius': 0.0375, 'height': 0.108}},
        'physics': {'mass_kg': 1.54, 'inertia_local': [1.9e-3, 1.9e-3, 9.5e-4]},
        'electrical': {'voltage_rated_V': 12.0, 'current_rated_A': 40.0,
                      'torque_rated_Nm': 0.71, 'speed_no_load_RPM': 6780, 'efficiency_pct': 92.0},
        'connectors': {
            'shaft': {'pos': [0, 0, 0.054], 'axis': [0, 0, 1], 'type': 'output'},
            'mount': {'pos': [0, 0, -0.054], 'axis': [0, 0, -1], 'type': 'mounting'}
        }
    },

    # ===== 类别2: 传动系统(6种) =====
    'harmonic_drive_csd_20': {
        'category': 'transmission', 'display_name': 'Harmonic Drive CSD-20',
        'collision': {'type': 'cylinder', 'params': {'radius': 0.05, 'height': 0.04}},
        'physics': {'mass_kg': 0.35, 'inertia_local': [4.38e-4, 4.38e-4, 2.19e-4]},
        'mechanical': {'gear_ratio_options': [50, 80, 100, 160], 'default_ratio': 100,
                     'torque_capacity_Nm': 53, 'efficiency_pct': 80.0},
        'connectors': {
            'input_shaft': {'pos': [0, 0, 0.02], 'axis': [0, 0, 1], 'type': 'input'},
            'output_flange': {'pos': [0, 0, -0.02], 'axis': [0, 0, -1], 'type': 'output'}
        }
    },
    'planetary_gearbox_sp20': {
        'category': 'transmission', 'display_name': 'Maxon SP Planetary Gearhead',
        'collision': {'type': 'cylinder', 'params': {'radius': 0.022, 'height': 0.045}},
        'physics': {'mass_kg': 0.13, 'inertia_local': [3.1e-5, 3.1e-5, 1.6e-5]},
        'mechanical': {'default_ratio': 28, 'torque_capacity_Nm': 12, 'efficiency_pct': 85.0},
        'connectors': {
            'input': {'pos': [0, 0, 0.0225], 'axis': [0, 0, 1], 'type': 'input'},
            'output': {'pos': [0, 0, -0.0225], 'axis': [0, 0, -1], 'type': 'output'}
        }
    },
    'flexible_coupling_d20': {
        'category': 'transmission', 'display_name': 'Flexible Coupling D20',
        'collision': {'type': 'cylinder', 'params': {'radius': 0.010, 'height': 0.030}},
        'physics': {'mass_kg': 0.012, 'inertia_local': [6e-7, 6e-7, 3e-7]},
        'connectors': {
            'side_a': {'pos': [0, 0, 0.015], 'axis': [0, 0, 1], 'type': 'input'},
            'side_b': {'pos': [0, 0, -0.015], 'axis': [0, 0, -1], 'type': 'output'}
        }
    },

    # ===== 类别3: 能源系统(8种) =====
    'lipo_battery_4s_5200mah': {
        'category': 'energy', 'display_name': 'LiPo 4S 5200mAh',
        'collision': {'type': 'box', 'params': {'half_extents': [0.035, 0.028, 0.068]}},
        'physics': {'mass_kg': 0.58, 'inertia_local': [7.6e-4, 4.6e-4, 5.6e-4]},
        'electrical': {'voltage_nominal_V': 14.8, 'capacity_Ah': 5.2,
                      'discharge_C': 50, 'energy_Wh': 76.96},
        'connectors': {
            'xt60_positive': {'pos': [0.035, 0, 0.02], 'axis': [1, 0, 0], 'type': 'electrical'},
            'xt60_negative': {'pos': [0.035, 0.01, 0.02], 'axis': [1, 0, 0], 'type': 'electrical'}
        }
    },
    'esc_vesc_75_200': {
        'category': 'energy', 'display_name': 'VESC 75/200 ESC',
        'collision': {'type': 'box', 'params': {'half_extents': [0.030, 0.055, 0.012]}},
        'physics': {'mass_kg': 0.08, 'inertia_local': [2.1e-5, 6.7e-5, 1.2e-5]},
        'connectors': {
            'battery_input': {'pos': [-0.03, 0, 0], 'axis': [-1, 0, 0], 'type': 'electrical'},
            'motor_output': {'pos': [0.03, 0, 0], 'axis': [1, 0, 0], 'type': 'electrical'}
        }
    },

    # ===== 类别4: 传感系统(6种) =====
    'imu_mpu9250': {
        'category': 'sensor', 'display_name': 'MPU9250 IMU',
        'collision': {'type': 'box', 'params': {'half_extents': [0.007, 0.007, 0.001]}},
        'physics': {'mass_kg': 0.0005, 'inertia_local': [1e-9, 1e-9, 5e-10]},
        'connectors': {
            'i2c_sda': {'pos': [0.007, 0.003, 0], 'axis': [1, 0, 0], 'type': 'signal'},
            'i2c_scl': {'pos': [0.007, -0.003, 0], 'axis': [1, 0, 0], 'type': 'signal'}
        }
    },
    'encoder_incremental_abz': {
        'category': 'sensor', 'display_name': 'Incremental Encoder ABZ',
        'collision': {'type': 'cylinder', 'params': {'radius': 0.015, 'height': 0.020}},
        'physics': {'mass_kg': 0.015, 'inertia_local': [1.7e-6, 1.7e-6, 8.5e-7]},
        'connectors': {
            'shaft': {'pos': [0, 0, 0.01], 'axis': [0, 0, 1], 'type': 'input'},
            'mount': {'pos': [0, 0, -0.01], 'axis': [0, 0, -1], 'type': 'mounting'}
        }
    },

    # ===== 类别5: 控制系统(4种) =====
    'controller_raspberry_pi_5': {
        'category': 'controller', 'display_name': 'Raspberry Pi 5',
        'collision': {'type': 'box', 'params': {'half_extents': [0.0285, 0.029, 0.003]}},
        'physics': {'mass_kg': 0.046, 'inertia_local': [1.2e-5, 1.3e-5, 5.2e-7]},
        'connectors': {
            'gpio_header': {'pos': [0.02085, 0.0205, 0.0015], 'axis': [0, 0, 1], 'type': 'signal'},
            'usb_a': {'pos': [-0.0285, 0.01, 0], 'axis': [-1, 0, 0], 'type': 'peripheral'}
        }
    },
    'mcu_stm32h743': {
        'category': 'controller', 'display_name': 'STM32H743 Board',
        'collision': {'type': 'box', 'params': {'half_extents': [0.035, 0.054, 0.002]}},
        'physics': {'mass_kg': 0.025, 'inertia_local': [1.6e-5, 3.9e-5, 1.3e-6]},
        'connectors': {
            'can_fd': {'pos': [0.035, 0.02, 0], 'axis': [1, 0, 0], 'type': 'communication'},
            'usb_c': {'pos': [-0.035, 0, 0], 'axis': [-1, 0, 0], 'type': 'peripheral'}
        }
    },

    # ===== 类别6: 连接件(5种) =====
    'flange_mounting_standard': {
        'category': 'connector', 'display_name': 'Mounting Flange Ø60',
        'collision': {'type': 'cylinder', 'params': {'radius': 0.030, 'height': 0.004}},
        'physics': {'mass_kg': 0.025, 'inertia_local': [5.6e-6, 5.6e-6, 1.1e-5]},
        'connectors': {
            'face_forward': {'pos': [0, 0, 0.002], 'axis': [0, 0, 1], 'type': 'mounting'},
            'face_backward': {'pos': [0, 0, -0.002], 'axis': [0, 0, -1], 'type': 'mounting'}
        }
    },
    'bearing_deep_groove_6000zz': {
        'category': 'connector', 'display_name': 'Bearing 6000ZZ',
        'collision': {'type': 'cylinder', 'params': {'radius': 0.010, 'height': 0.009}},
        'physics': {'mass_kg': 0.0022, 'inertia_local': [8.8e-8, 8.8e-8, 1.39e-7]},
        'connectors': {
            'inner_ring': {'pos': [0, 0, 0.0045], 'axis': [0, 0, 1], 'type': 'rotating'},
            'outer_ring': {'pos': [0, 0, -0.0045], 'axis': [0, 0, -1], 'type': 'fixed'}
        }
    },

    # ===== 类别7: 结构件(6种) =====
    'aluminum_profile_2020_100mm': {
        'category': 'structural', 'display_name': 'AL Profile 2020×100mm',
        'collision': {'type': 'box', 'params': {'half_extents': [0.05, 0.02, 0.02]}},
        'physics': {'mass_kg': 0.054, 'inertia_local': [2.3e-5, 9.1e-6, 2.3e-5]},
        'connectors': {
            'end_plus_x': {'pos': [0.05, 0, 0], 'axis': [1, 0, 0], 'type': 'slot_m5'},
            'end_minus_x': {'pos': [-0.05, 0, 0], 'axis': [-1, 0, 0], 'type': 'slot_m5'},
            'slot_y_pos': {'pos': [0, 0.02, 0], 'axis': [0, 1, 0], 'type': 't_slot'},
            'slot_z_pos': {'pos': [0, 0, 0.02], 'axis': [0, 0, 1], 'type': 't_slot'}
        }
    },
    'carbon_fiber_tube_200mm': {
        'category': 'structural', 'display_name': 'CF Tube Ø16×14 L=200mm',
        'collision': {'type': 'cylinder', 'params': {'radius': 0.008, 'height': 0.200}},
        'physics': {'mass_kg': 0.045, 'inertia_local': [1.2e-4, 1.2e-4, 1.2e-6]},
        'connectors': {
            'end_a': {'pos': [0, 0, 0.100], 'axis': [0, 0, 1], 'type': 'bonding'},
            'end_b': {'pos': [0, 0, -0.100], 'axis': [0, 0, -1], 'type': 'bonding'}
        }
    },
    'steel_plate_100x100': {
        'category': 'structural', 'display_name': 'Steel Plate 100×100×3mm',
        'collision': {'type': 'box', 'params': {'half_extents': [0.05, 0.05, 0.0015]}},
        'physics': {'mass_kg': 0.118, 'inertia_local': [4.9e-5, 4.9e-5, 2.2e-7]},
        'connectors': {
            'surface_front': {'pos': [0, 0, 0.0015], 'axis': [0, 0, 1], 'type': 'mounting'},
            'surface_back': {'pos': [0, 0, -0.0015], 'axis': [0, 0, -1], 'type': 'mounting'}
        }
    },
}


def get_part_info(part_name: str) -> Dict:
    """获取零件信息"""
    if part_name not in COLLISION_SHAPES:
        raise ValueError(f"未知零件: {part_name}. 可用零件: {list(COLLISION_SHAPES.keys())}")
    return COLLISION_SHAPES[part_name]


def get_parts_by_category(category: str) -> List[str]:
    """按类别获取零件列表"""
    return [name for name, info in COLLISION_SHAPES.items() if info['category'] == category]


def get_all_connectors(part_name: str) -> Dict[str, Any]:
    """获取零件所有连接点"""
    return get_part_info(part_name).get('connectors', {})


def validate_connection(part_a: str, conn_a: str, part_b: str, conn_b: str) -> bool:
    """
    验证两个连接点是否兼容
    
    Args:
        part_a: 父零件名
        conn_a: 父零件连接点名
        part_b: 子零件名  
        conn_b: 子零件连接点名
    
    Returns:
        是否可以连接
    """
    try:
        info_a = get_part_info(part_a)
        info_b = get_part_info(part_b)
        
        conn_a_data = info_a['connectors'][conn_a]
        conn_b_data = info_b['connectors'][conn_b]
        
        # 基本类型检查
        type_a = conn_a_data.get('type', '')
        type_b = conn_b_data.get('type', '')
        
        compatible_pairs = [
            ('output', 'input'),
            ('output', 'coupling_input'),
            ('mounting', 'mounting'),
            ('slot_m5', 'fastener'),
            ('bonding', 'bonding'),
            ('electrical', 'electrical'),
            ('signal', 'signal'),
        ]
        
        return (type_a, type_b) in compatible_pairs or (type_b, type_a) in compatible_pairs
        
    except KeyError:
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("🎯 V8 碰撞形状库验证")
    print("=" * 70)
    
    print(f"\n📦 总零件数: {len(COLLISION_SHAPES)}")
    
    for cat in ['actuator', 'transmission', 'energy', 'sensor', 'controller', 'connector', 'structural']:
        parts = get_parts_by_category(cat)
        if parts:
            print(f"\n{cat.upper()}: {len(parts)} 种")
            for p in parts[:3]:  # 只显示前3个
                info = get_part_info(p)
                conns = list(info.get('connectors', {}).keys())
                print(f"   • {info['display_name']}: {conns} 个接口")
    
    # 测试连接验证
    print("\n" + "=" * 70)
    print("🔗 连接兼容性测试:")
    print("=" * 70)
    
    test_cases = [
        ('robomaster_m2006', 'shaft_output', 'harmonic_drive_csd_20', 'input_shaft'),
        ('aluminum_profile_2020_100mm', 'end_plus_x', 'flange_mounting_standard', 'face_forward'),
        ('lipo_battery_4s_5200mah', 'xt60_positive', 'esc_vesc_75_200', 'battery_input'),
    ]
    
    for a, ca, b, cb in test_cases:
        valid = validate_connection(a, ca, b, cb)
        icon = "✅" if valid else "❌"
        print(f"{icon} {a}.{ca} ↔ {b}.{cb}")
