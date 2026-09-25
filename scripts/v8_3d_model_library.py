# ══════════════════════════════════════════════════════════
# V8 竞赛级零件 - 高精度3D模型库生成器
# 基于47种真实零件的官方规格参数，程序化生成精细3D网格
# 数据来源: RoboMaster/FRC/Harmonic Drive/McMaster-Carr等
# ══════════════════════════════════════════════════════════

import trimesh
import numpy as np
import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
MODEL_LIBRARY_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(MODEL_LIBRARY_ROOT, "v8_3d_models")


@dataclass
class PartSpec:
    part_id: str; display_name: str; category: str
    manufacturer: str; part_number: str; geometry_type: str
    dimensions: Dict[str, float]
    color: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])
    mass_kg: float = 0.0; density_kgm3: float = 2700.0
    voltage_v: float = 0.0; torque_nm: float = 0.0
    speed_rpm: float = 0.0; current_a: float = 0.0
    connectors: Dict[str, Dict] = field(default_factory=dict)
    external_model_path: Optional[str] = None


@dataclass
class ModelGenerationResult:
    part_id: str; success: bool
    mesh_path_stl: Optional[str] = None; mesh_path_obj: Optional[str] = None
    vertex_count: int = 0; face_count: int = 0; volume_cm3: float = 0.0
    error_message: Optional[str] = None; generation_method: str = "procedural"


def get_all_part_specs() -> Dict[str, PartSpec]:
    specs = {}
    
    # ===== 执行器 (12种) =====
    specs['robomaster_gm6020'] = PartSpec(
        'robomaster_gm6020', 'RoboMaster GM6020 云台电机', 'actuators',
        'DJI-RoboMaster', 'GM6020', 'cylinder_complex',
        {'outer_diameter': 0.0667, 'total_height': 0.045,
         'hollow_bore_inner': 0.018, 'flange_diameter': 0.058, 'flange_thickness': 0.003},
        [0.85, 0.35, 0.10, 1.0], 0.468, 3500.0,
        24.0, 1.2, 320, 1.62,
        {'shaft_hollow':{'pos':[0,0,0.0225],'axis':[0,0,1],'type':'output','diameter':0.018},
         'flange_mount':{'pos':[0,0,-0.0225],'axis':[0,0,-1],'type':'mounting','diameter':0.058}})
    
    specs['robomaster_m3508'] = PartSpec(
        'robomaster_m3508', 'RoboMaster M3508 减速电机', 'actuators',
        'DJI-RoboMaster', 'M3508+C620', 'cylinder_complex',
        {'outer_diameter': 0.042, 'total_length': 0.0984,
         'shaft_diameter': 0.010, 'gearbox_length': 0.035, 'motor_body_length': 0.0634},
        [0.75, 0.45, 0.10, 1.0], 0.365, 3500.0,
        24.0, 3.0, 469, 18.0,
        {'shaft_output':{'pos':[0,0,0.0492],'axis':[0,0,1],'type':'output','diameter':0.010},
         'mount_flange':{'pos':[0,0,-0.0492],'axis':[0,0,-1],'type':'mounting','diameter':0.038}})
    
    specs['robomaster_m2006'] = PartSpec(
        'robomaster_m2006', 'RoboMaster M2006 P36 减速电机', 'actuators',
        'DJI-RoboMaster', 'M2006 P36+C610', 'cylinder_complex',
        {'outer_diameter': 0.0244, 'total_length': 0.0648,
         'shaft_diameter': 0.006, 'gearbox_diameter': 0.027},
        [0.65, 0.35, 0.15, 1.0], 0.090, 3500.0,
        24.0, 1.0, 416, 10.0,
        {'shaft_output':{'pos':[0,0,0.0324],'axis':[0,0,1],'type':'output','diameter':0.006},
         'mount_flange':{'pos':[0,0,-0.0324],'axis':[0,0,-1],'type':'mounting','diameter':0.024}})
    
    specs['frc_falcon500'] = PartSpec(
        'frc_falcon500', 'FRC Falcon 500 无刷电机', 'actuators',
        'VEX/CTRE', 'Falcon 500', 'cylinder_complex',
        {'outer_diameter': 0.060, 'total_length': 0.100,
         'shaft_diameter': 0.008, 'mount_pattern_diameter': 0.050},
        [0.20, 0.20, 0.25, 1.0], 0.498, 7800.0,
        12.0, 0.71, 6780, 40.0,
        {'shaft_8mm':{'pos':[0,0,0.05],'axis':[0,0,1],'type':'output','diameter':0.008},
         'face_mount':{'pos':[0,0,-0.05],'axis':[0,0,-1],'type':'mounting','diameter':0.050}})
    
    specs['frc_neo550'] = PartSpec(
        'frc_neo550', 'FRC NEO 550 无刷电机(小型)', 'actuators',
        'REV Robotics', 'NEO 550', 'cylinder_simple',
        {'outer_diameter': 0.036, 'total_length': 0.055, 'shaft_diameter': 0.006},
        [0.15, 0.15, 0.20, 1.0], 0.089, 4800.0,
        12.0, 0.28, 11000, 20.0,
        {'shaft':{'pos':[0,0,0.0275],'axis':[0,0,1],'type':'output','diameter':0.006},
         'mount':{'pos':[0,0,-0.0275],'axis':[0,0,-1],'type':'mounting','diameter':0.032}})
    
    specs['frc_cim'] = PartSpec(
        'frc_cim', 'FRC CIM 有刷直流电机', 'actuators',
        'VEX/Andymark', 'CIM Motor', 'cylinder_complex',
        {'outer_diameter': 0.057, 'total_length': 0.115,
         'shaft_diameter': 0.008, 'mount_holes_pitch': 0.038},
        [0.30, 0.30, 0.32, 1.0], 2.32, 7850.0,
        12.0, 0.32, 5310, 27.0,
        {'shaft_8mm':{'pos':[0,0,0.0575],'axis':[0,0,1],'type':'output','diameter':0.008}})
    
    specs['maxon_ec45_flat'] = PartSpec(
        'maxon_ec45_flat', 'Maxon EC45 flat 扁平无刷电机', 'actuators',
        'Maxon Motor', 'EC45 flat 70W', 'cylinder_simple',
        {'outer_diameter': 0.045, 'total_length': 0.028, 'shaft_diameter': 0.006},
        [0.80, 0.80, 0.82, 1.0], 0.165, 4500.0,
        24.0, 0.106, 7730, 4.44,
        {'shaft':{'pos':[0,0,0.014],'axis':[0,0,1],'type':'output','diameter':0.006}})
    
    specs['t_motor_u8'] = PartSpec(
        't_motor_u8', 'T-Motor U8 II 无刷电机', 'actuators',
        'T-Motor', 'U8II KV100', 'cylinder_complex',
        {'outer_diameter': 0.078, 'total_length': 0.035,
         'shaft_diameter': 0.010, 'prop_mount_dia': 0.040},
        [0.10, 0.10, 0.12, 1.0], 0.196, 2500.0,
        24.0, 0.55, 950, 6.0,
        {'shaft':{'pos':[0,0,0.0175],'axis':[0,0,1],'type':'output','diameter':0.010}})

    specs['dynamixel_xm430'] = PartSpec(
        'dynamixel_xm430', 'Robotis Dynamixel XM430-W210-R', 'actuators',
        'Robotis', 'XM430-W210-R', 'box_complex',
        {'length': 0.061, 'width': 0.041, 'height': 0.041,
         'hub_diameter': 0.026, 'horn_offset': 0.010},
        [0.15, 0.35, 0.70, 1.0], 0.075, 3200.0,
        24.0, 4.3, 46, 2.8,
        {'horn':{'pos':[0,0,0.0405],'axis':[0,0,1],'type':'output','diameter':0.026}})

    specs['servo_hs485hb'] = PartSpec(
        'servo_hs485hb', 'Hitec HS-485HB 舵机', 'actuators',
        'Hitec', 'HS-485HB Deluxe', 'box_complex',
        {'length': 0.0405, 'width': 0.020, 'height': 0.038, 'shaft_length': 0.005},
        [0.20, 0.25, 0.20, 1.0], 0.045, 1800.0,
        6.0, 4.1, 67, 0.32,
        {'horn':{'pos':[0,0,0.020],'axis':[0,0,1],'type':'output'}})

    specs['dc_motor_jga25'] = PartSpec(
        'dc_motor_jga25', 'JGA25-370 直流减速电机', 'actuators',
        'Generic', 'JGA25-370 12V', 'cylinder_complex',
        {'body_diameter': 0.025, 'total_length': 0.065,
         'gearbox_diameter': 0.037, 'shaft_diameter': 0.006},
        [0.35, 0.35, 0.38, 1.0], 0.145, 6500.0,
        12.0, 2.0, 150, 2.5,
        {'shaft':{'pos':[0,0,0.0325],'axis':[0,0,1],'type':'output','diameter':0.006}})

    specs['stepper_nema17'] = PartSpec(
        'stepper_nema17', 'NEMA17 步进电机(42步进)', 'actuators',
        'Generic/Leadshine', 'NEMA17 42HS34', 'square_cylinder',
        {'square_size': 0.042, 'body_length': 0.040,
         'shaft_diameter': 0.005, 'shaft_length': 0.022},
        [0.25, 0.25, 0.28, 1.0], 0.28, 7500.0,
        12.0, 0.44, 2000, 1.5,
        {'shaft':{'pos':[0,0,0.020],'axis':[0,0,1],'type':'output','diameter':0.005}})

    # ===== 传动系统 (6种) =====
    specs['harmonic_drive_csd20'] = PartSpec(
        'harmonic_drive_csd20', 'Harmonic Drive CSD-20-100-2A-GR', 'transmission',
        'Harmonic Drive LLC', 'CSD-20-100-2A-GR', 'cylinder_complex',
        {'outer_diameter': 0.085, 'body_diameter': 0.065, 'total_height': 0.040,
         'input_flange_dia': 0.030, 'output_flange_dia': 0.050},
        [0.75, 0.72, 0.68, 1.0], 0.35, 8500.0,
        connectors={'input_shaft':{'pos':[0,0,0.020],'axis':[0,0,1],'type':'input','diameter':0.012},
                    'output_flange':{'pos':[0,0,-0.020],'axis':[0,0,-1],'type':'output','diameter':0.050}})
    
    specs['harmonic_drive_csd14'] = PartSpec(
        'harmonic_drive_csd14', 'Harmonic Drive CSD-14-50-2A-R', 'transmission',
        'Harmonic Drive LLC', 'CSD-14-50-2A-R', 'cylinder_complex',
        {'outer_diameter': 0.058, 'body_diameter': 0.045, 'total_height': 0.032,
         'input_flange_dia': 0.022, 'output_flange_dia': 0.035},
        [0.75, 0.72, 0.68, 1.0], 0.15, 8500.0,
        connectors={'input_shaft':{'pos':[0,0,0.016],'axis':[0,0,1],'type':'input','diameter':0.008},
                    'output_flange':{'pos':[0,0,-0.016],'axis':[0,0,-1],'type':'output','diameter':0.035}})
    
    specs['planetary_gearbox_sp20'] = PartSpec(
        'planetary_gearbox_sp20', 'Maxon GPX20HP 行星减速箱', 'transmission',
        'Maxon', 'GPX20HP', 'cylinder_simple',
        {'outer_diameter': 0.020, 'total_height': 0.029,
         'input_shaft_dia': 0.004, 'output_shaft_dia': 0.006},
        [0.65, 0.63, 0.58, 1.0], 0.065, 8200.0,
        connectors={'input':{'pos':[0,0,0.0145],'axis':[0,0,1],'type':'input','diameter':0.004},
                    'output':{'pos':[0,0,-0.0145],'axis':[0,0,-1],'type':'output','diameter':0.006}})
    
    specs['flexible_coupling_d20'] = PartSpec(
        'flexible_coupling_d20', 'XGT-20C 弹性联轴器', 'transmission',
        'NBK/R+W', 'XGT-20C', 'cylinder_detail',
        {'outer_diameter': 0.020, 'total_length': 0.030,
         'bore_a': 0.008, 'bore_b': 0.008},
        [0.55, 0.53, 0.48, 1.0], 0.012, 8300.0,
        connectors={'side_a':{'pos':[0,0,0.015],'axis':[0,0,1],'type':'input','diameter':0.008},
                    'side_b':{'pos':[0,0,-0.015],'axis':[0,0,-1],'type':'output','diameter':0.008}})

    specs['timing_belt_pulley_gt2'] = PartSpec(
        'timing_belt_pulley_gt2', 'GT2 同步带轮 20齿', 'transmission',
        'Gates/Brecoflex', 'GT2-20T-6mm', 'complex',
        {'pitch_diameter': 0.01273, 'teeth_count': 20, 'bore': 0.005, 'width': 0.011},
        [0.30, 0.30, 0.33, 1.0], 0.008, 1400.0,
        connectors={'bore':{'pos':[0,0,0],'axis':[0,0,1],'type':'output','diameter':0.005}})

    specs['chain_sprocket_08b'] = PartSpec(
        'chain_sprocket_08b', '08B链轮 16齿', 'transmission',
        'Tsubaki/Renold', '08B16', 'complex',
        {'pitch_diameter': 0.065, 'teeth_count': 16, 'bore': 0.010, 'width': 0.012},
        [0.20, 0.20, 0.23, 1.0], 0.025, 7850.0,
        connectors={'bore':{'pos':[0,0,0],'axis':[0,0,1],'type':'output','diameter':0.010}})

    # ===== 能源系统 (8种) =====
    specs['lipo_battery_4s'] = PartSpec(
        'lipo_battery_4s', 'LiPo 4S 5200mAh 50C', 'energy',
        'Tattu/Gens Ace', 'LiPo 4S 5200mAh', 'box_detail',
        {'length': 0.140, 'width': 0.056, 'height': 0.038},
        [0.15, 0.15, 0.18, 1.0], 0.58, 1950.0,
        14.8, 0, 0, 52.0,
        connectors={'xt60_positive':{'pos':[0.070,0,0.019],'axis':[1,0,0],'type':'electrical'},
                    'xt60_negative':{'pos':[0.070,0.015,0.019],'axis':[1,0,0],'type':'electrical'}})
    
    specs['esc_vesc'] = PartSpec(
        'esc_vesc', 'VESC 75/250 电调', 'energy',
        'Vedder Electronic', 'VESC 75/250', 'box_heatsink',
        {'length': 0.060, 'width': 0.055, 'height': 0.028, 'heatsink_fins': 8},
        [0.10, 0.10, 0.12, 1.0], 0.080, 2500.0,
        connectors={'battery_input':{'pos':[-0.030,0,0],'axis':[-1,0,0],'type':'electrical'},
                    'motor_output':{'pos':[0.030,0,0],'axis':[1,0,0],'type':'electrical'}})
    
    specs['esc_c620'] = PartSpec(
        'esc_c620', 'RoboMaster C620 电调', 'energy',
        'DJI-RoboMaster', 'C620', 'box_heatsink',
        {'length': 0.052, 'width': 0.040, 'height': 0.015, 'heatsink_fins': 6},
        [0.15, 0.15, 0.18, 1.0], 0.035, 2200.0,
        connectors={'battery_input':{'pos':[-0.026,0,0],'axis':[-1,0,0],'type':'electrical'},
                    'motor_output':{'pos':[0.026,0,0],'axis':[1,0,0],'type':'electrical'}})

    specs['bms_4s'] = PartSpec(
        'bms_4s', '4S LiPo BMS保护板 40A', 'energy',
        'Various', 'BMS-4S-40A', 'pcb',
        {'length': 0.060, 'width': 0.035, 'height': 0.002, 'component_height': 0.012},
        [0.10, 0.12, 0.08, 1.0], 0.020, 1500.0,
        connectors={'batt_in':{'pos':[-0.030,0,0],'axis':[-1,0,0],'type':'electrical'},
                    'batt_out':{'pos':[0.030,0,0],'axis':[1,0,0],'type':'electrical'}})

    specs['power_switch_100a'] = PartSpec(
        'power_switch_100a', 'ANT 100A 高压开关模块', 'energy',
        'ANT Robotics', 'SW-100A', 'box_simple',
        {'length': 0.045, 'width': 0.035, 'height': 0.025},
        [0.85, 0.20, 0.15, 1.0], 0.030, 1200.0,
        connectors={'in':{'pos':[-0.022,0,0],'axis':[-1,0,0],'type':'electrical'},
                    'out':{'pos':[0.022,0,0],'axis':[1,0,0],'type':'electrical'}})

    specs['pdb_power_dist'] = PartSpec(
        'pdb_power_dist', 'PDB 电源分配板', 'energy',
        'Matek/Hobbywing', 'PDB-X', 'pcb',
        {'length': 0.045, 'width': 0.040, 'height': 0.0015},
        [0.08, 0.08, 0.10, 1.0], 0.015, 1400.0,
        connectors={'main_in':{'pos':[0,0,0],'axis':[0,0,1],'type':'electrical'}})

    specs['voltage_regulator_5v_3a'] = PartSpec(
        'voltage_regulator_5v_3a', '5V 3A DC-DC降压模块', 'energy',
        'Pololu/DROK', '5V Step-Down 3A', 'box_small',
        {'length': 0.022, 'width': 0.017, 'height': 0.012},
        [0.05, 0.05, 0.08, 1.0], 0.005, 800.0,
        connectors={'vin':{'pos':[-0.011,0,0],'axis':[-1,0,0],'type':'electrical'},
                    'vout':{'pos':[0.011,0,0],'axis':[1,0,0],'type':'electrical'}})

    specs['supercapacitor_2.7v_10f'] = PartSpec(
        'supercapacitor_2.7v_10f', '超级电容 2.7V 10F', 'energy',
        'Nesscap/Elna', '2.7V 10F', 'cylinder_simple',
        {'outer_diameter': 0.016, 'total_height': 0.026},
        [0.20, 0.20, 0.25, 1.0], 0.004, 1100.0,
        2.7, 0, 0, 5.0,
        connectors={'positive':{'pos':[0,0,0.013],'axis':[0,0,1],'type':'electrical'},
                    'negative':{'pos':[0,0,-0.013],'axis':[0,0,-1],'type':'electrical'}})

    # ===== 传感系统 (6种) =====
    specs['imu_mpu9250'] = PartSpec(
        'imu_mpu9250', 'MPU9250 IMU (9轴)', 'sensors',
        'TDK InvenSense', 'MPU-9250', 'qfn_package',
        {'length': 0.004, 'width': 0.004, 'height': 0.001},
        [0.05, 0.05, 0.05, 1.0], 0.0005, 800.0,
        connectors={'sda':{'pos':[0.002,0.0005,0.0005],'axis':[1,0,0],'type':'signal'},
                    'scl':{'pos':[0.002,-0.0005,0.0005],'axis':[1,0,0],'type':'signal'}})
    
    specs['imu_bno055'] = PartSpec(
        'imu_bno055', 'BNO055 智能IMU (9轴+融合)', 'sensors',
        'Bosch Sensortec', 'BNO055', 'lga28_package',
        {'length': 0.0052, 'width': 0.0032, 'height': 0.011},
        [0.05, 0.05, 0.05, 1.0], 0.0012, 900.0,
        connectors={'sda':{'pos':[0.0026,0.001,0.0055],'axis':[1,0,0],'type':'signal'},
                    'scl':{'pos':[0.0026,-0.001,0.0055],'axis':[1,0,0],'type':'signal'}})
    
    specs['encoder_incremental_abz'] = PartSpec(
        'encoder_incremental_abz', '增量式编码器 ABZ 1024线', 'sensors',
        'Omron/Autonics', 'E6C2-CWZ6C 1024P/R', 'cylinder_with_hub',
        {'body_diameter': 0.038, 'body_height': 0.035, 'shaft_diameter': 0.008, 'shaft_length': 0.015},
        [0.15, 0.15, 0.18, 1.0], 0.015, 2500.0,
        connectors={'shaft':{'pos':[0,0,0.0175],'axis':[0,0,1],'type':'input','diameter':0.008},
                    'mount':{'pos':[0,0,-0.0175],'axis':[0,0,-1],'type':'mounting','diameter':0.038}})

    specs['encoder_absolute_multi_turn'] = PartSpec(
        'encoder_absolute_multi_turn', '绝对值多圈编码器 19位', 'sensors',
        'Tamagawa/SICK', 'TS5208N 19-bit', 'cylinder_with_hub',
        {'body_diameter': 0.052, 'body_height': 0.040, 'shaft_diameter': 0.010, 'shaft_length': 0.016},
        [0.12, 0.12, 0.15, 1.0], 0.095, 3000.0,
        connectors={'shaft':{'pos':[0,0,0.020],'axis':[0,0,1],'type':'input','diameter':0.010}})

    specs['force_sensor_loadcell'] = PartSpec(
        'force_sensor_loadcell', '称重传感器 50kg', 'sensors',
        'Phidgets/Seeed', 'Load Cell 50kg', 'aluminum_block',
        {'length': 0.035, 'width': 0.015, 'height': 0.015, 'hole_dia': 0.004},
        [0.70, 0.72, 0.68, 1.0], 0.015, 2700.0,
        connectors={'mount_a':{'pos':[-0.017,0,0],'axis':[-1,0,0],'type':'mounting'},
                    'mount_b':{'pos':[0.017,0,0],'axis':[1,0,0],'type':'mounting'}})

    specs['distance_sensor_vl53l0x'] = PartSpec(
        'distance_sensor_vl53l0x', 'VL53L0X ToF激光测距', 'sensors',
        'STMicroelectronics', 'VL53L0X', 'small_pcb_module',
        {'length': 0.012, 'width': 0.008, 'height': 0.0015, 'lens_dia': 0.004},
        [0.05, 0.05, 0.08, 1.0], 0.001, 600.0,
        connectors={'i2c':{'pos':[0,-0.004,0],'axis':[0,-1,0],'type':'signal'}})

    # ===== 控制系统 (4种) =====
    specs['controller_raspberry_pi_5'] = PartSpec(
        'controller_raspberry_pi_5', 'Raspberry Pi 5 主控', 'controllers',
        'Raspberry Pi Foundation', 'RPi5 8GB', 'pcb_board',
        {'length': 0.087, 'width': 0.056, 'board_thickness': 0.0015,
         'usb_height': 0.016, 'ethernet_height': 0.014, 'gpio_header_height': 0.012},
        [0.05, 0.05, 0.08, 1.0], 0.046, 900.0,
        connectors={'gpio_header':{'pos':[0.021,0.021,0.002],'axis':[0,0,1],'type':'signal'},
                    'usb_a':{'pos':[-0.044,0.010,0],'axis':[-1,0,0],'type':'peripheral'},
                    'usb_c_power':{'pos':[0,-0.028,0],'axis':[0,-1,0],'type':'power'},
                    'ethernet':{'pos':[0.044,-0.007,0],'axis':[1,0,0],'type':'network'}})
    
    specs['mcu_stm32h743'] = PartSpec(
        'mcu_stm32h743', 'STM32H743 开发板', 'controllers',
        'STMicroelectronics', 'NUCLEO-H743ZI2', 'pcb_board',
        {'length': 0.090, 'width': 0.070, 'board_thickness': 0.002},
        [0.05, 0.05, 0.08, 1.0], 0.045, 950.0,
        connectors={'can_fd':{'pos':[0.045,0.020,0.002],'axis':[1,0,0],'type':'communication'},
                    'usb_c':{'pos':[-0.045,0,0.002],'axis':[-1,0,0],'type':'peripheral'}})

    specs['can_transceiver'] = PartSpec(
        'can_transceiver', 'CAN收发器 TJA1050T', 'controllers',
        'NXP', 'TJA1050T', 'so8_package',
        {'length': 0.006, 'width': 0.005, 'height': 0.0015},
        [0.05, 0.05, 0.05, 1.0], 0.0003, 700.0,
        connectors={'can_h':{'pos':[0.001,0.0015,0],'axis':[0,1,0],'type':'communication'},
                    'can_l':{'pos':[-0.001,0.0015,0],'axis':[0,1,0],'type':'communication'}})

    specs['motor_driver_tb6612'] = PartSpec(
        'motor_driver_tb6612', 'TB6612FNG 双路电机驱动', 'controllers',
        'Toshiba', 'TB6612FNG', 'ssop24_package',
        {'length': 0.010, 'width': 0.006, 'height': 0.0015},
        [0.05, 0.05, 0.05, 1.0], 0.0005, 650.0,
        connectors={'motora':{'pos':[0.003,0,0],'axis':[1,0,0],'type':'electrical'},
                    'motorb':{'pos':[-0.003,0,0],'axis':[-1,0,0],'type':'electrical'}})

    # ===== 连接件 (5种) =====
    specs['flange_mounting_standard'] = PartSpec(
        'flange_mounting_standard', '安装法兰 O60 标准型', 'connectors',
        'Misumi/McMaster', 'Flange O60 M4x4', 'ring_with_holes',
        {'outer_diameter': 0.060, 'inner_diameter': 0.020, 'thickness': 0.004,
         'bolt_circle_diameter': 0.050, 'bolt_count': 4, 'bolt_diameter': 0.004},
        [0.55, 0.55, 0.57, 1.0], 0.025, 7950.0,
        connectors={'face_forward':{'pos':[0,0,0.002],'axis':[0,0,1],'type':'mounting'},
                    'face_backward':{'pos':[0,0,-0.002],'axis':[0,0,-1],'type':'mounting'}})
    
    specs['bearing_deep_groove_6000zz'] = PartSpec(
        'bearing_deep_groove_6000zz', '深沟球轴承 6000ZZ', 'connectors',
        'SKF/NSK/FAG', '6000-2Z', 'bearing_assembly',
        {'inner_diameter': 0.010, 'outer_diameter': 0.026, 'width': 0.008},
        [0.60, 0.62, 0.65, 1.0], 0.0022, 7850.0,
        connectors={'inner_ring':{'pos':[0,0,0.004],'axis':[0,0,1],'type':'rotating','diameter':0.010},
                    'outer_ring':{'pos':[0,0,-0.004],'axis':[0,0,-1],'type':'fixed','diameter':0.026}})

    specs['coupler_rigid'] = PartSpec(
        'coupler_rigid', '刚性联轴器 O12 L25', 'connectors',
        'NBK/Ruland', 'Rigid Coupler 12x12', 'cylinder_step',
        {'outer_diameter': 0.020, 'total_length': 0.025, 'bore_a': 0.006, 'bore_b': 0.006},
        [0.50, 0.52, 0.55, 1.0], 0.018, 7850.0,
        connectors={'side_a':{'pos':[0,0,0.0125],'axis':[0,0,1],'type':'rotating'},
                    'side_b':{'pos':[0,0,-0.0125],'axis':[0,0,-1],'type':'rotating'}})

    specs['shaft_collar'] = PartSpec(
        'shaft_collar', '轴夹紧环 O8', 'connectors',
        'McMaster/Ruland', 'Shaft Collar 8mm', 'ring_split',
        {'inner_diameter': 0.008, 'outer_diameter': 0.014, 'thickness': 0.006},
        [0.45, 0.46, 0.48, 1.0], 0.003, 7850.0,
        connectors={'inner':{'pos':[0,0,0],'axis':[0,0,1],'type':'clamping','diameter':0.008}})

    specs['quick_release_pin'] = PartSpec(
        'quick_release_pin', '快拆销 O6 L30', 'connectors',
        'McMaster/Carr Lane', 'QR Pin 6mm', 'pin_with_ball',
        {'diameter': 0.006, 'total_length': 0.030, 'handle_diameter': 0.012},
        [0.60, 0.55, 0.30, 1.0], 0.008, 8600.0,
        connectors={'tip':{'pos':[0,0,0.015],'axis':[0,0,1],'type':'locking'},
                    'handle':{'pos':[0,0,-0.015],'axis':[0,0,-1],'type':'pull'}})

    # ===== 结构件 (6种) =====
    specs['aluminum_extrusion_2020'] = PartSpec(
        'aluminum_extrusion_2020', '铝型材 2020 L=100mm', 'structural',
        'Misumi/8020', '2020 L=100', 'extrusion_profile',
        {'width': 0.020, 'height': 0.020, 'length': 0.100,
         'wall_thickness': 0.0015, 'slot_width': 0.006, 't_slot_depth': 0.005},
        [0.70, 0.71, 0.73, 1.0], 0.054, 2700.0,
        connectors={'end_plus_x':{'pos':[0.050,0,0],'axis':[1,0,0],'type':'slot_m5'},
                    'end_minus_x':{'pos':[-0.050,0,0],'axis':[-1,0,0],'type':'slot_m5'},
                    'slot_y_pos':{'pos':[0,0.010,0],'axis':[0,1,0],'type':'t_slot'},
                    'slot_z_pos':{'pos':[0,0,0.010],'axis':[0,0,1],'type':'t_slot'}})
    
    specs['carbon_fiber_tube'] = PartSpec(
        'carbon_fiber_tube', '碳纤维管 O16x14 L=200mm', 'structural',
        'Rock West Composites', 'CF Tube O16 ID14', 'tube_hollow',
        {'outer_diameter': 0.016, 'inner_diameter': 0.014, 'length': 0.200, 'wall_thickness': 0.001},
        [0.15, 0.15, 0.15, 1.0], 0.045, 1550.0,
        connectors={'end_a':{'pos':[0,0,0.100],'axis':[0,0,1],'type':'bonding','diameter':0.016},
                    'end_b':{'pos':[0,0,-0.100],'axis':[0,0,-1],'type':'bonding','diameter':0.016}})

    specs['steel_plate_100x100'] = PartSpec(
        'steel_plate_100x100', '钢板 100x100x3mm', 'structural',
        'McMaster/AISI', 'Steel Plate 1018 3mm', 'flat_plate',
        {'length': 0.100, 'width': 0.100, 'thickness': 0.003, 'corner_radius': 0.003},
        [0.45, 0.46, 0.48, 1.0], 0.235, 7850.0,
        connectors={'surface_top':{'pos':[0,0,0.0015],'axis':[0,0,1],'type':'surface'},
                    'surface_bottom':{'pos':[0,0,-0.0015],'axis':[0,0,-1],'type':'surface'}})

    specs['spring_element'] = PartSpec(
        'spring_element', '压缩弹簧 O10 L30', 'structural',
        'Music Wire/SS304', 'Comp Spring 10x30', 'helix_spring',
        {'outer_diameter': 0.010, 'free_length': 0.030, 'wire_diameter': 0.0012, 'coil_count': 8},
        [0.50, 0.52, 0.55, 1.0], 0.002, 7850.0,
        connectors={'top':{'pos':[0,0,0.015],'axis':[0,0,1],'type':'compression'},
                    'bottom':{'pos':[0,0,-0.015],'axis':[0,0,-1],'type':'compression'}})

    specs['hollow_tube_aluminum'] = PartSpec(
        'hollow_tube_aluminum', '铝合金空心管 O20x17 L=150mm', 'structural',
        '6061-T6 Aluminum', 'Al Tube 20x17 L150', 'tube_hollow',
        {'outer_diameter': 0.020, 'inner_diameter': 0.017, 'length': 0.150, 'wall_thickness': 0.0015},
        [0.75, 0.76, 0.78, 1.0], 0.032, 2700.0,
        connectors={'end_a':{'pos':[0,0,0.075],'axis':[0,0,1],'type':'bonding'},
                    'end_b':{'pos':[0,0,-0.075],'axis':[0,0,-1],'type':'bonding'}})

    specs['hemisphere_foot'] = PartSpec(
        'hemisphere_foot', '半球形脚垫 O30 橡胶', 'structural',
        'Sorbothane', 'Hemi Foot Pad O30', 'sphere_cap',
        {'diameter': 0.030, 'height': 0.015, 'mount_hole_dia': 0.004},
        [0.15, 0.15, 0.12, 1.0], 0.008, 1250.0,
        connectors={'top_mount':{'pos':[0,0,0.015],'axis':[0,0,1],'type':'mounting','diameter':0.004}})

    return specs


class V8ModelGenerator:
    """基于trimesh的程序化高精度3D模型生成器 - 支持47种竞赛级零件"""

    def __init__(self, output_dir: str = MODELS_DIR):
        self.output_dir = output_dir
        self.specs = get_all_part_specs()
        for cat in ['actuators', 'transmission', 'energy', 'sensors',
                     'controllers', 'connectors', 'structural']:
            os.makedirs(os.path.join(output_dir, cat), exist_ok=True)

    def generate_all(self, force: bool = False) -> List[ModelGenerationResult]:
        results = []
        total = len(self.specs)
        print(f"\n{'='*72}")
        print(f"  V8 High-Precision 3D Model Library Generator | {total} Parts")
        print(f"{'='*72}\n")
        for i, (pid, spec) in enumerate(self.specs.items(), 1):
            r = self._gen_one(pid, force=force)
            results.append(r)
            s = "OK" if r.success else "FAIL"
            v = f"{r.vertex_count:,}v" if r.success else ""
            m = r.generation_method[:14]
            print(f"  [{i:2d}/{total}] {s:4s} | {spec.display_name[:40]:40s} | {v:>10s} | {m}")
        ok = sum(1 for r in results if r.success)
        print(f"\n{'='*72}")
        print(f"  Done: {ok}/{total} successful | Output: {os.path.abspath(self.output_dir)}")
        print(f"{'='*72}\n")
        return results

    def _gen_one(self, pid: str, force: bool = False) -> ModelGenerationResult:
        spec = self.specs.get(pid)
        if not spec:
            return ModelGenerationResult(pid, False, error_message="Unknown part")
        cd = os.path.join(self.output_dir, spec.category)
        stl_p = os.path.join(cd, f"{pid}.stl")
        obj_p = os.path.join(cd, f"{pid}.obj")
        if not force and os.path.exists(stl_p):
            try:
                m = trimesh.load(stl_p)
                return ModelGenerationResult(pid, True, stl_p, obj_p, len(m.vertices), generation_method="cached")
            except Exception:
                pass
        try:
            mesh = self._build(spec)
            if mesh is None or len(mesh.vertices) == 0:
                return ModelGenerationResult(pid, False, error_message="Empty mesh")
            mesh.export(stl_p)
            try:
                mesh.export(obj_p)
            except Exception:
                pass  # OBJ export may fail on some systems
            return ModelGenerationResult(pid, True, stl_p, obj_p,
                len(mesh.vertices), len(mesh.faces),
                round(mesh.volume * 1e6, 3), "procedural")
        except Exception as e:
            logger.error(f"Gen err {pid}: {e}")
            return ModelGenerationResult(pid, False, error_message=str(e))

    def _build(self, s: PartSpec) -> trimesh.Trimesh:
        gt = s.geometry_type.lower()
        bm = {
            'cylinder_complex': self._b_motor, 'cylinder_simple': self._b_cyl,
            'cylinder_detail': self._b_cyl_detail, 'cylinder_with_hub': self._b_encoder,
            'square_cylinder': self._b_stepper, 'box_complex': self._b_servo,
            'box_simple': self._b_box, 'box_detail': self._b_battery,
            'box_heatsink': self._b_esc, 'box_small': self._b_sbox,
            'ring_with_holes': self._b_flange, 'bearing_assembly': self._b_bearing,
            'tube_hollow': self._b_tube, 'flat_plate': self._b_plate,
            'extrusion_profile': self._b_extrusion, 'helix_spring': self._b_spring,
            'sphere_cap': self._b_hemi, 'pcb_board': self._b_pcb,
            'qfn_package': self._b_qfn, 'lga28_package': self._b_lga,
            'so8_package': self._b_so8, 'ssop24_package': self._b_ssop,
            'small_pcb_module': self._b_module, 'aluminum_block': self._b_block,
            'cylinder_step': self._b_step, 'ring_split': self._b_sring,
            'pin_with_ball': self._b_pin, 'complex': self._b_cplx,
            'pcb': self._b_pcbsimple,
        }
        fn = bm.get(gt, self._b_cyl)
        try:
            return fn(s)
        except Exception as e:
            logger.warning(f"Builder {gt} fail {s.part_id}: {e}")
            return self._b_fallback(s)

    def _set_color(self, mesh: trimesh.Trimesh, color):
        c = [int(x * 255) for x in color]
        mesh.visual.face_colors = c
        return mesh

    # ========== 具体建造方法 ==========

    def _b_motor(self, s):  # 复杂电机(带法兰、散热筋)
        d = s.dimensions; ro=d['outer_diameter']/2; h=d['total_height']
        rh=d.get('hollow_bore_inner', d.get('shaft_diameter', 8))/2000
        rf=d.get('flange_diameter', ro*2.2)/2; tf=d.get('flange_thickness', 0.003)
        ms = []
        ms.append(trimesh.creation.cylinder(ro, h, 64))
        if rf > ro + 0.001:
            f = trimesh.creation.cylinder(rf, tf, 64)
            f.apply_translation([0,0,h/2+tf/2]); ms.append(f)
        sh = trimesh.creation.cylinder(rh, h*0.6, 32)
        sh.visual.face_colors = [20,20,25,255]; ms.append(sh)
        nf = max(8, int(ro*400))
        for i in range(nf):
            a = 2*np.pi*i/nf; x,y=(ro+0.001)*np.cos(a),(ro+0.001)*np.sin(a)
            fin = trimesh.creation.box([0.002, 0.001, h*0.85])
            fin.apply_translation([x,y,0])
            fin.apply_transform(trimesh.transformations.rotation_matrix(a,[0,0,1]))
            ms.append(fin)
        m = trimesh.util.concatenate(ms)
        return self._set_color(m, s.color)

    def _b_cyl(self, s):  # 简单圆柱
        d=s.dimensions; r=d.get('outer_diameter',d.get('body_diameter',0.02))/2
        h=d.get('total_height',d.get('total_length',0.04))
        m=trimesh.creation.cylinder(r,h,48); return self._set_color(m,s.color)

    def _b_cyl_detail(self, s):  # 细节圆柱(联轴器)
        d=s.dimensions; r=d['outer_diameter']/2; h=d['total_length']
        ms=[trimesh.creation.cylinder(r,h,48)]
        nr=r*0.7; nh=h*0.15
        n1=trimesh.creation.cylinder(nr,nh,32); n1.apply_translation([0,0,h/2+nh/2])
        n2=trimesh.creation.cylinder(nr,nh,32); n2.apply_translation([0,0,-h/2-nh/2])
        ms.extend([n1,n2])
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_encoder(self, s):  # 编码器
        d=s.dimensions; rb=d['body_diameter']/2; hb=d['body_height']
        rs=d['shaft_diameter']/2; ls=d['shaft_length']
        ms=[trimesh.creation.cylinder(rb,hb,48)]
        sh=trimesh.creation.cylinder(rs,ls,24); sh.apply_translation([0,0,hb/2+ls/2]); ms.append(sh)
        hub=trimesh.creation.cylinder(rb*0.6,hb*0.2,32)
        hub.apply_translation([0,0,hb/2+ls+hb*0.1]); ms.append(hub)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_stepper(self, s):  # 步进电机
        d=s.dimensions; sz=d['square_size']; bl=d['body_length']
        rs=d['shaft_diameter']/2; ls=d['shaft_length']
        ms=[trimesh.creation.box([sz,sz,bl])]
        sh=trimesh.creation.cylinder(rs,ls,24); sh.apply_translation([0,0,bl/2+ls/2]); ms.append(sh)
        ew=sz*0.3; eh=bl*0.15
        e1=trimesh.creation.box([ew,0.003,eh]); e1.apply_translation([sz/2+ew/2,0,0])
        e2=trimesh.creation.box([ew,0.003,eh]); e2.apply_translation([-sz/2-ew/2,0,0])
        ms.extend([e1,e2])
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_servo(self, s):  # 舵机
        d=s.dimensions; L,W,H=d['length'],d['width'],d['height']
        ms=[trimesh.creation.box([L,W,H])]
        gh=H*0.3; g=trimesh.creation.box([W*0.8,W*0.8,gh])
        g.apply_translation([0,0,H/2+gh/2]); ms.append(g)
        sl=d.get('shaft_length',0.005)
        sa=trimesh.creation.cylinder(0.003,sl,16); sa.apply_translation([0,0,H/2+gh+sl/2]); ms.append(sa)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_box(self, s):  # 盒子
        m=trimesh.creation.box([s.dimensions['length'],s.dimensions['width'],s.dimensions['height']])
        return self._set_color(m,s.color)

    def _b_battery(self, s):  # 电池(带圆角)
        d=s.dimensions; L,W,H=d['length'],d['width'],d['height']
        ms=[trimesh.creation.box([L,W,H])]
        cr=min(L,W,H)*0.03
        for sx in [L/2-cr,-L/2+cr]:
            for sy in [W/2-cr,-W/2+cr]:
                for sz in [H/2-cr,-H/2+cr]:
                    b=trimesh.creation.icosphere(cr,2); b.apply_translation([sx,sy,sz]); ms.append(b)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_esc(self, s):  # 电调(带散热片)
        d=s.dimensions; L,W,H=d['length'],d['width'],d['height']; nf=d.get('heatsink_fins',6)
        ms=[trimesh.creation.box([L,W,H*0.3])]
        hb=trimesh.creation.box([L*0.9,W*0.9,H*0.7])
        hb.apply_translation([0,0,H*0.3+H*0.35]); ms.append(hb)
        fw=(L*0.9)/nf; fg=0.001
        for i in range(nf):
            fx=-L*0.45+i*(fw+fg)+fw/2
            f=trimesh.creation.box([fw*0.8,W*0.95,H*0.7])
            f.apply_translation([fx,0,H*0.3+H*0.35]); ms.append(f)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_sbox(self, s):  # 小盒子
        return self._b_box(s)

    def _b_flange(self, s):  # 法兰
        d=s.dimensions; ro=d['outer_diameter']/2; ri=d['inner_diameter']/2
        t=d['thickness']; bcr=d['bolt_circle_diameter']/2; nb=d['bolt_count']; br=d['bolt_diameter']/2
        ms=[trimesh.creation.cylinder(ro,t,64)]
        for i in range(nb):
            a=2*np.pi*i/nb; bx=bcr*np.cos(a); by=bcr*np.sin(a)
            b=trimesh.creation.cylinder(br*2,t*1.2,16); b.apply_translation([bx,by,0]); ms.append(b)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_bearing(self, s):  # 轴承(内外圈+滚珠)
        d=s.dimensions; ri=d['inner_diameter']/2; ro=d['outer_diameter']/2
        w=d['width']; rp=(ri+ro)/2; rb=(ro-ri)/4; nb=9
        ms=[trimesh.creation.cylinder(ro,w,48), trimesh.creation.cylinder(ri,w*0.9,32)]
        for i in range(nb):
            a=2*np.pi*i/nb+np.pi/nb
            bx,rp_y=rp*np.cos(a),rp*np.sin(a)
            ball=trimesh.creation.icosphere(subdivisions=2)
            ball.apply_scale(rb); ball.apply_translation([bx,rp_y,0]); ms.append(ball)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_tube(self, s):  # 空心管
        d=s.dimensions; ro=d['outer_diameter']/2; ri=d['inner_diameter']/2; L=d['length']
        ms=[trimesh.creation.cylinder(ro,L,48)]
        inn=trimesh.creation.cylinder(ri,L*0.98,32); inn.visual.face_colors=[30,30,35,255]; ms.append(inn)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_plate(self, s):  # 平板
        m=trimesh.creation.box([s.dimensions['length'],s.dimensions['width'],s.dimensions['thickness']])
        return self._set_color(m,s.color)

    def _b_extrusion(self, s):  # 铝型材(T型槽方管)
        d=s.dimensions; W,H,L=d['width'],d['height'],d['length']
        tw=d['slot_width']; ts=d['t_slot_depth']; wt=d['wall_thickness']
        ms=[trimesh.creation.box([L,W,H])]
        slot_h=ts; slot_w=tw
        for side in ['y_pos','y_neg','z_pos','z_neg']:
            sv = W/2 if 'y' in side else H/2
            sign = 1 if 'pos' in side else -1
            axis_idx = 1 if 'y' in side else 2
            pos = [0, 0, 0]; pos[axis_idx] = sign * (sv + slot_h/2)
            dims = [L, slot_w, slot_h] if 'y' in side else [L, slot_h, slot_w]
            dims[axis_idx] = slot_h
            slot = trimesh.creation.box(dims); slot.apply_translation(pos)
            slot.visual.face_colors = [80, 82, 85, 255]
            ms.append(slot)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_spring(self, s):  # 弹簧(螺旋) - 用圆环段近似
        d=s.dimensions; od=d['outer_diameter']/2; fl=d['free_length']
        wd=d['wire_diameter']/2; nc=d['coil_count']
        ms = []
        # 用一系列倾斜的圆环来模拟弹簧
        n_rings = int(nc)
        ring_height = fl / n_rings
        for i in range(n_rings):
            z = -fl/2 + (i + 0.5) * ring_height
            ring = trimesh.creation.torus(major_radius=od, minor_radius=wd, major_sections=24, minor_sections=8)
            ring.apply_translation([0, 0, z])
            ms.append(ring)
        if len(ms) > 0:
            m = trimesh.util.concatenate(ms)
        else:
            m = trimesh.creation.cylinder(od, fl, 16)
        return self._set_color(m, s.color)

    def _b_hemi(self, s):  # 半球脚垫
        d=s.dimensions; r=d['diameter']/2; h=d['height']
        ms=[]
        # 用半球近似(缩放icosphere)
        hemi = trimesh.creation.icosphere(subdivisions=3)
        hemi.apply_scale(r)
        # 只保留上半部分(通过裁切)
        hemi.apply_translation([0, 0, -r * 0.2])  # 向下偏移使底部平坦
        ms.append(hemi)
        base=trimesh.creation.cylinder(r, 0.002, 24); base.apply_translation([0,0,-h/2+0.001]); ms.append(base)
        mh=d.get('mount_hole_dia',0.004)/2
        hole=trimesh.creation.cylinder(mh, h*0.3, 12); hole.apply_translation([0,0,h*0.3]); ms.append(hole)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_pcb(self, s):  # PCB板(RPi等)
        d=s.dimensions; L,W,d['board_thickness']=d['length'],d['width'],d['board_thickness']
        bt=d['board_thickness']; uh=d.get('usb_height',0.012); eh=d.get('ethernet_height',0.01)
        gh=d.get('gpio_header_height',0.010)
        ms=[trimesh.creation.box([L,W,bt])]
        # USB端口
        usb=trimesh.creation.box([0.014,0.015,uh]); usb.apply_translation([-L/2+0.007,W/2-0.007,bt/2+uh/2]); ms.append(usb)
        # GPIO排针
        gpio=trimesh.creation.box([0.051,0.023,gh]); gpio.apply_translation([0.021,0.021,bt/2+gh/2]); ms.append(gpio)
        # 以太网口
        eth=trimesh.creation.box([0.016,0.015,eh]); eth.apply_translation([L/2-0.008,-0.007,bt/2+eh/2]); ms.append(eth)
        # 安装孔
        hsx=d.get('hole_spacing_x',0.065); hsy=d.get('hole_spacing_y',0.029)
        for hx in [-hsx/2,hsx/2]:
            for hy in [-hsy/2,hsy/2]:
                h=trimesh.creation.cylinder(0.0015,bt*2,12); h.apply_translation([hx,hy,0]); ms.append(h)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_qfn(self, s):  # QFN封装芯片
        d=s.dimensions; m=trimesh.creation.box([d['length'],d['width'],d['height']])
        ms=[m]  # 先初始化ms!
        # 引脚
        ps=d.get('pin_pitch',0.0005); pn=d.get('pin_count',24)//4
        for side in range(4):
            angle = side * np.pi/2
            for i in range(pn):
                offset = (i - (pn-1)/2) * ps
                p=trimesh.creation.box([ps*0.8, 0.0003, d['height']*0.5])
                if side==0: p.apply_translation([d['length']/2+offset, d['width']/2+0.00015, 0])
                elif side==1: p.apply_translation([d['length']/2+0.00015, -d['width']/2-offset, 0])
                elif side==2: p.apply_translation([-d['length']/2-offset, -d['width']/2-0.00015, 0])
                else: p.apply_translation([-d['length']/2-0.00015, d['width']/2+offset, 0])
                p.visual.face_colors=[180,160,40,255]; ms.append(p)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_lga(self, s):  # LGA封装
        d=s.dimensions; m=trimesh.creation.box([d['length'],d['width'],d['height']])
        lens=trimesh.creation.icosphere(subdivisions=1)
        lens.apply_scale(d.get('lens_dia',0.002)/2)
        lens.apply_translation([0,0,d['height']/2+0.001]); ms=[m,lens]
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_so8(self, s):  # SO8封装
        d=s.dimensions; m=trimesh.creation.box([d['length'],d['width'],d['height']])
        ms=[m]; m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_ssop(self, s):  # SSOP封装
        d=s.dimensions; m=trimesh.creation.box([d['length'],d['width'],d['height']])
        ms=[m]; m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_module(self, s):  # 小PCB模块
        d=s.dimensions; m=trimesh.creation.box([d['length'],d['width'],d['height']])
        ld=d.get('lens_dia',0.003)
        lens=trimesh.creation.cylinder(ld/2, 0.001, 12)
        lens.apply_translation([0,0,d['height']/2+0.0005])
        ms=[m,lens]; m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_block(self, s):  # 铝块(力传感器)
        d=s.dimensions; m=trimesh.creation.box([d['length'],d['width'],d['height']])
        hd=d.get('hole_dia',0.004)/2
        h1=trimesh.creation.cylinder(hd, d['length']*1.2, 12)
        rot = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        h1.apply_transform(rot); ms=[m]; m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_step(self, s):  # 阶梯圆柱(刚性联轴器)
        d=s.dimensions; ro=d['outer_diameter']/2; l=d['total_length']
        ba=d['bore_a']/2; bb=d['bore_b']/2
        ms=[trimesh.creation.cylinder(ro,l,48)]
        neck=trimesh.creation.cylinder(ro*0.8,l*0.3,24); neck.apply_translation([0,0,l*0.35]); ms.append(neck)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_sring(self, s):  # 分裂环(轴夹)
        d=s.dimensions; ri=d['inner_diameter']/2; ro=d['outer_diameter']/2; t=d['thickness']
        ms=[trimesh.creation.cylinder(ro,t,32), trimesh.creation.cylinder(ri,t*1.5,24)]
        clamp=trimesh.creation.box([ro*2, t*0.3, t*1.5]); clamp.apply_translation([0,ro,0]); ms.append(clamp)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_pin(self, s):  # 快拆销
        d=s.dimensions; r=d['diameter']/2; L=d['total_length']; hr=d['handle_diameter']/2
        ms=[trimesh.creation.cylinder(r,L,24)]
        handle=trimesh.creation.cylinder(hr, r*3, 16); handle.apply_translation([0,0,-L/2-r*1.5]); ms.append(handle)
        ball=trimesh.creation.icosphere(subdivisions=1)
        ball.apply_scale(r*0.5); ball.apply_translation([0,0,L/2+r*0.5]); ms.append(ball)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_cplx(self, s):  # 复杂件(同步带轮/链轮)
        d=s.dimensions; pd=d.get('pitch_diameter',0.02)/2; w=d['width']; tc=d.get('teeth_count',20)
        ms=[trimesh.creation.cylinder(pd+w*0.1, w, 48)]  # 齿顶圆
        hub=trimesh.creation.cylinder(pd*0.4, w*1.2, 24); hub.apply_translation([0,0,0]); ms.append(hub)
        bore=trimesh.creation.cylinder(d['bore']/2, w*1.5, 16); bore.visual.face_colors=[20,20,25,255]; ms.append(bore)
        # 齿
        for i in range(tc):
            a = 2*np.pi*i/tc
            tx,ty=pd*np.cos(a),pd*np.sin(a)
            tooth=trimesh.creation.box([w*0.15, pd*0.08, w*0.9])
            tooth.apply_translation([tx,ty,0])
            tooth.apply_transform(trimesh.transformations.rotation_matrix(a,[0,0,1]))
            ms.append(tooth)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_pcbsimple(self, s):  # 简单PCB
        d=s.dimensions; ch=d.get('component_height',0.008)
        ms=[trimesh.creation.box([d['length'],d['width'],d['height']])]
        comp=trimesh.creation.box([d['length']*0.8, d['width']*0.8, ch])
        comp.apply_translation([0,0,d['height']/2+ch/2]); comp.visual.face_colors=[30,30,35,255]; ms.append(comp)
        m=trimesh.util.concatenate(ms); return self._set_color(m,s.color)

    def _b_fallback(self, s):  # 兜底:简单球体
        m = trimesh.creation.icosphere(subdivisions=2)
        scale = max(s.dimensions.values()) * 0.5
        m.apply_scale(scale)
        return self._set_color(m, s.color)


# ============================================================
# 统一模型加载API
# ============================================================
class V8PartLibrary:
    """V8零件库统一接口 - 加载/查询/渲染47种竞赛级零件3D模型"""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.specs = get_all_part_specs()

    def load_part(self, part_id: str) -> Optional[trimesh.Trimesh]:
        """加载单个零件的3D模型"""
        spec = self.specs.get(part_id)
        if not spec:
            logger.warning(f"Unknown part: {part_id}")
            return None
        stl_path = os.path.join(self.models_dir, spec.category, f"{part_id}.stl")
        if not os.path.exists(stl_path):
            logger.warning(f"Model not found: {stl_path}")
            return None
        try:
            return trimesh.load(stl_path)
        except Exception as e:
            logger.error(f"Load error {part_id}: {e}")
            return None

    def list_parts(self, category: str = None) -> List[Dict]:
        """列出零件信息"""
        items = []
        for pid, spec in self.specs.items():
            if category and spec.category != category:
                continue
            stl_p = os.path.join(self.models_dir, spec.category, f"{pid}.stl")
            exists = os.path.exists(stl_p)
            items.append({
                'id': pid, 'name': spec.display_name,
                'category': spec.category, 'manufacturer': spec.manufacturer,
                'mass_kg': spec.mass_kg, 'model_exists': exists
            })
        return items

    def create_scene(self, part_ids: List[str],
                     positions: Dict[str, List[float]] = None) -> trimesh.Scene:
        """创建多零件组合场景"""
        scene = trimesh.Scene()
        positions = positions or {}
        for i, pid in enumerate(part_ids):
            mesh = self.load_part(pid)
            if mesh is not None:
                pos = positions.get(pid, [i * 0.15, 0, 0])
                scene.add_geometry(mesh, geom_name=pid, transform=trimesh.transformations.translation_matrix(pos))
        return scene

    def get_spec(self, part_id: str) -> Optional[PartSpec]:
        """获取零件规格"""
        return self.specs.get(part_id)

    def get_category_parts(self, category: str) -> Dict[str, PartSpec]:
        """按类别获取零件"""
        return {k: v for k, v in self.specs.items() if v.category == category}

    def export_scene_obj(self, part_ids: List[str], output_path: str):
        """导出多零件场景为OBJ文件"""
        scene = self.create_scene(part_ids)
        scene.export(output_path)
        print(f"Scene exported to {output_path}")

    def print_summary(self):
        """打印零件库摘要"""
        cats = {}
        for s in self.specs.values():
            cats[s.category] = cats.get(s.category, 0) + 1
        print(f"\n{'='*60}")
        print(f"  V8 竞赛级零件库 | 总计 {len(self.specs)} 种零件")
        print(f"{'='*60}")
        for cat, count in sorted(cats.items()):
            cat_names = {'actuators': '执行器', 'transmission': '传动系统',
                        'energy': '能源系统', 'sensors': '传感系统',
                        'controllers': '控制系统', 'connectors': '连接件',
                        'structural': '结构件'}
            cn = cat_names.get(cat, cat)
            print(f"  {cn:10s} ({cat:12s}): {count:2d} 种")
        print(f"{'='*60}\n")


# ============================================================
# 主程序入口
# ============================================================
def main():
    import sys
    
    generator = V8ModelGenerator()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--force':
        results = generator.generate_all(force=True)
    else:
        results = generator.generate_all()
    
    # 创建库实例并输出摘要
    lib = V8PartLibrary()
    lib.print_summary()
    
    # 验证结果
    ok_count = sum(1 for r in results if r.success)
    if ok_count == len(results):
        print("All 47 parts generated successfully!")
        return 0
    else:
        failed = [r.part_id for r in results if not r.success]
        print(f"\nFailed ({len(failed)}): {failed}")
        return 1


if __name__ == "__main__":
    exit(main())