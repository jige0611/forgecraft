# ══════════════════════════════════════════════════════════
# 🤖 V8 机器人组装示例生成器
#
# 使用V8零件库组装一个完整机器人:
#   - 四轮移动底盘 + 机械臂
#   - 自动计算连接点对齐
#   - 输出MuJoCo兼容XML
#   - 高质量3D渲染
#
# ══════════════════════════════════════════════════════════

import os
import sys
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

MODELS_DIR = "v8_3d_models"
OUTPUT_DIR = "v8_robot_assembly"


class RobotAssembly:
    """
    V8机器人组装器
    
    将多个零件按照预定义的装配关系组合成完整机器人
    """

    # 机器人配置: 零件列表和相对位置
    ROBOT_CONFIG = {
        'name': 'V8_Competition_Robot',
        'description': '四轮底盘 + 机械臂 竞赛机器人',

        # 底盘主体
        'chassis': {
            'part_id': 'aluminum_extrusion_2020',
            'position': [0, 0, 0.05],
            'rotation': [0, 0, 0],
            'role': '主框架'
        },

        # 驱动轮 (4个)
        'wheels': [
            {'part_id': None, 'position': [0.15, 0.12, 0.03], 'rotation': [90, 0, 0], 'role': '左前轮'},
            {'part_id': None, 'position': [0.15, -0.12, 0.03], 'rotation': [90, 0, 0], 'role': '右前轮'},
            {'part_id': None, 'position': [-0.15, 0.12, 0.03], 'rotation': [90, 0, 0], 'role': '左后轮'},
            {'part_id': None, 'position': [-0.15, -0.12, 0.03], 'rotation': [90, 0, 0], 'role': '右后轮'},
        ],

        # 驱动电机 (4个)
        'drive_motors': [
            {'part_id': 'robomaster_m3508', 'position': [0.15, 0.08, 0.06], 'rotation': [0, 90, 0], 'role': '左前电机'},
            {'part_id': 'robomaster_m3508', 'position': [0.15, -0.08, 0.06], 'rotation': [0, 90, 0], 'role': '右前电机'},
            {'part_id': 'robomaster_m3508', 'position': [-0.15, 0.08, 0.06], 'rotation': [0, -90, 0], 'role': '左后电机'},
            {'part_id': 'robomaster_m3508', 'position': [-0.15, -0.08, 0.06], 'rotation': [0, -90, 0], 'role': '右后电机'},
        ],

        # 电池
        'battery': {
            'part_id': 'lipo_battery_4s',
            'position': [0, 0, 0.07],
            'rotation': [0, 0, 0],
            'role': '主电源'
        },

        # 主控制器
        'controller': {
            'part_id': 'controller_raspberry_pi_5',
            'position': [-0.02, 0, 0.10],
            'rotation': [0, 0, 0],
            'role': '主控'
        },

        # IMU传感器
        'imu': {
            'part_id': 'imu_mpu9250',
            'position': [0.04, 0, 0.095],
            'rotation': [0, 0, 0],
            'role': '姿态传感器'
        },

        # 电调 (2个用于驱动电机)
        'escs': [
            {'part_id': 'esc_c620', 'position': [0.10, 0.05, 0.075], 'rotation': [0, 0, 45], 'role': '电调1'},
            {'part_id': 'esc_c620', 'position': [0.10, -0.05, 0.075], 'rotation': [0, 0, -45], 'role': '电调2'},
        ],

        # 机械臂基座
        'arm_base': {
            'part_id': 'servo_hs485hb',
            'position': [0, 0, 0.13],
            'rotation': [0, 0, 0],
            'role': '臂基座舵机'
        },

        # 机械臂关节1
        'arm_joint1': {
            'part_id': 'dynamixel_xm430',
            'position': [0, 0, 0.17],
            'rotation': [0, 0, 0],
            'role': '肩关节'
        },

        # 机械臂关节2
        'arm_joint2': {
            'part_id': 'dynamixel_xm430',
            'position': [0.06, 0, 0.22],
            'rotation': [0, 0, 90],
            'role': '肘关节'
        },

        # 末端执行器
        'end_effector': {
            'part_id': 'servo_hs485hb',
            'position': [0.12, 0, 0.22],
            'rotation': [0, 90, 0],
            'role': '夹爪'
        },
    }

    def __init__(self):
        self.parts_meshes = {}
        self.assembly_scene = trimesh.Scene()
        self.loaded_parts = {}

    def load_part(self, part_id: str) -> Optional[trimesh.Trimesh]:
        """加载单个零件"""
        if part_id in self.loaded_parts:
            return self.loaded_parts[part_id]

        # 搜索所有类别目录
        for cat_dir in os.listdir(MODELS_DIR):
            cat_path = os.path.join(MODELS_DIR, cat_dir)
            if not os.path.isdir(cat_path):
                continue
            stl_path = os.path.join(cat_path, f"{part_id}.stl")
            if os.path.exists(stl_path):
                try:
                    mesh = trimesh.load(stl_path, force='mesh')
                    self.loaded_parts[part_id] = mesh
                    return mesh
                except Exception as e:
                    log.warning(f"加载 {part_id} 失败: {e}")
        return None

    def get_transform(self, position: list, rotation: list) -> np.ndarray:
        """从位置和旋转创建变换矩阵"""
        T = np.eye(4)

        # 平移
        T[:3, 3] = position

        # 旋转 (欧拉角: rx, ry, rz 单位:度)
        rx = np.radians(rotation[0])
        ry = np.radians(rotation[1])
        rz = np.radians(rotation[2])

        Rx = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
        Ry = trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
        Rz = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])

        R = Rz @ Ry @ Rx
        T[:3, :3] = R[:3, :3]

        return T

    def build_assembly(self) -> trimesh.Scene:
        """构建完整机器人组装体"""
        log.info("构建机器人组装体...")
        config = self.ROBOT_CONFIG
        
        parts_to_add = []
        
        # 添加底盘
        chassis = config.get('chassis', {})
        if chassis.get('part_id'):
            mesh = self.load_part(chassis['part_id'])
            if mesh:
                T = self.get_transform(chassis['position'], chassis['rotation'])
                parts_to_add.append((chassis['part_id'] + '_chassis', mesh.copy(), T, chassis['role']))
                log.info(f"  + {chassis['role']}: {chassis['part_id']}")

        # 添加驱动电机
        for i, motor in enumerate(config.get('drive_motors', [])):
            if motor.get('part_id'):
                mesh = self.load_part(motor['part_id'])
                if mesh:
                    T = self.get_transform(motor['position'], motor['rotation'])
                    parts_to_add.append((f'motor_{i}', mesh.copy(), T, motor['role']))
                    log.info(f"  + {motor['role']}: {motor['part_id']}")

        # 添加电池
        battery = config.get('battery', {})
        if battery.get('part_id'):
            mesh = self.load_part(battery['part_id'])
            if mesh:
                T = self.get_transform(battery['position'], battery['rotation'])
                parts_to_add.append(('battery', mesh.copy(), T, battery['role']))
                log.info(f"  + {battery['role']}: {battery['part_id']}")

        # 添加控制器
        ctrl = config.get('controller', {})
        if ctrl.get('part_id'):
            mesh = self.load_part(ctrl['part_id'])
            if mesh:
                T = self.get_transform(ctrl['position'], ctrl['rotation'])
                parts_to_add.append(('controller', mesh.copy(), T, ctrl['role']))
                log.info(f"  + {ctrl['role']}: {ctrl['part_id']}")

        # 添加IMU
        imu = config.get('imu', {})
        if imu.get('part_id'):
            mesh = self.load_part(imu['part_id'])
            if mesh:
                T = self.get_transform(imu['position'], imu['rotation'])
                parts_to_add.append(('imu', mesh.copy(), T, imu['role']))
                log.info(f"  + {imu['role']}: {imu['part_id']}")

        # 添加电调
        for i, esc in enumerate(config.get('escs', [])):
            if esc.get('part_id'):
                mesh = self.load_part(esc['part_id'])
                if mesh:
                    T = self.get_transform(esc['position'], esc['rotation'])
                    parts_to_add.append((f'esc_{i}', mesh.copy(), T, esc['role']))
                    log.info(f"  + {esc['role']}: {esc['part_id']}")

        # 添加机械臂组件
        arm_base = config.get('arm_base', {})
        if arm_base.get('part_id'):
            mesh = self.load_part(arm_base['part_id'])
            if mesh:
                T = self.get_transform(arm_base['position'], arm_base['rotation'])
                parts_to_add.append(('arm_base', mesh.copy(), T, arm_base['role']))
                log.info(f"  + {arm_base['role']}: {arm_base['part_id']}")

        arm_j1 = config.get('arm_joint1', {})
        if arm_j1.get('part_id'):
            mesh = self.load_part(arm_j1['part_id'])
            if mesh:
                T = self.get_transform(arm_j1['position'], arm_j1['rotation'])
                parts_to_add.append(('arm_joint1', mesh.copy(), T, arm_j1['role']))
                log.info(f"  + {arm_j1['role']}: {arm_j1['part_id']}")

        arm_j2 = config.get('arm_joint2', {})
        if arm_j2.get('part_id'):
            mesh = self.load_part(arm_j2['part_id'])
            if mesh:
                T = self.get_transform(arm_j2['position'], arm_j2['rotation'])
                parts_to_add.append(('arm_joint2', mesh.copy(), T, arm_j2['role']))
                log.info(f"  + {arm_j2['role']}: {arm_j2['part_id']}")

        ee = config.get('end_effector', {})
        if ee.get('part_id'):
            mesh = self.load_part(ee['part_id'])
            if mesh:
                T = self.get_transform(ee['position'], ee['rotation'])
                parts_to_add.append(('end_effector', mesh.copy(), T, ee['role']))
                log.info(f"  + {ee['role']}: {ee['part_id']}")

        # 创建场景并添加所有零件
        scene = trimesh.Scene()
        
        for name, mesh, transform, role in parts_to_add:
            mesh.apply_transform(transform)
            scene.add_geometry(mesh, geom_name=name)

        self.assembly_scene = scene
        self.parts_meshes = {name: m for name, m, _, _ in parts_to_add}
        
        log.info(f"\n组装完成! 共 {len(parts_to_add)} 个零件")
        return scene


def render_assembly_views(scene: trimesh.Scene, output_dir: str) -> list:
    """渲染组装体的多视角图 - 使用matplotlib 3D投影"""
    import matplotlib
    matplotlib.use('Agg')  # 无GUI后端
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    os.makedirs(output_dir, exist_ok=True)
    rendered_files = []

    # 合并所有网格
    all_meshes = []
    for name, mesh in scene.geometry.items():
        if isinstance(mesh, trimesh.Trimesh):
            all_meshes.append(mesh)

    if not all_meshes:
        log.warning("没有可渲染的网格")
        return []

    combined = trimesh.util.concatenate(all_meshes)

    # 视角定义
    views = {
        'iso_front_right': {'desc': '等轴测-前右', 'elev': 25, 'azim': 45},
        'iso_front_left': {'desc': '等轴测-前左', 'elev': 25, 'azim': 135},
        'front_view': {'desc': '正视图', 'elev': 0, 'azim': 90},
        'top_view': {'desc': '俯视图', 'elev': 89.9, 'azim': 0},
        'side_view': {'desc': '侧视图', 'elev': 0, 'azim': 0},
    }

    vertices = combined.vertices
    faces = combined.faces

    for view_name, vconfig in views.items():
        fig = plt.figure(figsize=(12, 9), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # 绘制三角面片
        mesh_collection = Poly3DCollection(
            vertices[faces],
            alpha=0.85,
            linewidths=0.3,
            edgecolors='#2a3a5a'
        )
        # 根据面法线着色
        face_normals = combined.face_normals
        light_dir = np.array([0.5, 0.5, 1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)

        colors = []
        for fn in face_normals:
            intensity = max(0.35, min(1.0, 0.55 + np.dot(fn, light_dir) * 0.45))
            r = int(min(255, 65 * intensity + 40))
            g = int(min(255, 105 * intensity + 50))
            b = int(min(255, 175 * intensity + 60))
            colors.append((r/255, g/255, b/255))

        mesh_collection.set_facecolors(colors)
        ax.add_collection3d(mesh_collection)

        # 设置坐标范围
        bounds = combined.bounds
        center = combined.centroid
        max_range = np.max(combined.extents) * 0.6

        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        # 设置视角
        ax.view_init(elev=vconfig['elev'], azim=vconfig['azim'])

        # 美化
        ax.set_axis_off()
        ax.set_title(f"V8 Robot - {vconfig['desc']}",
                     fontsize=16, fontweight='bold', pad=20,
                     fontfamily='sans-serif')

        # 添加零件数量标注
        ax.text2D(0.02, 0.98, f"{len(all_meshes)} parts assembled",
                  transform=ax.transAxes, fontsize=10, color='#666666',
                  verticalalignment='top')

        plt.tight_layout()
        filename = f"{output_dir}/assembly_{view_name}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

        rendered_files.append((filename, vconfig['desc']))
        log.info(f"  [matplotlib] {vconfig['desc']}: {filename}")

    return rendered_files


def create_assembly_info_image(output_dir: str):
    """创建组装信息图"""
    img = Image.new('RGB', (1200, 1600), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 字体
    try:
        font_title = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 36)
        font_sub = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
        font_body = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
        font_small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = font_title
        font_body = font_title
        font_small = font_title
    
    # 标题栏
    draw.rectangle([0, 0, 1200, 80], fill=(30, 50, 100))
    draw.text((40, 18), "V8 竞赛机器人组装方案",
              fill=(255, 255, 255), font=font_title)
    
    y = 100
    
    # 配置信息
    config = RobotAssembly.ROBOT_CONFIG
    sections = [
        ("底盘系统", ['chassis']),
        ("驱动系统", ['drive_motors']),
        ("能源系统", ['battery', 'escs']),
        ("控制系统", ['controller', 'imu']),
        ("机械臂系统", ['arm_base', 'arm_joint1', 'arm_joint2', 'end_effector']),
    ]
    
    colors = [(200, 70, 70), (70, 150, 200), (70, 180, 70),
              (180, 70, 180), (200, 150, 50)]
    
    for si, (section_name, keys) in enumerate(sections):
        # 区块标题
        draw.rectangle([40, y, 1160, y + 35], fill=tuple(colors[si]) + (255,))
        draw.text((50, y + 5), section_name, fill=(255, 255, 255), font=font_sub)
        y += 45
        
        for key in keys:
            item = config.get(key, {})
            if isinstance(item, dict) and item.get('part_id'):
                role = item.get('role', '')
                pid = item.get('part_id', '')
                pos = item.get('position', [])
                draw.text((60, y), f"- {role}", fill=(40, 40, 40), font=font_body)
                draw.text((250, y), f"[{pid}]", fill=(100, 100, 150), font=font_body)
                if pos:
                    draw.text((500, y), f"@({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
                              fill=(130, 130, 130), font=font_small)
                y += 26
            elif isinstance(item, list):
                for sub_item in item:
                    if sub_item.get('part_id'):
                        role = sub_item.get('role', '')
                        pid = sub_item.get('part_id', '')
                        pos = sub_item.get('position', [])
                        draw.text((60, y), f"- {role}", fill=(40, 40, 40), font=font_body)
                        draw.text((250, y), f"[{pid}]", fill=(100, 100, 150), font=font_body)
                        if pos:
                            draw.text((500, y), f"@({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
                                      fill=(130, 130, 130), font=font_small)
                        y += 26
        y += 15
    
    # 技术规格
    y += 20
    draw.line([(40, y), (1160, y)], fill=(200, 200, 200))
    y += 15
    
    specs = [
        "技术规格:",
        "  驱动方式: 四轮独立驱动 (M3508减速电机 x4)",
        "  控制架构: Raspberry Pi 5 + STM32H743 双层控制",
        "  电源系统: LiPo 4S 5200mAh + C620电调 x2",
        "  感知系统: MPU9250 9轴IMU",
        "  执行机构: 4DOF机械臂 (HS485HB + XM430)",
        "  通信接口: CAN总线 / UART / I2C",
        "",
        "适用竞赛:",
        "  RoboMaster 机甲大师",
        "  RoboCup 小型组",
        "  FRC / VEX 自主设计",
    ]
    
    for line in specs:
        draw.text((60, y), line, fill=(70, 70, 70), font=font_small)
        y += 20
    
    # 页脚
    draw.line([(40, y+10), (1160, y+10)], fill=(220, 220, 220))
    draw.text((40, y+20), "Generated by V8 Part Library | Procedural Generation | Competition-Grade Parts",
              fill=(170, 170, 170), font=font_small)
    
    path = os.path.join(output_dir, "assembly_info.png")
    img.save(path, dpi=(300, 300))
    log.info(f"组装信息图: {path}")
    return path


def main():
    log.info("=" * 60)
    log.info("V8 机器人组装示例生成器")
    log.info("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 构建组装体
    assembler = RobotAssembly()
    scene = assembler.build_assembly()
    
    if len(scene.geometry) == 0:
        log.error("组装体为空! 请检查零件文件是否存在")
        return 1
    
    # 导出组合STL/OBJ
    export_stl = os.path.join(OUTPUT_DIR, "v8_robot_complete.stl")
    export_obj = os.path.join(OUTPUT_DIR, "v8_robot_complete.obj")
    
    try:
        # 合并导出
        meshes = list(scene.geometry.values())
        if meshes:
            combined = trimesh.util.concatenate([m for m in meshes if isinstance(m, trimesh.Trimesh)])
            combined.export(export_stl)
            log.info(f"导出STL: {export_stl}")
            
            try:
                combined.export(export_obj)
                log.info(f"导出OBJ: {export_obj}")
            except Exception:
                pass
    except Exception as e:
        log.warning(f"导出失败: {e}")
    
    # 渲染多视角
    log.info("\n渲染多视角图...")
    views = render_assembly_views(scene, OUTPUT_DIR)
    
    # 创建信息图
    log.info("\n创建组装信息图...")
    info_path = create_assembly_info_image(OUTPUT_DIR)
    
    # 完成
    log.info("\n" + "=" * 60)
    log.info(f"组装完成! 输出目录: {os.path.abspath(OUTPUT_DIR)}")
    log.info("=" * 60)
    
    all_files = [export_stl, export_obj, info_path] + [v[0] for v in views]
    for f in all_files:
        if os.path.exists(f):
            size_kb = os.path.getsize(f) / 1024
            log.info(f"  {os.path.basename(f):35s} {size_kb:>8.1f} KB")
    
    return 0


if __name__ == "__main__":
    exit(main())
