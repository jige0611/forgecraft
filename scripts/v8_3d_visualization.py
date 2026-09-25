# ══════════════════════════════════════════════════════════
# 🎨 V8 零件库 3D 可视化系统 (纯PIL线框版)
#
# 功能:
#   ✅ 全部47种零件网格渲染预览
#   ✅ 按7个类别分组展示
#   ✅ 典型机器人装配体3D场景
#   ✅ 完整机器人STL/OBJ导出
#   ✅ 模型质量验证报告
#
# ══════════════════════════════════════════════════════════

import os
import sys
import math
import numpy as np
from datetime import datetime

import trimesh
from trimesh import transformations
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 配置
# ============================================================
MODELS_DIR = "v8_3d_models"
OUTPUT_DIR = "v8_visualizations"
PREVIEW_SIZE = 200
CATEGORIES = {
    'actuators': '执行器 (12)',
    'transmission': '传动系统 (6)',
    'energy': '能源系统 (8)',
    'sensors': '传感系统 (6)',
    'controllers': '控制系统 (4)',
    'connectors': '连接件 (5)',
    'structural': '结构件 (6)'
}

CATEGORY_COLORS = {
    'actuators': (220, 60, 50),
    'transmission': (50, 140, 220),
    'energy': (230, 190, 30),
    'sensors': (60, 200, 80),
    'controllers': (150, 60, 210),
    'connectors': (160, 120, 70),
    'structural': (110, 125, 140)
}


class WireframeRenderer:
    """纯PIL线框渲染器 - 不依赖OpenGL"""

    def __init__(self):
        pass

    @staticmethod
    def render_mesh(mesh: trimesh.Trimesh, size: int = PREVIEW_SIZE,
                   angle_x: float = 25, angle_y: float = 35,
                   line_color: tuple = None,
                   bg_color: tuple = (248, 250, 252),
                   line_width: int = 1) -> Image.Image:
        """
        渲染单个网格为线框图
        
        Args:
            mesh: trimesh对象
            size: 输出尺寸
            angle_x: X轴旋转角度(度)
            angle_y: Y轴旋转角度(度)
        """
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return Image.new('RGB', (size, size), bg_color)

        img = Image.new('RGB', (size, size), bg_color)
        draw = ImageDraw.Draw(img)

        if line_color is None:
            line_color = (70, 100, 160)

        # 复制并变换网格
        vertices = mesh.vertices.copy()
        
        # 居中
        center = vertices.mean(axis=0)
        vertices -= center
        
        # 缩放以适应画布
        extents = np.abs(vertices).max(axis=0)
        max_extent = max(extents) if max(extents) > 0 else 1.0
        scale = (size * 0.4) / max_extent
        vertices *= scale
        
        # 旋转
        rx = transformations.rotation_matrix(np.radians(angle_x), [1, 0, 0])
        ry = transformations.rotation_matrix(np.radians(angle_y), [0, 1, 0])
        vertices = transformations.transform_points(vertices, rx @ ry)
        
        # 投影到2D (正交投影，忽略Y轴深度)
        points_2d = []
        for v in vertices:
            x = int(v[0] + size / 2)
            y = int(size / 2 - v[2])  # 翻转Y轴
            x = max(2, min(size-2, x))
            y = max(2, min(size-2, y))
            points_2d.append((x, y))

        # 提取边
        edges = set()
        for face in mesh.faces:
            n = len(face) if hasattr(face, '__len__') else 3
            for i in range(n):
                e = tuple(sorted([int(face[i]), int(face[(i+1) % n])]))
                edges.add(e)

        # 绘制边框(限制数量避免太慢)
        edge_list = list(edges)
        if len(edge_list) > 800:
            # 对边进行采样
            indices = np.linspace(0, len(edge_list)-1, 800, dtype=int)
            edge_list = [edge_list[i] for i in indices]

        for e in edge_list:
            if e[0] < len(points_2d) and e[1] < len(points_2d):
                draw.line([points_2d[e[0]], points_2d[e[1]]],
                         fill=line_color, width=line_width)

        return img


class V8Visualizer:
    """V8零件库3D可视化系统"""

    def __init__(self, models_dir: str = MODELS_DIR, output_dir: str = OUTPUT_DIR):
        self.models_dir = models_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.renderer = WireframeRenderer()
        self.parts = {}
        self._load_all_parts()

    def _load_all_parts(self):
        """加载所有STL模型"""
        for cat in CATEGORIES:
            cat_dir = os.path.join(self.models_dir, cat)
            if not os.path.exists(cat_dir):
                continue
            for fname in sorted(os.listdir(cat_dir)):
                if fname.endswith('.stl'):
                    pid = fname.replace('.stl', '')
                    path = os.path.join(cat_dir, fname)
                    try:
                        mesh = trimesh.load(path)
                        if mesh is not None and len(mesh.vertices) > 0:
                            self.parts[pid] = {
                                'mesh': mesh,
                                'category': cat,
                                'path': path,
                                'vertices': len(mesh.vertices),
                                'faces': len(mesh.faces),
                                'volume_mm3': round(mesh.volume * 1e6, 2)
                            }
                    except Exception as e:
                        print(f"  ⚠️ Load error {pid}: {e}")

        print(f"✅ Loaded {len(self.parts)} parts from {self.models_dir}")

    # ================================================================
    # 单个零件渲染
    # ================================================================
    def render_part_preview(self, part_id: str, size: int = PREVIEW_SIZE) -> Image.Image:
        """渲染单个零件预览"""
        if part_id not in self.parts:
            return None

        mesh = self.parts[part_id]['mesh']
        color = CATEGORY_COLORS.get(self.parts[part_id]['category'], (70, 100, 160))

        return self.renderer.render_mesh(
            mesh, size=size,
            angle_x=20 + hash(part_id) % 30,
            angle_y=30 + (hash(part_id) >> 8) % 40,
            line_color=color
        )

    # ================================================================
    # 分类展示图
    # ================================================================
    def render_category_grid(self, category: str, cols: int = 4,
                             preview_size: int = PREVIEW_SIZE) -> str:
        """渲染某个类别的所有零件"""
        cat_parts = [(pid, info) for pid, info in self.parts.items()
                     if info['category'] == category]
        
        if not cat_parts:
            print(f"  ⚠️ No parts in category: {category}")
            return None

        n = len(cat_parts)
        rows = math.ceil(n / cols)
        
        padding = 12
        label_height = 38
        cell_size = preview_size + padding + label_height
        
        img_w = cols * cell_size + padding * 2
        img_h = rows * cell_size + padding * 2 + 55

        canvas = Image.new('RGB', (img_w, img_h), (245, 247, 250))
        draw = ImageDraw.Draw(canvas)

        try:
            font_title = ImageFont.truetype("arial.ttf", 22)
            font_label = ImageFont.truetype("arial.ttf", 13)
            font_sub = ImageFont.truetype("arial.ttf", 11)
        except:
            font_title = font_label = font_sub = ImageFont.load_default()

        # 标题
        title = f"{CATEGORIES.get(category, category)}"
        draw.text((padding, 10), title, fill=(35, 40, 55), font=font_title)
        
        # 类别颜色条
        cat_color = CATEGORY_COLORS.get(category, (100, 100, 100))
        draw.rectangle([padding, 42, padding + 200, 48], fill=cat_color)

        for i, (pid, info) in enumerate(cat_parts):
            col = i % cols
            row = i // cols
            
            x = padding + col * cell_size
            y = padding + 58 + row * cell_size
            
            # 渲染预览
            preview = self.render_part_preview(pid, size=preview_size)
            
            if preview is not None:
                canvas.paste(preview, (x, y))
                # 边框
                draw.rectangle([x-1, y-1, x+preview_size, y+preview_size],
                             outline=(200, 205, 215), width=1)
            else:
                draw.rectangle([x, y, x+preview_size, y+preview_size],
                             fill=(225, 228, 232), outline=(195, 198, 205))

            # 标签
            display_name = pid.replace('_', '\n')[:18]
            draw.text((x + 3, y + preview_size + 3),
                     display_name, fill=(45, 50, 65), font=font_label)
            
            stats = f"{info['vertices']}v | {info['volume_mm3']:.0f}mm³"
            draw.text((x + 3, y + preview_size + 21),
                     stats, fill=(120, 125, 135), font=font_sub)

        filename = f"{OUTPUT_DIR}/category_{category}.png"
        canvas.save(filename, quality=95)
        file_kb = os.path.getsize(filename) / 1024
        print(f"  ✅ {category}: {n} parts -> {filename} ({file_kb:.0f}KB)")
        
        return filename

    def render_all_categories(self) -> list:
        """渲染所有类别"""
        results = []
        for cat in CATEGORIES:
            filepath = self.render_category_grid(cat)
            if filepath:
                results.append(filepath)
        return results

    # ================================================================
    # 全零件总览图
    # ================================================================
    def render_overview_grid(self, cols: int = 8, preview_size: int = 180) -> str:
        """渲染全部47种零件总览大图"""
        n = len(self.parts)
        rows = math.ceil(n / cols)
        
        padding = 8
        label_height = 32
        cell_size = preview_size + padding + label_height
        
        img_w = cols * cell_size + padding * 2
        img_h = rows * cell_size + padding * 2 + 70

        canvas = Image.new('RGB', (img_w, img_h), (252, 253, 255))
        draw = ImageDraw.Draw(canvas)

        try:
            font_title = ImageFont.truetype("arial.ttf", 26)
            font_label = ImageFont.truetype("arial.ttf", 10)
            font_sub = ImageFont.truetype("arial.ttf", 13)
        except:
            font_title = font_label = font_sub = ImageFont.load_default()

        # 标题区
        timestamp = datetime.now().strftime("%Y-%m-%d")
        draw.text((padding, 8),
                 "V8 Competition-Grade Part Library",
                 fill=(25, 30, 45), font=font_title)
        draw.text((padding, 36),
                 f"{n} Parts Generated Procedurally | {timestamp}",
                 fill=(90, 95, 105), font=font_sub)
        
        # 图例
        legend_y = 54
        legend_x = padding
        for cat, color in CATEGORY_COLORS.items():
            cn = {'actuators':'Act','transmission':'Tran','energy':'Egy',
                  'sensors':'Sen','controllers':'Ctrl','connectors':'Con',
                  'structural':'Str'}.get(cat, cat[:3])
            count = sum(1 for p in self.parts.values() if p['category'] == cat)
            draw.rectangle([legend_x, legend_y, legend_x+12, legend_y+10], fill=color)
            draw.text((legend_x + 15, legend_y - 1), f"{cn}({count})",
                     fill=(70, 75, 85), font=font_label)
            legend_x += 75

        # 排列顺序
        ordered_cats = ['actuators', 'transmission', 'energy', 'sensors',
                       'controllers', 'connectors', 'structural']
        ordered_parts = []
        for cat in ordered_cats:
            cat_parts = [(pid, info) for pid, info in self.parts.items()
                        if info['category'] == cat]
            ordered_parts.extend(cat_parts)

        base_y = padding + 68
        for i, (pid, info) in enumerate(ordered_parts):
            col = i % cols
            row = i // cols
            
            x = padding + col * cell_size
            y = base_y + row * cell_size
            
            # 渲染预览
            preview = self.render_part_preview(pid, size=preview_size)
            
            if preview is not None:
                canvas.paste(preview, (x, y))
                # 类别颜色边框
                cat_color = CATEGORY_COLORS.get(info['category'], (150, 150, 150))
                draw.rectangle([x-1, y-1, x+preview_size+1, y+preview_size+1],
                             outline=cat_color, width=2)
            else:
                draw.rectangle([x, y, x+preview_size, y+preview_size],
                             fill=(230, 233, 237), outline=(200, 203, 210))

            # 标签
            short_name = pid.replace('_', '\n')[:16]
            draw.text((x + 2, y + preview_size + 2),
                     short_name, fill=(40, 45, 60), font=font_label)

        filename = f"{OUTPUT_DIR}/v8_complete_library.png"
        canvas.save(filename, quality=95)
        total_kb = os.path.getsize(filename) / 1024
        print(f"\n📊 Overview: {filename} ({total_kb:.0f} KB, {img_w}x{img_h}px)")

        return filename

    # ================================================================
    # 典型机器人装配体
    # ================================================================
    def create_assembly_scene(self, assembly_type: str = "mobile_robot") -> dict:
        """创建典型机器人装配体"""
        print(f"\n🔧 Creating assembly: {assembly_type}")

        scene = trimesh.Scene()
        added_parts = []

        if assembly_type == "mobile_robot":
            positions = {
                'aluminum_extrusion_2020': [0, 0, 0.01],
                'steel_plate_100x100': [0, 0, 0.02],
                'robomaster_m3508': [-0.06, 0.08, 0.04],
                'harmonic_drive_csd14': [-0.06, 0.08, 0.07],
                'robomaster_m3508': [0.06, 0.08, 0.04],
                'harmonic_drive_csd14': [0.06, 0.08, 0.07],
                'lipo_battery_4s': [0, -0.03, 0.04],
                'esc_vesc': [0.05, -0.01, 0.035],
                'bms_4s': [-0.05, -0.01, 0.035],
                'controller_raspberry_pi_5': [0, 0.02, 0.045],
                'mcu_stm32h743': [0.04, 0.02, 0.045],
                'imu_mpu9250': [0, -0.05, 0.05],
                'encoder_incremental_abz': [-0.06, 0.12, 0.06],
                'distance_sensor_vl53l0x': [0.09, 0, 0.05],
                'bearing_deep_groove_6000zz': [-0.06, 0.13, 0.04],
                'bearing_deep_groove_6000zz': [0.06, 0.13, 0.04],
                'coupler_rigid': [0, 0.08, 0.06],
                'carbon_fiber_tube': [0, 0, 0.08],
                'hemisphere_foot': [-0.08, -0.06, 0.005],
                'hemisphere_foot': [0.08, -0.06, 0.005],
            }
        elif assembly_type == "arm":
            positions = {
                'aluminum_extrusion_2020': [0, 0, 0],
                'robomaster_gm6020': [0, 0, 0.05],
                'harmonic_drive_csd20': [0, 0, 0.09],
                'aluminum_extrusion_2020': [0, 0, 0.14],
                'dynamixel_xm430': [0, 0, 0.17],
                'flexible_coupling_d20': [0, 0, 0.21],
                't_motor_u8': [0, 0, 0.26],
                'force_sensor_loadcell': [0, 0, 0.31],
                'imu_bno055': [0.03, 0, 0.16],
                'encoder_absolute_multi_turn': [0, 0, 0.23],
            }
        elif assembly_type == "drone":
            positions = {
                'carbon_fiber_tube': [0.1, 0, 0],
                'carbon_fiber_tube': [-0.1, 0, 0],
                'carbon_fiber_tube': [0, 0.1, 0],
                'carbon_fiber_tube': [0, -0.1, 0],
                'aluminum_extrusion_2020': [0, 0, 0],
                't_motor_u8': [0.1, 0, 0.02],
                't_motor_u8': [-0.1, 0, 0.02],
                't_motor_u8': [0, 0.1, 0.02],
                't_motor_u8': [0, -0.1, 0.02],
                'lipo_battery_4s': [0, 0, -0.02],
                'controller_raspberry_pi_5': [0, 0.02, 0.01],
                'esc_vesc': [0.05, 0, 0],
                'esc_vesc': [-0.05, 0, 0],
                'imu_mpu9250': [0, -0.02, 0.01],
            }
        else:
            positions = {}

        for pid, pos in positions.items():
            if pid not in self.parts:
                matches = [p for p in self.parts if pid.split('_')[0] in p or pid in p]
                if matches:
                    pid = matches[0]
                else:
                    continue

            mesh = self.parts[pid]['mesh'].copy()
            scale = 0.001
            mesh.apply_scale(scale)
            mesh.apply_translation(pos)
            
            scene.add_geometry(mesh, geom_name=f"{pid}_{len(added_parts)}",
                             transform=transformations.translation_matrix(pos))
            added_parts.append(pid)

        print(f"  📦 Added {len(added_parts)} components")

        result = {}

        # 导出STL
        stl_path = f"{OUTPUT_DIR}/assembly_{assembly_type}.stl"
        try:
            if len(scene.geometry) > 0:
                combined = trimesh.util.concatenate(list(scene.geometry.values()))
                combined.export(stl_path)
                stl_kb = os.path.getsize(stl_path) / 1024
                print(f"  💾 STL: {stl_path} ({stl_kb:.0f}KB)")
                result['stl'] = stl_path
        except Exception as e:
            print(f"  ⚠️ STL error: {e}")

        # 导出OBJ
        obj_path = f"{OUTPUT_DIR}/assembly_{assembly_type}.obj"
        try:
            scene.export(obj_path)
            obj_kb = os.path.getsize(obj_path) / 1024
            print(f"  💾 OBJ: {obj_path} ({obj_kb:.0f}KB)")
            result['obj'] = obj_path
        except Exception as e:
            print(f"  ⚠️ OBJ error: {e}")

        # 线框渲染
        render_path = f"{OUTPUT_DIR}/assembly_{assembly_type}_render.png"
        try:
            if len(scene.geometry) > 0:
                combined = trimesh.util.concatenate(list(scene.geometry.values()))
                render_img = self.renderer.render_mesh(combined, size=1200,
                                                       angle_x=20, angle_y=30)
                render_img.save(render_path)
                r_kb = os.path.getsize(render_path) / 1024
                print(f"  🖼️ Render: {render_path} ({r_kb:.0f}KB)")
                result['render'] = render_path
        except Exception as e:
            print(f"  ⚠️ Render error: {e}")

        return result

    # ================================================================
    # 质量验证报告
    # ================================================================
    def generate_quality_report(self) -> dict:
        """生成模型质量验证报告"""
        report = {
            'total_parts': len(self.parts),
            'timestamp': datetime.now().isoformat(),
            'categories': {},
            'quality_stats': {
                'total_vertices': 0, 'total_faces': 0,
                'avg_vertices': 0, 'avg_faces': 0,
                'min_vertices': float('inf'), 'max_vertices': 0,
                'watertight_count': 0, 'manifold_count': 0,
            },
            'issues': []
        }

        for cat in CATEGORIES:
            cat_parts = [(pid, info) for pid, info in self.parts.items()
                        if info['category'] == cat]
            report['categories'][cat] = {'count': len(cat_parts), 'parts': []}

            for pid, info in cat_parts:
                mesh = info['mesh']
                part_info = {
                    'id': pid, 'vertices': info['vertices'],
                    'faces': info['faces'], 'volume_mm3': info['volume_mm3'],
                    'is_watertight': mesh.is_watertight,
                    'is_convex': mesh.is_convex,
                    'bounds': [round(b, 4) for b in mesh.bounds.flatten()],
                    'extents': [round(e, 4) for e in mesh.extents.tolist()],
                }

                qs = report['quality_stats']
                qs['total_vertices'] += info['vertices']
                qs['total_faces'] += info['faces']
                qs['min_vertices'] = min(qs['min_vertices'], info['vertices'])
                qs['max_vertices'] = max(qs['max_vertices'], info['vertices'])

                if mesh.is_watertight:
                    qs['watertight_count'] += 1
                try:
                    if mesh.is_volume:
                        qs['manifold_count'] += 1
                except:
                    pass

                if info['vertices'] < 20:
                    report['issues'].append(
                        f"⚠️ {pid}: Low vertex count ({info['vertices']})")
                if not mesh.is_watertight:
                    report['issues'].append(
                        f"⚠️ {pid}: Not watertight")

                report['categories'][cat]['parts'].append(part_info)

        n = len(self.parts)
        if n > 0:
            qs = report['quality_stats']
            qs['avg_vertices'] = round(qs['total_vertices'] / n, 1)
            qs['avg_faces'] = round(qs['total_faces'] / n, 1)

        return report

    def save_quality_report(self, report: dict) -> str:
        """保存质量报告"""
        filepath = f"{OUTPUT_DIR}/v8_quality_report.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("  V8 Competition-Grade Part Library - Quality Report\n")
            f.write("=" * 70 + "\n")
            f.write(f"  Timestamp: {report['timestamp']}\n")
            f.write(f"  Total Parts: {report['total_parts']}\n\n")

            qs = report['quality_stats']
            f.write("-" * 50 + "\n")
            f.write("  Statistics Summary\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Total Vertices:     {qs['total_vertices']:,}\n")
            f.write(f"  Total Faces:         {qs['total_faces']:,}\n")
            f.write(f"  Avg Vertices/Part:   {qs['avg_vertices']:.1f}\n")
            f.write(f"  Avg Faces/Part:      {qs['avg_faces']:.1f}\n")
            f.write(f"  Min Vertices:        {qs['min_vertices']}\n")
            f.write(f"  Max Vertices:        {qs['max_vertices']}\n")
            f.write(f"  Watertight Models:   {qs['watertight_count']}/{report['total_parts']}\n")
            f.write(f"  Manifold Models:     {qs['manifold_count']}/{report['total_parts']}\n\n")

            f.write("-" * 50 + "\n")
            f.write("  Category Details\n")
            f.write("-" * 50 + "\n")
            
            cat_cn_map = {
                'actuators': 'Actuators', 'transmission': 'Transmission',
                'energy': 'Energy Systems', 'sensors': 'Sensors',
                'controllers': 'Controllers', 'connectors': 'Connectors',
                'structural': 'Structural Components'
            }

            for cat, data in report['categories'].items():
                cn = cat_cn_map.get(cat, cat)
                f.write(f"\n  [{cn}] {data['count']} parts:\n")
                for p in data['parts']:
                    wt = "OK" if p['is_watertight'] else "NG"
                    f.write(f"    {p['id']:42s} | {p['vertices']:5d}v "
                           f"| {p['faces']:5d}f | {p['volume_mm3']:8.1f}mm³ "
                           f"| watertight:{wt}\n")

            if report['issues']:
                f.write("\n" + "-" * 50 + "\n")
                f.write("  Issues Found\n")
                f.write("-" * 50 + "\n")
                for issue in report['issues']:
                    f.write(f"  {issue}\n")
            else:
                f.write("\n  All models passed quality checks!\n")

            f.write("\n" + "=" * 70 + "\n")

        kb = os.path.getsize(filepath) / 1024
        print(f"📋 Report: {filepath} ({kb:.0f}KB)")
        return filepath


def main():
    print("=" * 70)
    print("  V8 Part Library 3D Visualization System")
    print("=" * 70)

    start_time = datetime.now()
    viz = V8Visualizer()
    results = {}

    # Task 1: 全零件总览
    print("\n" + "-" * 50)
    print("Task 1/5: Overview grid...")
    results['overview'] = viz.render_overview_grid(cols=8, preview_size=180)

    # Task 2: 分类展示
    print("\n" + "-" * 50)
    print("Task 2/5: Category grids...")
    results['categories'] = viz.render_all_categories()

    # Task 3: 装配体
    print("\n" + "-" * 50)
    print("Task 3/5: Robot assemblies...")
    results['assemblies'] = {}
    for asm in ['mobile_robot', 'arm', 'drone']:
        try:
            results['assemblies'][asm] = viz.create_assembly_scene(asm)
        except Exception as e:
            print(f"  ⚠️ {asm}: {e}")

    # Task 4: 质量报告
    print("\n" + "-" * 50)
    print("Task 4/5: Quality report...")
    report = viz.generate_quality_report()
    results['report'] = viz.save_quality_report(report)

    # Task 5: 文件清单
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print(f"  Complete! Time: {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"\n📁 Output: {os.path.abspath(OUTPUT_DIR)}")
    
    output_files = []
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for fn in files:
            fp = os.path.join(root, fn)
            sz = os.path.getsize(fp) / 1024
            output_files.append((fn, sz))
    
    total_kb = sum(s for _, s in output_files)
    print(f"\n📊 {len(output_files)} files, {total_kb:.0f} KB total")
    
    for fn, sz in sorted(output_files):
        print(f"   • {fn} ({sz:.0f}KB)")

    return 0


if __name__ == "__main__":
    exit(main())
