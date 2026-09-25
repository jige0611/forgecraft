# ══════════════════════════════════════════════════════════
# 🏗️ V8 机器人完整3D建模系统 (STL/OBJ格式)
#
# 功能:
#   ✅ 基于真实几何参数构建3D模型
#   ✅ 支持STL格式 (通用3D打印/仿真)
#   ✅ 支持OBJ格式 (Blender/SolidWorks兼容)
#   ✅ 按部件分色/分组
#   ✅ 可直接导入Blender/Fusion360/SolidWorks
#
# 输出:
#   - v8_robot_complete.stl (完整组装模型)
#   - v8_parts/ (各零件单独文件)
#
# ══════════════════════════════════════════════════════════

import trimesh
import numpy as np
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class RobotPart:
    """机器人的一个部件"""
    name: str
    part_type: str  # box/cylinder/sphere/capsule
    size: List[float]
    position: List[float] = None
    rotation: List[float] = None  # Euler angles [rx, ry, rz] in degrees
    color: Tuple[int, int, int, int] = (128, 128, 128, 255)
    mass: float = 1.0


class V8Robot3DBuilder:
    """
    V8竞赛级人形机器人3D建模器
    
    使用trimesh构建真实的3D几何模型，可导出为STL/OBJ。
    """
    
    def __init__(self):
        self.parts: List[RobotPart] = []
        self.meshes: dict = {}  # {name: trimesh.Trimesh}
        
        self._define_parts()
    
    def _define_parts(self):
        """定义所有机器人部件"""
        
        # ═══ 躯干主体 ═══
        self.parts.append(RobotPart(
            name="torso_main",
            part_type="box",
            size=[0.24, 0.16, 0.36],  # 宽×深×高 (2x显示尺寸)
            position=[0, 0, 0.85],
            color=(180, 180, 190, 255),
            mass=5.0
        ))
        
        # ═══ 头部 ═══
        self.parts.append(RobotPart(
            name="head",
            part_type="sphere",
            size=[0.12],
            position=[0, 0, 1.15],
            color=(215, 205, 195, 255),
            mass=1.5
        ))
        
        # ═══ 左臂 ═══
        # 肩关节电机
        self.parts.append(RobotPart(
            name="left_shoulder_motor",
            part_type="cylinder",
            size=[0.06, 0.08],
            position=[0.14, 0, 1.0],
            rotation=[90, 0, 0],
            color=(217, 89, 26, 255),  # M2006橙色
            mass=0.5
        ))
        
        # 上臂
        self.parts.append(RobotPart(
            name="left_upper_arm",
            part_type="capsule",
            size=[0.05, 0.24],
            position=[0.22, 0, 1.0],
            rotation=[0, 0, 90],
            color=(153, 153, 166, 255),
            mass=0.8
        ))
        
        # 肘关节电机
        self.parts.append(RobotPart(
            name="left_elbow_motor",
            part_type="cylinder",
            size=[0.05, 0.06],
            position=[0.34, 0, 1.0],
            rotation=[90, 0, 0],
            color=(217, 89, 26, 255),
            mass=0.4
        ))
        
        # 前臂
        self.parts.append(RobotPart(
            name="left_forearm",
            part_type="capsule",
            size=[0.04, 0.20],
            position=[0.42, 0, 1.0],
            rotation=[0, 0, 90],
            color=(140, 140, 153, 255),
            mass=0.6
        ))
        
        # ═══ 右臂 (镜像) ═══
        self.parts.append(RobotPart(
            name="right_shoulder_motor",
            part_type="cylinder",
            size=[0.06, 0.08],
            position=[-0.14, 0, 1.0],
            rotation=[90, 0, 0],
            color=(217, 89, 26, 255),
            mass=0.5
        ))
        
        self.parts.append(RobotPart(
            name="right_upper_arm",
            part_type="capsule",
            size=[0.05, 0.24],
            position=[-0.22, 0, 1.0],
            rotation=[0, 0, 90],
            color=(153, 153, 166, 255),
            mass=0.8
        ))
        
        self.parts.append(RobotPart(
            name="right_elbow_motor",
            part_type="cylinder",
            size=[0.05, 0.06],
            position=[-0.34, 0, 1.0],
            rotation=[90, 0, 0],
            color=(217, 89, 26, 255),
            mass=0.4
        ))
        
        self.parts.append(RobotPart(
            name="right_forearm",
            part_type="capsule",
            size=[0.04, 0.20],
            position=[-0.42, 0, 1.0],
            rotation=[0, 0, 90],
            color=(140, 140, 153, 255),
            mass=0.6
        ))
        
        # ═══ 左腿 ═══
        # 髋关节电机 (CIM)
        self.parts.append(RobotPart(
            name="left_hip_motor",
            part_type="cylinder",
            size=[0.08, 0.10],
            position=[0.05, 0, 0.69],
            rotation=[90, 0, 0],
            color=(51, 102, 204, 255),  # CIM蓝色
            mass=1.0
        ))
        
        # 大腿
        self.parts.append(RobotPart(
            name="left_thigh",
            part_type="capsule",
            size=[0.07, 0.30],
            position=[0.05, 0, 0.42],
            rotation=[0, 0, 0],
            color=(128, 128, 140, 255),
            mass=2.0
        ))
        
        # 膝关节电机
        self.parts.append(RobotPart(
            name="left_knee_motor",
            part_type="cylinder",
            size=[0.06, 0.08],
            position=[0.05, 0, 0.14],
            rotation=[90, 0, 0],
            color=(217, 89, 26, 255),
            mass=0.7
        ))
        
        # 小腿
        self.parts.append(RobotPart(
            name="left_shin",
            part_type="capsule",
            size=[0.06, 0.26],
            position=[0.05, 0, -0.10],
            rotation=[0, 0, 0],
            color=(115, 115, 128, 255),
            mass=1.5
        ))
        
        # 脚踝
        self.parts.append(RobotPart(
            name="left_ankle",
            part_type="cylinder",
            size=[0.05, 0.05],
            position=[0.05, 0, -0.25],
            rotation=[90, 0, 0],
            color=(102, 102, 115, 255),
            mass=0.4
        ))
        
        # 脚
        self.parts.append(RobotPart(
            name="left_foot",
            part_type="box",
            size=[0.10, 0.16, 0.04],
            position=[0.05, 0.01, -0.28],
            color=(77, 77, 89, 255),
            mass=0.5
        ))
        
        # ═══ 右腿 (镜像) ═══
        self.parts.append(RobotPart(
            name="right_hip_motor",
            part_type="cylinder",
            size=[0.08, 0.10],
            position=[-0.05, 0, 0.69],
            rotation=[90, 0, 0],
            color=(51, 102, 204, 255),
            mass=1.0
        ))
        
        self.parts.append(RobotPart(
            name="right_thigh",
            part_type="capsule",
            size=[0.07, 0.30],
            position=[-0.05, 0, 0.42],
            rotation=[0, 0, 0],
            color=(128, 128, 140, 255),
            mass=2.0
        ))
        
        self.parts.append(RobotPart(
            name="right_knee_motor",
            part_type="cylinder",
            size=[0.06, 0.08],
            position=[-0.05, 0, 0.14],
            rotation=[90, 0, 0],
            color=(217, 89, 26, 255),
            mass=0.7
        ))
        
        self.parts.append(RobotPart(
            name="right_shin",
            part_type="capsule",
            size=[0.06, 0.26],
            position=[-0.05, 0, -0.10],
            rotation=[0, 0, 0],
            color=(115, 115, 128, 255),
            mass=1.5
        ))
        
        self.parts.append(RobotPart(
            name="right_ankle",
            part_type="cylinder",
            size=[0.05, 0.05],
            position=[-0.05, 0, -0.25],
            rotation=[90, 0, 0],
            color=(102, 102, 115, 255),
            mass=0.4
        ))
        
        self.parts.append(RobotPart(
            name="right_foot",
            part_type="box",
            size=[0.10, 0.16, 0.04],
            position=[-0.05, 0.01, -0.28],
            color=(77, 77, 89, 255),
            mass=0.5
        ))
        
        print(f"   部件定义完成: {len(self.parts)} 个")
    
    def _create_geometry(self, part: RobotPart) -> trimesh.Trimesh:
        """
        根据部件参数创建trimesh几何体
        
        Args:
            part: 部件定义
            
        Returns:
            trimesh对象
        """
        if part.part_type == "box":
            w, h, d = part.size
            mesh = trimesh.creation.box(extents=[w, h, d])
            
        elif part.part_type == "sphere":
            r = part.size[0] / 2
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=r)
            
        elif part.part_type == "cylinder":
            r, h = part.size[0] / 2, part.size[1]
            mesh = trimesh.creation.cylinder(radius=r, height=h, sections=32)
            
        elif part.part_type == "capsule":
            r = part.size[0] / 2
            h = part.size[1]
            # capsule = cylinder + two hemispheres
            mesh = trimesh.creation.capsule(radius=r, height=h, count=[32, 16])
            
        else:
            # 默认使用box
            mesh = trimesh.creation.box(extents=part.size)
        
        return mesh
    
    def build_complete_model(self) -> trimesh.Scene:
        """
        构建完整的3D场景
        
        Returns:
            trimesh.Scene包含所有部件
        """
        print("\n🏗️ 构建3D模型...")
        
        scene = trimesh.Scene()
        
        for i, part in enumerate(self.parts):
            print(f"   [{i+1:2d}/{len(self.parts)}] 创建: {part.name}")
            
            # 创建几何体
            mesh = self._create_geometry(part)
            
            # 设置位置
            if part.position:
                translation = np.array(part.position)
                mesh.apply_translation(translation)
            
            # 设置旋转
            if part.rotation:
                rx, ry, rz = [np.radians(a) for a in part.rotation]
                rotation_matrix = trimesh.transformations.euler_matrix(rx, ry, rz)
                # 应用旋转（需要先回到原点）
                center = mesh.centroid.copy()
                mesh.apply_translation(-center)
                mesh.apply_transform(rotation_matrix)
                mesh.apply_translation(center + np.array(part.position))
            
            # 设置颜色
            mesh.visual.vertex_colors = part.color
            
            # 存储到场景
            scene.add_geometry(mesh, geom_name=part.name)
            
            # 同时存储到meshes字典
            self.meshes[part.name] = mesh
        
        print(f"\n   场景构建完成: {len(scene.geometry)} 个几何体")
        
        return scene
    
    def export_stl(self, output_file: str = "v8_robot_complete.stl"):
        """
        导出为STL格式 (二进制)
        
        STL是3D打印和仿真的标准格式，支持颜色(通过材质)。
        """
        print(f"\n💾 导出STL: {output_file}")
        
        # 合并所有网格为一个
        combined_meshes = list(self.meshes.values())
        
        if len(combined_meshes) > 1:
            # 使用concatenate合并所有网格
            try:
                complete_mesh = trimesh.util.concatenate(combined_meshes)
            except Exception as e:
                print(f"   ⚠️ 合并失败，尝试单独保存: {e}")
                complete_mesh = combined_meshes[0]
        else:
            complete_mesh = combined_meshes[0]
        
        # 导出STL
        complete_mesh.export(output_file)
        
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"   ✅ STL保存成功: {file_size:.1f} KB")
        print(f"   面数: {len(complete_mesh.faces)}")
        print(f"   顶点数: {len(complete_mesh.vertices)}")
        
        return output_file
    
    def export_obj(self, output_file: str = "v8_robot_complete.obj"):
        """
        导出为OBJ格式
        
        OBJ支持多物体分组和材质，适合Blender等软件。
        """
        print(f"\n💾 导出OBJ: {output_file}")
        
        # 构建场景并导出
        scene = trimesh.Scene()
        for name, mesh in self.meshes.items():
            scene.add_geometry(mesh, geom_name=name)
        
        # 导出OBJ (包含MTL材质文件)
        scene.export(output_file)
        
        file_size = os.path.getsize(output_file) / 1024
        mtl_file = output_file.replace('.obj', '.mtl')
        mtl_size = os.path.getsize(mtl_file) / 1024 if os.path.exists(mtl_file) else 0
        
        print(f"   ✅ OBJ保存成功: {file_size:.1f} KB")
        print(f"   材质文件(MTL): {mtl_size:.1f} KB")
        
        return output_file
    
    def export_individual_parts(self, output_dir: str = "v8_3d_parts"):
        """
        导出每个零件单独的STL文件
        
        用于单独查看或修改某个零件。
        """
        print(f"\n📦 导出单个零件到: {output_dir}/")
        
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = []
        
        for name, mesh in self.meshes.items():
            filename = f"{output_dir}/{name}.stl"
            mesh.export(filename)
            exported_files.append(filename)
        
        print(f"   ✅ 共导出 {len(exported_files)} 个零件文件")
        
        return exported_files


def main():
    """主函数：构建并导出完整的3D模型"""
    print("=" * 70)
    print("🏗️ V8 竞赛级人形机器人 - 3D建模系统")
    print("=" * 70)
    
    try:
        # 创建构建器
        builder = V8Robot3DBuilder()
        
        # 构建完整模型
        scene = builder.build_complete_model()
        
        # 导出各种格式
        stl_file = builder.export_stl("v8_robot_complete.stl")
        obj_file = builder.export_obj("v8_robot_complete.obj")
        parts = builder.export_individual_parts("v8_3d_parts")
        
        # 统计信息
        total_vertices = sum(len(m.vertices) for m in builder.meshes.values())
        total_faces = sum(len(m.faces) for m in builder.meshes.values())
        
        print("\n" + "=" * 70)
        print("✅ 3D建模全部完成!")
        print("=" * 70)
        
        print(f"\n📊 模型统计:")
        print(f"   部件数量: {len(builder.parts)} 个")
        print(f"   总顶点数: {total_vertices:,}")
        print(f"   总面数: {total_faces:,}")
        
        print(f"\n📁 生成的文件:")
        print(f"   🔷 完整STL: {os.path.abspath(stl_file)}")
        print(f"   🔶 完整OBJ: {os.path.abspath(obj_file)}")
        print(f"   📦 单个零件: {os.path.abspath('v8_3d_parts')}/ ({len(parts)}个)")
        
        print(f"\n💡 使用方法:")
        print(f"   Blender: 文件 → 导入 → Wavefront (.obj)")
        print(f"   SolidWorks: 文件 → 打开 → 选择STL")
        print(f"   Fusion360: 插入 → 插入网格 → 选择STL")
        print(f"   3D打印: 直接导入STL到切片软件")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 建模出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
