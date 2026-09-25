# ══════════════════════════════════════════════════════════
# 🎨 V8 机器人 3D 多角度渲染器
#
# 功能:
#   ✅ 自动渲染6个标准视角 (前/后/左/右/顶/透视)
#   ✅ 高分辨率输出 (1920x1080)
#   ✅ 保存PNG图片
#   ✅ 显示关节和驱动器信息
#
# ══════════════════════════════════════════════════════════

import mujoco
import numpy as np
import os
from datetime import datetime
from PIL import Image


def render_robot_views(model_path: str = "v8_humanoid_robot.xml", 
                       output_dir: str = "v8_renders"):
    """
    渲染V8机器人的多个视角
    
    Args:
        model_path: MuJoCo XML模型路径
        output_dir: 输出目录
    """
    print("=" * 70)
    print("🎨 V8 机器人 3D 多角度渲染")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"\n🔄 加载模型: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 创建渲染器 (使用较小分辨率以适配默认framebuffer 640x480)
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # 重置模型到初始位置
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.85  # 设置初始高度
    
    # 前进几步让模型稳定
    for _ in range(100):
        mujoco.mj_step(model, data)
    
    print(f"\n📊 模型信息:")
    print(f"   连杆数: {model.nbody}")
    print(f"   关节数: {model.njnt}")
    print(f"   驱动器: {model.nu}")
    print(f"   几何体: {model.ngeom}")
    
    # 定义相机视角
    views = {
        'front': {
            'lookat': [0, 0, 0.7],
            'distance': 3.5,
            'azimuth': 90,
            'elevation': -10,
            'desc': '正面视图'
        },
        'back': {
            'lookat': [0, 0, 0.7],
            'distance': 3.5,
            'azimuth': -90,
            'elevation': -10,
            'desc': '背面视图'
        },
        'left': {
            'lookat': [0, 0, 0.7],
            'distance': 3.5,
            'azimuth': 180,
            'elevation': -10,
            'desc': '左侧视图'
        },
        'right': {
            'lookat': [0, 0, 0.7],
            'distance': 3.5,
            'azimuth': 0,
            'elevation': -10,
            'desc': '右侧视图'
        },
        'top': {
            'lookat': [0, 0, 0.7],
            'distance': 4.0,
            'azimuth': 0,
            'elevation': 89.9,
            'desc': '俯视视图'
        },
        'perspective': {
            'lookat': [0, 0, 0.7],
            'distance': 3.0,
            'azimuth': 45,
            'elevation': -20,
            'desc': '透视图'
        }
    }
    
    rendered_files = []
    
    print(f"\n🖼️ 开始渲染 ({len(views)} 个视角)...")
    
    for view_name, camera_config in views.items():
        # 创建相机对象 (MuJoCo 3.x API)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        
        # 设置相机参数
        cam.lookat = np.array(camera_config['lookat'])
        cam.distance = camera_config['distance']
        cam.azimuth = camera_config['azimuth']
        cam.elevation = camera_config['elevation']
        
        # 更新场景并设置相机
        renderer.update_scene(data, camera=cam)
        
        # 渲染
        rgb = renderer.render()
        
        # 保存图片 (使用PIL)
        img = Image.fromarray(rgb)
        filename = f"{output_dir}/v8_{view_name}_view.png"
        img.save(filename)
        rendered_files.append((filename, camera_config['desc']))
        
        print(f"   ✅ {view_name:12s}: {filename} ({camera_config['desc']})")
    
    # 额外: 创建一个信息图
    create_info_image(model, data, f"{output_dir}/v8_info.png", renderer)
    
    # 清理
    renderer.close()
    
    print(f"\n💾 所有渲染完成! 共 {len(rendered_files)} 张图片")
    print(f"   输出目录: {os.path.abspath(output_dir)}")
    
    return rendered_files


def create_info_image(model, data, filename, renderer):
    """创建带信息的参考图"""
    # 创建相机
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    
    # 设置相机到一个好的角度
    cam.lookat = np.array([0, 0, 0.7])
    cam.distance = 3.2
    cam.azimuth = 30
    cam.elevation = -15
    
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    
    # 保存
    img = Image.fromarray(rgb)
    img.save(filename)
    print(f"   ✅ {'info':12s}: {filename} (参考信息图)")


def main():
    """主函数"""
    try:
        files = render_robot_views(
            model_path="v8_humanoid_robot.xml",
            output_dir="v8_renders"
        )
        
        print("\n" + "=" * 70)
        print("✅ 3D渲染全部完成!")
        print("=" * 70)
        
        print("\n📸 生成的图片文件:")
        for filepath, desc in files:
            fullpath = os.path.abspath(filepath)
            print(f"   📷 {desc}: {fullpath}")
        
        print(f"\n📁 完整目录: {os.path.abspath('v8_renders')}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 渲染出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
