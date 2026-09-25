# ═══════════════════════════════════════════════════════════════
#  V8 交互式3D查看器 — 真正的实时旋转/缩放/平移
#  ═══════════════════════════════════════════════════════════════
#
#  🖱️ 鼠标操作：
#     左键拖拽 → 旋转视角 (Orbit)
#     右键拖拽 → 平移 (Pan)
#     滚轮     → 缩放 (Zoom)
#
#  ⌨️ 键盘操作：
#     Tab       → 切换7种预设视角
#     R         → 截图保存
#     Space     → 暂停/继续仿真
#     1-7       → 快速切换视角
#     ESC / Q   → 退出
#
#  启动：python viewer_interactive.py [模型文件.xml]
# ═══════════════════════════════════════════════════════════════

import mujoco
import numpy as np
import glfw
from OpenGL.GL import *
from pathlib import Path
from PIL import Image
import sys

# ============ 配置 ============
DEFAULT_MODEL = "v8_humanoid_robot.xml"
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
INIT_HEIGHT = 0.85
STEPS_TO_STABILIZE = 200

# 相机预设 (7种)
CAMERA_PRESETS = [
    {"name": "正面",    "lookat": np.array([0, 0, 0.7]),  "distance": 2.5, "azimuth": 90,   "elevation": -15},
    {"name": "背面",    "lookat": np.array([0, 0, 0.7]),  "distance": 2.5, "azimuth": -90,  "elevation": -15},
    {"name": "左侧",    "lookat": np.array([0, 0, 0.7]),  "distance": 2.5, "azimuth": 180,  "elevation": -15},
    {"name": "右侧",    "lookat": np.array([0, 0, 0.7]),  "distance": 2.5, "azimuth": 0,    "elevation": -15},
    {"name": "俯视",    "lookat": np.array([0, 0, 0.7]),  "distance": 3.5, "azimuth": 0,    "elevation": 89},
    {"name": "等轴",    "lookat": np.array([0, 0, 0.7]),  "distance": 2.0, "azimuth": 45,   "elevation": -30},
    {"name": "特写",    "lookat": np.array([0, 0, 0.85]), "distance": 1.2, "azimuth": 30,   "elevation": -20},
]


class InteractiveViewer:
    """交互式3D查看器"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        
        # 加载模型
        print(f"📦 加载模型: {self.model_path.name}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        
        # 打印模型信息
        print(f"\n{'='*50}")
        print(f"  模型规格:")
        print(f"    连杆数: {self.model.nbody}")
        print(f"    关节数: {self.model.njnt}")
        print(f"    驱动器: {self.model.nu}")
        print(f"    几何体: {self.model.ngeom}")
        print(f"    自由度: {self.model.nq}")
        print(f"{'='*50}\n")
        
        # 初始化相机和选项（不需要OpenGL上下文）
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        
        # 设置默认相机
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)
        self._apply_preset(5)  # 默认等轴视角
        
        # 鼠标状态
        self.mouse_left = False
        self.mouse_right = False
        self.last_mouse_pos = (0, 0)
        
        # 控制状态
        self.paused = False
        self.current_preset_idx = 5
        self.screenshot_count = 0
        
        # 输出目录
        self.out_dir = Path("viewer_captures")
        self.out_dir.mkdir(exist_ok=True)
    
    def _apply_preset(self, idx: int):
        """应用相机预设"""
        if 0 <= idx < len(CAMERA_PRESETS):
            preset = CAMERA_PRESETS[idx]
            self.cam.lookat[:] = preset["lookat"]
            self.cam.distance = preset["distance"]
            self.cam.azimuth = preset["azimuth"]
            self.cam.elevation = preset["elevation"]
            self.current_preset_idx = idx
            print(f"  📷 切换到 [{preset['name']}] 视角")
    
    def _init_glfw(self):
        """初始化GLFW窗口"""
        if not glfw.init():
            raise RuntimeError("无法初始化 GLFW")
        
        # 创建窗口
        self.window = glfw.create_window(
            WINDOW_WIDTH, WINDOW_HEIGHT,
            f"V8 机器人 3D查看器 — {self.model_path.name}",
            None, None
        )
        
        if not self.window:
            glfw.terminate()
            raise RuntimeError("无法创建 GLFW 窗口")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # 垂直同步
        
        # 现在有了OpenGL上下文，可以初始化MuJoCo渲染组件
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        
        # 设置回调函数
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_window_size_callback(self.window, self._window_size_callback)
        
        # 获取framebuffer大小
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        self.viewport = mujoco.MjrRect(0, 0, fb_width, fb_height)
    
    def _cursor_pos_callback(self, window, xpos, ypos):
        """鼠标移动回调"""
        dx = xpos - self.last_mouse_pos[0]
        dy = ypos - self.last_mouse_pos[1]
        
        if self.mouse_left:
            # 左键：旋转
            self.cam.azimuth += dx * 0.5
            self.cam.elevation += dy * 0.5
            # 限制仰角范围
            self.cam.elevation = max(-89, min(89, self.cam.elevation))
        
        if self.mouse_right:
            # 右键：平移
            # 计算平移量（基于距离）
            scale = self.cam.distance * 0.001
            # 根据相机方位计算平移方向
            az = np.radians(self.cam.azimuth)
            self.cam.lookat[0] -= (dx * np.cos(az) + dy * np.sin(az)) * scale
            self.cam.lookat[1] -= (-dx * np.sin(az) + dy * np.cos(az)) * scale
        
        self.last_mouse_pos = (xpos, ypos)
    
    def _mouse_button_callback(self, window, button, action, mods):
        """鼠标按钮回调"""
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_left = (action == glfw.PRESS)
            if self.mouse_left:
                self.last_mouse_pos = glfw.get_cursor_pos(window)
        
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse_right = (action == glfw.PRESS)
            if self.mouse_right:
                self.last_mouse_pos = glfw.get_cursor_pos(window)
    
    def _scroll_callback(self, window, xoffset, yoffset):
        """滚轮回调 — 缩放"""
        # 缩放因子
        zoom_speed = 0.1
        
        if yoffset > 0:
            # 放大
            self.cam.distance *= (1 - zoom_speed)
        else:
            # 缩小
            self.cam.distance *= (1 + zoom_speed)
        
        # 限制缩放范围
        self.cam.distance = max(0.5, min(10.0, self.cam.distance))
    
    def _key_callback(self, window, key, scancode, action, mods):
        """键盘回调"""
        if action != glfw.PRESS and action != glfw.REPEAT:
            return
        
        # Tab：切换视角
        if key == glfw.KEY_TAB:
            self.current_preset_idx = (self.current_preset_idx + 1) % len(CAMERA_PRESETS)
            self._apply_preset(self.current_preset_idx)
        
        # 数字键1-7：快速切换
        elif glfw.KEY_1 <= key <= glfw.KEY_7:
            idx = key - glfw.KEY_1
            self._apply_preset(idx)
        
        # R：截图
        elif key == glfw.KEY_R:
            self._take_screenshot()
        
        # Space：暂停/继续
        elif key == glfw.KEY_SPACE:
            self.paused = not self.paused
            status = "⏸️ 已暂停" if self.paused else "▶️ 继续运行"
            print(f"\n  {status}")
        
        # ESC/Q：退出
        elif key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
            glfw.set_window_should_close(window, True)
        
        # H：帮助信息
        elif key == glfw.KEY_H:
            self._print_help()
    
    def _window_size_callback(self, window, width, height):
        """窗口大小改变回调"""
        fb_width, fb_height = glfw.get_framebuffer_size(window)
        self.viewport = mujoco.MjrRect(0, 0, fb_width, fb_height)
    
    def _take_screenshot(self):
        """截图保存"""
        self.screenshot_count += 1
        filename = f"screenshot_{self.screenshot_count:03d}.png"
        filepath = self.out_dir / filename
        
        # 获取当前帧缓冲区数据
        width = self.viewport.width
        height = self.viewport.height
        
        # 从OpenGL读取像素
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        
        # PIL需要翻转Y轴
        from PIL import Image
        img = Image.frombytes('RGB', (width, height), pixels)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(filepath)
        
        size_kb = filepath.stat().st_size / 1024
        print(f"  📸 截图已保存: {filepath.name} ({size_kb:.0f}KB)")
    
    def _print_help(self):
        """打印帮助信息"""
        print("\n" + "="*55)
        print("  🎮 操作说明")
        print("="*55)
        print("  🖱️ 鼠标:")
        print("     左键拖拽 → 旋转视角")
        print("     右键拖拽 → 平移视图")
        print("     滚轮     → 缩放")
        print("")
        print("  ⌨️ 键盘:")
        print("     Tab     → 切换视角")
        print("     1-7     → 快速切换")
        print("     R       → 截图")
        print("     Space   → 暂停/继续")
        print("     H       → 帮助")
        print("     ESC/Q   → 退出")
        print("="*55 + "\n")
    
    def run(self):
        """运行主循环"""
        try:
            # 初始化GLFW
            self._init_glfw()
            
            # 重置并稳定模型
            mujoco.mj_resetData(self.model, self.data)
            if self.model.nq > 2:
                self.data.qpos[2] = INIT_HEIGHT
            
            for _ in range(STEPS_TO_STABILIZE):
                mujoco.mj_step(self.model, self.data)
            
            # 打印帮助
            self._print_help()
            
            print("  ✅ 交互式查看器已启动!")
            print("  💡 拖拽鼠标旋转，滚轮缩放，按H查看帮助\n")
            
            # 主循环
            frame_count = 0
            while not glfw.window_should_close(self.window):
                # 物理步进（如果未暂停）
                if not self.paused:
                    for _ in range(2):  # 每帧模拟2步
                        mujoco.mj_step(self.model, self.data)
                
                # 更新场景
                mujoco.mjv_updateScene(
                    self.model, self.data, self.opt,
                    None, self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scene
                )
                
                # 渲染
                mujoco.mjr_render(self.viewport, self.scene, self.ctx)
                
                # 显示HUD信息
                self._render_hud(frame_count)
                
                # 交换缓冲区
                glfw.swap_buffers(self.window)
                glfw.poll_events()
                
                frame_count += 1
            
        finally:
            # 清理
            if hasattr(self, 'window'):
                glfw.destroy_window(self.window)
            glfw.terminate()
            
            print(f"\n  👋 查看器已关闭")
            print(f"  📁 截图保存在: {self.out_dir.absolute()}\n")
    
    def _render_hud(self, frame_count: int):
        """渲染HUD信息"""
        # 使用MuJoCo的文本渲染功能
        overlay_text = f"Frame: {frame_count} | Preset: {CAMERA_PRESETS[self.current_preset_idx]['name']}"
        if self.paused:
            overlay_text += " | PAUSED"
        
        # 可以在这里添加更多HUD信息
        # MuJoCo的mjr_overlay可以用于文本叠加


def main():
    """主入口"""
    print("=" * 60)
    print("  V8 人形机器人 — 交互式3D查看器")
    print("=" * 60)
    print()
    
    # 确定模型路径
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    
    if not Path(model_path).exists():
        # 尝试其他可能的文件
        for alt in ["v8_example_robot.xml", "best_body.xml"]:
            if Path(alt).exists():
                model_path = alt
                break
        else:
            print(f"❌ 错误: 找不到模型文件 '{model_path}'")
            print(f"   可用文件: v8_humanoid_robot.xml, v8_example_robot.xml, best_body.xml")
            return
    
    # 创建并运行查看器
    try:
        viewer = InteractiveViewer(model_path)
        viewer.run()
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
