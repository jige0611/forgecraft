# ═══════════════════════════════════════════════════════════════
#  零件库3D交互式浏览器 — 逐一浏览47种工业级零件
#  ═══════════════════════════════════════════════════════════════
#
#  🖱️ 操作：
#     左键拖拽 → 旋转视角
#     右键拖拽 → 平移视图
#     滚轮     → 缩放
#
#  ⌨️ 键盘：
#     ← / A    → 上一个零件
#     → / D    → 下一个零件
#     Home      → 第一个零件
#     End       → 最后一个零件
#     1-5       → 跳转到分类 (执行器/传动/能源/传感/控制/连接)
#     R         → 截图当前零件
#     I         → 显示/隐藏零件信息面板
#     Space     → 自动旋转开关
#     ESC/Q     → 退出
#
#  启动：python part_library_viewer.py
# ═══════════════════════════════════════════════════════════════

import mujoco
import numpy as np
import glfw
from OpenGL.GL import *
from pathlib import Path
from PIL import Image
import sys
import trimesh
import trimesh.creation
import trimesh.transformations as ttf

# 导入零件库
sys.path.insert(0, str(Path(__file__).parent))
from v8_3d_model_library import get_all_part_specs, PartSpec


# ============ 配置 ============
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
AUTO_ROTATE_SPEED = 0.3  # 自动旋转速度（度/帧）


class PartLibraryViewer:
    """零件库交互式浏览器"""
    
    def __init__(self):
        # 加载所有零件规格
        print("📦 加载零件库...")
        self.all_parts = get_all_part_specs()
        self.part_ids = list(self.all_parts.keys())
        self.total_parts = len(self.part_ids)
        
        print(f"   共 {self.total_parts} 个零件")
        
        # 分类
        self.categories = {
            'actuators': ('执行器', [0.85, 0.35, 0.10]),
            'transmission': ('传动系统', [0.75, 0.72, 0.68]),
            'energy': ('能源系统', [0.15, 0.15, 0.18]),
            'sensors': ('传感系统', [0.05, 0.50, 0.05]),
            'controllers': ('控制系统', [0.05, 0.05, 0.80]),
            'connectors': ('连接件', [0.55, 0.55, 0.57]),
        }
        
        # 当前状态
        self.current_idx = 0
        self.auto_rotate = False
        self.show_info = True
        self.screenshot_count = 0
        
        # 鼠标状态
        self.mouse_left = False
        self.mouse_right = False
        self.last_mouse_pos = (0, 0)
        
        # 输出目录
        self.out_dir = Path("part_library_captures")
        self.out_dir.mkdir(exist_ok=True)
    
    def _get_current_part(self) -> PartSpec:
        """获取当前零件"""
        return self.all_parts[self.part_ids[self.current_idx]]
    
    def _create_mujoco_model_for_part(self, spec: PartSpec):
        """为单个零件创建MuJoCo模型XML"""
        dims = spec.dimensions
        color = spec.color
        geo_type = spec.geometry_type
        
        # 基础缩放因子（让零件在视图中大小合适）
        scale = 1.0
        
        # 根据几何类型选择基础形状
        if 'cylinder' in geo_type:
            if geo_type == 'cylinder_simple':
                r = dims.get('outer_diameter', 0.02) / 2 * scale
                h = dims.get('total_height', dims.get('total_length', 0.04)) * scale
                geom_xml = f'  <geom type="cylinder" size="{r} {h/2}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>'
            elif geo_type == 'cylinder_complex':
                r = dims.get('outer_diameter', dims.get('body_diameter', 0.03)) / 2 * scale
                h = dims.get('total_height', dims.get('total_length', 0.05)) * scale
                geom_xml = f'''  <geom type="cylinder" size="{r} {h/2}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>
  <!-- 法兰 -->
  <geom name="flange" pos="0 0 {-h/2 + 0.002*scale}" type="cylinder" size="{r*0.9} 0.001" rgba="{0.4} {0.4} {0.45} 1"/>'''
            elif geo_type == 'square_cylinder':
                s = dims.get('square_size', 0.03) / 2 * scale
                h = dims.get('body_length', 0.03) * scale
                geom_xml = f'  <geom type="box" size="{s} {s} {h/2}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>'
            elif geo_type == 'cylinder_detail':
                r = dims.get('outer_diameter', 0.01) / 2 * scale
                h = dims.get('total_length', 0.02) * scale
                geom_xml = f'''  <geom type="cylinder" size="{r} {h/2}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>
  <!-- 内孔 -->
  <geom type="cylinder" size="{r*0.4} {h*0.6}" pos="0 0 0" rgba="0.1 0.1 0.1 1"/>'''
            elif geo_type == 'cylinder_with_hub':
                r = dims.get('body_diameter', 0.03) / 2 * scale
                h = dims.get('body_height', 0.03) * scale
                geom_xml = f'''  <geom type="cylinder" size="{r} {h/2}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>
  <!-- 轴 -->
  <geom name="shaft" pos="0 0 {h/2}" type="cylinder" size="{dims.get("shaft_diameter",0.005)/2*scale} {dims.get("shaft_length",0.01)*scale}" rgba="0.7 0.7 0.7 1"/>'''
            else:
                r = 0.02 * scale
                h = 0.04 * scale
                geom_xml = f'  <geom type="cylinder" size="{r} {h/2}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>'
                
        elif 'box' in geo_type:
            if geo_type == 'box_simple':
                l = dims.get('length', 0.02) / 2 * scale
                w = dims.get('width', 0.015) / 2 * scale
                h = dims.get('height', 0.01) / 2 * scale
                geom_xml = f'  <geom type="box" size="{l} {w} {h}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>'
            elif geo_type == 'box_complex':
                l = dims.get('length', 0.04) / 2 * scale
                w = dims.get('width', 0.025) / 2 * scale
                h = dims.get('height', 0.025) / 2 * scale
                hub_r = dims.get('hub_diameter', 0.015) / 2 * scale
                geom_xml = f'''  <geom type="box" size="{l} {w} {h}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>
  <!-- 输出轴 -->
  <geom name="horn" pos="0 0 {h + 0.005*scale}" type="cylinder" size="{hub_r} 0.003" rgba="0.7 0.65 0.6 1"/>'''
            elif geo_type == 'box_heatsink':
                l = dims.get('length', 0.04) / 2 * scale
                w = dims.get('width', 0.03) / 2 * scale
                h_base = dims.get('height', 0.01) / 2 * scale
                fins = dims.get('heatsink_fins', 6)
                fin_w = (w * 2) / (fins * 2)
                fin_geoms = '\n'.join([f'  <geom name="fin{i}" pos="{-w + fin_w*(i*2+1)} 0 {h_base+0.003*scale}" type="box" size="{fin_w*0.4} {w*0.9} 0.003" rgba="0.85 0.75 0.70 1"/>' for i in range(fins)])
                geom_xml = f'''  <geom type="box" size="{l} {w} {h_base}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>
{fin_geoms}'''
            elif geo_type == 'box_small':
                l = dims.get('length', 0.015) / 2 * scale
                w = dims.get('width', 0.012) / 2 * scale
                h = dims.get('height', 0.008) / 2 * scale
                geom_xml = f'  <geom type="box" size="{l} {w} {h}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>'
            elif geo_type == 'box_detail':
                l = dims.get('length', 0.08) / 2 * scale
                w = dims.get('width', 0.04) / 2 * scale
                h = dims.get('height', 0.025) / 2 * scale
                geom_xml = f'''  <geom type="box" size="{l} {w} {h}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>
  <!-- 标签区域 -->
  <geom name="label_area" pos="{-l*0.5} 0 {h}" type="box" size="{l*0.3} {w*0.8} 0.001" rgba="0.1 0.1 0.1 0.8"/>'''
            else:
                l, w, h = 0.02, 0.015, 0.01
                geom_xml = f'  <geom type="box" size="{l} {w} {h}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>'
                
        elif 'bearing' in geo_type or geo_type == 'bearing_assembly':
            ri = dims.get('inner_diameter', 0.008) / 2 * scale
            ro = dims.get('outer_diameter', 0.02) / 2 * scale
            hw = dims.get('width', 0.006) / 2 * scale
            geom_xml = f'''  <!-- 外圈 -->
  <geom name="outer_ring" type="cylinder" size="{ro} {hw}" rgba="0.55 0.55 0.58 1"/>
  <!-- 内圈 -->
  <geom name="inner_ring" type="cylinder" size="{ri} {hw*0.9}" rgba="0.60 0.62 0.65 1"/>
  <!-- 滚珠 -->
  <geom name="balls" type="sphere" size="{(ro+ri)/4}" pos="{(ro+ri)/3} 0 0" rgba="0.75 0.77 0.80 1"/>'''
            
        elif 'ring' in geo_type or geo_type == 'ring_with_holes':
            ro = dims.get('outer_diameter', 0.04) / 2 * scale
            ri = dims.get('inner_diameter', 0.015) / 2 * scale
            th = dims.get('thickness', 0.003) / 2 * scale
            bolt_c = dims.get('bolt_circle_diameter', ro)
            n_bolts = dims.get('bolt_count', 4)
            bolt_geom = '\n'.join([
                f'  <geom name="bolt{i}" pos="{bolt_c/2*np.cos(2*np.pi*i/n_bolts)} {bolt_c/2*np.sin(2*np.pi*i/n_bolts)} 0" type="cylinder" size="{dims.get("bolt_diameter",0.003)/2*scale} {th*1.5}" rgba="0.3 0.3 0.32 1"/>'
                for i in range(n_bolts)
            ])
            geom_xml = f'''  <geom type="cylinder" size="{ro} {th}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>
  <!-- 中心孔 -->
  <geom type="cylinder" size="{ri} {th*2}" rgba="0.1 0.1 0.1 1"/>
{bolt_geom}'''
            
        elif 'pcb' in geo_type or geo_type == 'pcb_board':
            l = dims.get('length', 0.06) / 2 * scale
            w = dims.get('width', 0.04) / 2 * scale
            th = dims.get('board_thickness', 0.0015) / 2 * scale
            usb_h = dims.get('usb_height', 0.012) * scale
            eth_h = dims.get('ethernet_height', 0.01) * scale
            gpio_h = dims.get('gpio_header_height', 0.01) * scale
            geom_xml = f'''  <!-- PCB基板 -->
  <geom type="box" size="{l} {w} {th*3}" rgba="0.0 0.30 0.10 1" material="mat_part"/>
  <!-- USB接口 -->
  <geom name="usb" pos="{-l*0.8} {w*0.2} {th*3}" type="box" size="{l*0.15} {w*0.12} {usb_h/2}" rgba="0.70 0.70 0.72 1"/>
  <!-- GPIO排针 -->
  <geom name="gpio" pos="{l*0.3} {w*0.5} {th*3+gpio_h/2}" type="box" size="{l*0.08} {w*0.25} {gpio_h/2}" rgba="0.1 0.1 0.1 1"/>
  <!-- 网口 -->
  <geom name="ethernet" pos="{l*0.8} {-w*0.1} {th*3+eth_h/2}" type="box" size="{l*0.15} {w*0.22} {eth_h/2}" rgba="0.15 0.20 0.25 1"/>'''
            
        elif 'package' in geo_type:
            if geo_type in ('qfn_package', 'so8_package', 'ssop24_package'):
                l = dims.get('length', 0.005) * 5 * scale  # 放大以便可见
                w = dims.get('width', 0.004) * 5 * scale
                h = dims.get('height', 0.001) * 10 * scale
                pin_count = 8 if 'so8' in geo_type else (24 if 'ssop' in geo_type else 16)
                pin_geom = ''
                if pin_count <= 16:
                    pins_per_side = pin_count // 2
                    pin_spacing = (l * 2) / (pins_per_side + 1)
                    for side in [-1, 1]:
                        for p in range(pins_per_side):
                            px = -l + pin_spacing * (p + 1)
                            pin_geom += f'\n  <geom name="pin_{side}_{p}" pos="{px} {side*w} 0" type="box" size="{l*0.06} {w*0.15} {h*0.8}" rgba="0.75 0.70 0.65 1"/>'
                geom_xml = f'''  <geom type="box" size="{l} {w} {h}" rgba="0.05 0.05 0.05 1" material="mat_part"/>
{pin_geom}'''
            elif geo_type == 'lga28_package':
                l = dims.get('length', 0.005) * 5 * scale
                w = dims.get('width', 0.003) * 5 * scale
                h = dims.get('height', 0.01) * scale
                geom_xml = f'''  <geom type="box" size="{l} {w} {h}" rgba="0.05 0.05 0.05 1" material="mat_part"/>
  <!-- 封装标记 -->
  <geom name="dot" pos="-{l*0.6} {w*0.6} {h}" type="sphere" size="{min(l,w)*0.1}" rgba="0.9 0.1 0.1 1"/>'''
            else:
                l, w, h = 0.015, 0.012, 0.005
                geom_xml = f'  <geom type="box" size="{l} {w} {h}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>'
                
        elif geo_type == 'aluminum_block':
            l = dims.get('length', 0.03) / 2 * scale
            w = dims.get('width', 0.015) / 2 * scale
            h = dims.get('height', 0.015) / 2 * scale
            hole_r = dims.get('hole_dia', 0.003) / 2 * scale
            geom_xml = f'''  <geom type="box" size="{l} {w} {h}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>
  <!-- 安装孔 -->
  <geom name="hole_a" pos="-{l*0.7} 0 0" type="cylinder" size="{hole_r} {h*1.1}" rgba="0.1 0.1 0.1 1"/>
  <geom name="hole_b" pos="{l*0.7} 0 0" type="cylinder" size="{hole_r} {h*1.1}" rgba="0.1 0.1 0.1 1"/>'''
            
        elif geo_type == 'small_pcb_module':
            l = dims.get('length', 0.01) * 3 * scale
            w = dims.get('width', 0.008) * 3 * scale
            lens_r = dims.get('lens_dia', 0.003) * 2 * scale
            geom_xml = f'''  <geom type="box" size="{l} {w} 0.001" rgba="0.0 0.0 0.0 1" material="mat_part"/>
  <!-- 激光透镜 -->
  <geom name="lens" pos="0 0 0.003" type="cylinder" size="{lens_r} 0.0015" rgba="0.85 0.88 0.92 0.8"/>'''
            
        elif geo_type == 'complex':
            # 同步带轮或链轮
            pd = dims.get('pitch_diameter', 0.02) / 2 * scale
            bore = dims.get('bore', 0.005) / 2 * scale
            width = dims.get('width', 0.01) / 2 * scale
            teeth = dims.get('teeth_count', 16)
            geom_xml = f'''  <geom type="cylinder" size="{pd} {width}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>
  <!-- 中心孔 -->
  <geom type="cylinder" size="{bore} {width*1.5}" rgba="0.1 0.1 0.1 1"/>
  <!-- 齿廓示意 -->
  <geom name="teeth" type="cylinder" size="{pd*1.05} {width*0.5}" rgba="{color[0]*0.7} {color[1]*0.7} {color[2]*0.7} 1"/>'''
            
        elif geo_type == 'square_cylinder':
            s = dims.get('square_size', 0.03) / 2 * scale
            h = dims.get('body_length', 0.03) * scale
            shaft_r = dims.get('shaft_diameter', 0.005) / 2 * scale
            shaft_l = dims.get('shaft_length', 0.02) * scale
            geom_xml = f'''  <geom type="box" size="{s} {s} {h/2}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>
  <!-- 轴伸 -->
  <geom name="shaft" pos="0 0 {h/2+shaft_l/2}" type="cylinder" size="{shaft_r} {shaft_l/2}" rgba="0.7 0.7 0.7 1"/>'''
            
        else:
            # 默认：通用盒子
            l, w, h = 0.02, 0.015, 0.01
            geom_xml = f'  <geom type="box" size="{l} {w} {h}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}" material="mat_part"/>'
        
        # 构建完整MJCF XML
        xml = f"""<mujoco model="{spec.part_id}">
  <option gravity="0 0 -9.81" timestep="0.001"/>

  <asset>
    <material name="mat_part" specular="0.5" shininess="0.3"/>
    <material name="mat_grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.15 0.15 0.15" rgb2="0.25 0.25 0.25" width="512" height="512"/>
  </asset>

  <worldbody>
    <!-- 地面网格 -->
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="ground" type="plane" size="2 2 0.1" rgba="0.2 0.2 0.22 1" material="mat_grid"/>

    <!-- 当前零件: {spec.display_name} -->
    <body name="{spec.part_id}" pos="0 0 0.15">
{geom_xml}
    </body>
  </worldbody>
</mujoco>"""
        
        return xml
    
    def _load_part_model(self, idx: int):
        """加载指定索引的零件模型"""
        part_id = self.part_ids[idx]
        spec = self.all_parts[part_id]
        
        # 生成MJCF XML
        xml_str = self._create_mujoco_model_for_part(spec)
        
        # 临时保存XML文件
        temp_xml_path = self.out_dir / "_current_part.xml"
        temp_xml_path.write_text(xml_str, encoding='utf-8')
        
        # 加载模型
        model = mujoco.MjModel.from_xml_path(str(temp_xml_path))
        data = mujoco.MjData(model)
        
        return model, data, spec
    
    def _init_glfw(self):
        """初始化GLFW窗口"""
        if not glfw.init():
            raise RuntimeError("无法初始化 GLFW")
        
        self.window = glfw.create_window(
            WINDOW_WIDTH, WINDOW_HEIGHT,
            "🔧 零件库3D浏览器 — 47种工业级零件",
            None, None
        )
        
        if not self.window:
            glfw.terminate()
            raise RuntimeError("无法创建 GLFW 窗口")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        # 设置回调
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_window_size_callback(self.window, self._window_size_callback)
        
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        self.viewport = mujoco.MjrRect(0, 0, fb_width, fb_height)
        
        # 初始化渲染组件（需要OpenGL上下文）
        model, data, spec = self._load_part_model(0)
        self.model = model
        self.data = data
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        self.cam.lookat[:] = [0, 0, 0.15]
        self.cam.distance = 0.5
        self.cam.azimuth = 45
        self.cam.elevation = -20
    
    def _switch_to_part(self, idx: int):
        """切换到指定零件"""
        idx = max(0, min(idx, self.total_parts - 1))
        if idx != self.current_idx:
            self.current_idx = idx
            
            # 重新加载模型
            model, data, spec = self._load_part_model(idx)
            self.model = model
            self.data = data
            
            # 重建场景和上下文
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            
            # 重置相机
            mujoco.mjv_defaultFreeCamera(model, self.cam)
            self.cam.lookat[:] = [0, 0, 0.15]
            self.cam.distance = 0.5
            self.cam.azimuth = 45
            self.cam.elevation = -20
            
            self._print_part_info(spec)
    
    def _print_part_info(self, spec: PartSpec):
        """打印零件信息"""
        cat_name, _ = self.categories.get(spec.category, (spec.category, [0.5, 0.5, 0.5]))
        
        print(f"\n{'='*60}")
        print(f"  [{self.current_idx+1:2d}/{self.total_parts}] {spec.display_name}")
        print(f"{'='*60}")
        print(f"  ID:          {spec.part_id}")
        print(f"  分类:        {cat_name}")
        print(f"  制造商:      {spec.manufacturer}")
        print(f"  型号:        {spec.part_number}")
        print(f"  几何类型:    {spec.geometry_type}")
        print(f"  质量:        {spec.mass_kg:.3f} kg")
        print(f"  密度:        {spec.density_kgm3:.0f} kg/m³")
        
        if spec.voltage_v > 0:
            print(f"  电压:        {spec.voltage_v}V")
        if spec.torque_nm > 0:
            print(f"  扭矩:        {spec.torque_nm:.2f} N·m")
        if spec.speed_rpm > 0:
            print(f"  转速:        {spec.speed_rpm:.0f} RPM")
        if spec.current_a > 0:
            print(f"  电流:        {spec.current_a:.1f} A")
        
        print(f"  尺寸:")
        for k, v in spec.dimensions.items():
            print(f"    {k}: {v*100:.1f} cm")
        print(f"{'='*60}\n")
    
    def _cursor_pos_callback(self, window, xpos, ypos):
        dx = xpos - self.last_mouse_pos[0]
        dy = ypos - self.last_mouse_pos[1]
        
        if self.mouse_left:
            self.cam.azimuth += dx * 0.5
            self.cam.elevation += dy * 0.5
            self.cam.elevation = max(-89, min(89, self.cam.elevation))
        
        if self.mouse_right:
            scale = self.cam.distance * 0.001
            az = np.radians(self.cam.azimuth)
            self.cam.lookat[0] -= (dx * np.cos(az) + dy * np.sin(az)) * scale
            self.cam.lookat[1] -= (-dx * np.sin(az) + dy * np.cos(az)) * scale
        
        self.last_mouse_pos = (xpos, ypos)
    
    def _mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_left = (action == glfw.PRESS)
            if self.mouse_left:
                self.last_mouse_pos = glfw.get_cursor_pos(window)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse_right = (action == glfw.PRESS)
            if self.mouse_right:
                self.last_mouse_pos = glfw.get_cursor_pos(window)
    
    def _scroll_callback(self, window, xoffset, yoffset):
        zoom_speed = 0.1
        if yoffset > 0:
            self.cam.distance *= (1 - zoom_speed)
        else:
            self.cam.distance *= (1 + zoom_speed)
        self.cam.distance = max(0.1, min(5.0, self.cam.distance))
    
    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.PRESS and action != glfw.REPEAT:
            return
        
        spec = self._get_current_part()
        
        # 上一个/下一个
        if key == glfw.KEY_LEFT or key == glfw.KEY_A:
            self._switch_to_part(self.current_idx - 1)
        elif key == glfw.KEY_RIGHT or key == glfw.KEY_D:
            self._switch_to_part(self.current_idx + 1)
        
        # 首尾跳转
        elif key == glfw.KEY_HOME:
            self._switch_to_part(0)
        elif key == glfw.KEY_END:
            self._switch_to_part(self.total_parts - 1)
        
        # 分类跳转 (1-6)
        elif key == glfw.KEY_1:
            self._jump_to_category('actuators')
        elif key == glfw.KEY_2:
            self._jump_to_category('transmission')
        elif key == glfw.KEY_3:
            self._jump_to_category('energy')
        elif key == glfw.KEY_4:
            self._jump_to_category('sensors')
        elif key == glfw.KEY_5:
            self._jump_to_category('controllers')
        elif key == glfw.KEY_6:
            self._jump_to_category('connectors')
        
        # 截图
        elif key == glfw.KEY_R:
            self._take_screenshot(spec)
        
        # 信息面板切换
        elif key == glfw.KEY_I:
            self.show_info = not self.show_info
        
        # 自动旋转
        elif key == glfw.KEY_SPACE:
            self.auto_rotate = not self.auto_rotate
            status = "开启" if self.auto_rotate else "关闭"
            print(f"\n  🔄 自动旋转: {status}")
        
        # 退出
        elif key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
            glfw.set_window_should_close(window, True)
        
        # 帮助
        elif key == glfw.KEY_H:
            self._print_help()
    
    def _jump_to_category(self, category: str):
        """跳转到指定分类的第一个零件"""
        for i, pid in enumerate(self.part_ids):
            if self.all_parts[pid].category == category:
                self._switch_to_part(i)
                return
        print(f"\n  ⚠️ 未找到分类: {category}")
    
    def _window_size_callback(self, window, width, height):
        fb_width, fb_height = glfw.get_framebuffer_size(window)
        self.viewport = mujoco.MjrRect(0, 0, fb_width, fb_height)
    
    def _take_screenshot(self, spec: PartSpec):
        """截图保存"""
        self.screenshot_count += 1
        filename = f"{spec.part_id}_{self.screenshot_count:03d}.png"
        filepath = self.out_dir / filename
        
        width = self.viewport.width
        height = self.viewport.height
        
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        
        img = Image.frombytes('RGB', (width, height), pixels)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(filepath)
        
        size_kb = filepath.stat().st_size / 1024
        print(f"  📸 截图保存: {filename} ({size_kb:.0f}KB)")
    
    def _print_help(self):
        print("\n" + "="*62)
        print("  🔧 零件库浏览器 — 操作指南")
        print("="*62)
        print("  🖱️ 鼠标:")
        print("     左键拖拽 → 旋转 | 右键拖拽 → 平移 | 滚轮 → 缩放")
        print("")
        print("  ⌨️ 键盘:")
        print("     ←/A  → 上一个零件    →/D  → 下一个零件")
        print("     Home → 第一个         End  → 最后一个")
        print("     1-6  → 跳转分类      R    → 截图")
        print("     I    → 信息面板       Space→ 自动旋转")
        print("     H    → 帮助           Q/ESC→ 退出")
        print("")
        print("  📂 分类:")
        print("     1=执行器(12)  2=传动(6)  3=能源(8)")
        print("     4=传感(6)    5=控制(4)  6=连接(5)")
        print("="*62 + "\n")
    
    def run(self):
        """运行主循环"""
        try:
            self._init_glfw()
            
            spec = self._get_current_part()
            self._print_help()
            self._print_part_info(spec)
            
            print("  ✅ 零件库浏览器已启动!\n")
            
            frame_count = 0
            while not glfw.window_should_close(self.window):
                # 自动旋转
                if self.auto_rotate:
                    self.cam.azimuth += AUTO_ROTATE_SPEED
                
                # 更新场景
                mujoco.mjv_updateScene(
                    self.model, self.data, self.opt,
                    None, self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scene
                )
                
                # 渲染
                mujoco.mjr_render(self.viewport, self.scene, self.ctx)
                
                # 交换缓冲区
                glfw.swap_buffers(self.window)
                glfw.poll_events()
                
                frame_count += 1
            
        finally:
            if hasattr(self, 'window'):
                glfw.destroy_window(self.window)
            glfw.terminate()
            print(f"\n  👋 浏览器已关闭")
            print(f"  📁 截图保存在: {self.out_dir.absolute()}\n")


def main():
    print("=" * 64)
    print("  🔧 V8 零件库 — 3D交互式浏览器")
    print("  共47种工业级零件，支持旋转/缩放/逐个浏览")
    print("=" * 64)
    
    viewer = PartLibraryViewer()
    viewer.run()


if __name__ == "__main__":
    main()
