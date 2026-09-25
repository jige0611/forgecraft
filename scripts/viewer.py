# ═══════════════════════════════════════════════════════════════
#  ForgeCraft 实时3D仿真查看器
#  ═══════════════════════════════════════════════════════════════
#
#  功能：
#  - 加载V11最佳机器人模型
#  - MuJoCo实时物理仿真 + 渲染
#  - 键盘交互控制 (WASD驱动, 空格重置, R记录)
#  - HUD显示性能指标
#  - 支持无头模式和窗口模式
#
#  用法：
#    python viewer.py                    # 查看V11最佳个体
#    python viewer.py --template diff   # 使用模板生成
#    python viewer.py --headless        # 无头模式(截图)
#    python viewer.py --record demo.mp4 # 录制视频
# ═══════════════════════════════════════════════════════════════

import sys
import os
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

import mujoco
from mujoco import viewer as mj_viewer

# ForgeCraft模块
from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
from forgecraft.core.morphology import MechanicalBody
from forgecraft.simulation.builder import build_mjcf_model
from locomotion_templates import LocomotionTemplateGenerator


@dataclass
class ViewerConfig:
    """查看器配置"""
    # 仿真参数
    timestep: float = 0.002       # 物理时间步长 (500Hz)
    substeps: int = 4            # 子步数
    
    # 显示参数
    width: int = 1200            # 窗口宽度
    height: int = 800            # 窗口高度
    camera_distance: float = 2.0 # 相机距离
    camera_azimuth: float = 90   # 方位角
    camera_elevation: float = -20 # 仰角
    
    # 驱动参数
    drive_speed: float = 0.8     # 基础驱动速度
    auto_drive: bool = True      # 自动施加驱动信号
    
    # 录制参数
    record: bool = False         # 是否录制
    output_path: str = "viewer_output"  # 输出目录
    
    # HUD显示
    show_hud: bool = True        # 显示性能指标
    show_contacts: bool = True   # 显示碰撞点
    show_com: bool = True        # 显示质心


class RealtimeViewer:
    """
    ForgeCraft实时3D仿真查看器
    
    集成MuJoCo原生渲染器，提供：
    - 实时物理仿真
    - 交互式相机控制
    - 键盘驱动机器人
    - 性能指标HUD
    """
    
    def __init__(
        self,
        body: MechanicalBody,
        config: ViewerConfig = None,
        catalog: Dict[str, Any] = None
    ):
        self.config = config or ViewerConfig()
        self.catalog = catalog or load_catalog()
        self.body = body
        
        self.logger = logging.getLogger("Viewer")
        
        # MuJoCo模型和数据
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.xml_string: str = ""
        self.joint_map: Dict[str, Tuple[str, str]] = {}
        self.motor_map: Dict[str, str] = {}
        
        # 运行状态
        self.running = False
        self.paused = False
        self.step_count = 0
        self.start_time: float = 0
        self.sim_time: float = 0
        
        # 性能追踪
        self.initial_pos: np.ndarray = np.zeros(3)
        self.positions: List[np.ndarray] = []
        self.speeds: List[float] = []
        self.max_height: float = 0
        self.fall_count: int = 0
        
        # 驱动状态
        self.drive_mode: str = "auto"  # auto / manual / none
        self.manual_ctrl: np.ndarray = np.array([])
        
        # 录制状态
        self.frames: List[Any] = []
        
        # 初始化模型
        self._build_model()
    
    def _build_model(self) -> bool:
        """构建MuJoCo模型"""
        try:
            self.logger.info(f"Building model for body: {self.body.name}")
            
            result = build_mjcf_model(self.body, self.catalog)
            if isinstance(result, tuple):
                self.xml_string, self.joint_map, self.motor_map, _ = result
            else:
                self.xml_string = result
                self.joint_map = {}
                self.motor_map = {}
            
            # 编译模型
            self.model = mujoco.MjModel.from_xml_string(self.xml_string)
            self.data = mujoco.MjData(self.model)
            
            # 设置时间步长
            self.model.opt.timestep = self.config.timestep
            
            # 记录初始位置
            mujoco.mj_forward(self.model, self.data)
            self._update_initial_pos()
            
            # 初始化手动控制
            n_actuators = self.model.nu
            self.manual_ctrl = np.zeros(n_actuators)
            
            self.logger.info(
                f"Model loaded: {self.model.nbody} bodies, "
                f"{self.model.njnt} joints, {self.model.nu} actuators, "
                f"{self.model.ngeom} geoms"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build model: {e}")
            return False
    
    def _update_initial_pos(self):
        """更新初始位置（质心）"""
        mass = np.array(self.model.body_mass)
        xpos = np.array(self.data.xpos[:, :3])
        if mass.sum() > 0:
            self.initial_pos = (mass[:, None] * xpos).sum(axis=0) / mass.sum()
        else:
            self.initial_pos = xpos[0].copy()
    
    def _get_current_position(self) -> np.ndarray:
        """获取当前质心位置"""
        mass = np.array(self.model.body_mass)
        xpos = np.array(self.data.xpos[:, :3])
        if mass.sum() > 0:
            return (mass[:, None] * xpos).sum(axis=0) / mass.sum()
        return xpos[0].copy()
    
    def _get_displacement(self) -> float:
        """获取位移（X方向）"""
        current = self._get_current_position()
        return abs(current[0] - self.initial_pos[0])
    
    def _get_speed(self) -> float:
        """获取当前速度"""
        current = self._get_current_position()
        if len(self.positions) >= 2:
            dt = self.config.timestep * self.config.substeps
            prev = self.positions[-1]
            return np.linalg.norm(current - prev) / dt
        return 0.0
    
    def _generate_drive_signal(self) -> np.ndarray:
        """生成自动驱动信号"""
        n_actuators = self.model.nu
        if n_actuators == 0:
            return np.array([])
        
        t = self.sim_time
        
        # 根据机器人类型选择驱动模式
        robot_type = getattr(self.body, 'robot_type', 'unknown')
        
        if robot_type == "diff_drive":
            # 差速驱动：左右轮同向旋转
            ctrl = np.ones(n_actuators) * self.config.drive_speed
            if n_actuators >= 2:
                # 轻微差异产生转向趋势
                ctrl[0] *= (1.0 + 0.15 * np.sin(2 * t))
                ctrl[1] *= (1.0 + 0.15 * np.sin(2 * t + 0.5))
            return ctrl
        
        elif robot_type in ["bipedal", "quadruped"]:
            # 步态驱动：交替摆动
            freq = 3.0
            amp = self.config.drive_speed
            phases = np.linspace(0, 2*np.pi, n_actuators+1)[:-1]
            ctrl = amp * np.sin(2*np.pi*freq*t + phases)
            
            if robot_type == "bipedal" and n_actuators >= 2:
                for i in range(n_actuators):
                    if i % 2 == 1:
                        ctrl[i] *= -1  # 左右相反
            elif robot_type == "quadruped":
                for i in range(n_actuators):
                    leg_group = i % 4
                    if leg_group in [1, 2]:
                        ctrl[i] *= -1  # 对角组相反
            return ctrl
        
        else:
            # 默认混合信号
            freq = 3.0
            phases = np.linspace(0, 2*np.pi, max(n_actuators, 1)+1)[:-1]
            ctrl = np.sin(2*np.pi*freq*t + phases) * self.config.drive_speed
            ctrl *= np.cos(2*np.pi*freq*0.7*t + phases*0.5)
            return ctrl
    
    def step(self) -> Dict[str, Any]:
        """执行单步仿真"""
        if self.paused or not self.running:
            return {}
        
        # 应用驱动信号
        if self.drive_mode == "auto" and self.config.auto_drive:
            self.data.ctrl[:] = self._generate_drive_signal()
        elif self.drive_mode == "manual":
            self.data.ctrl[:] = self.manual_ctrl
        else:
            self.data.ctrl[:] = 0
        
        # 执行物理步骤
        mujoco.mj_step(self.model, self.data, nstep=self.config.substeps)
        
        # 更新状态
        self.step_count += 1
        self.sim_time += self.config.timestep * self.config.substeps
        
        # 记录数据
        pos = self._get_current_position()
        self.positions.append(pos.copy())
        speed = self._get_speed()
        self.speeds.append(speed)
        
        # 追踪高度
        height = pos[2]
        if height > self.max_height:
            self.max_height = height
        
        # 检测倒地
        if height < 0.02:
            self.fall_count += 1
        
        # 构建返回信息
        info = {
            "step": self.step_count,
            "sim_time": self.sim_time,
            "position": pos.tolist(),
            "displacement": self._get_displacement(),
            "speed": speed,
            "height": height,
            "max_height": self.max_height,
            "n_contacts": self.data.ncon,
            "ctrl": self.data.ctrl.copy().tolist(),
        }
        
        return info
    
    def reset(self):
        """重置仿真"""
        if self.model is None or self.data is None:
            return
        
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.sim_time = 0
        self.start_time = time.time()
        self.positions = []
        self.speeds = []
        self.max_height = 0
        self.fall_count = 0
        self._update_initial_pos()
        self.logger.info("Simulation reset")
    
    def run_headless(self, duration: float = 5.0, save_images: bool = True) -> Dict[str, Any]:
        """
        无头模式运行（截图/录制）
        
        Args:
            duration: 仿真时长（秒）
            save_images: 是否保存截图
        """
        if self.model is None:
            self.logger.error("No model loaded")
            return {}
        
        self.logger.info(f"Running headless mode for {duration}s...")
        
        # 创建渲染器（使用较小分辨率以兼容framebuffer）
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # 输出目录
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(exist_ok=True)
        
        # 运行仿真
        self.running = True
        self.reset()
        
        n_steps = int(duration / (self.config.timestep * self.config.substeps))
        frames = []
        stats = {
            "positions": [],
            "speeds": [],
            "displacements": [],
            "heights": [],
            "times": [],
        }
        
        start_time = time.time()
        
        for step_i in range(n_steps):
            # 步进
            info = self.step()
            
            # 记录统计
            stats["times"].append(info.get("sim_time", 0))
            stats["positions"].append(info.get("position", [0,0,0]))
            stats["speeds"].append(info.get("speed", 0))
            stats["displacements"].append(info.get("displacement", 0))
            stats["heights"].append(info.get("height", 0))
            
            # 定期渲染和保存
            if step_i % 10 == 0 and save_images:
                renderer.update_scene(self.data)
                
                frame = renderer.render()
                frames.append(frame.copy())
                
                # 保存图片
                from PIL import Image
                img = Image.fromarray(frame[::-1, :, :])  # 翻转Y轴
                img_path = output_dir / f"frame_{step_i:05d}.png"
                img.save(img_path)
        
        elapsed = time.time() - start_time
        
        # 最终统计
        final_stats = {
            "duration": duration,
            "sim_time": self.sim_time,
            "total_steps": self.step_count,
            "elapsed_real": elapsed,
            "final_displacement": self._get_displacement(),
            "avg_speed": np.mean(self.speeds) if self.speeds else 0,
            "max_height": self.max_height,
            "n_frames": len(frames),
            "output_dir": str(output_dir),
            "stats": stats,
        }
        
        # 保存结果
        results_path = output_dir / "headless_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_stats, f, indent=2, default=str)
        
        renderer.close()
        self.running = False
        
        self.logger.info(
            f"Headless complete: {len(frames)} frames saved to {output_dir}"
        )
        
        return final_stats
    
    def run_interactive(self) -> None:
        """交互模式运行（带MuJoCo viewer）"""
        if self.model is None:
            self.logger.error("No model loaded")
            return
        
        self.logger.info("Starting interactive viewer...")
        self.logger.info("Controls:")
        self.logger.info("  Space: Pause/Resume")
        self.logger.info("  R: Reset simulation")
        self.logger.info("  WASD: Manual control")
        self.logger.info("  Q/E: Increase/decrease speed")
        self.logger.info("  1-5: Switch drive mode")
        self.logger.info("  ESC: Exit")
        
        self.running = True
        self.reset()
        
        # 创建viewer窗口
        with mj_viewer.launch_passive(
            self.model, self.data,
            key_callback=self._key_callback
        ) as viewer:
            
            # 配置相机
            viewer.cam.distance = self.config.camera_distance
            viewer.cam.azimuth = self.config.camera_azimuth
            viewer.cam.elevation = self.config.camera_elevation
            
            # 主循环
            last_hud_update = 0
            hud_interval = 0.5  # 每0.5秒更新一次HUD
            
            while self.running:
                step_start = time.time()
                
                # 步进仿真
                if not self.paused:
                    self.step()
                    
                    # 更新相机跟踪
                    pos = self._get_current_position()
                    viewer.cam.lookat[0] = pos[0]
                    viewer.cam.lookat[1] = pos[1]
                    viewer.cam.lookat[2] = max(pos[2], 0.05)
                
                # 渲染
                viewer.sync()
                
                # 打印HUD（定期）
                if self.config.show_hud and (time.time() - last_hud_update > hud_interval):
                    self._print_hud()
                    last_hud_update = time.time()
                
                # 控制帧率
                step_time = time.time() - step_start
                target_dt = self.config.timestep * self.config.substeps
                if step_time < target_dt:
                    time.sleep(target_dt - step_time)
        
        self.running = False
        self.logger.info("Viewer closed")
    
    def _key_callback(self, keycode: int) -> None:
        """键盘回调处理"""
        import glfw
        
        # 空格键：暂停/继续
        if keycode == glfw.KEY_SPACE:
            self.paused = not self.paused
            self.logger.info(f"{'Paused' if self.paused else 'Resumed'}")
        
        # R键：重置
        elif keycode == glfw.KEY_R:
            self.reset()
        
        # 数字键：切换驱动模式
        elif keycode == glfw.KEY_1:
            self.drive_mode = "auto"
            self.logger.info("Drive mode: Auto")
        elif keycode == glfw.KEY_2:
            self.drive_mode = "manual"
            self.logger.info("Drive mode: Manual")
        elif keycode == glfw.KEY_3:
            self.drive_mode = "none"
            self.logger.info("Drive mode: None (free fall)")
        
        # Q/E：调整速度
        elif keycode == glfw.KEY_Q:
            self.config.drive_speed = min(self.config.drive_speed + 0.1, 2.0)
            self.logger.info(f"Drive speed: {self.config.drive_speed:.1f}")
        elif keycode == glfw.KEY_E:
            self.config.drive_speed = max(self.config.drive_speed - 0.1, 0.0)
            self.logger.info(f"Drive speed: {self.config.drive_speed:.1f}")
        
        # 手动控制 (WASD)
        elif keycode == glfw.KEY_W:
            if len(self.manual_ctrl) >= 2:
                self.manual_ctrl[:min(2, len(self.manual_ctrl))] += 0.1
                self.manual_ctrl = np.clip(self.manual_ctrl, -1, 1)
        elif keycode == glfw.KEY_S:
            if len(self.manual_ctrl) >= 2:
                self.manual_ctrl[:min(2, len(self.manual_ctrl))] -= 0.1
                self.manual_ctrl = np.clip(self.manual_ctrl, -1, 1)
        elif keycode == glfw.KEY_A:
            if len(self.manual_ctrl) >= 2:
                self.manual_ctrl[0] -= 0.1
                self.manual_ctrl[1] += 0.1
                self.manual_ctrl = np.clip(self.manual_ctrl, -1, 1)
        elif keycode == glfw.KEY_D:
            if len(self.manual_ctrl) >= 2:
                self.manual_ctrl[0] += 0.1
                self.manual_ctrl[1] -= 0.1
                self.manual_ctrl = np.clip(self.manual_ctrl, -1, 1)
    
    def _print_hud(self):
        """打印HUD信息"""
        disp = self._get_displacement() * 100  # cm
        speed = self.speeds[-1] if self.speeds else 0
        height = self._get_current_position()[2]
        
        print(f"\r"
              f"[{self.sim_time:>6.2f}s] "
              f"Disp: {disp:>6.2f}cm | "
              f"Speed: {speed:>5.3f}m/s | "
              f"Height: {height:>5.3f}m | "
              f"Contacts: {self.data.ncon:>3} | "
              f"{'PAUSED' if self.paused else 'RUNNING':>7}", end="", flush=True)


def load_best_v11_robot() -> Tuple[MechanicalBody, Dict[str, Any]]:
    """加载V11实验的最佳机器人"""
    results_path = Path("v11_results/v11_final_results.json")
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        best_info = results.get("best_result", {})
        logging.info(f"Loaded V11 best: fitness={best_info.get('fitness', 0):.4f}")
    
    # 使用模板生成一个类似最佳的机器人
    template_gen = LocomotionTemplateGenerator(seed=42)
    population = template_gen.generate_population(size=1)
    body = population[0] if population else None
    
    catalog = load_catalog()
    return body, catalog


def main():
    parser = argparse.ArgumentParser(
        description="ForgeCraft Real-time 3D Simulation Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python viewer.py                  # View V11 best robot
  python viewer.py --template quad  # Generate quadruped
  python viewer.py --headless       # Headless mode (screenshots)
  python viewer.py --record video   # Record simulation
        """
    )
    
    parser.add_argument(
        "--template", "-t",
        type=str,
        choices=["diff", "four_wheel", "bipedal", "quadruped", "crawler"],
        help="Robot template to generate"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (save screenshots)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=10.0,
        help="Simulation duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--record",
        type=str,
        metavar="OUTPUT_PATH",
        help="Record simulation to path"
    )
    parser.add_argument(
        "--no-auto-drive",
        action="store_true",
        help="Disable automatic motor driving"
    )
    parser.add_argument(
        "--drive-speed",
        type=float,
        default=0.8,
        help="Base drive signal amplitude (default: 0.8)"
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable HUD overlay"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 加载或生成机器人
    if args.template:
        template_gen = LocomotionTemplateGenerator(seed=42)
        template_map = {
            "diff": "differential_wheeled",
            "four_wheel": "quad_wheeled",
            "bipedal": "bipedal",
            "quadruped": "quadruped",
            "crawler": "crawler",
        }
        # 强制生成指定类型
        original_weights = template_gen.template_weights.copy()
        for k in template_gen.template_weights:
            template_gen.template_weights[k] = 0.0
        if args.template in template_map:
            template_gen.template_weights[template_map[args.template]] = 1.0
        population = template_gen.generate_population(size=1)
        body = population[0] if population else None
        print(f"Generated template: {args.template}")
    else:
        body, _ = load_best_v11_robot()
    
    # 配置查看器
    config = ViewerConfig(
        auto_drive=not args.no_auto_drive,
        drive_speed=args.drive_speed,
        show_hud=not args.no_hud,
        record=bool(args.record),
        output_path=args.record or "viewer_output",
    )
    
    # 创建查看器
    viewer = RealtimeViewer(body, config)
    
    if viewer.model is None:
        print("ERROR: Failed to create model!")
        sys.exit(1)
    
    # 运行
    if args.headless:
        results = viewer.run_headless(duration=args.duration, save_images=True)
        print(f"\nHeadless Results:")
        print(f"  Duration: {results['duration']}s sim, {results['elapsed_real']:.1f}s real")
        print(f"  Displacement: {results['final_displacement']*100:.2f}cm")
        print(f"  Avg Speed: {results['avg_speed']:.3f}m/s")
        print(f"  Max Height: {results['max_height']:.3f}m")
        print(f"  Frames: {results['n_frames']}")
        print(f"  Output: {results['output_dir']}")
    else:
        viewer.run_interactive()


if __name__ == "__main__":
    main()
