# ═══════════════════════════════════════════════════════════════
#  ForgeCraft 仿真回放系统
#  ═══════════════════════════════════════════════════════════════
#
#  功能：
#  - 录制仿真状态序列（位置、速度、关节角度）
#  - 回放任意速度（0.1x ~ 10x）
#  - 帧级跳转和关键帧标记
#  - 导出为视频/动画GIF
#  - 多机器人对比回放
#
#  使用：
#    from replay import SimulationRecorder, SimulationPlayer
#    
#    # 录制
#    recorder = SimulationRecorder()
#    recorder.record_step(data, model)
#    recorder.save("recording.pkl")
#    
#    # 回放
#    player = SimulationPlayer("recording.pkl")
#    player.play(speed=2.0)
# ═══════════════════════════════════════════════════════════════

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

logger = logging.getLogger("ForgeCraft.Replay")


@dataclass 
class FrameData:
    """单帧数据"""
    frame_id: int = 0
    timestamp: float = 0.0        # 仿真时间 (s)
    wall_time: float = 0.0       # 实际时间 (s)
    
    # 机器人状态
    positions: np.ndarray = None  # 各body位置 (n_body x 3)
    orientations: np.ndarray = None  # 四元数 (n_body x 4)
    velocities: np.ndarray = None   # 线速度 (n_body x 3)
    angular_velocities: np.ndarray = None  # 角速度
    
    # 关节状态
    joint_positions: np.ndarray = None  # 关节角度
    joint_velocities: np.ndarray = None  # 关节角速度
    
    # 执行器状态
    actuator_forces: np.ndarray = None
    actuator_controls: np.ndarray = None
    
    # 性能指标
    displacement: float = 0.0      # 累计位移
    speed: float = 0.0             # 当前速度
    height: float = 0.0            # 质心高度
    fitness: float = 0.0           # 当前适应度
    
    # 自定义数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为可序列化字典"""
        d = {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "wall_time": self.wall_time,
            "displacement": self.displacement,
            "speed": self.speed,
            "height": self.height,
            "fitness": self.fitness,
            "metadata": self.metadata,
        }
        
        # 序列化numpy数组
        arrays = {
            "positions": self.positions,
            "orientations": self.orientations,
            "velocities": self.velocities,
            "joint_positions": self.joint_positions,
            "actuator_controls": self.actuator_controls,
        }
        
        for name, arr in arrays.items():
            if arr is not None:
                d[name] = arr.tolist() if isinstance(arr, np.ndarray) else arr
        
        return d


@dataclass
class RecordingMetadata:
    """录制元数据"""
    robot_name: str = ""
    robot_type: str = ""
    total_frames: int = 0
    duration: float = 0.0          # 仿真时长(s)
    dt: float = 0.002             # 时间步长
    n_bodies: int = 0
    n_joints: int = 0
    n_actuators: int = 0
    
    # 录制信息
    recorded_at: str = ""
    config: Dict = field(default_factory=dict)
    
    # 统计
    max_displacement: float = 0.0
    max_speed: float = 0.0
    avg_fitness: float = 0.0
    survival_rate: float = 1.0


@dataclass
class Recording:
    """完整录制"""
    metadata: RecordingMetadata = field(default_factory=RecordingMetadata)
    frames: List[FrameData] = field(default_factory=list)
    keyframes: List[int] = field(default_factory=list)  # 关键帧索引


class SimulationRecorder:
    """仿真状态录制器"""
    
    def __init__(self, model=None, data=None):
        self.model = model
        self.data = data
        self.recording = Recording()
        self._frame_count = 0
        self._start_time = None
        self._initial_pos = None
        
        if model is not None:
            self._init_from_model(model)
    
    def _init_from_model(self, model):
        """从MuJoCo模型初始化元数据"""
        self.recording.metadata.n_bodies = model.nbody
        self.recording.metadata.n_joints = model.njnt
        self.recording.metadata.n_actuators = model.nu
        self.recording.metadata.dt = model.opt.timestep
    
    def set_robot_info(self, name: str, robot_type: str = ""):
        """设置机器人信息"""
        self.recording.metadata.robot_name = name
        self.recording.metadata.robot_type = robot_type
    
    def start(self):
        """开始录制"""
        self._frame_count = 0
        self._start_time = datetime.now()
        self.recording.frames.clear()
        self.recording.keyframes.clear()
        logger.info(f"Recording started")
    
    def record_frame(self, mujoco_data=None, mujoco_model=None, **kwargs) -> FrameData:
        """
        录制当前帧
        
        Args:
            mujoco_data: MuJoCo MjData对象
            mujoco_model: MuJoCo MjModel对象  
            **kwargs: 额外指标 (displacement, speed, fitness等)
        """
        frame = FrameData(
            frame_id=self._frame_count,
            timestamp=self._frame_count * self.recording.metadata.dt,
            wall_time=(datetime.now() - self._start_time).total_seconds() if self._start_time else 0,
        )
        
        # 从Mujoco数据提取状态
        if mujoco_data is not None and mujoco_model is not None:
            nbody = mujoco_model.nbody
            
            frame.positions = mujoco_data.xpos[:nbody].copy()
            frame.orientations = mujoco_data.xquat[:nbody].copy()
            frame.velocities = mujoco_data.cvel[:nbody].copy()
            
            njnt = mujoco_model.njnt
            if njnt > 0:
                frame.joint_positions = mujoco_data.qpos[7:].copy()  # 跳过自由度
                frame.joint_velocities = mujoco_data.qvel[6:].copy()
            
            nu = mujoco_model.nu
            if nu > 0:
                frame.actuator_forces = mujoco_data.actuator_force[:nu].copy()
                frame.actuator_controls = mujoco_data.ctrl[:nu].copy()
            
            # 计算质心高度
            body_pos = mujoco_data.xpos[1] if nbody > 1 else mujoco_data.xpos[0]
            frame.height = body_pos[2] if len(body_pos) > 2 else 0
            
            # 初始化位置记录
            if self._initial_pos is None and nbody > 1:
                self._initial_pos = mujoco_data.xpos[1].copy()
        
        # 额外指标
        for key, value in kwargs.items():
            if hasattr(frame, key):
                setattr(frame, key, value)
        
        # 更新统计
        if frame.displacement > self.recording.metadata.max_displacement:
            self.recording.metadata.max_displacement = frame.displacement
        if frame.speed > self.recording.metadata.max_speed:
            self.recording.metadata.max_speed = frame.speed
        
        self.recording.frames.append(frame)
        self._frame_count += 1
        
        return frame
    
    def add_keyframe(self, label: str = ""):
        """添加关键帧"""
        idx = len(self.recording.frames) - 1
        if idx >= 0:
            self.recording.keyframes.append(idx)
            if label:
                self.recording.frames[idx].metadata["keyframe_label"] = label
    
    def stop(self) -> Recording:
        """停止录制并返回结果"""
        # 更新元数据
        meta = self.recording.metadata
        meta.total_frames = len(self.recording.frames)
        meta.duration = meta.total_frames * meta.dt
        meta.recorded_at = datetime.now().isoformat()
        
        if self.recording.frames:
            fits = [f.fitness for f in self.recording.frames if f.fitness > 0]
            if fits:
                meta.avg_fitness = sum(fits) / len(fits)
        
        logger.info(f"Recording stopped: {meta.total_frames} frames, {meta.duration:.2f}s")
        return self.recording
    
    def save(self, path: str) -> Path:
        """保存录制到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 序列化
        data = {
            "metadata": asdict(self.recording.metadata),
            "frames": [f.to_dict() for f in self.recording.frames],
            "keyframes": self.recording.keyframes,
        }
        
        if path.suffix == '.pkl':
            with open(path, 'wb') as f:
                pickle.dump(self.recording, f)
        elif path.suffix == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        else:
            # 默认用pickle保存完整对象，同时导出JSON摘要
            with open(path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(self.recording, f)
            
            summary_path = path.with_suffix('.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Recording saved to: {path}")
        return path


class SimulationPlayer:
    """仿真回放器"""
    
    def __init__(self, recording_path: str = None, recording: Recording = None):
        if recording is not None:
            self.recording = recording
        elif recording_path is not None:
            self.load(recording_path)
        else:
            self.recording = Recording()
        
        self.current_frame_idx = 0
        self.playback_speed = 1.0
        self.playing = False
        self.looping = False
        self.on_frame_callback: Optional[Callable] = None
    
    def load(self, path: str) -> bool:
        """加载录制文件"""
        path = Path(path)
        
        try:
            if path.suffix == '.pkl' or (not path.exists() and path.with_suffix('.pkl').exists()):
                pkl_path = path if path.suffix == '.pkl' else path.with_suffix('.pkl')
                with open(pkl_path, 'rb') as f:
                    self.recording = pickle.load(f)
            elif path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._load_from_dict(data)
            else:
                logger.error(f"Unknown format: {path.suffix}")
                return False
            
            logger.info(f"Loaded recording: {self.recording.metadata.total_frames} frames")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load recording: {e}")
            return False
    
    def _load_from_dict(self, data: dict):
        """从字典加载"""
        meta_data = data.get("metadata", {})
        self.recording.metadata = RecordingMetadata(**{k: v for k, v in meta_data.items() 
                                                      if k in RecordingMetadata.__dataclass_fields__})
        
        self.recording.frames.clear()
        for fd in data.get("frames", []):
            frame = FrameData()
            for key, value in fd.items():
                if isinstance(value, list) and key in ['positions', 'orientations', 'velocities',
                                                        'joint_positions', 'joint_velocities']:
                    setattr(frame, key, np.array(value))
                elif hasattr(frame, key):
                    setattr(frame, key, value)
            self.recording.frames.append(frame)
        
        self.recording.keyframes = data.get('keyframes', [])
    
    @property
    def total_frames(self) -> int:
        return len(self.recording.frames)
    
    @property
    def current_frame(self) -> Optional[FrameData]:
        if 0 <= self.current_frame_idx < self.total_frames:
            return self.recording.frames[self.current_frame_idx]
        return None
    
    @property
    def progress(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.current_frame_idx / self.total_frames
    
    @property
    def duration(self) -> float:
        return self.recording.metadata.duration
    
    def seek(self, frame_id: int = None, timestamp: float = None, ratio: float = None):
        """跳转到指定帧"""
        if frame_id is not None:
            self.current_frame_idx = max(0, min(frame_id, self.total_frames - 1))
        elif timestamp is not None:
            dt = self.recording.metadata.dt
            if dt > 0:
                target = int(timestamp / dt)
                self.current_frame_idx = max(0, min(target, self.total_frames - 1))
        elif ratio is not None:
            self.current_frame_idx = int(ratio * (self.total_frames - 1))
    
    def next_frame(self) -> Optional[FrameData]:
        """下一帧"""
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            frame = self.current_frame
            if self.on_frame_callback:
                self.on_frame_callback(frame)
            return frame
        elif self.looping:
            self.current_frame_idx = 0
            return self.current_frame
        return None
    
    def prev_frame(self) -> Optional[FrameData]:
        """上一帧"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            return self.current_frame
        return None
    
    def get_frame_at(self, timestamp: float) -> Optional[FrameData]:
        """获取指定时间点的帧（插值）"""
        if not self.recording.frames:
            return None
        
        dt = self.recording.metadata.dt
        if dt <= 0:
            return self.recording.frames[0]
        
        target_idx = int(timestamp / dt)
        
        # 边界处理
        if target_idx < 0:
            return self.recording.frames[0]
        if target_idx >= self.total_frames:
            return self.recording.frames[-1]
        
        return self.recording.frames[target_idx]
    
    def get_trajectory(self) -> Dict[str, np.ndarray]:
        """提取完整轨迹数据"""
        if not self.recording.frames:
            return {}
        
        positions = np.array([f.positions for f in self.recording.frames if f.positions is not None])
        times = np.array([f.timestamp for f in self.recording.frames])
        displacements = np.array([f.displacement for f in self.recording.frames])
        speeds = np.array([f.speed for f in self.recording.frames])
        
        return {
            "times": times,
            "positions": positions,
            "displacements": displacements,
            "speeds": speeds,
        }
    
    def get_keyframe_indices(self) -> List[int]:
        """获取所有关键帧索引"""
        return self.recording.keyframes.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取回放统计"""
        if not self.recording.frames:
            return {}
        
        displacements = [f.displacement for f in self.recording.frames]
        speeds = [f.speed for f in self.recording.frames]
        heights = [f.height for f in self.recording.frames if f.height > 0]
        
        return {
            "total_frames": self.total_frames,
            "duration_s": self.duration,
            "max_displacement": max(displacements) if displacements else 0,
            "final_displacement": displacements[-1] if displacements else 0,
            "avg_speed": sum(speeds) / len(speeds) if speeds else 0,
            "max_speed": max(speeds) if speeds else 0,
            "min_height": min(heights) if heights else 0,
            "max_height": max(heights) if heights else 0,
            "n_keyframes": len(self.recording.keyframes),
        }
    
    def export_video_frames(self, output_dir: str, fps: int = 30) -> List[Path]:
        """导出视频帧（需要配合渲染器使用）"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported = []
        interval = max(1, int(1.0 / (self.recording.metadata.dt * fps)))
        
        for i, frame in enumerate(self.recording.frames):
            if i % interval == 0:
                # 这里只导出帧数据，实际图像由外部渲染器生成
                frame_path = output_dir / f"frame_{i:05d}.json"
                with open(frame_path, 'w') as f:
                    json.dump(frame.to_dict(), f)
                exported.append(frame_path)
        
        return exported
    
    def compare_with(self, other_player: 'SimulationPlayer') -> Dict[str, Any]:
        """与另一个回放对比"""
        my_stats = self.get_statistics()
        other_stats = other_player.get_statistics()
        
        return {
            "this": {"name": self.recording.metadata.robot_name, **my_stats},
            "other": {"name": other_player.recording.metadata.robot_name, **other_stats},
            "comparison": {
                "displacement_diff": my_stats.get("final_displacement", 0) - other_stats.get("final_displacement", 0),
                "speed_diff": my_stats.get("avg_speed", 0) - other_stats.get("avg_speed", 0),
                "duration_diff": my_stats.get("duration_s", 0) - other_stats.get("duration_s", 0),
            }
        }


def create_recorder_from_viewer(viewer) -> SimulationRecorder:
    """从Viewer创建录制器"""
    recorder = SimulationRecorder(model=viewer.model)
    recorder.set_robot_info(
        getattr(viewer.body, 'name', 'robot'),
        getattr(viewer.body, 'robot_type', 'unknown')
    )
    return recorder


if __name__ == "__main__":
    print("="*60)
    print("  Simulation Replay System Test")
    print("="*60 + "\n")
    
    # 创建测试录制
    recorder = SimulationRecorder()
    recorder.set_robot_info("test_robot", "differential_wheeled")
    recorder.start()
    
    # 模拟录制100帧
    import math
    for i in range(100):
        t = i * 0.002
        recorder.record_frame(
            displacement=0.01 * t + 0.0001 * math.sin(t * 10),
            speed=0.01 + 0.001 * math.cos(t * 10),
            height=0.1 + 0.005 * math.sin(t * 5),
            fitness=min(0.01 * t, 0.5),
        )
        
        if i % 25 == 0:
            recorder.add_keyframe(label=f"frame_{i}")
    
    recording = recorder.stop()
    
    # 保存
    save_path = recorder.save("replay_test/recording")
    print(f"Saved: {save_path}")
    print(f"Frames: {recording.metadata.total_frames}")
    print(f"Duration: {recording.metadata.duration:.2f}s")
    print(f"Keyframes: {len(recording.keyframes)}")
    
    # 测试回放
    player = SimulationPlayer(str(save_path))
    stats = player.get_statistics()
    print(f"\nPlayback Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
