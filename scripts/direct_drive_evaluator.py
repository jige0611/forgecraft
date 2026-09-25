"""
直接驱动评估系统 (Direct-Drive Evaluator)
==========================================
绕过PPO训练，直接给电机施加驱动信号（正弦波/方波），
在MuJoCo物理仿真中测量机器人的实际运动能力。

核心优势：
- 不依赖RL训练，评估速度快10-100倍
- 直接测试物理可行性，结果可重复
- 对电机数量和配置敏感
"""

import numpy as np
import mujoco
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class DirectDriveConfig:
    """直接驱动评估配置"""
    # 仿真参数
    sim_steps: int = 1500           # 每个episode的仿真步数（增加到1500）
    sim_dt: float = 0.002          # 时间步长 (500Hz)
    substeps: int = 4              # 每步子步数
    
    # 稳定期参数：让机器人先站稳
    settling_steps: int = 200      # 稳定期步数（不施加驱动信号）
    grounding_steps: int = 300     # 落地步数（让悬空机器人落到地面）
    
    # 驱动信号参数
    drive_mode: str = "mixed"      # mixed / sine / square / random
    base_frequency: float = 3.0    # 基础频率 (Hz) - 提高频率
    frequency_range: Tuple[float, float] = (1.0, 10.0)  # 频率范围
    amplitude: float = 1.0         # 信号幅度 [-1, 1]
    
    # 多频率测试
    n_frequencies: int = 3         # 测试的不同频率数
    episodes_per_freq: int = 2     # 每个频率的episode数
    
    # 物理约束
    max_force: float = 200.0       # 最大驱动力 (N) - 大幅提高
    fall_height: float = 0.03      # 倒地判定高度 (m) - 更宽容
    
    # 评分权重
    weight_displacement: float = 0.5   # 位移奖励权重（最重要）
    weight_speed: float = 0.25         # 平均速度权重
    weight_survival: float = 0.15      # 存活时间权重
    weight_stability: float = 0.10     # 稳定性(不倒)权重


class DirectDriveEvaluator:
    """
    直接驱动评估器
    
    使用预设的驱动信号模式测试机器人运动能力，
    无需任何RL训练过程。
    """
    
    def __init__(self, config: DirectDriveConfig = None):
        self.config = config or DirectDriveConfig()
        self.logger = logging.getLogger("DirectDrive")
        
        # 缓存编译后的MJCF模型
        self._model_cache: Dict[int, Any] = {}
    
    def evaluate_body(
        self,
        body,
        catalog: Dict[str, Any],
        n_episodes: int = 6,
    ) -> Dict[str, float]:
        """
        评估单个机器人的直接驱动性能
        
        Args:
            body: MechanicalBody对象
            catalog: 零件目录
            n_episodes: 评估episode数
            
        Returns:
            包含各项指标的字典
        """
        try:
            # 构建MuJoCo模型
            from forgecraft.simulation.builder import build_mjcf_model
            
            xml_string, joint_map, motor_map, sensor_info = build_mjcf_model(body, catalog)
            
            # 编译XML为MuJoCo模型
            model = mujoco.MjModel.from_xml_string(xml_string)
            data = mujoco.MjData(model)
            
            # 检查是否有驱动关节
            n_actuators = model.nu
            if n_actuators == 0:
                return {
                    "fitness": 0.0,
                    "displacement": 0.0,
                    "avg_speed": 0.0,
                    "survival_ratio": 0.0,
                    "stability": 0.0,
                    "max_height": 0.0,
                    "n_actuators": 0,
                    "n_motors": len(body.actuated_joints()),
                }
            
            # 运行多episode评估
            results = []
            freqs = np.linspace(
                self.config.frequency_range[0],
                self.config.frequency_range[1],
                min(self.config.n_frequencies, n_episodes)
            )
            
            for ep in range(n_episodes):
                freq = freqs[ep % len(freqs)]
                result = self._run_episode(model, data, ep, freq)
                if result is not None:
                    results.append(result)
            
            if not results:
                return {
                    "fitness": 0.0, "displacement": 0.0, "avg_speed": 0.0,
                    "survival_ratio": 0.0, "stability": 0.0,
                    "max_height": 0.0, "n_actuators": n_actuators,
                    "n_motors": len(body.actuated_joints()),
                }
            
            # 聚合多episode结果
            avg_disp = np.mean([r["displacement"] for r in results])
            avg_speed = np.mean([r["avg_speed"] for r in results])
            avg_survival = np.mean([r["survival_ratio"] for r in results])
            avg_stability = np.mean([r["stability"] for r in results])
            max_height = np.max([r["max_height"] for r in results])
            
            # 综合适应度
            fitness = (
                self.config.weight_displacement * min(avg_disp * 5, 1.0) +
                self.config.weight_speed * min(avg_speed * 2, 1.0) +
                self.config.weight_survival * avg_survival +
                self.config.weight_stability * avg_stability
            )
            
            # 电机数量加成
            n_motors = len(body.actuated_joints())
            motor_bonus = 1.0 + min(n_motors * 0.1, 0.5)  # 最多+50%
            fitness *= motor_bonus
            
            return {
                "fitness": fitness,
                "displacement": avg_disp,
                "avg_speed": avg_speed,
                "survival_ratio": avg_survival,
                "stability": avg_stability,
                "max_height": max_height,
                "n_actuators": n_actuators,
                "n_motors": n_motors,
                "motor_bonus": motor_bonus,
            }
            
        except Exception as e:
            self.logger.warning(f"Direct drive evaluation failed: {e}")
            return {
                "fitness": 0.0, "displacement": 0.0, "avg_speed": 0.0,
                "survival_ratio": 0.0, "stability": 0.0,
                "max_height": 0.0, "n_actuators": 0,
                "n_motors": 0, "error": str(e),
            }
    
    def _run_episode(
        self,
        model,
        data,
        episode_id: int,
        frequency: float,
    ) -> Optional[Dict[str, float]]:
        """运行单个直接驱动episode（含稳定期）"""
        try:
            # 重置仿真
            mujoco.mj_resetData(model, data)
            
            # 设置初始状态：轻微随机化
            data.qpos[:] += np.random.uniform(-0.005, 0.005, model.nq)
            data.qvel[:] = np.random.uniform(-0.05, 0.05, model.nv)
            mujoco.mj_forward(model, data)
            
            n_actuators = model.nu
            n_steps = self.config.sim_steps
            grounding = self.config.grounding_steps
            settling = self.config.settling_steps
            dt = self.config.dt
            
            # 调试：记录初始状态
            init_com_raw = self._get_com_pos(model, data)
            
            # === 阶段1：落地期（让悬空机器人落到地面） ===
            for gstep in range(grounding):
                data.ctrl[:] = 0.0
                mujoco.mj_step(model, data, nstep=self.config.substeps)
                # 检查是否已落地
                com_g = self._get_com_pos(model, data)
                if com_g[2] < 0.15:  # 接近地面就停止落地期
                    break
            
            # === 阶段2：稳定期（落地后稳定） ===
            init_com = None
            
            # 跟踪变量
            total_displacement = 0.0
            prev_pos = None
            speed_sum = 0.0
            survival_count = 0
            fell = False
            max_height = 0.0
            drive_started = False
            
            t = 0.0
            for step in range(n_steps):
                # 稳定期：不施加驱动信号，让机器人自然稳定
                if step < settling:
                    data.ctrl[:] = 0.0
                else:
                    # 首次进入驱动期，记录初始位置
                    if not drive_started:
                        drive_started = True
                        init_com = self._get_com_pos(model, data).copy()
                        prev_pos = init_com[:2].copy()
                    
                    # 生成驱动信号
                    ctrl = self._generate_signal(t, n_actuators, frequency, episode_id)
                    data.ctrl[:] = ctrl
                
                # 仿真步进
                mujoco.mj_step(model, data, nstep=self.config.substeps)
                t += dt * self.config.substeps
                
                # 获取当前状态
                com_pos = self._get_com_pos(model, data)
                com_vel = self._get_com_vel(model, data)
                
                # 驱动期才开始跟踪运动指标
                if drive_started:
                    step_disp = np.linalg.norm(com_pos[:2] - prev_pos)
                    total_displacement += step_disp
                    prev_pos = com_pos[:2].copy()
                    speed_sum += np.linalg.norm(com_vel[:2])
                
                max_height = max(max_height, com_pos[2])
                
                # 存活计数
                if com_pos[2] > self.config.fall_height:
                    survival_count += 1
                else:
                    fell = True
                    break  # 提前终止
            
            # 如果从未进入驱动期（稳定期内就倒了）
            if init_com is None:
                return {
                    "displacement": 0.0, "total_path_length": 0.0,
                    "avg_speed": 0.0, "survival_ratio": 0.0,
                    "stability": 0.0, "max_height": max_height,
                    "fell": True, "n_steps": min(step+1, settling),
                }
            
            # 计算指标
            actual_steps = step + 1 if fell else n_steps
            drive_steps = actual_steps - settling
            survival_ratio = survival_count / n_steps
            final_com = self._get_com_pos(model, data)
            net_displacement = np.linalg.norm(final_com[:2] - init_com[:2])
            avg_speed = speed_sum / max(drive_steps, 1) if drive_steps > 0 else 0.0
            stability = 1.0 if not fell else survival_ratio
            
            return {
                "displacement": net_displacement,
                "total_path_length": total_displacement,
                "avg_speed": avg_speed,
                "survival_ratio": survival_ratio,
                "stability": stability,
                "max_height": max_height,
                "fell": fell,
                "n_steps": actual_steps,
                "drive_steps": drive_steps,
            }
            
        except Exception as e:
            self.logger.debug(f"Episode failed: {e}")
            return None
    
    def _generate_signal(
        self,
        t: float,
        n_actuators: int,
        frequency: float,
        episode_id: int,
    ) -> np.ndarray:
        """生成驱动信号"""
        amp = self.config.amplitude
        
        if self.config.drive_mode == "sine":
            # 正弦波，不同电机有相位偏移
            phases = np.linspace(0, 2*np.pi, n_actuators+1)[:-1]
            signal = amp * np.sin(2 * np.pi * frequency * t + phases)
        
        elif self.config.drive_mode == "square":
            # 方波
            phases = np.linspace(0, 2*np.pi, n_actuators+1)[:-1]
            signal = amp * np.sign(np.sin(2 * np.pi * frequency * t + phases))
        
        elif self.config.drive_mode == "random":
            # 随机信号（每100步更新一次）
            seed = int(t * 10 + episode_id * 1000)
            rng = np.random.RandomState(seed)
            signal = rng.uniform(-amp, amp, n_actuators)
        
        elif self.config.drive_mode == "mixed":
            # 混合模式：部分正弦、部分方波
            signal = np.zeros(n_actuators)
            for i in range(n_actuators):
                phase = 2 * np.pi * i / max(n_actuators, 1)
                if i < n_actuators // 2:
                    signal[i] = amp * np.sin(2 * np.pi * frequency * t + phase)
                else:
                    signal[i] = amp * 0.7 * np.sign(np.sin(2 * np.pi * frequency * 1.5 * t + phase))
        
        else:
            # 默认正弦波
            signal = amp * np.sin(2 * np.pi * frequency * t) * np.ones(n_actuators)
        
        return signal.astype(np.float32)
    
    @property
    def dt(self) -> float:
        """有效时间步长"""
        return self.config.sim_dt * self.config.substeps
    
    def _get_com_pos(self, model, data) -> np.ndarray:
        """获取质心位置"""
        mass = np.array(model.body_mass)
        pos = np.array(data.xpos[:, :3])
        if mass.sum() > 0:
            return (mass[:, None] * pos).sum(axis=0) / mass.sum()
        return pos[0]
    
    def _get_com_vel(self, model, data) -> np.ndarray:
        """获取质心速度"""
        mass = np.array(model.body_mass)
        vel = np.array(data.cvel[:, :3])
        if mass.sum() > 0:
            return (mass[:, None] * vel).sum(axis=0) / mass.sum()
        return vel[0]


def direct_drive_fitness_batch(
    bodies: List[Any],
    catalog: Dict[str, Any],
    evaluator: DirectDriveEvaluator = None,
    n_episodes: int = 6,
) -> List[Dict[str, float]]:
    """
    批量直接驱动评估（用于种群评估）
    
    Args:
        bodies: 机器人个体列表
        catalog: 零件目录
        evaluator: 评估器实例
        n_episodes: 每个个体的episode数
        
    Returns:
        结果列表
    """
    if evaluator is None:
        evaluator = DirectDriveEvaluator()
    
    results = []
    for body in bodies:
        res = evaluator.evaluate_body(body, catalog, n_episodes)
        res["_body"] = body
        results.append(res)
    
    return results


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    print("=" * 60)
    print("  DIRECT DRIVE EVALUATOR TEST")
    print("=" * 60)
    
    # 加载forgecraft模块
    sys.path.insert(0, str(Path(__file__).parent))
    
    from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
    from forgecraft.core.generator import BodyGenerator
    
    catalog = load_catalog()
    generator = BodyGenerator(catalog)
    
    # 创建评估器
    config = DirectDriveConfig(
        sim_steps=800,
        drive_mode="mixed",
        n_frequencies=4,
        episodes_per_freq=2,
    )
    evaluator = DirectDriveEvaluator(config)
    
    # 生成测试种群
    print("\nGenerating test population...")
    population = generator.generate_initial_population(size=8, min_parts=3, max_parts=12)
    
    # 评估
    print(f"\nEvaluating {len(population)} individuals with direct drive...")
    start = time.time()
    
    results = []
    for i, body in enumerate(population):
        res = evaluator.evaluate_body(body, catalog, n_episodes=6)
        results.append(res)
        
        motors = res.get("n_motors", 0)
        actuators = res.get("n_actuators", 0)
        fit = res.get("fitness", 0)
        disp = res.get("displacement", 0)
        speed = res.get("avg_speed", 0)
        
        status = "MOTION" if disp > 0.001 else "static"
        print(f"  [{i+1:>2}] fit={fit:.4f} | disp={disp:.4f}m | "
              f"speed={speed:.4f}m/s | motors={motors} | acts={actuators} | [{status}]")
    
    elapsed = time.time() - start
    print(f"\nTotal evaluation time: {elapsed:.1f}s ({elapsed/len(population):.1f}s/body)")
    
    # 找最佳
    best_idx = np.argmax([r.get("fitness", 0) for r in results])
    best = results[best_idx]
    print(f"\nBest individual #{best_idx+1}:")
    print(f"  Fitness: {best['fitness']:.4f}")
    print(f"  Displacement: {best['displacement']:.4f} m")
    print(f"  Avg Speed: {best['avg_speed']:.4f} m/s")
    print(f"  Motors: {best['n_motors']}")
    print(f"  Survival: {best['survival_ratio']:.2%}")
    print(f"  Stability: {best['stability']:.2%}")
