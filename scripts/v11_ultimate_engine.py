#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 Ultimate Evolution Engine - Template-Based Evolution
========================================================
集成运动能力模板 + 直接驱动评估 + 完整进化流程

核心改进（vs V10）：
1. 使用LocomotionTemplateGenerator生成具有运动能力的机器人
2. 使用DirectDriveEvaluator直接测量位移/速度
3. 类型专用驱动信号（差速/步态/波浪）
4. 更强的选择压力 + 智能变异

里程碑：获得能实际移动的最佳机器人模型
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# ============================================================
# 项目路径设置
# ============================================================
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 导入核心模块
# ============================================================
from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
from forgecraft.core.morphology import MechanicalBody, Part, Joint
from forgecraft.simulation.builder import build_mjcf_model
from locomotion_templates import LocomotionTemplateGenerator, TemplateConfig


@dataclass
class V11Config:
    """V11实验配置"""
    # 进化参数
    total_generations: int = 100       # 总代数
    population_size: int = 40          # 种群大小
    
    # 评估参数
    eval_episodes: int = 6             # 每个体评估次数
    sim_steps: int = 3000             # 仿真步数
    settling_steps: int = 200         # 稳定期步数
    grounding_steps: int = 500        # 落地期步数
    
    # 选择参数
    tournament_size: int = 5          # 锦标赛选择大小
    elite_count: int = 5              # 精英保留数量
    crossover_rate: float = 0.3       # 交叉概率
    mutation_rate: float = 0.8        # 变异概率
    
    # 早停参数
    patience: int = 20                # 停滞耐心值
    min_improvement: float = 0.001    # 最小改进阈值
    
    # 输出
    output_dir: str = "v11_results"
    save_every: int = 10              # 每N代保存


@dataclass 
class RobotResult:
    """单个机器人的评估结果"""
    body: MechanicalBody
    displacement: float = 0.0
    speed: float = 0.0
    survival: float = 0.0
    stability: float = 0.0
    fitness: float = 0.0
    robot_type: str = ""              # diff_wheel / quad_wheel / bipedal / quadruped / crawler
    n_motors: int = 0
    n_parts: int = 0


class DirectDriveEvaluator:
    """
    直接驱动评估器 - 核心评估组件
    
    直接给电机施加驱动信号，测量实际运动能力。
    绕过PPO训练，速度快10-100倍。
    """
    
    def __init__(self, config: V11Config):
        self.config = config
        self.logger = logging.getLogger("DirectDriveEval")
        self.catalog = load_catalog()
        
        # 驱动信号配置（按类型）
        self.drive_configs = {
            "diff_drive": {"type": "differential", "base_speed": 0.9},
            "quad_wheeled": {"type": "differential", "base_speed": 0.8},
            "bipedal": {"type": "gait", "freq": 2.5, "amplitude": 0.7},
            "quadruped": {"type": "gait", "freq": 2.0, "amplitude": 0.6},
            "crawler": {"type": "wave", "freq": 3.5, "amplitude": 0.8},
        }
    
    def evaluate(self, body: MechanicalBody) -> Optional[RobotResult]:
        """
        评估单个机器人
        
        返回包含各项指标的RobotResult，或None如果评估失败
        """
        try:
            # 构建MuJoCo模型
            xml_string, jm, mm, si = build_mjcf_model(body, self.catalog)
            
            import mujoco
            model = mujoco.MjModel.from_xml_string(xml_string)
            data = mujoco.MjData(model)
            
            if model.nu == 0:
                return None
            
            # 识别机器人类型
            robot_type = self._detect_type(body)
            drive_cfg = self.drive_configs.get(robot_type, {"type": "mixed"})
            
            n_motors = len(body.actuated_joints())
            results_list = []
            
            # 多次评估取平均
            for ep in range(self.config.eval_episodes):
                result = self._run_episode(model, data, ep, robot_type, drive_cfg)
                if result is not None:
                    results_list.append(result)
            
            if not results_list:
                return None
            
            # 计算平均指标
            avg_disp = np.mean([r["disp"] for r in results_list])
            avg_speed = np.mean([r["speed"] for r in results_list])
            avg_survival = np.mean([r["survival"] for r in results_list])
            max_disp = max(r["disp"] for r in results_list)
            
            # 综合适应度计算
            fitness = self._compute_fitness(avg_disp, avg_speed, avg_survival, 
                                           max_disp, n_motors)
            
            return RobotResult(
                body=body,
                displacement=avg_disp,
                speed=avg_speed,
                survival=avg_survival,
                stability=avg_survival,
                fitness=fitness,
                robot_type=robot_type,
                n_motors=n_motors,
                n_parts=body.num_parts(),
            )
            
        except Exception as e:
            self.logger.debug(f"Evaluation failed: {e}")
            return None
    
    def _detect_type(self, body: MechanicalBody) -> str:
        """根据身体结构识别机器人类型"""
        name = body.name.lower()
        if "diff" in name or "drive" in name:
            return "diff_drive"
        elif "quad" in name:
            if "wheel" in name:
                return "quad_wheeled"
            return "quadruped"
        elif "biped" in name or "bi" in name:
            return "bipedal"
        elif "crawl" in name:
            return "crawler"
        else:
            # 根据零件数量和结构推断
            n = body.num_parts()
            if n <= 5:
                return "diff_drive"  # 小型→可能是轮式
            elif n <= 8:
                return "bipedal"
            else:
                return "quadruped"
    
    def _run_episode(
        self,
        model,
        data,
        episode_id: int,
        robot_type: str,
        drive_cfg: dict,
    ) -> Optional[Dict[str, float]]:
        """运行单个评估episode"""
        try:
            import mujoco
            
            mujoco.mj_resetData(model, data)
            
            # 轻微随机化初始状态
            data.qpos[:] += np.random.uniform(-0.003, 0.003, model.nq)
            data.qvel[:] += np.random.uniform(-0.02, 0.02, model.nv)
            mujoco.mj_forward(model, data)
            
            mass = np.array(model.body_mass)
            def get_com():
                xpos = np.array(data.xpos[:, :3])
                return (mass[:, None] * xpos).sum(axis=0) / mass.sum()
            
            n_actuators = model.nu
            dt = 0.002 * 4  # dt * substeps
            
            # === 阶段1：落地期 ===
            for _ in range(self.config.grounding_steps):
                data.ctrl[:] = 0.0
                mujoco.mj_step(model, data, nstep=4)
            
            # === 阶段2：稳定+驱动 ===
            settling = self.config.settling_steps
            init_com = None
            prev_pos = None
            speed_sum = 0.0
            drive_steps = 0
            fell = False
            
            t = 0.0
            for step in range(self.config.sim_steps):
                if step < settling:
                    data.ctrl[:] = 0.0
                else:
                    # 首次进入驱动期
                    if init_com is None:
                        init_com = get_com().copy()
                        prev_pos = init_com[:2].copy()
                    
                    # 生成驱动信号
                    ctrl = self._generate_signal(t, n_actuators, episode_id, 
                                                robot_type, drive_cfg)
                    data.ctrl[:] = ctrl
                
                mujoco.mj_step(model, data, nstep=4)
                t += dt
                
                com_now = get_com()
                
                if init_com is not None:
                    step_disp = np.linalg.norm(com_now[:2] - prev_pos)
                    speed_sum += step_disp / dt if dt > 0 else 0
                    prev_pos = com_now[:2].copy()
                    drive_steps += 1
                    
                    if com_now[2] < 0.03:
                        fell = True
                        break
            
            if init_com is None:
                return {"disp": 0, "speed": 0, "survival": 0}
            
            final_com = get_com()
            net_disp = np.linalg.norm(final_com[:2] - init_com[:2])
            avg_speed = speed_sum / max(drive_steps, 1)
            actual_steps = min(step + 1, self.config.sim_steps)
            survival = (actual_steps - settling) / max(self.config.sim_steps - settling, 1)
            
            return {
                "disp": net_disp,
                "speed": avg_speed,
                "survival": survival,
            }
            
        except Exception as e:
            return None
    
    def _generate_signal(
        self, t: float, n_actuators: int, episode_id: int,
        robot_type: str, cfg: dict
    ) -> np.ndarray:
        """生成类型专用的驱动信号"""
        dtype = cfg.get("type", "mixed")
        
        if dtype == "differential":
            # 差速驱动：所有轮子同向旋转 → 前进
            base = cfg.get("base_speed", 0.8)
            ctrl = np.ones(n_actuators) * base
            # 添加轻微差异以产生转向趋势
            if n_actuators >= 2:
                ctrl[0] *= (1.0 + 0.1 * np.sin(0.5 * t))
                ctrl[1] *= (1.0 + 0.1 * np.sin(0.5 * t + 0.5))
            return ctrl
        
        elif dtype == "gait":
            # 步态驱动：交替摆动
            freq = cfg.get("freq", 2.0)
            amp = cfg.get("amplitude", 0.7)
            phases = np.linspace(0, 2*np.pi, n_actuators+1)[:-1]
            
            if robot_type == "bipedal":
                # 双足：左右腿相位相反
                ctrl = np.zeros(n_actuators)
                for i in range(n_actuators):
                    phase = phases[i]
                    if i % 2 == 0:
                        ctrl[i] = amp * np.sin(2*np.pi*freq*t + phase)
                    else:
                        ctrl[i] = amp * np.sin(2*np.pi*freq*t + phase + np.pi)
                        
            elif robot_type == "quadruped":
                # 四足：对角步态
                ctrl = np.zeros(n_actuators)
                for i in range(n_actuators):
                    phase = phases[i]
                    leg_group = i % 4
                    if leg_group in [0, 3]:  # 对角组1
                        ctrl[i] = amp * np.sin(2*np.pi*freq*t + phase)
                    else:  # 对角组2
                        ctrl[i] = amp * np.sin(2*np.pi*freq*t + phase + np.pi)
            else:
                ctrl = amp * np.sin(2*np.pi*freq*t + phases)
            
            return ctrl
        
        elif dtype == "wave":
            # 波浪驱动：相邻关节有相位延迟
            freq = cfg.get("freq", 3.0)
            amp = cfg.get("amplitude", 0.8)
            ctrl = np.zeros(n_actuators)
            for i in range(n_actuators):
                phase_delay = i * 0.7  # 相邻关节相位延迟
                ctrl[i] = amp * np.sin(2*np.pi*freq*t - phase_delay)
            return ctrl
        
        else:  # mixed
            freq = 3.0
            phases = np.linspace(0, 2*np.pi, n_actuators+1)[:-1]
            ctrl = np.sin(2*np.pi*freq*t + phases) * \
                  np.cos(2*np.pi*freq*0.7*t + phases*0.5)
            return ctrl
    
    def _compute_fitness(
        self, disp: float, speed: float, survival: float,
        max_disp: float, n_motors: int
    ) -> float:
        """
        综合适应度计算
        
        权重：
        - 位移（最重要）：40%
        - 速度：25%
        - 存活率：20%
        - 最大位移奖励：15%
        """
        w_disp = 0.40
        w_speed = 0.25
        w_survival = 0.20
        w_max = 0.15
        
        # 归一化到合理范围
        disp_score = min(disp / 0.5, 1.0)  # 0.5m为满分
        speed_score = min(speed / 0.3, 1.0)  # 0.3m/s为满分
        surv_score = survival
        max_score = min(max_disp / 0.5, 1.0)
        
        # 无电机惩罚
        motor_factor = 1.0
        if n_motors == 0:
            motor_factor = 0.05
        elif n_motors == 1:
            motor_factor = 0.3
        
        fitness = (
            w_disp * disp_score +
            w_speed * speed_score +
            w_survival * surv_score +
            w_max * max_score
        ) * motor_factor
        
        return fitness


class V11EvolutionEngine:
    """
    V11终极进化引擎
    
    核心流程：
    1. 用模板生成初始种群（保证运动潜力）
    2. 直接驱动评估每个体
    3. 锦标赛选择 + 精英保留
    4. 参数变异 + 结构变异
    5. 重复直到收敛或达到最大代数
    """
    
    def __init__(self, config: V11Config = None):
        self.config = config or V11Config()
        self.logger = self._setup_logging()
        
        # 初始化组件
        self.template_gen = LocomotionTemplateGenerator(seed=42)
        self.evaluator = DirectDriveEvaluator(self.config)
        self.catalog = load_catalog()
        
        # 状态变量
        self.population: List[RobotResult] = []
        self.generation = 0
        self.history: List[Dict] = []
        self.best_result: Optional[RobotResult] = None
        self.best_fitness = -float('inf')
        self.stagnation_count = 0
        
        # 创建输出目录
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """配置日志"""
        logger = logging.getLogger("V11Engine")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(handler)
        
        return logger
    
    def initialize_population(self):
        """使用模板生成器初始化种群"""
        self.logger.info(f"Generating population of size {self.config.population_size}...")
        
        from forgecraft.core.generator import BodyGenerator
        fallback_gen = BodyGenerator(self.catalog, seed=42)
        
        bodies = self.template_gen.generate_population(
            size=self.config.population_size,
            min_parts=4,
            max_parts=12,
        )
        
        # 如果模板生成的不足，用传统生成器补充
        while len(bodies) < self.config.population_size:
            extra = fallback_gen.generate_initial_population(
                size=1, min_parts=4, max_parts=10
            )
            bodies.extend(extra)
        
        # 截断到目标大小
        bodies = bodies[:self.config.population_size]
        
        # 评估所有个体
        self.logger.info("Evaluating initial population...")
        self.population = []
        
        for i, body in enumerate(bodies):
            result = self.evaluator.evaluate(body)
            if result is not None:
                self.population.append(result)
                
                if result.fitness > self.best_fitness:
                    self.best_fitness = result.fitness
                    self.best_result = result
                
                status = f"disp={result.displacement:.4f}m mot={result.n_motors}"
                icon = "*" if result.fitness > 0.01 else " "
                self.logger.info(f"  [{i+1:>2}/{len(bodies)}] {icon} type={result.robot_type:<12} fit={result.fitness:.4f} {status}")
            else:
                self.logger.info(f"  [{i+1:>2}/{len(bodies)}] ✗ failed")
        
        self.generation = 0
        self.logger.info(f"\nInitial pop: {len(self.population)} valid, best_fit={self.best_fitness:.4f}")
    
    def run(self):
        """运行完整进化实验"""
        start_time = time.time()
        self.logger.info("=" * 65)
        self.logger.info("  V11 ULTIMATE EVOLUTION ENGINE")
        self.logger.info("=" * 65)
        self.logger.info(f"Config: gen={self.config.total_generations} pop={self.config.population_size}")
        
        # 初始化
        self.initialize_population()
        
        # 进化主循环
        for gen_idx in range(1, self.config.total_generations + 1):
            self.generation = gen_idx
            gen_start = time.time()
            
            self.logger.info(f"\n{'─'*50}")
            self.logger.info(f"Generation {gen_idx}/{self.config.total_generations}")
            
            # 选择
            selected = self._select()
            
            # 交叉+变异产生新种群
            new_pop = self._breed(selected)
            
            # 评估新种群
            self.logger.info("Evaluating new population...")
            evaluated = []
            for i, body in enumerate(new_pop):
                result = self.evaluator.evaluate(body)
                if result is not None:
                    evaluated.append(result)
                    
                    if result.fitness > self.best_fitness:
                        improvement = result.fitness - self.best_fitness
                        self.best_fitness = result.fitness
                        self.best_result = result
                        self.stagnation_count = 0
                        self.logger.info(f"  *** NEW BEST! fit={result.fitness:.4f} (+{improvement:.4f}) "
                                       f"type={result.robot_type} disp={result.displacement:.4f}m")
                    
                    if (i+1) % 10 == 0:
                        self.logger.info(f"  Evaluated {i+1}/{len(new_pop)}")
            
            # 精英保留
            elite = self._get_elite()
            combined = evaluated + elite
            combined.sort(key=lambda r: r.fitness, reverse=True)
            self.population = combined[:self.config.population_size]
            
            # 记录历史
            gen_time = time.time() - gen_start
            gen_stats = self._compute_stats()
            gen_stats["generation"] = gen_idx
            gen_stats["time"] = gen_time
            self.history.append(gen_stats)
            
            # 打印统计
            self.logger.info(f"Gen {gen_idx}: "
                           f"avg_fit={gen_stats['avg_fitness']:.4f} "
                           f"best_fit={gen_stats['best_fitness']:.4f} "
                           f"avg_disp={gen_stats['avg_displacement']*100:.2f}cm "
                           f"valid={gen_stats['valid_count']}/{self.config.population_size} "
                           f"time={gen_time:.1f}s")
            
            # 定期保存
            if gen_idx % self.config.save_every == 0:
                self._save_checkpoint()
            
            # 早停检查
            if self.stagnation_count >= self.config.patience:
                self.logger.info(f"\nEarly stop at gen {gen_idx} (stagnation={self.stagnation_count})")
                break
        
        # 最终报告
        total_time = time.time() - start_time
        self._final_report(total_time)
        
        return self.best_result
    
    def _select(self) -> List[RobotResult]:
        """锦标赛选择"""
        selected = []
        valid = [r for r in self.population if r.fitness > 0]
        
        if len(valid) < 2:
            valid = self.population
        
        for _ in range(self.config.population_size):
            # 锦标赛
            candidates = [valid[np.random.randint(len(valid))] 
                         for _ in range(self.config.tournament_size)]
            winner = max(candidates, key=lambda r: r.fitness)
            selected.append(winner)
        
        return selected
    
    def _get_elite(self) -> List[RobotResult]:
        """获取精英个体"""
        elite = sorted(self.population, key=lambda r: r.fitness, reverse=True)
        return elite[:self.config.elite_count]
    
    def _breed(self, selected: List[RobotResult]) -> List[MechanicalBody]:
        """交叉+变异产生新种群"""
        new_bodies = []
        
        for i in range(0, len(selected), 2):
            parent1 = selected[i].body
            parent2 = selected[i+1].body if i+1 < len(selected) else selected[0].body
            
            # 交叉
            if np.random.random() < self.config.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = self._mutate_body(parent1.copy() if hasattr(parent1, 'copy') else parent1)
            
            if child is not None:
                new_bodies.append(child)
            
            # 变异第二个父代
            mutated = self._mutate_body(parent2)
            if mutated is not None:
                new_bodies.append(mutated)
        
        # 补充新随机个体（保持多样性）
        while len(new_bodies) < self.config.population_size:
            fresh = self.template_gen.generate_population(size=1, min_parts=4, max_parts=12)
            if fresh:
                new_bodies.extend(fresh)
        
        return new_bodies[:self.config.population_size]
    
    def _crossover(self, p1: MechanicalBody, p2: MechanicalBody) -> Optional[MechanicalBody]:
        """交叉两个父代"""
        try:
            # 简单交叉：从p1取结构，从p2取部分参数
            child = p1  # 复制结构
            
            # 交叉参数
            for node_id in list(child.graph.nodes)[:min(len(child.graph.nodes), len(p2.graph.nodes))]:
                part = child.graph.nodes[node_id]["part"]
                
                if node_id in p2.graph.nodes:
                    p2_part = p2.graph.nodes[node_id]["part"]
                    
                    # 以50%概率从p2继承参数
                    for key in ["mass", "length", "radius"]:
                        if key in part.params and key in p2_part.params:
                            if np.random.random() > 0.5:
                                part.params[key] = p2_part.params[key]
            
            return child
        except Exception as e:
            return None
    
    def _mutate_body(self, body: MechanicalBody) -> Optional[MechanicalBody]:
        """变异单个机器人"""
        try:
            for node_id in body.graph.nodes:
                part = body.graph.nodes[node_id]["part"]
                
                # 参数变异
                if np.random.random() < self.config.mutation_rate:
                    for key in list(part.params.keys()):
                        if key in ("actuated", "touch_sensor", "imu_sensor"):
                            continue
                        
                        val = part.params[key]
                        if isinstance(val, (int, float)) and val != 0:
                            # 高斯变异，幅度随停滞增加
                            scale = 0.15 * (1 + self.stagnation_count * 0.05)
                            mutation = np.random.normal(0, abs(val) * scale)
                            part.params[key] = max(val * 0.3, val + mutation)
            
            return body
        except Exception as e:
            return None
    
    def _compute_stats(self) -> Dict:
        """计算当前代统计信息"""
        if not self.population:
            return {
                "avg_fitness": 0, "best_fitness": 0, "avg_displacement": 0,
                "valid_count": 0, "motion_capable": 0,
            }
        
        fits = [r.fitness for r in self.population]
        disps = [r.displacement for r in self.population]
        moving = sum(1 for r in self.population if r.displacement > 0.001)
        
        return {
            "avg_fitness": np.mean(fits),
            "best_fitness": max(fits),
            "avg_displacement": np.mean(disps),
            "valid_count": len(self.population),
            "motion_capable": moving,
        }
    
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "history": self.history,
            "population_summary": [
                {
                    "fitness": r.fitness,
                    "displacement": r.displacement,
                    "robot_type": r.robot_type,
                    "n_motors": r.n_motors,
                }
                for r in sorted(self.population, key=lambda x: x.fitness, reverse=True)[:10]
            ],
        }
        
        path = self.output_dir / f"checkpoint_gen{self.generation}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Saved checkpoint: {path}")
    
    def _final_report(self, total_time: float):
        """生成最终报告"""
        self.logger.info("\n" + "=" * 65)
        self.logger.info("  V11 EVOLUTION COMPLETE")
        self.logger.info("=" * 65)
        
        if self.best_result is not None:
            self.logger.info(f"\nBest Individual:")
            self.logger.info(f"  Fitness:     {self.best_result.fitness:.4f}")
            self.logger.info(f"  Displacement:{self.best_result.displacement:.4f} m")
            self.logger.info(f"  Speed:       {self.best_result.speed:.4f} m/s")
            self.logger.info(f"  Survival:    {self.best_result.survival:.1%}")
            self.logger.info(f"  Type:        {self.best_result.robot_type}")
            self.logger.info(f"  Parts:       {self.best_result.n_parts}")
            self.logger.info(f"  Motors:      {self.best_result.n_motors}")
        
        self.logger.info(f"\nTotal time: {total_time/60:.1f} minutes")
        self.logger.info(f"Generations: {self.generation}")
        
        # 保存最终结果
        final_data = {
            "best_result": {
                "fitness": self.best_result.fitness if self.best_result else 0,
                "displacement": self.best_result.displacement if self.best_result else 0,
                "speed": self.best_result.speed if self.best_result else 0,
                "survival": self.best_result.survival if self.best_result else 0,
                "robot_type": self.best_result.robot_type if self.best_result else "",
                "n_motors": self.best_result.n_motors if self.best_result else 0,
                "n_parts": self.best_result.n_parts if self.best_result else 0,
            },
            "history": self.history,
            "config": {
                "total_generations": self.config.total_generations,
                "population_size": self.config.population_size,
                "eval_episodes": self.config.eval_episodes,
            },
            "total_time_seconds": total_time,
        }
        
        path = self.output_dir / "v11_final_results.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"\nResults saved to: {path}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    # 配置实验
    config = V11Config(
        total_generations=100,
        population_size=30,
        eval_episodes=4,           # 每个体评估4次
        sim_steps=3000,            # 3秒仿真
        settling_steps=200,
        grounding_steps=500,
        tournament_size=5,
        elite_count=4,
        patience=25,
        output_dir="v11_results",
        save_every=10,
    )
    
    # 运行进化
    engine = V11EvolutionEngine(config)
    best = engine.run()
    
    print("\nok啦鸡哥！")
