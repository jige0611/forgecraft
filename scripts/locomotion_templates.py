"""
运动能力机器人模板生成器 (Locomotion-Capable Template Generator)
==============================================================
生成具有基本运动能力的机器人模板，包括：
- 轮式机器人（差速驱动/全向）
- 腿式机器人（双足/四足）

每个模板保证：
1. 有稳定的地面接触（轮子或脚）
2. 有驱动关节（电机连接到运动部件）
3. 物理上可行（不会立即倒塌）
4. 参数可变（支持进化优化）
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from forgecraft.core.morphology import MechanicalBody, Part, Joint


@dataclass
class TemplateConfig:
    """模板配置"""
    # 尺寸变化范围（增大最小值确保MuJoCo兼容）
    body_size_range: Tuple[float, float] = (0.12, 0.25)   # 主体尺寸（提高下限）
    wheel_radius_range: Tuple[float, float] = (0.04, 0.10)  # 轮半径（提高下限）
    leg_length_range: Tuple[float, float] = (0.10, 0.20)    # 腿长（提高下限）
    motor_torque_range: Tuple[float, float] = (1.0, 15.0)   # 电机扭矩
    
    # 结构参数
    max_variations: float = 0.3       # 最大随机变异比例
    ensure_symmetry: bool = True      # 保证对称性


class LocomotionTemplateGenerator:
    """
    运动能力模板生成器
    
    生成具有基本运动能力的机器人，而非随机树结构。
    """
    
    def __init__(
        self,
        config: TemplateConfig = None,
        seed: int = 42,
        catalog: Dict = None,
    ):
        self.config = config or TemplateConfig()
        self.rng = np.random.RandomState(seed)
        self.catalog = catalog
        self.logger = logging.getLogger("LocoTemplate")
        
        # 模板类型权重
        self.template_weights = {
            "differential_wheeled": 0.35,   # 差速轮式
            "quad_wheeled": 0.25,           # 四轮
            "bipedal": 0.20,                # 双足
            "quadruped": 0.15,             # 四足
            "crawler": 0.05,               # 履带式（简化）
        }
    
    def generate_population(
        self,
        size: int = 40,
        min_parts: int = 4,
        max_parts: int = 12,
    ) -> List[MechanicalBody]:
        """生成具有运动能力的种群"""
        population = []
        
        template_names = list(self.template_weights.keys())
        weights = list(self.template_weights.values())
        
        for _ in range(size):
            # 选择模板类型
            ttype = self.rng.choice(template_names, p=weights)
            
            # 生成机器人
            if ttype == "differential_wheeled":
                body = self._create_differential_wheeled()
            elif ttype == "quad_wheeled":
                body = self._create_quad_wheeled()
            elif ttype == "bipedal":
                body = self._create_bipedal()
            elif ttype == "quadruped":
                body = self._create_quadruped()
            elif ttype == "crawler":
                body = self._create_crawler()
            else:
                body = self._create_differential_wheeled()
            
            if body is not None and body.num_parts() >= min_parts:
                population.append(body)
        
        # 如果不够，补充差速轮式
        while len(population) < size:
            body = self._create_differential_wheeled()
            if body is not None:
                population.append(body)
        
        self.logger.info(f"Generated {len(population)} locomotion-capable robots")
        return population
    
    def _create_part(self, part_type: str, position: np.ndarray, 
                     params: Dict = None, is_actuated: bool = False) -> Part:
        """创建零件"""
        if params is None:
            params = {}
        
        # 基础物理参数
        default_params = {
            "actuated": 1.0 if is_actuated else 0.0,
        }
        
        if is_actuated or part_type in ("hinge_motor",):
            default_params.update({
                "actuated": 1.0,
                "max_torque": self.rng.uniform(*self.config.motor_torque_range),
                "max_velocity": self.rng.uniform(5.0, 30.0),
            })
        
        default_params.update(params)
        return Part(part_type=part_type, params=default_params, position=position.copy())
    
    # ══════════════════════════════════════════════════════════
    # 模板1：差速轮式机器人（最简单、最可靠的运动方式）
    # ══════════════════════════════════════════════════════════
    
    def _create_differential_wheeled(self) -> Optional[MechanicalBody]:
        """
        创建差速轮式机器人
        
        结构：
              [主体(box_body)]
             /              \
        [左轮(motor)]  [右轮(motor)]
        
        运动：左右轮差速转向+前进
        """
        try:
            body = MechanicalBody(name="diff_drive")
            
            # 随机化参数
            body_len = self.rng.uniform(*self.config.body_size_range)
            body_h = body_len * self.rng.uniform(0.5, 0.8)
            wheel_r = self.rng.uniform(*self.config.wheel_radius_range)
            wheel_w = wheel_r * self.rng.uniform(0.4, 0.8)
            track_width = body_len * self.rng.uniform(1.2, 1.8)  # 轮距
            
            # 主体位置：让轮子接触地面
            ground_clearance = wheel_r * 0.9  # 轮子稍微压入地面
            body_z = wheel_r + body_h / 2 + 0.01
            
            # === 主体 ===
            root = self._create_part("box_body", np.array([0.0, 0.0, body_z]), {
                "length": body_len, "height": body_h, "width": body_len * 0.6,
                "mass": max(body_len * body_h * 2000, 0.1),  # 最小0.1kg
            })
            body.add_part(root)
            
            # === 左轮（用cylinder/rod代替sphere，惯量更大） ===
            left_wheel_pos = np.array([-track_width/2, 0.0, wheel_r])
            left_wheel = self._create_part("rod", left_wheel_pos, {
                "length": wheel_w,           # 轮宽
                "radius": wheel_r,           # 轮半径
                "mass": max(wheel_r**2 * wheel_w * 6000, 0.1),  # 密度~6000kg/m³
            }, is_actuated=True)
            body.add_part(left_wheel)
            
            # 左轮电机关节
            left_joint = Joint(
                joint_type="hinge",
                parent_id=root.part_id,
                child_id=left_wheel.part_id,
                anchor=left_wheel_pos.copy(),
                axis=np.array([0.0, 1.0, 0.0]),  # Y轴旋转（轮子前后滚动）
                params={
                    "range_min": -999, "range_max": 999,  # 无限旋转
                    "damping": 0.3,
                    "actuated": 1.0,
                    "max_torque": max(self.rng.uniform(5.0, 20.0), 1.0),
                },
            )
            body.add_joint(left_joint)
            
            # === 右轮（用cylinder/rod） ===
            right_wheel_pos = np.array([track_width/2, 0.0, wheel_r])
            right_wheel = self._create_part("rod", right_wheel_pos, {
                "length": wheel_w,
                "radius": wheel_r,
                "mass": max(wheel_r**2 * wheel_w * 6000, 0.1),
            }, is_actuated=True)
            body.add_part(right_wheel)
            
            # 右轮电机关节
            right_joint = Joint(
                joint_type="hinge",
                parent_id=root.part_id,
                child_id=right_wheel.part_id,
                anchor=right_wheel_pos.copy(),
                axis=np.array([0.0, 1.0, 0.0]),
                params={
                    "range_min": -999, "range_max": 999,
                    "damping": 0.3,
                    "actuated": 1.0,
                    "max_torque": max(self.rng.uniform(5.0, 20.0), 1.0),
                },
            )
            body.add_joint(right_joint)
            
            # 可选：添加 caster wheel（后部小轮用于平衡）
            if self.rng.random() > 0.3:
                caster_pos = np.array([0.0, -body_len*0.35, wheel_r * 0.5])
                caster_r = wheel_r * 0.3
                caster = self._create_part("rod", caster_pos, {
                    "length": caster_r * 0.5,
                    "radius": caster_r,
                    "mass": max(0.05, 0.05),
                })
                body.add_part(caster)
                
                caster_joint = Joint(
                    joint_type="hinge",
                    parent_id=root.part_id,
                    child_id=caster.part_id,
                    anchor=caster_pos.copy(),
                    axis=np.array([1.0, 0.0, 0.0]),
                    params={"range_min": -1.5, "range_max": 1.5, "damping": 0.5},
                )
                body.add_joint(caster_joint)
            
            return body
            
        except Exception as e:
            self.logger.debug(f"Failed to create diff-drive: {e}")
            return None
    
    # ══════════════════════════════════════════════════════════
    # 模板2：四轮机器人（更像小车）
    # ══════════════════════════════════════════════════════════
    
    def _create_quad_wheeled(self) -> Optional[MechanicalBody]:
        """
        创建四轮机器人（汽车式布局）
        
        结构：
           [前左轮] ---- [主体] ---- [前右轮]
                                       |
           [后左轮]  -----------       [后右轮]
        """
        try:
            body = MechanicalBody(name="quad_drive")
            
            body_len = self.rng.uniform(*self.config.body_size_range) * 1.3
            body_w = body_len * self.rng.uniform(0.6, 0.9)
            wheel_r = self.rng.uniform(*self.config.wheel_radius_range)
            wheelbase = body_len * 0.75  # 轴距
            track = body_w * 1.3          # 轮距
            
            body_z = wheel_r + 0.03
            
            # 主体
            root = self._create_part("box_body", np.array([0.0, 0.0, body_z]), {
                "length": body_len, "width": body_w, "height": body_w * 0.4,
                "mass": body_len * body_w * 400,
            })
            body.add_part(root)
            
            # 四个轮子
            wheel_positions = [
                (-track/2,  wheelbase/2, wheel_r),  # 前左
                ( track/2,  wheelbase/2, wheel_r),  # 前右
                (-track/2, -wheelbase/2, wheel_r),  # 后左
                ( track/2, -wheelbase/2, wheel_r),  # 后右
            ]
            
            for i, (wx, wy, wz) in enumerate(wheel_positions):
                wpos = np.array([wx, wy, wz])
                wheel_w = wheel_r * self.rng.uniform(0.4, 0.8)
                wheel = self._create_part("rod", wpos, {
                    "length": wheel_w,
                    "radius": wheel_r,
                    "mass": max(wheel_r**2 * wheel_w * 6000, 0.1),
                }, is_actuated=True)
                body.add_part(wheel)
                
                # 所有轮子都有驱动
                wjoint = Joint(
                    joint_type="hinge",
                    parent_id=root.part_id,
                    child_id=wheel.part_id,
                    anchor=wpos.copy(),
                    axis=np.array([0.0, 1.0, 0.0]),
                    params={
                        "range_min": -999, "range_max": 999,
                        "damping": 0.2 + self.rng.uniform(0, 0.3),
                        "actuated": 1.0,
                        "max_torque": self.rng.uniform(2.0, 12.0),
                    },
                )
                body.add_joint(wjoint)
            
            return body
            
        except Exception as e:
            self.logger.debug(f"Failed to create quad-wheeled: {e}")
            return None
    
    # ══════════════════════════════════════════════════════════
    # 模板3：双足机器人
    # ══════════════════════════════════════════════════════════
    
    def _create_bipedal(self) -> Optional[MechanicalBody]:
        """
        创建简化双足机器人
        
        结构：
                 [头/传感器]
                    |
                [躯干(box)]
               /          \
          [左髋关节]  [右髋关节]
             |              |
          [左大腿]      [右大腿]
             |              |
          [左膝关节]  [右膝关节]
             |              |
          [小腿]        [小腿]
             |              |
          [左脚]        [右脚]
        """
        try:
            body = MechanicalBody(name="bipedal")
            
            torso_h = self.rng.uniform(0.10, 0.18)
            torso_w = torso_h * self.rng.uniform(0.35, 0.5)
            thigh_l = self.rng.uniform(*self.config.leg_length_range) * 0.55
            shin_l = self.rng.uniform(*self.config.leg_length_range) * 0.45
            foot_r = self.rng.uniform(0.025, 0.05)
            hip_w = torso_w * self.rng.uniform(1.5, 2.0)
            
            # 躯干（高度让脚着地）
            total_leg = thigh_l + shin_l
            torso_z = total_leg - thigh_l * 0.3 + 0.02
            
            # 躯干
            root = self._create_part("box_body", np.array([0.0, 0.0, torso_z]), {
                "length": torso_w, "height": torso_h, "width": torso_w * 0.7,
                "mass": max(torso_h * torso_w * 2000, 0.15),
            })
            body.add_part(root)
            
            # 双腿
            for side, x_sign in enumerate([-1.0, 1.0]):
                hip_x = x_sign * hip_w / 2
                
                # 大腿（驱动）
                thigh_end_y = -thigh_l * 0.4
                thigh_pos = np.array([hip_x, thigh_end_y, torso_z - torso_h/2])
                thigh = self._create_part("rod", thigh_pos, {
                    "length": thigh_l, "radius": 0.012, "mass": max(thigh_l * 1.5, 0.05),
                }, is_actuated=True)
                body.add_part(thigh)
                
                # 髋关节（驱动）
                hip_joint = Joint(
                    joint_type="hinge",
                    parent_id=root.part_id,
                    child_id=thigh.part_id,
                    anchor=np.array([hip_x, torso_z - torso_h/2, 0.0]),
                    axis=np.array([1.0, 0.0, 0.0]),  # X轴旋转（前后摆动）
                    params={
                        "range_min": -1.2, "range_max": 1.2,
                        "damping": 0.8, "actuated": 1.0,
                        "max_torque": self.rng.uniform(3.0, 10.0),
                    },
                )
                body.add_joint(hip_joint)
                
                # 小腿（驱动）
                shin_end_y = thigh_end_y - shin_l * 0.5
                shin_pos = np.array([hip_x, shin_end_y, thigh_pos[2] - thigh_l*0.4])
                shin = self._create_part("rod", shin_pos, {
                    "length": shin_l, "radius": 0.010, "mass": max(shin_l * 1.2, 0.04),
                }, is_actuated=True)
                body.add_part(shin)
                
                # 膝关节（驱动）
                knee_joint = Joint(
                    joint_type="hinge",
                    parent_id=thigh.part_id,
                    child_id=shin.part_id,
                    anchor=thigh_pos - np.array([0, thigh_l*0.45, 0]),
                    axis=np.array([1.0, 0.0, 0.0]),
                    params={
                        "range_min": -0.1, "range_max": 1.5,  # 只向后弯
                        "damping": 0.6, "actuated": 1.0,
                        "max_torque": self.rng.uniform(2.0, 8.0),
                    },
                )
                body.add_joint(knee_joint)
                
                # 脚
                foot_pos = np.array([hip_x, shin_pos[1] - shin_l*0.4, foot_r])
                foot = self._create_part("foot_contact", foot_pos, {
                    "radius": foot_r, "mass": 0.03,
                })
                body.add_part(foot)
                
                # 踝关节（被动）
                ankle_joint = Joint(
                    joint_type="hinge",
                    parent_id=shin.part_id,
                    child_id=foot.part_id,
                    anchor=shin_pos - np.array([0, shin_l*0.4, 0]),
                    axis=np.array([1.0, 0.0, 0.0]),
                    params={
                        "range_min": -0.5, "range_max": 0.5,
                        "damping": 1.0, "actuated": 0.0,
                    },
                )
                body.add_joint(ankle_joint)
            
            return body
            
        except Exception as e:
            self.logger.debug(f"Failed to create bipedal: {e}")
            return None
    
    # ══════════════════════════════════════════════════════════
    # 模板4：四足机器人
    # ══════════════════════════════════════════════════════════
    
    def _create_quadruped(self) -> Optional[MechanicalBody]:
        """
        创建四足机器人（简化版）
        
        结构：
              [前左腿]    [躯体]    [前右腿]
                  |         |          |
              [后左腿] ------------ [后右腿]
        """
        try:
            body = MechanicalBody(name="quadruped")
            
            body_len = self.rng.uniform(0.12, 0.22)
            body_w = body_len * self.rng.uniform(0.5, 0.7)
            body_h = body_w * 0.6
            leg_l = self.rng.uniform(*self.config.leg_length_range) * 0.7
            foot_r = self.rng.uniform(0.02, 0.04)
            
            body_z = leg_l * 0.7 + 0.02
            
            # 躯干
            root = self._create_part("box_body", np.array([0.0, 0.0, body_z]), {
                "length": body_len, "width": body_w, "height": body_h,
                "mass": max(body_len * body_w * body_h * 2000, 0.15),
            })
            body.add_part(root)
            
            # 四条腿（四角）
            leg_positions = [
                (-body_len*0.4,  body_w*0.35, body_z - body_h/2),  # 前左
                ( body_len*0.4,  body_w*0.35, body_z - body_h/2),  # 前右
                (-body_len*0.4, -body_w*0.35, body_z - body_h/2),  # 后左
                ( body_len*0.4, -body_w*0.35, body_z - body_h/2),  # 后右
            ]
            
            for lx, ly, lz in leg_positions:
                # 上腿（驱动）
                upper_leg = self._create_part("rod", np.array([lx, ly - leg_l*0.3, lz]), {
                    "length": leg_l * 0.55, "radius": 0.010, "mass": max(0.06, 0.05),
                }, is_actuated=True)
                body.add_part(upper_leg)
                
                # 肩/髋关节（驱动）
                shoulder = Joint(
                    joint_type="hinge",
                    parent_id=root.part_id,
                    child_id=upper_leg.part_id,
                    anchor=np.array([lx, ly, lz]),
                    axis=np.array([1.0, 0.0, 0.0]),
                    params={
                        "range_min": -0.8, "range_max": 0.8,
                        "damping": 0.5, "actuated": 1.0,
                        "max_torque": self.rng.uniform(2.0, 8.0),
                    },
                )
                body.add_joint(shoulder)
                
                # 下腿（驱动）
                lower_leg = self._create_part("rod", 
                    np.array([lx, ly - leg_l*0.75, lz - leg_l*0.25]), {
                    "length": leg_l * 0.45, "radius": 0.008, "mass": max(0.03, 0.03),
                }, is_actuated=True)
                body.add_part(lower_leg)
                
                # 肘/膝关节（驱动）
                elbow = Joint(
                    joint_type="hinge",
                    parent_id=upper_leg.part_id,
                    child_id=lower_leg.part_id,
                    anchor=np.array([lx, ly - leg_l*0.55, lz - leg_l*0.15]),
                    axis=np.array([1.0, 0.0, 0.0]),
                    params={
                        "range_min": 0.0, "range_max": 1.4,
                        "damping": 0.4, "actuated": 1.0,
                        "max_torque": self.rng.uniform(1.5, 6.0),
                    },
                )
                body.add_joint(elbow)
                
                # 脚
                foot = self._create_part("foot_contact",
                    np.array([lx, ly - leg_l, foot_r]), {
                    "radius": foot_r, "mass": 0.02,
                })
                body.add_part(foot)
                
                # 腕/踝关节（被动）
                wrist = Joint(
                    joint_type="hinge",
                    parent_id=lower_leg.part_id,
                    child_id=foot.part_id,
                    anchor=np.array([lx, ly - leg_l*0.85, lz - leg_l*0.35]),
                    axis=np.array([1.0, 0.0, 0.0]),
                    params={"range_min": -0.5, "range_max": 0.5, "damping": 0.8},
                )
                body.add_joint(wrist)
            
            return body
            
        except Exception as e:
            self.logger.debug(f"Failed to create quadruped: {e}")
            return None
    
    # ══════════════════════════════════════════════════════════
    # 模板5：爬行机器人（蛇形/蠕虫形简化版）
    # ══════════════════════════════════════════════════════════
    
    def _create_crawler(self) -> Optional[MechanicalBody]:
        """
        创建爬行机器人（多节铰链+摩擦足）
        
        结构：
        [头]-[节1]-[节2]-[节3]-...-[尾]
         |     |     |     |        |
        (足) (足) (足) (足)     (足)
        """
        try:
            body = MechanicalBody(name="crawler")
            
            n_segments = self.rng.randint(4, 7)
            seg_len = self.rng.uniform(0.05, 0.10)
            seg_r = seg_len * self.rng.uniform(0.25, 0.35)
            
            # 第一节（头部）在地面附近
            head_z = seg_r + 0.015
            
            prev_part = None
            prev_pos = np.array([0.0, 0.0, head_z])
            
            for i in range(n_segments):
                # 体节
                seg_pos = prev_pos + np.array([seg_len * 0.8, 0, 0]) if i > 0 else prev_pos.copy()
                
                if i == 0:
                    seg = self._create_part("box_body", seg_pos, {
                        "length": seg_len, "height": seg_r*2, "width": seg_r*2,
                        "mass": max(seg_len * seg_r**2 * 3000, 0.08),
                    })
                else:
                    # 后续体节都是驱动的（产生波浪运动）
                    seg = self._create_part("rod", seg_pos, {
                        "length": seg_len, "radius": seg_r, "mass": max(seg_len * seg_r**2 * 2000, 0.05),
                    }, is_actuated=True)
                body.add_part(seg)
                
                if prev_part is not None:
                    # 节间关节（驱动，产生波浪运动）
                    inter_joint = Joint(
                        joint_type="hinge",
                        parent_id=prev_part.part_id,
                        child_id=seg.part_id,
                        anchor=(prev_pos + seg_pos) / 2,
                        axis=np.array([0.0, 0.0, 1.0]),  # Z轴（垂直弯曲）
                        params={
                            "range_min": -0.6, "range_max": 0.6,
                            "damping": 0.6, "actuated": 1.0,
                            "max_torque": self.rng.uniform(1.0, 5.0),
                        },
                    )
                    body.add_joint(inter_joint)
                    
                    # 侧向足（增加摩擦）
                    if self.rng.random() > 0.3:
                        foot = self._create_part("foot_contact",
                            seg_pos - np.array([0, 0, seg_r]), {
                            "radius": seg_r * 0.4, "mass": 0.015,
                        })
                        body.add_part(foot)
                        
                        foot_joint = Joint(
                            joint_type="hinge",
                            parent_id=seg.part_id,
                            child_id=foot.part_id,
                            anchor=seg_pos - np.array([0, 0, seg_r*0.5]),
                            axis=np.array([1.0, 0.0, 0.0]),
                            params={"range_min": -0.3, "range_max": 0.3, "damping": 1.0},
                        )
                        body.add_joint(foot_joint)
                
                prev_part = seg
                prev_pos = seg_pos
            
            return body
            
        except Exception as e:
            self.logger.debug(f"Failed to create crawler: {e}")
            return None


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import time
    
    sys.path.insert(0, str(Path(__file__).parent))
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    
    print("=" * 60)
    print("  LOCOMOTION TEMPLATE GENERATOR TEST")
    print("=" * 60)
    
    gen = LocomotionTemplateGenerator(seed=42)
    
    # 测试各种模板
    print("\n--- Testing Templates ---")
    templates = {
        "Differential Wheeled": gen._create_differential_wheeled,
        "Quad Wheeled": gen._create_quad_wheeled,
        "Bipedal": gen._create_bipedal,
        "Quadruped": gen._create_quadruped,
        "Crawler": gen._create_crawler,
    }
    
    for name, creator in templates.items():
        try:
            robot = creator()
            if robot:
                n_motors = len(robot.actuated_joints())
                n_parts = robot.num_parts()
                print(f"  {name}: OK (parts={n_parts}, motors={n_motors})")
            else:
                print(f"  {name}: FAILED (returned None)")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
    
    # 生成种群并测试MuJoCo加载
    print("\n--- Population Test ---")
    population = gen.generate_population(size=8, min_parts=4, max_parts=12)
    
    from forgecraft.simulation.builder import build_mjcf_model
    from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
    import mujoco
    
    catalog = load_catalog()
    
    valid_count = 0
    motor_count = 0
    
    for i, body in enumerate(population):
        try:
            xml_string, jm, mm, si = build_mjcf_model(body, catalog)
            model = mujoco.MjModel.from_xml_string(xml_string)
            
            n_motors = len(body.actuated_joints())
            n_actuators = model.nu
            motor_count += n_actuators
            
            status = f"acts={n_actuators}"
            if n_actuators > 0:
                status += " [MOTION CAPABLE]"
                valid_count += 1
            
            print(f"  [{i+1}] parts={body.num_parts():>2} motors={n_motors} {status}")
            
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
    
    print(f"\nSummary: {valid_count}/{len(population)} motion-capable, "
          f"total actuators={motor_count}")
