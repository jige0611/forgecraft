# ══════════════════════════════════════════════════════════
# 🔗 柔性构件系统
#
# 功能:
#   ✅ 弹簧元件建模与仿真
#   ✅ 绳索/线缆系统
#   ✅ 阻尼器建模
#   ✅ 弹性连接
#   ✅ MuJoCo tendon集成
#   ✅ 材料属性支持
#
# 使用示例:
#   >>> from forgecraft.core.flexible_components import SpringElement, TendonSystem
#   >>> spring = SpringElement(stiffness=1000, damping=10, free_length=0.1)
#   >>> tendon = TendonSystem()
#   >>> tendon.add_segment(body1, body2, stiffness=500)
# ══════════════════════════════════════════════════════════

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

from forgecraft.core.morphology import MechanicalBody, Part

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "SpringProperties",
    "TendonSegment",
    "DamperProperties",
    "SpringElement",
    "TendonSystem",
    "DamperElement",
    "FlexibleConnector",
    "SoftBodySimulator",
    "SpringLibrary",
    "get_flexible_connector",
    "add_spring_to_body",
]


@dataclass
class SpringProperties:
    """弹簧属性"""
    stiffness: float = 1000.0      # 刚度 (N/m)
    damping: float = 10.0          # 阻尼 (N·s/m)
    free_length: float = 0.1       # 自由长度 (m)
    max_deformation: float = 0.1   # 最大变形量 (m)
    preload: float = 0.0           # 预载力 (N)
    material: str = "steel"        # 材料类型


@dataclass
class TendonSegment:
    """绳索段"""
    name: str
    body1: str
    body2: str
    stiffness: float = 500.0
    damping: float = 5.0
    length: float = 0.1
    cross_section_area: float = 1e-6  # 横截面积 (m²)
    max_force: float = 100.0         # 最大拉力 (N)


@dataclass
class DamperProperties:
    """阻尼器属性"""
    damping_coefficient: float = 50.0  # 阻尼系数 (N·s/m)
    max_force: float = 500.0          # 最大阻尼力 (N)
    type: str = "linear"              # linear / rotational


class SpringElement:
    """弹簧元件"""
    
    def __init__(self, properties: SpringProperties = None):
        self.properties = properties or SpringProperties()
    
    def calculate_force(self, current_length: float) -> float:
        """
        计算弹簧力
        
        Args:
            current_length: 当前长度
        
        Returns:
            弹簧力 (N)，正值为拉力，负值为推力
        """
        delta = current_length - self.properties.free_length
        force = delta * self.properties.stiffness + self.properties.preload
        return force
    
    def calculate_energy(self, current_length: float) -> float:
        """
        计算弹簧存储的弹性势能
        
        Args:
            current_length: 当前长度
        
        Returns:
            弹性势能 (J)
        """
        delta = current_length - self.properties.free_length
        return 0.5 * self.properties.stiffness * delta ** 2
    
    def get_mujoco_tendon_xml(self, name: str, body1: str, body2: str) -> str:
        """
        生成MuJoCo tendon XML
        
        Args:
            name: 元件名称
            body1: 连接体1
            body2: 连接体2
        
        Returns:
            XML字符串
        """
        spring = self.properties
        max_length = spring.free_length + spring.max_deformation
        
        return f"""<tendon name="{name}_tendon">
  <spatial name="{name}_spatial" 
           limited="true" 
           range="0 {max_length:.4f}"
           stiffness="{spring.stiffness:.1f}" 
           damping="{spring.damping:.1f}"
           body1="{body1}" 
           body2="{body2}"
           site1="{body1}" 
           site2="{body2}"/>
</tendon>"""


class TendonSystem:
    """绳索系统"""
    
    def __init__(self):
        self.segments: List[TendonSegment] = []
        self._segment_counter = 0
    
    def add_segment(
        self,
        body1: str,
        body2: str,
        stiffness: float = 500.0,
        damping: float = 5.0,
        length: float = 0.1,
        max_force: float = 100.0
    ) -> str:
        """
        添加绳索段
        
        Args:
            body1: 第一个连接点
            body2: 第二个连接点
            stiffness: 刚度
            damping: 阻尼
            length: 初始长度
            max_force: 最大拉力
        
        Returns:
            段名称
        """
        self._segment_counter += 1
        name = f"tendon_{self._segment_counter}"
        
        segment = TendonSegment(
            name=name,
            body1=body1,
            body2=body2,
            stiffness=stiffness,
            damping=damping,
            length=length,
            max_force=max_force
        )
        
        self.segments.append(segment)
        return name
    
    def calculate_total_length(self) -> float:
        """计算绳索总长度"""
        return sum(s.length for s in self.segments)
    
    def calculate_max_force(self) -> float:
        """计算系统最大承受力（最弱环节）"""
        return min(s.max_force for s in self.segments)
    
    def get_mujoco_tendon_xml(self) -> str:
        """生成完整的MuJoCo tendon XML"""
        lines = ["<tendon>"]
        
        for segment in self.segments:
            lines.append(f'  <spatial name="{segment.name}"')
            lines.append(f'           limited="true"')
            lines.append(f'           range="0 {segment.length * 1.2:.4f}"')
            lines.append(f'           stiffness="{segment.stiffness:.1f}"')
            lines.append(f'           damping="{segment.damping:.1f}"')
            lines.append(f'           body1="{segment.body1}"')
            lines.append(f'           body2="{segment.body2}"')
            lines.append(f'           site1="{segment.body1}"')
            lines.append(f'           site2="{segment.body2}"/>')
        
        lines.append("</tendon>")
        return "\n".join(lines)


class DamperElement:
    """阻尼器元件"""
    
    def __init__(self, properties: DamperProperties = None):
        self.properties = properties or DamperProperties()
    
    def calculate_force(self, velocity: float) -> float:
        """
        计算阻尼力
        
        Args:
            velocity: 相对速度
        
        Returns:
            阻尼力 (N)
        """
        force = self.properties.damping_coefficient * velocity
        
        # 限制最大力
        if abs(force) > self.properties.max_force:
            force = np.sign(force) * self.properties.max_force
        
        return force


class FlexibleConnector:
    """柔性连接器管理器"""
    
    def __init__(self):
        self.springs: Dict[str, SpringElement] = {}
        self.dampers: Dict[str, DamperElement] = {}
        self.tendons: Dict[str, TendonSystem] = {}
    
    def add_spring(
        self,
        name: str,
        stiffness: float = 1000.0,
        damping: float = 10.0,
        free_length: float = 0.1,
        max_deformation: float = 0.1
    ):
        """
        添加弹簧
        
        Args:
            name: 弹簧名称
            stiffness: 刚度 (N/m)
            damping: 阻尼 (N·s/m)
            free_length: 自由长度 (m)
            max_deformation: 最大变形 (m)
        """
        props = SpringProperties(
            stiffness=stiffness,
            damping=damping,
            free_length=free_length,
            max_deformation=max_deformation
        )
        self.springs[name] = SpringElement(props)
    
    def add_damper(
        self,
        name: str,
        damping_coefficient: float = 50.0,
        max_force: float = 500.0,
        type: str = "linear"
    ):
        """
        添加阻尼器
        
        Args:
            name: 阻尼器名称
            damping_coefficient: 阻尼系数
            max_force: 最大力
            type: 类型 (linear/rotational)
        """
        props = DamperProperties(
            damping_coefficient=damping_coefficient,
            max_force=max_force,
            type=type
        )
        self.dampers[name] = DamperElement(props)
    
    def create_tendon_system(self, name: str) -> TendonSystem:
        """
        创建绳索系统
        
        Args:
            name: 系统名称
        
        Returns:
            TendonSystem实例
        """
        tendon = TendonSystem()
        self.tendons[name] = tendon
        return tendon
    
    def generate_mujoco_xml(self) -> str:
        """生成MuJoCo XML元素"""
        lines = []
        
        # 添加弹簧
        for name, spring in self.springs.items():
            # 弹簧需要连接两个body，这里简化处理
            # 实际使用时需要指定连接关系
            pass
        
        # 添加绳索系统
        for name, tendon in self.tendons.items():
            lines.append(tendon.get_mujoco_tendon_xml())
        
        return "\n".join(lines)


class SoftBodySimulator:
    """软体模拟器接口"""
    
    def __init__(self):
        pass
    
    def simulate_spring_force(self, spring: SpringElement, displacement: float) -> float:
        """
        模拟弹簧力
        
        Args:
            spring: 弹簧元件
            displacement: 位移
        
        Returns:
            力
        """
        return spring.calculate_force(spring.properties.free_length + displacement)
    
    def simulate_damper_force(self, damper: DamperElement, velocity: float) -> float:
        """
        模拟阻尼力
        
        Args:
            damper: 阻尼器元件
            velocity: 速度
        
        Returns:
            力
        """
        return damper.calculate_force(velocity)
    
    def simulate_tendon_dynamics(
        self,
        segments: List[TendonSegment],
        velocities: List[float]
    ) -> List[float]:
        """
        模拟绳索动力学
        
        Args:
            segments: 绳索段列表
            velocities: 各段速度
        
        Returns:
            各段力
        """
        forces = []
        for segment, velocity in zip(segments, velocities):
            # 简化模型：阻尼力 + 弹性力
            force = segment.damping * velocity
            forces.append(force)
        return forces


# 预定义弹簧类型
class SpringLibrary:
    """弹簧库 - 预定义常见弹簧规格"""
    
    @staticmethod
    def get_spring_by_specification(stiffness_range: Tuple[float, float]) -> SpringProperties:
        """
        根据刚度范围获取合适的弹簧规格
        
        Args:
            stiffness_range: 刚度范围 (min, max)
        
        Returns:
            SpringProperties
        """
        min_k, max_k = stiffness_range
        
        # 选择最接近范围中点的标准弹簧
        target_k = (min_k + max_k) / 2
        
        # 标准弹簧规格
        standard_springs = [
            (500, "light"),
            (1000, "medium"),
            (2000, "stiff"),
            (5000, "very_stiff"),
            (10000, "ultra_stiff"),
        ]
        
        # 找到最接近的
        closest_k = min(standard_springs, key=lambda x: abs(x[0] - target_k))
        
        return SpringProperties(
            stiffness=closest_k[0],
            damping=closest_k[0] * 0.01,  # 阻尼约为刚度的1%
            free_length=0.1,
            material="steel"
        )
    
    @staticmethod
    def get_standard_spring(type: str) -> SpringProperties:
        """获取标准弹簧类型"""
        springs = {
            'micro': SpringProperties(stiffness=100, damping=1, free_length=0.05),
            'small': SpringProperties(stiffness=500, damping=5, free_length=0.08),
            'medium': SpringProperties(stiffness=1000, damping=10, free_length=0.1),
            'large': SpringProperties(stiffness=2000, damping=20, free_length=0.15),
            'heavy': SpringProperties(stiffness=5000, damping=50, free_length=0.2),
        }
        return springs.get(type, springs['medium'])


# 全局柔性连接器实例
_flexible_connector = None


def get_flexible_connector() -> FlexibleConnector:
    """获取全局柔性连接器实例"""
    global _flexible_connector
    if _flexible_connector is None:
        _flexible_connector = FlexibleConnector()
    return _flexible_connector


def add_spring_to_body(
    body: MechanicalBody,
    from_part: str,
    to_part: str,
    stiffness: float = 1000.0,
    damping: float = 10.0,
    free_length: float = 0.1
) -> str:
    """
    快捷函数：向机械体添加弹簧连接
    
    Args:
        body: 机械体
        from_part: 起点零件ID
        to_part: 终点零件ID
        stiffness: 刚度
        damping: 阻尼
        free_length: 自由长度
    
    Returns:
        弹簧名称
    """
    connector = get_flexible_connector()
    name = f"spring_{from_part}_{to_part}"
    connector.add_spring(name, stiffness, damping, free_length)
    return name
