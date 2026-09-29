# ══════════════════════════════════════════════════════════
# ⚙️ 关节锚点优化器
#
# 功能:
#   ✅ 基于质心自动计算最佳关节锚点位置
#   ✅ 优化关节轴方向
#   ✅ 计算惯性张量
#   ✅ 动力学性能分析
#   ✅ 关节配置优化建议
#
# 使用示例:
#   >>> from forgecraft.core.joint_optimizer import JointOptimizer
#   >>> optimizer = JointOptimizer()
#   >>> optimized_joint = optimizer.optimize_joint(body, parent_id, child_id)
# ══════════════════════════════════════════════════════════

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import math
import logging

from forgecraft.core.morphology import MechanicalBody, Part, Joint

__all__ = [
    "JointOptimizationResult",
    "JointOptimizer",
    "get_joint_optimizer",
    "optimize_body_joints",
]

_logger = logging.getLogger(__name__)


from forgecraft.core.materials_database import get_part_density


@dataclass
class JointOptimizationResult:
    """关节优化结果"""
    anchor: np.ndarray
    axis: np.ndarray
    suggested_range: Tuple[float, float]
    suggested_damping: float
    inertia_tensor: np.ndarray
    center_of_mass: np.ndarray
    torque_arm_length: float
    optimization_score: float
    suggestions: List[str] = field(default_factory=list)


class JointOptimizer:
    """关节锚点优化器"""
    
    def __init__(self):
        pass
    
    def optimize_joint(
        self,
        body: MechanicalBody,
        parent_id: str,
        child_id: str,
        target_axis: Optional[np.ndarray] = None
    ) -> JointOptimizationResult:
        """
        优化关节配置
        
        Args:
            body: 机械体
            parent_id: 父零件ID
            child_id: 子零件ID
            target_axis: 目标关节轴方向 (可选)
        
        Returns:
            JointOptimizationResult 包含优化后的关节配置
        """
        parent_part = body.get_part(parent_id)
        child_part = body.get_part(child_id)
        
        # 1. 计算两个零件的质心
        parent_com = self._calculate_part_com(parent_part)
        child_com = self._calculate_part_com(child_part)
        
        # 2. 计算最佳锚点位置 (基于两质心连线)
        anchor = self._calculate_optimal_anchor(parent_part, child_part, parent_com, child_com)
        
        # 3. 确定关节轴方向
        axis = self._determine_joint_axis(body, parent_part, child_part, target_axis)
        
        # 4. 计算惯性张量
        inertia_tensor = self._calculate_inertia_tensor(child_part)
        
        # 5. 建议关节范围和阻尼
        suggested_range = self._suggest_joint_range(child_part)
        suggested_damping = self._suggest_damping(child_part)
        
        # 6. 计算扭矩臂长度
        torque_arm = np.linalg.norm(child_com - anchor)
        
        # 7. 计算优化分数
        score = self._calculate_optimization_score(anchor, child_com, axis)
        
        # 8. 生成建议
        suggestions = self._generate_suggestions(body, parent_part, child_part, anchor, axis)
        
        return JointOptimizationResult(
            anchor=anchor,
            axis=axis,
            suggested_range=suggested_range,
            suggested_damping=suggested_damping,
            inertia_tensor=inertia_tensor,
            center_of_mass=child_com,
            torque_arm_length=torque_arm,
            optimization_score=score,
            suggestions=suggestions
        )
    
    def _calculate_part_com(self, part: Part) -> np.ndarray:
        """
        计算零件的质心位置
        
        Args:
            part: 零件
        
        Returns:
            质心位置 (世界坐标系)
        """
        # 对于简单几何形状，质心通常在几何中心
        # 对于复杂零件，我们使用位置加上偏移
        com = np.array(part.position, dtype=np.float32)
        
        # 根据零件类型调整质心位置
        part_type = part.part_type.lower()
        
        # 获取零件尺寸
        size = self._get_part_dimensions(part)
        
        # 对于电机等圆柱形零件，质心可能偏向一端
        if 'motor' in part_type or 'cylinder' in part_type:
            # 电机质心通常在电机壳体中心
            length = size[1]
            # 假设输出轴在z方向，质心向后偏移
            com[2] -= length * 0.1  # 轻微向后偏移
        
        # 对于传感器等扁平零件
        elif 'sensor' in part_type or 'chip' in part_type:
            thickness = size[2]
            com[2] += thickness / 2.0
        
        return com
    
    def _get_part_dimensions(self, part: Part) -> np.ndarray:
        """获取零件的尺寸向量"""
        params = part.params
        length = params.get('length', 0.1)
        width = params.get('width', params.get('thickness', 0.02))
        height = params.get('height', params.get('thickness', 0.02))
        radius = params.get('radius', 0.03)
        
        part_type = part.part_type.lower()
        if 'cylinder' in part_type or 'motor' in part_type:
            return np.array([radius * 2, length, radius * 2])
        else:
            return np.array([length, width, height])
    
    def _calculate_optimal_anchor(
        self,
        parent_part: Part,
        child_part: Part,
        parent_com: np.ndarray,
        child_com: np.ndarray
    ) -> np.ndarray:
        """
        计算最佳关节锚点位置
        
        原则:
        1. 锚点应位于两零件的连接面上
        2. 尽量靠近两质心连线
        3. 考虑零件的实际几何形状
        
        Args:
            parent_part: 父零件
            child_part: 子零件
            parent_com: 父零件质心
            child_com: 子零件质心
        
        Returns:
            优化后的锚点位置
        """
        # 两质心连线
        direction = child_com - parent_com
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return parent_com.copy()
        
        direction_normalized = direction / distance
        
        # 获取零件尺寸
        parent_size = self._get_part_dimensions(parent_part)
        child_size = self._get_part_dimensions(child_part)
        
        # 计算从父零件到连接面的距离
        # 假设连接面在父零件的"前端"
        parent_extent = np.max(parent_size) * 0.4
        
        # 计算从子零件到连接面的距离
        child_extent = np.max(child_size) * 0.4
        
        # 锚点位置 = 父质心 + 方向 * 父延伸
        anchor = parent_com + direction_normalized * parent_extent
        
        # 验证锚点是否在合理范围内
        # 如果超出子零件范围，调整到子零件边缘
        child_to_anchor = anchor - child_com
        child_dist = np.linalg.norm(child_to_anchor)
        
        if child_dist > child_extent:
            # 将锚点调整到子零件边缘
            anchor = child_com + (child_to_anchor / child_dist) * child_extent
        
        return anchor
    
    def _determine_joint_axis(
        self,
        body: MechanicalBody,
        parent_part: Part,
        child_part: Part,
        target_axis: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        确定最佳关节轴方向
        
        Args:
            body: 机械体
            parent_part: 父零件
            child_part: 子零件
            target_axis: 目标轴方向 (可选)
        
        Returns:
            关节轴方向向量
        """
        if target_axis is not None:
            return target_axis / np.linalg.norm(target_axis)
        
        # 根据零件类型和位置关系推断最佳轴方向
        part_type = child_part.part_type.lower()
        
        # 默认使用z轴作为旋转轴
        default_axis = np.array([0.0, 0.0, 1.0])
        
        # 根据位置推断轴方向
        if body.root_id is not None:
            root_pos = body.get_part(body.root_id).position
            child_pos = child_part.position
            
            # 计算到根部的方向
            to_root = root_pos - child_pos
            
            # 如果子零件在根部上方，优先使用水平轴
            if to_root[2] > 0.1:  # 明显在上方
                # 使用垂直于to_root的方向作为旋转轴
                if abs(to_root[0]) > abs(to_root[1]):
                    default_axis = np.array([0.0, 1.0, 0.0])  # y轴
                else:
                    default_axis = np.array([1.0, 0.0, 0.0])  # x轴
        
        # 根据零件类型调整
        if 'shoulder' in part_type or 'arm' in part_type:
            # 肩部和手臂通常绕y轴旋转
            default_axis = np.array([0.0, 1.0, 0.0])
        elif 'elbow' in part_type:
            # 肘部通常绕x轴旋转
            default_axis = np.array([1.0, 0.0, 0.0])
        elif 'hip' in part_type or 'leg' in part_type:
            # 腿部关节通常绕y轴旋转
            default_axis = np.array([0.0, 1.0, 0.0])
        elif 'wheel' in part_type or 'roller' in part_type:
            # 轮子绕x或y轴旋转
            default_axis = np.array([1.0, 0.0, 0.0])
        
        return default_axis
    
    def _calculate_inertia_tensor(self, part: Part) -> np.ndarray:
        """
        计算零件的惯性张量
        
        Args:
            part: 零件
        
        Returns:
            3x3惯性张量矩阵
        """
        # 获取零件尺寸
        size = self._get_part_dimensions(part)
        length, width, height = size
        
        # 获取材料密度
        density = get_part_density(part.part_type)
        
        # 计算质量
        volume = length * width * height
        mass = volume * density
        
        # 计算惯性张量 (假设为长方体)
        # I_xx = (1/12) * m * (w^2 + h^2)
        # I_yy = (1/12) * m * (l^2 + h^2)
        # I_zz = (1/12) * m * (l^2 + w^2)
        
        I_xx = (1.0 / 12.0) * mass * (width**2 + height**2)
        I_yy = (1.0 / 12.0) * mass * (length**2 + height**2)
        I_zz = (1.0 / 12.0) * mass * (length**2 + width**2)
        
        inertia_tensor = np.diag([I_xx, I_yy, I_zz])
        
        return inertia_tensor
    
    def _suggest_joint_range(self, part: Part) -> Tuple[float, float]:
        """
        建议关节运动范围
        
        Args:
            part: 子零件
        
        Returns:
            (最小值, 最大值) 弧度
        """
        part_type = part.part_type.lower()
        
        # 默认范围
        default_range = (-np.pi * 0.8, np.pi * 0.8)  # ±144度
        
        # 根据零件类型调整
        if 'shoulder' in part_type:
            # 肩关节：大范围
            return (-np.pi, np.pi)  # ±180度
        elif 'elbow' in part_type:
            # 肘关节：单向弯曲
            return (0.0, np.pi * 0.8)  # 0~144度
        elif 'hip' in part_type:
            # 髋关节：大范围
            return (-np.pi * 0.9, np.pi * 0.9)  # ±162度
        elif 'knee' in part_type:
            # 膝关节：单向弯曲
            return (0.0, np.pi * 0.7)  # 0~126度
        elif 'wheel' in part_type:
            # 轮子：无限制
            return (-np.inf, np.inf)
        elif 'neck' in part_type:
            # 颈部：有限范围
            return (-np.pi * 0.5, np.pi * 0.5)  # ±90度
        
        return default_range
    
    def _suggest_damping(self, part: Part) -> float:
        """
        建议关节阻尼值
        
        Args:
            part: 子零件
        
        Returns:
            阻尼值
        """
        # 根据零件尺寸和类型建议阻尼
        size = self._get_part_dimensions(part)
        volume = np.prod(size)
        part_type = part.part_type.lower()
        
        # 基础阻尼
        base_damping = 0.5
        
        # 根据体积调整
        if volume < 0.0001:  # 小型零件
            base_damping = 0.1
        elif volume > 0.001:  # 大型零件
            base_damping = 1.0
        
        # 根据零件类型调整
        if 'motor' in part_type or 'drive' in part_type:
            # 驱动关节需要较低阻尼
            base_damping *= 0.5
        elif 'joint' in part_type:
            # 被动关节需要较高阻尼
            base_damping *= 1.5
        
        return base_damping
    
    def _calculate_optimization_score(
        self,
        anchor: np.ndarray,
        com: np.ndarray,
        axis: np.ndarray
    ) -> float:
        """
        计算优化分数
        
        Args:
            anchor: 锚点位置
            com: 质心位置
            axis: 关节轴
        
        Returns:
            优化分数 (0-1)
        """
        # 计算扭矩臂长度
        torque_arm = np.linalg.norm(com - anchor)
        
        # 理想情况下，扭矩臂应适中
        # 太小：力臂太短，效率低
        # 太大：惯性大，响应慢
        
        optimal_arm = 0.05  # 5cm
        arm_score = max(0.0, 1.0 - abs(torque_arm - optimal_arm) / 0.1)
        
        # 轴方向稳定性分数
        # 如果轴接近垂直或水平，稳定性更好
        axis_score = max(abs(axis[0]), abs(axis[1]), abs(axis[2]))
        
        # 综合分数
        score = (arm_score * 0.6 + axis_score * 0.4)
        
        return score
    
    def _generate_suggestions(
        self,
        body: MechanicalBody,
        parent_part: Part,
        child_part: Part,
        anchor: np.ndarray,
        axis: np.ndarray
    ) -> List[str]:
        """
        生成优化建议
        
        Returns:
            建议列表
        """
        suggestions = []
        
        # 检查锚点是否过远
        distance_to_parent = np.linalg.norm(anchor - parent_part.position)
        distance_to_child = np.linalg.norm(anchor - child_part.position)
        
        parent_size = np.max(self._get_part_dimensions(parent_part))
        child_size = np.max(self._get_part_dimensions(child_part))
        
        if distance_to_parent > parent_size * 0.8:
            suggestions.append("警告：锚点距离父零件过远，可能导致不稳定")
        
        if distance_to_child > child_size * 0.8:
            suggestions.append("警告：锚点距离子零件过远，可能导致不稳定")
        
        # 检查关节轴是否合理
        if np.linalg.norm(axis) < 0.9:
            suggestions.append("警告：关节轴方向向量长度异常")
        
        # 检查是否为驱动关节
        if child_part.params.get('actuated', 0.0) > 0.5:
            suggestions.append("提示：此关节为驱动关节，建议设置合适的扭矩限制")
        
        return suggestions
    
    def optimize_all_joints(self, body: MechanicalBody) -> Dict[Tuple[str, str], JointOptimizationResult]:
        """
        优化机械体中所有关节
        
        Args:
            body: 机械体
        
        Returns:
            关节优化结果字典 {(parent_id, child_id): result}
        """
        results = {}
        
        for parent_id, child_id in body.actuated_joints():
            result = self.optimize_joint(body, parent_id, child_id)
            results[(parent_id, child_id)] = result
            
            # 应用优化结果到关节
            joint = body.get_joint(parent_id, child_id)
            joint.anchor = result.anchor
            joint.axis = result.axis
            if result.suggested_range is not None:
                lo, hi = result.suggested_range[0], result.suggested_range[1]
                joint.params['range_min'], joint.params['range_max'] = min(lo, hi), max(lo, hi)
            if result.suggested_damping is not None:
                joint.params['damping'] = result.suggested_damping
        
        return results
    
    def analyze_kinematic_chain(self, body: MechanicalBody) -> Dict[str, Any]:
        """
        分析运动链的动力学性能
        
        Args:
            body: 机械体
        
        Returns:
            分析结果
        """
        total_mass = 0.0
        total_inertia = 0.0
        chain_length = 0
        max_torque_arm = 0.0
        
        for part in body.parts():
            size = self._get_part_dimensions(part)
            density = get_part_density(part.part_type)
            volume = np.prod(size)
            mass = volume * density
            total_mass += mass
            
            # 简化惯性计算
            total_inertia += mass * np.mean(size)**2
        
        for parent_id, child_id in body.actuated_joints():
            chain_length += 1
            result = self.optimize_joint(body, parent_id, child_id)
            max_torque_arm = max(max_torque_arm, result.torque_arm_length)
        
        # 计算动力学性能指标
        # 功率/质量比 (简化版)
        performance_ratio = 1.0 / (total_mass * max_torque_arm) if total_mass > 0 and max_torque_arm > 0 else 0.0
        
        return {
            'total_mass': total_mass,
            'total_inertia': total_inertia,
            'chain_length': chain_length,
            'max_torque_arm': max_torque_arm,
            'performance_ratio': performance_ratio,
            'suggestions': self._generate_chain_suggestions(total_mass, chain_length, max_torque_arm)
        }
    
    def _generate_chain_suggestions(self, mass: float, length: int, torque_arm: float) -> List[str]:
        """生成运动链优化建议"""
        suggestions = []
        
        if mass > 10.0:  # 超过10kg
            suggestions.append("建议：减轻结构质量，考虑使用轻质材料如碳纤维")
        
        if length > 5:  # 超过5个关节
            suggestions.append("建议：减少运动链长度，或增加支撑结构")
        
        if torque_arm > 0.3:  # 超过30cm
            suggestions.append("建议：缩短力臂长度以提高响应速度")
        
        return suggestions


# 全局关节优化器实例
_joint_optimizer = None


def get_joint_optimizer() -> JointOptimizer:
    """获取全局关节优化器实例"""
    global _joint_optimizer
    if _joint_optimizer is None:
        _joint_optimizer = JointOptimizer()
    return _joint_optimizer


def optimize_body_joints(body: MechanicalBody) -> Dict[Tuple[str, str], JointOptimizationResult]:
    """快捷函数：优化机械体所有关节"""
    return get_joint_optimizer().optimize_all_joints(body)
