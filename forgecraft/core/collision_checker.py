# ══════════════════════════════════════════════════════════
# 🛡️ 碰撞预检查系统
#
# 功能:
#   ✅ 自碰撞检测 (零件间干涉检测)
#   ✅ 运动链验证 (检查是否为有效运动结构)
#   ✅ 自由度分析 (DOF计算)
#   ✅ 关节限制验证
#   ✅ 稳定性分析 (重心检查)
#   ✅ 无效结构过滤
#
# 使用示例:
#   >>> from forgecraft.core.collision_checker import CollisionChecker
#   >>> checker = CollisionChecker()
#   >>> result = checker.check_body(body)
#   >>> if not result.is_valid:
#   ...     print(result.errors)
# ══════════════════════════════════════════════════════════

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

from forgecraft.core.morphology import MechanicalBody, Part, Joint

logger = logging.getLogger(__name__)

__all__ = [
    "CollisionResult",
    "CollisionChecker",
    "get_collision_checker",
    "is_body_valid",
    "filter_valid_bodies",
]


@dataclass
class CollisionResult:
    """碰撞检查结果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    collision_pairs: List[Tuple[str, str]] = field(default_factory=list)
    dof_count: int = 0
    is_connected: bool = True
    stability_score: float = 1.0
    centroid: np.ndarray = None
    
    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)


class CollisionChecker:
    """碰撞预检查器"""
    
    def __init__(self, tolerance: float = 0.001):
        """
        初始化碰撞检查器
        
        Args:
            tolerance: 碰撞检测容差 (米)
        """
        self.tolerance = tolerance
    
    def check_body(self, body: MechanicalBody) -> CollisionResult:
        """
        全面检查机械体的有效性
        
        Returns:
            CollisionResult 包含检查结果
        """
        result = CollisionResult(is_valid=True, centroid=np.zeros(3))
        
        # 1. 检查连通性
        if not body.is_connected():
            result.add_error("机械体不是连通图")
            result.is_connected = False
            return result
        
        # 2. 自由度分析
        dof_count = self._calculate_dof(body)
        result.dof_count = dof_count
        
        # 3. 自碰撞检测
        collisions = self._detect_self_collisions(body)
        result.collision_pairs = collisions
        if collisions:
            result.add_error(f"检测到 {len(collisions)} 对零件碰撞")
        
        # 4. 关节限制验证
        joint_errors = self._validate_joint_limits(body)
        for error in joint_errors:
            result.add_error(error)
        
        # 5. 稳定性分析
        stability, centroid = self._analyze_stability(body)
        result.stability_score = stability
        result.centroid = centroid
        if stability < 0.5:
            result.add_warning(f"稳定性较低: {stability:.2f}")
        
        # 6. 运动链验证
        chain_errors = self._validate_kinematic_chain(body)
        for error in chain_errors:
            result.add_error(error)
        
        return result
    
    def _calculate_dof(self, body: MechanicalBody) -> int:
        """
        计算机械体的自由度
        
        Returns:
            自由度数
        """
        dof = 0
        for joint in body.joints():
            if joint.joint_type == 'hinge':
                dof += 1
            elif joint.joint_type == 'slide':
                dof += 1
            elif joint.joint_type == 'ball':
                dof += 3
            elif joint.joint_type == 'free':
                dof += 6
            # fixed 关节不贡献自由度
        
        return dof
    
    def _detect_self_collisions(self, body: MechanicalBody) -> List[Tuple[str, str]]:
        """
        检测零件间的自碰撞
        
        Returns:
            碰撞的零件ID对列表
        """
        collisions = []
        parts = body.parts()
        part_ids = [p.part_id for p in parts]
        
        for i, part1 in enumerate(parts):
            for j, part2 in enumerate(parts):
                if i >= j:
                    continue
                
                if self._check_part_collision(part1, part2):
                    collisions.append((part_ids[i], part_ids[j]))
        
        return collisions
    
    def _check_part_collision(self, part1: Part, part2: Part) -> bool:
        """
        检查两个零件是否碰撞
        
        Args:
            part1: 第一个零件
            part2: 第二个零件
        
        Returns:
            是否碰撞
        """
        # 获取零件尺寸
        size1 = self._get_part_size(part1)
        size2 = self._get_part_size(part2)
        
        # 计算距离
        pos1 = part1.position
        pos2 = part2.position
        distance = np.linalg.norm(pos1 - pos2)
        
        # 计算最小距离 (考虑尺寸)
        min_distance = (np.max(size1) + np.max(size2)) / 2.0 - self.tolerance
        
        return distance < min_distance
    
    def _get_part_size(self, part: Part) -> np.ndarray:
        """获取零件的尺寸向量"""
        params = part.params
        length = params.get('length', 0.1)
        width = params.get('width', params.get('thickness', 0.02))
        height = params.get('height', params.get('thickness', 0.02))
        radius = params.get('radius', 0.03)
        
        # 根据零件类型返回合适的尺寸
        part_type = part.part_type.lower()
        if 'cylinder' in part_type or 'motor' in part_type:
            return np.array([radius * 2, length, radius * 2])
        else:
            return np.array([length, width, height])
    
    def _validate_joint_limits(self, body: MechanicalBody) -> List[str]:
        """
        验证关节限制是否合理
        
        Returns:
            错误信息列表
        """
        errors = []
        
        for joint in body.joints():
            # 从 params 字典获取范围
            range_min = joint.params.get('range_min', None)
            range_max = joint.params.get('range_max', None)
            
            if range_min is not None and range_max is not None:
                if range_min >= range_max:
                    errors.append(f"关节 {joint.parent_id}-{joint.child_id} 范围无效: [{range_min}, {range_max}]")
                
                # 检查角度范围是否过大 (> 2π 通常不合理)
                if abs(range_max - range_min) > 2 * np.pi * 2:
                    errors.append(f"关节 {joint.parent_id}-{joint.child_id} 范围过大: {range_max - range_min:.2f} rad")
        
        return errors
    
    def _analyze_stability(self, body: MechanicalBody) -> Tuple[float, np.ndarray]:
        """
        分析机械体的稳定性 (基于重心位置)
        
        Returns:
            (稳定性分数, 重心位置)
        """
        total_mass = 0.0
        weighted_pos = np.zeros(3)
        
        for part in body.parts():
            # 估算质量
            size = self._get_part_size(part)
            volume = np.prod(size)
            density = self._get_part_density(part)
            mass = volume * density
            
            total_mass += mass
            weighted_pos += part.position * mass
        
        if total_mass > 0:
            centroid = weighted_pos / total_mass
        else:
            centroid = np.zeros(3)
        
        # 稳定性分数: 重心越靠近支撑面越高
        # 简化模型：重心高度越低越稳定
        max_height = max(p.position[2] for p in body.parts())
        if max_height > 0:
            stability = max(0.0, 1.0 - centroid[2] / max_height)
        else:
            stability = 1.0
        
        return stability, centroid
    
    def _get_part_density(self, part: Part) -> float:
        """获取零件密度 (简化版)"""
        part_type = part.part_type.lower()
        if 'aluminum' in part_type or 'extrusion' in part_type:
            return 2700.0
        elif 'steel' in part_type or 'bearing' in part_type or 'gear' in part_type:
            return 7850.0
        elif 'carbon' in part_type:
            return 1800.0
        elif 'lipo' in part_type or 'battery' in part_type:
            return 2600.0
        elif 'plastic' in part_type or 'sensor' in part_type or 'chip' in part_type:
            return 1000.0
        elif 'rubber' in part_type or 'foot' in part_type:
            return 920.0
        else:
            return 1000.0
    
    def _validate_kinematic_chain(self, body: MechanicalBody) -> List[str]:
        """
        验证运动链是否合理
        
        Returns:
            错误信息列表
        """
        errors = []
        
        # 检查是否存在闭环 (一般不应存在)
        if self._has_closed_loop(body):
            errors.append("检测到运动闭环，可能导致仿真不稳定")
        
        # 检查是否存在悬空零件 (无父节点且不是根节点)
        for part_id in body.graph.nodes:
            if part_id != body.root_id:
                parent = body.parent_of(part_id)
                if parent is None:
                    errors.append(f"零件 {part_id} 没有父节点且不是根节点")
        
        # 检查驱动关节数量是否合理
        actuated_count = len(body.actuated_joints())
        total_joints = body.num_joints()
        
        if actuated_count == 0 and total_joints > 0:
            errors.append("没有驱动关节，机械体无法主动运动")
        
        if actuated_count > total_joints:
            errors.append("驱动关节数量超过总关节数")
        
        return errors
    
    def _has_closed_loop(self, body: MechanicalBody) -> bool:
        """检查是否存在闭环"""
        # 使用DFS检测环
        visited = set()
        for node in body.graph.nodes:
            if node not in visited:
                if self._detect_cycle_dfs(body, node, visited, set()):
                    return True
        return False
    
    def _detect_cycle_dfs(self, body: MechanicalBody, node: str, visited: set, rec_stack: set) -> bool:
        """DFS检测环"""
        visited.add(node)
        rec_stack.add(node)
        
        for child in body.children_of(node):
            if child not in visited:
                if self._detect_cycle_dfs(body, child, visited, rec_stack):
                    return True
            elif child in rec_stack:
                return True
        
        rec_stack.discard(node)
        return False
    
    def check_collision_between_bodies(self, body1: MechanicalBody, body2: MechanicalBody) -> bool:
        """
        检查两个机械体之间是否碰撞
        
        Args:
            body1: 第一个机械体
            body2: 第二个机械体
        
        Returns:
            是否碰撞
        """
        for part1 in body1.parts():
            for part2 in body2.parts():
                if self._check_part_collision(part1, part2):
                    return True
        return False
    
    def get_collision_report(self, body: MechanicalBody) -> str:
        """获取碰撞检查报告"""
        result = self.check_body(body)
        
        lines = ["=" * 60]
        lines.append("🛡️ 碰撞检查报告")
        lines.append("=" * 60)
        lines.append(f"状态: {'✅ 有效' if result.is_valid else '❌ 无效'}")
        lines.append(f"连通性: {'连通' if result.is_connected else '不连通'}")
        lines.append(f"自由度: {result.dof_count}")
        lines.append(f"稳定性: {result.stability_score:.2f}")
        lines.append(f"重心: {result.centroid}")
        
        if result.collision_pairs:
            lines.append("\n⚠️ 碰撞对:")
            for pair in result.collision_pairs:
                lines.append(f"  - {pair[0]} ↔ {pair[1]}")
        
        if result.errors:
            lines.append("\n❌ 错误:")
            for error in result.errors:
                lines.append(f"  - {error}")
        
        if result.warnings:
            lines.append("\n⚠️ 警告:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# 全局碰撞检查器实例
_collision_checker = None


def get_collision_checker() -> CollisionChecker:
    """获取全局碰撞检查器实例"""
    global _collision_checker
    if _collision_checker is None:
        _collision_checker = CollisionChecker()
    return _collision_checker


def is_body_valid(body: MechanicalBody) -> bool:
    """快捷函数：检查机械体是否有效"""
    return get_collision_checker().check_body(body).is_valid


def filter_valid_bodies(bodies: List[MechanicalBody]) -> List[MechanicalBody]:
    """过滤出有效的机械体"""
    checker = get_collision_checker()
    return [body for body in bodies if checker.check_body(body).is_valid]
