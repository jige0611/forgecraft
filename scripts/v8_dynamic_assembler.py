# ══════════════════════════════════════════════════════════
# ⚙️ V8 动态零件组装器 (Dynamic Assembler)
#
# 核心能力:
#   ✅ 将零件加载到PyBullet物理世界
#   ✅ 自动计算连接点对齐变换矩阵
#   ✅ 创建物理约束(固定关节/旋转关节)
#   ✅ 支持运行时拆卸/重组装
#   ✅ 装配关系图生成与可视化
#
# 使用示例:
#   >>> assembler = DynamicAssembler()
#   >>> motor_id = assembler.load_part('robomaster_m2006', [0, 0, 0.5])
#   >>> gear_id = assembler.load_part('harmonic_drive_csd_20', [0, 0, 0.6])
#   >>> constraint = assembler.assemble('robomaster_m2006', 'shaft_output',
#   ...                               'harmonic_drive_csd_20', 'input_shaft')
#
# ══════════════════════════════════════════════════════════

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("⚠️ PyBullet未安装，使用模拟模式")

from v8_collision_shapes import (
    COLLISION_SHAPES,
    get_part_info,
    get_all_connectors,
    validate_connection
)


@dataclass 
class AssemblyRecord:
    """单次装配记录"""
    constraint_id: int
    parent_part: str
    parent_connector: str
    child_part: str
    child_connector: str
    joint_type: str
    timestamp: float


@dataclass
class PartInstance:
    """零件实例信息"""
    part_name: str
    body_id: int
    position: np.ndarray
    orientation: np.ndarray
    connectors_world: Dict[str, Dict]  # 连接点世界坐标


class DynamicAssembler:
    """
    V8竞赛级零件动态组装器
    
    核心算法:
    1. 解析AI的装配指令
    2. 从连接点库获取空间坐标
    3. 计算对齐变换矩阵 (旋转+平移)
    4. 创建物理约束锁死或链接零件
    """
    
    def __init__(self, physics_client_id=0, use_gui=False):
        """
        初始化组装器
        
        Args:
            physics_client_id: PyBullet客户端ID
            use_gui: 是否启用GUI渲染
        """
        self.client = physics_client_id
        self.use_gui = use_gui
        
        # 存储已加载的零件实例
        self.loaded_parts: Dict[str, PartInstance] = {}
        
        # 存储所有装配约束
        self.constraints: Dict[int, AssemblyRecord] = {}
        self._constraint_counter = 0
        
        # 碰撞形状库引用
        self.part_library = COLLISION_SHAPES
        
        # 统计信息
        self.stats = {
            'parts_loaded': 0,
            'assemblies_created': 0,
            'disassemblies': 0,
            'failed_assemblies': 0
        }
        
        # 初始化PyBullet (如果可用且需要)
        if PYBULLET_AVAILABLE and use_gui:
            self._init_pybullet()
    
    def _init_pybullet(self):
        """初始化PyBullet物理环境"""
        if self.use_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        # 设置基本参数
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0/500,  # 500Hz仿真频率
            numSolverIterations=10,
            physicsClientId=self.client
        )
        
        print(f"✅ PyBullet环境初始化完成 (client={self.client})")
    
    def load_part(self, 
                  part_name: str, 
                  position: List[float],
                  orientation: List[float] = None,
                  color: List[float] = None) -> int:
        """
        将零件加载到物理世界
        
        Args:
            part_name: 零件名称 (如 "robomaster_m2006")
            position: 初始位置 [x,y,z]
            orientation: 初始姿态四元数 [w,x,y,z]
            color: RGBA颜色 [r,g,b,a]
        
        Returns:
            body_id: PyBullet刚体ID (-1表示失败)
        """
        if not PYBULLET_AVAILABLE:
            print(f"⚠️ 模拟加载: {part_name}")
            return -1
        
        try:
            # 获取零件定义
            part_def = get_part_info(part_name)
            
            if orientation is None:
                orientation = [0, 0, 0, 1]  # 无旋转
            
            # 根据碰撞形状类型创建几何体
            collision_data = part_def['collision']
            collision_type = collision_data['type']
            params = collision_data['params']
            
            # 创建碰撞几何体
            if collision_type == 'box':
                half_extents = params['half_extents']
                col_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    physicsClientId=self.client
                )
                vis_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    rgbaColor=color or self._get_default_color(part_name),
                    physicsClientId=self.client
                )
                
            elif collision_type == 'cylinder':
                radius = params['radius']
                height = params['height']
                col_shape = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    height=height,
                    physicsClientId=self.client
                )
                vis_shape = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    length=height,
                    rgbaColor=color or self._get_default_color(part_name),
                    physicsClientId=self.client
                )
                
            elif collision_type == 'sphere':
                radius = params['radius']
                col_shape = p.createCollisionShape(
                    p.GEOM_SPHERE,
                    radius=radius,
                    physicsClientId=self.client
                )
                vis_shape = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=radius,
                    rgbaColor=color or self._get_default_color(part_name),
                    physicsClientId=self.client
                )
                
            else:
                raise ValueError(f"不支持的碰撞类型: {collision_type}")
            
            # 获取质量
            mass = part_def['physics']['mass_kg']
            
            # 创建多刚体
            body_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=position,
                baseOrientation=orientation,
                physicsClientId=self.client
            )
            
            # 存储实例信息
            instance = PartInstance(
                part_name=part_name,
                body_id=body_id,
                position=np.array(position),
                orientation=np.array(orientation),
                connectors_world=self._compute_connectors_world(
                    part_name, position, orientation
                )
            )
            
            self.loaded_parts[part_name + f"_{body_id}"] = instance
            
            # 更新统计
            self.stats['parts_loaded'] += 1
            
            print(f"✅ 已加载: {part_name} (body_id={body_id}, mass={mass:.3f}kg)")
            
            return body_id
            
        except Exception as e:
            print(f"❌ 加载失败 {part_name}: {e}")
            self.stats['failed_assemblies'] += 1
            return -1
    
    def assemble(self,
                 parent_part_key: str,
                 parent_connector: str,
                 child_part_key: str,
                 child_connector: str,
                 joint_type: str = 'fixed',
                 joint_axis: List[float] = None) -> Optional[int]:
        """
        执行装配操作 (核心算法!)
        
        Args:
            parent_part_key: 父零件的key (name_bodyid格式)
            parent_connector: 父零件连接点名 ("shaft_output", etc.)
            child_part_key: 子零件的key
            child_connector: 子零件连接点名
            joint_type: "fixed" | "revolute" | "prismatic"
            joint_axis: 旋转轴 (仅revolute需要)
        
        Returns:
            constraint_id: 约束ID (None表示失败)
        """
        if not PYBULLET_AVAILABLE:
            print(f"⚠️ 模拟装配: {parent_part_key}.{parent_connector} ↔ {child_part_key}.{child_connector}")
            return None
        
        try:
            # 验证连接兼容性
            parent_name = parent_part_key.rsplit('_', 1)[0]
            child_name = child_part_key.rsplit('_', 1)[0]
            
            if not validate_connection(parent_name, parent_connector, 
                                     child_name, child_connector):
                print(f"❌ 连接不兼容: {parent_name}.{parent_connector} ↔ {child_name}.{child_connector}")
                self.stats['failed_assemblies'] += 1
                return None
            
            # 获取零件实例
            parent_inst = self.loaded_parts[parent_part_key]
            child_inst = self.loaded_parts[child_part_key]
            
            parent_id = parent_inst.body_id
            child_id = child_inst.body_id
            
            # 获取连接点定义
            parent_conn_def = get_all_connectors(parent_name)[parent_connector]
            child_conn_def = get_all_connectors(child_name)[child_connector]
            
            # ★★★ 核心计算: 对齐变换 ★★★
            aligned_pos, aligned_orn = self._calculate_alignment(
                parent_inst.position, parent_inst.orientation,
                parent_conn_def['pos'], parent_conn_def['axis'],
                child_conn_def['pos'], child_conn_def['axis']
            )
            
            # 移动子零件到对齐位置
            p.resetBasePositionAndOrientation(
                child_id, aligned_pos, aligned_orn,
                physicsClientId=self.client
            )
            
            # 更新子零件实例信息
            child_inst.position = np.array(aligned_pos)
            child_inst.orientation = np.array(aligned_orn)
            child_inst.connectors_world = self._compute_connectors_world(
                child_name, aligned_pos, aligned_orn
            )
            
            # 确定PyBullet约束类型
            joint_type_map = {
                'fixed': p.JOINT_FIXED,
                'revolute': p.JOINT_REVOLUTE,
                'prismatic': p.JOINT_PRISMATIC,
                'ball': p.JOINT_POINT2POINT
            }
            pb_joint_type = joint_type_map.get(joint_type, p.JOINT_FIXED)
            
            # 设置转轴 (默认Z轴)
            if joint_axis is None:
                joint_axis = [0, 0, 1]
            
            # 创建物理约束
            constraint_id = p.createConstraint(
                parentBodyUniqueId=parent_id,
                parentLinkIndex=-1,  # 基座坐标系
                childBodyUniqueId=child_id,
                childLinkIndex=-1,
                jointType=pb_joint_type,
                jointAxis=joint_axis,
                parentFramePosition=parent_conn_def['pos'],
                childFramePosition=child_conn_def['pos'],
                physicsClientId=self.client
            )
            
            # 记录装配关系
            record = AssemblyRecord(
                constraint_id=constraint_id,
                parent_part=parent_part_key,
                parent_connector=parent_connector,
                child_part=child_part_key,
                child_connector=child_connector,
                joint_type=joint_type,
                timestamp=time.time()
            )
            
            self.constraints[constraint_id] = record
            self._constraint_counter += 1
            
            # 更新统计
            self.stats['assemblies_created'] += 1
            
            print(f"🔗 已装配: {parent_name}.{parent_connector} ↔ {child_name}.{child_connector}")
            print(f"   类型: {joint_type}, ID: {constraint_id}")
            
            return constraint_id
            
        except Exception as e:
            print(f"❌ 装配错误: {e}")
            import traceback
            traceback.print_exc()
            self.stats['failed_assemblies'] += 1
            return None
    
    def disassemble(self, constraint_id: int) -> bool:
        """
        拆卸约束
        
        Args:
            constraint_id: 要移除的约束ID
        
        Returns:
            是否成功
        """
        if not PYBULLET_AVAILABLE:
            return True
        
        try:
            p.removeConstraint(constraint_id, physicsClientId=self.client)
            
            if constraint_id in self.constraints:
                del self.constraints[constraint_id]
            
            self.stats['disassemblies'] += 1
            print(f"🔧 已拆卸约束: {constraint_id}")
            return True
            
        except Exception as e:
            print(f"❌ 拆卸失败: {e}")
            return False
    
    def _calculate_alignment(self,
                            parent_pos: np.ndarray,
                            parent_orn: np.ndarray,
                            parent_conn_local: List[float],
                            parent_conn_axis: List[float],
                            child_conn_local: List[float],
                            child_conn_axis: List[float]) -> Tuple[List[float], List[float]]:
        """
        计算让两个连接点完美对齐所需的变换 (核心数学!)
        
        数学原理:
        1. 计算父连接点的世界坐标
        2. 计算子零件的目标位置 (使两连接点重合)
        3. 计算旋转使子连接轴与父连接轴对齐
        
        Returns:
            (aligned_position, aligned_orientation) 世界坐标
        """
        # 1. 将父连接点局部坐标转换为世界坐标
        parent_conn_world = self._transform_point(parent_pos, parent_orn, parent_conn_local)
        
        # 2. 子零件的目标位置 = 父连接点世界位置 - 子连接点局部偏移
        target_pos = np.array(parent_conn_world) - np.array(child_conn_local)
        
        # 3. 计算旋转对齐
        # 将父连接轴转换到世界坐标
        parent_axis_world = self._transform_direction(parent_orn, parent_conn_axis)
        child_axis_local = np.array(child_conn_axis)
        
        # 使用四元数旋转使两轴对齐
        rotation_euler = self._align_vectors(parent_axis_world, child_axis_local)
        
        # 转换为四元数 (PyBullet格式)
        if PYBULLET_AVAILABLE:
            target_orn = list(p.getQuaternionFromEuler(rotation_euler, 
                                                       physicsClientId=self.client))
        else:
            # 简化: 无旋转
            target_orn = [0, 0, 0, 1]
        
        return list(target_pos), target_orn
    
    def _transform_point(self, pos, orn, local_point) -> np.ndarray:
        """将局部坐标点转换为世界坐标"""
        local_pt = np.array(local_point)
        
        # 四元数旋转
        rotated = self._quaternion_rotate(orn, local_pt)
        
        # 平移
        world = np.array(pos) + rotated
        
        return world
    
    def _transform_direction(self, orn, local_dir) -> np.ndarray:
        """将局部方向向量转换为世界方向 (只旋转不平移)"""
        return self._quaternion_rotate(orn, np.array(local_dir))
    
    def _quaternion_rotate(self, q, v) -> np.ndarray:
        """用四元数旋转向量"""
        q = np.array(q)
        v = np.array(v)
        
        # 提取四元数分量
        w, x, y, z = q
        
        # 旋转公式: v' = q * v * q^(-1)
        # 展开为矩阵形式
        t2 =   w * x
        t3 =   w * y
        t4 =   w * z
        t5 =  -x * x
        t6 =   x * y
        t7 =   x * z
        t8 =  -y * y
        t9 =   y * z
        t10 = -z * z
        
        rotated = np.array([
            2 * (t6 + t4 - z) * v[1] + 2 * (t7 - t3 - y) * v[2] + (1 + 2*(t5 + t10)) * v[0],
            2 * (t6 - t4 + z) * v[0] + 2 * (t9 + t2 - x) * v[2] + (1 + 2*(t5 + t8)) * v[1],
            2 * (t7 + t3 - y) * v[0] + 2 * (t9 - t2 + x) * v[1] + (1 + 2*(t8 + t10)) * v[2]
        ])
        
        return rotated
    
    def _align_vectors(self, v_target, v_source) -> List[float]:
        """
        计算让v_source旋转到v_target所需的欧拉角 (简化版)
        
        使用Rodrigues旋转公式近似
        """
        v_target = np.array(v_target)
        v_source = np.array(v_source)
        
        # 归一化
        v_target = v_target / (np.linalg.norm(v_target) + 1e-8)
        v_source = v_source / (np.linalg.norm(v_source) + 1e-8)
        
        # 计算旋转轴 (叉积)
        axis = np.cross(v_source, v_target)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            # 已经平行
            return [0, 0, 0]
        
        axis = axis / axis_norm
        
        # 计算旋转角度 (点积)
        cos_angle = np.clip(np.dot(v_source, v_target), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # 转换为绕该轴的旋转 (简化为欧拉角)
        # 这里做近似处理: 主要旋转发生在垂直于两向量的平面内
        euler = [
            angle * axis[0],  # 绕X轴
            angle * axis[1],  # 绕Y轴  
            angle * axis[2]   # 绕Z轴
        ]
        
        return euler
    
    def _compute_connectors_world(self, part_name, position, orientation) -> Dict[str, Dict]:
        """计算零件所有连接点的世界坐标"""
        connectors_local = get_all_connectors(part_name)
        connectors_world = {}
        
        for conn_name, conn_data in connectors_local.items():
            world_pos = self._transform_point(position, orientation, conn_data['pos'])
            world_axis = self._transform_direction(orientation, conn_data['axis'])
            
            connectors_world[conn_name] = {
                'pos': world_pos.tolist(),
                'axis': world_axis.tolist(),
                'type': conn_data.get('type', 'unknown')
            }
        
        return connectors_world
    
    def _get_default_color(self, part_name: str) -> List[float]:
        """根据类别返回默认颜色"""
        category_colors = {
            'actuator': [0.85, 0.35, 0.10, 1.0],     # 橙色 (电机)
            'transmission': [0.70, 0.70, 0.75, 1.0],  # 银灰 (传动)
            'energy': [0.20, 0.60, 0.20, 1.0],       # 绿色 (能源)
            'sensor': [0.30, 0.30, 0.80, 1.0],       # 蓝色 (传感)
            'controller': [0.50, 0.00, 0.80, 1.0],   # 紫色 (控制)
            'connector': [0.80, 0.80, 0.20, 1.0],    # 黄色 (连接件)
            'structural': [0.60, 0.60, 0.65, 1.0],   # 灰色 (结构)
        }
        
        try:
            cat = get_part_info(part_name)['category']
            return category_colors.get(cat, [0.5, 0.5, 0.5, 1.0])
        except:
            return [0.5, 0.5, 0.5, 1.0]
    
    def get_assembly_graph(self) -> Dict:
        """获取当前装配关系图 (用于可视化/分析)"""
        graph = {
            'nodes': [],      # 零件节点
            'edges': [],      # 装配边
            'stats': self.stats.copy(),
            'timestamp': time.time()
        }
        
        # 收集所有节点
        for key, inst in self.loaded_parts.items():
            graph['nodes'].append({
                'id': inst.body_id,
                'part_name': inst.part_name,
                'position': inst.position.tolist(),
                'n_connectors': len(inst.connectors_world)
            })
        
        # 收集所有边
        for cid, record in self.constraints.items():
            graph['edges'].append({
                'constraint_id': cid,
                'source': record.parent_part,
                'target': record.child_part,
                'joint_type': record.joint_type,
                'connectors': f"{record.parent_connector} → {record.child_connector}"
            })
        
        return graph
    
    def run_stability_test(self, duration_s: float = 1.0) -> float:
        """
        运行稳定性测试 (短时间仿真检测结构完整性)
        
        Args:
            duration_s: 测试时长 (秒)
        
        Returns:
            stability_score: 稳定性评分 (-10 到 +10)
        """
        if not PYBULLET_AVAILABLE:
            return 0.0
        
        steps = int(duration_s / (1.0/500))  # 500Hz
        
        min_heights = []
        
        for step in range(steps):
            p.stepSimulation(physicsClientId=self.client)
            
            # 检查所有零件高度
            for key, inst in self.loaded_parts.items():
                pos, _ = p.getBasePositionAndOrientation(
                    inst.body_id, physicsClientId=self.client
                )
                min_heights.append(pos[2])
        
        if not min_heights:
            return 0.0
        
        min_height = min(min_heights)
        
        # 评分逻辑
        if min_height < 0.01:      # 掉入地面以下
            return -10.0           # 完全崩溃
        elif min_height < 0.05:    # 很低但不一定崩溃
            return -5.0
        elif min_height < 0.15:    # 不稳定
            return -2.0
        else:                      # 正常
            return 10.0
    
    def reset(self):
        """重置整个组装状态"""
        if PYBULLET_AVAILABLE:
            # 移除所有约束
            for cid in list(self.constraints.keys()):
                self.disassemble(cid)
            
            # 移除所有刚体
            for key, inst in list(self.loaded_parts.items()):
                try:
                    p.removeBody(inst.body_id, physicsClientId=self.client)
                except:
                    pass
        
        # 清空数据
        self.loaded_parts.clear()
        self.constraints.clear()
        
        print("🔄 组装器已重置")
    
    def summary(self) -> str:
        """生成摘要报告"""
        lines = [
            "=" * 60,
            "📊 V8 动态组装器状态报告",
            "=" * 60,
            f"\n📦 已加载零件: {len(self.loaded_parts)}",
            f"🔗 活跃约束: {len(self.constraints)}",
            "",
            "统计:",
            f"  • 总加载次数: {self.stats['parts_loaded']}",
            f"  • 成功装配: {self.stats['assemblies_created']}",
            f"  • 失败装配: {self.stats['failed_assemblies']}",
            f"  • 拆卸次数: {self.stats['disassemblies']}",
        ]
        
        if self.loaded_parts:
            lines.append("\n零件列表:")
            for key, inst in self.loaded_parts.items():
                lines.append(f"  • {inst.part_name} (ID={inst.body_id})")
        
        if self.constraints:
            lines.append("\n装配关系:")
            for cid, rec in self.constraints.items():
                lines.append(f"  [{cid}] {rec.parent_part} → {rec.child_part} ({rec.joint_type})")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════
# 便捷函数
# ══════════════════════════════════════════════════════════

def quick_assemble_motor_gearbox(assembler: DynamicAssembler,
                                motor_name: str = 'robomaster_m2006',
                                gearbox_name: str = 'harmonic_drive_csd_20') -> Optional[int]:
    """
    快速装配电机+减速器的常用组合
    
    Args:
        assembler: 组装器实例
        motor_name: 电机型号
        gearbox_name: 减速器型号
    
    Returns:
        约束ID
    """
    # 加载电机
    motor_id = assembler.load_part(motor_name, position=[0, 0, 0.5])
    if motor_id < 0:
        return None
    
    # 加载减速器 (初始位置在电机上方)
    gear_id = assembler.load_part(gearbox_name, position=[0, 0, 0.7])
    if gear_id < 0 < 0:
        return None
    
    # 执行装配
    motor_key = f"{motor_name}_{motor_id}"
    gear_key = f"{gearbox_name}_{gear_id}"
    
    constraint = assembler.assemble(
        parent_part_key=motor_key,
        parent_connector='shaft_output',
        child_part_key=gear_key,
        child_connector='input_shaft'
    )
    
    return constraint


if __name__ == "__main__":
    print("=" * 70)
    print("⚙️ V8 动态组装器测试")
    print("=" * 70)
    
    # 创建组装器
    assembler = DynamicAssembler(use_gui=True)
    
    # 测试1: 快速装配电机+减速器
    print("\n🔧 测试: 电机 + 减速器装配")
    constraint = quick_assemble_motor_gearbox(assembler)
    
    if constraint is not None:
        print(f"✅ 装配成功! 约束ID={constraint}")
        
        # 运行稳定性测试
        stability = assembler.run_stability_test(duration_s=0.5)
        print(f"📊 稳定性评分: {stability:+.1f}")
    
    # 打印摘要
    print("\n" + assembler.summary())
    
    # 获取装配图
    graph = assembler.get_assembly_graph()
    print(f"\n🕸️ 装配图节点数: {len(graph['nodes'])}")
    print(f"   边数: {len(graph['edges'])}")
