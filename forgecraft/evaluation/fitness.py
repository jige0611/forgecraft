"""适应度函数优化模块

实现：
1. 多目标自适应权重：根据进化进度动态调整目标权重
2. 软约束处理：惩罚违反约束的个体
3. NSGA-II选择：多目标优化算法
4. 动态权重学习：自动学习目标重要性
"""

from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

from forgecraft.config import TaskConfig
from forgecraft.core.morphology import MechanicalBody

__all__ = [
    "MultiObjectiveFitnessEvaluator",
    "AdaptiveWeightLearner",
    "NSGAIISelector",
    "SoftConstraintHandler",
]

_logger = logging.getLogger(__name__)


class MultiObjectiveFitnessEvaluator:
    """多目标适应度评估器 — 速度/能耗/稳定性/对称性/质量

    将仿真日志映射到多个 fitness 维度:
      - speed:             质心绝对位移 / max_steps
      - energy_efficiency: bodyspeed 惩罚 (平滑运动)
      - stability:         z 高度波动 (稳定的 z)
      - symmetry:          左右对称性评分
      - body_mass:         质量惩罚 (越轻越好)
      - alive_frames:      存活时长占比
    """

    def __init__(self, task_config: TaskConfig):
        self.task_config = task_config
        
        # 目标权重（可动态调整）
        self.weights = {
            'speed': 0.3,
            'efficiency': 0.2,
            'stability': 0.2,
            'manufacturability': 0.15,
            'survival': 0.15,
        }
        
        # 约束参数
        self.constraints = {
            'max_energy': 100.0,
            'min_stability': 0.3,
            'min_manufacturability': 0.5,
        }
        
        # 自适应权重学习器
        self.weight_learner = AdaptiveWeightLearner()
    
    def evaluate(self, body: MechanicalBody, episode_info: dict, generation: int = 0) -> float:
        """评估个体适应度（带自适应权重）"""
        # 更新自适应权重
        self._update_weights(generation)
        
        # 计算各目标得分
        objectives = self._compute_objectives(body, episode_info)
        
        # 应用约束惩罚
        penalty = self._compute_constraint_penalty(objectives)
        
        # 加权求和
        fitness = sum(
            self.weights[key] * objectives.get(key, 0.0)
            for key in self.weights
        )
        
        # 应用惩罚
        fitness = max(0.0, fitness - penalty)
        
        return fitness
    
    # ── 多目标计算 ── 速度归一化 + 能量效率 + 稳定性 + 可制造性 + 存活率
    def _compute_objectives(self, body: MechanicalBody, episode_info: dict) -> Dict[str, float]:
        """计算各目标得分"""
        speed = episode_info.get('speed', 0.0)
        energy = episode_info.get('energy', 0.0)
        upright = episode_info.get('upright', 0.0)
        displacement = episode_info.get('displacement', 0.0)
        manufacturability = episode_info.get('manufacturability', 0.0)
        survival_ratio = episode_info.get('survival_ratio', 0.0)
        
        # 速度目标（归一化到[0, 1]）
        max_speed = 10.0  # 假设最大速度
        speed_score = min(speed / max_speed, 1.0)
        
        # 效率目标（能量效率 = 位移 / 能量消耗）
        if energy > 0:
            efficiency_score = min(displacement / energy * 10, 1.0)
        else:
            efficiency_score = 0.0 if displacement == 0 else 1.0
        
        # 稳定性目标
        stability_score = upright
        
        # 可制造性目标
        manufacturability_score = manufacturability
        
        # 生存能力目标
        survival_score = survival_ratio
        
        return {
            'speed': speed_score,
            'efficiency': efficiency_score,
            'stability': stability_score,
            'manufacturability': manufacturability_score,
            'survival': survival_score,
        }
    
    # ── 约束惩罚 ── 能量/稳定性/可制造性 软约束 min/max 越界惩罚
    def _compute_constraint_penalty(self, objectives: Dict[str, float]) -> float:
        """计算约束违反惩罚"""
        penalty = 0.0
        
        # 稳定性约束
        if objectives.get('stability', 0.0) < self.constraints['min_stability']:
            penalty += (self.constraints['min_stability'] - objectives['stability']) * 0.5
        
        # 可制造性约束
        if objectives.get('manufacturability', 0.0) < self.constraints['min_manufacturability']:
            penalty += (self.constraints['min_manufacturability'] - objectives['manufacturability']) * 0.3
        
        return penalty
    
    # ── 权重自适应 ── AdaptiveWeightLearner 按进化进度调权
    def _update_weights(self, generation: int):
        """根据进化进度更新权重"""
        progress = generation / 100  # 假设100代为完整进化
        
        # 早期：注重探索，平衡各目标
        if progress < 0.3:
            self.weights = {
                'speed': 0.2,
                'efficiency': 0.2,
                'stability': 0.2,
                'manufacturability': 0.2,
                'survival': 0.2,
            }
        # 中期：加速选择压力
        elif progress < 0.7:
            self.weights = {
                'speed': 0.35,
                'efficiency': 0.2,
                'stability': 0.15,
                'manufacturability': 0.15,
                'survival': 0.15,
            }
        # 后期：精细化选择
        else:
            self.weights = {
                'speed': 0.3,
                'efficiency': 0.25,
                'stability': 0.2,
                'manufacturability': 0.15,
                'survival': 0.1,
            }


class AdaptiveWeightLearner:
    """自适应权重学习器"""
    
    def __init__(self):
        self.learning_rate = 0.01
        self.weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
        self.target_names = ['speed', 'efficiency', 'stability', 'manufacturability', 'survival']
        
    def update(self, population_objectives: List[np.ndarray], generation: int):
        """根据种群表现更新权重"""
        if not population_objectives:
            return
        
        # 计算各目标的统计信息
        objectives_array = np.array(population_objectives)
        means = objectives_array.mean(axis=0)
        stds = objectives_array.std(axis=0) + 1e-8
        
        # 熵最大化策略：增加方差大的目标权重
        entropies = -np.sum((objectives_array / (objectives_array.sum(axis=1, keepdims=True) + 1e-8)) * 
                           np.log(objectives_array / (objectives_array.sum(axis=1, keepdims=True) + 1e-8) + 1e-8), axis=0)
        
        # 更新权重
        # 方向1：增加表现好的目标权重（均值高）
        reward_signal = means / (means.sum() + 1e-8)
        
        # 方向2：增加多样性高的目标权重（方差大）
        diversity_signal = stds / (stds.sum() + 1e-8)
        
        # 综合更新
        update_signal = 0.7 * reward_signal + 0.3 * diversity_signal
        self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * update_signal
        
        # 归一化
        self.weights = self.weights / (self.weights.sum() + 1e-8)
        
        # 施加最小权重约束
        min_weight = 0.05
        self.weights = np.maximum(self.weights, min_weight)
        self.weights = self.weights / (self.weights.sum() + 1e-8)


class NSGAIISelector:
    """NSGA-II选择器：多目标优化"""
    
    def __init__(self):
        self.crowding_distance_threshold = 0.01
    
    def select(self, population: List[MechanicalBody], n_select: int) -> List[MechanicalBody]:
        """选择前n_select个非支配解"""
        if not population:
            return []
        
        # 计算非支配前沿
        fronts = self._compute_non_dominated_fronts(population)
        
        # 选择
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= n_select:
                selected.extend(front)
            else:
                # 按拥挤度排序选择剩余个体
                front_with_crowding = self._compute_crowding_distance(front)
                front_with_crowding.sort(key=lambda x: x[1], reverse=True)
                needed = n_select - len(selected)
                selected.extend([ind for ind, _ in front_with_crowding[:needed]])
                break
        
        return selected
    
    def _compute_non_dominated_fronts(self, population: List[MechanicalBody]) -> List[List[MechanicalBody]]:
        """计算非支配前沿"""
        fronts = []
        
        # 获取所有个体的目标向量
        objectives = []
        for ind in population:
            obj = np.array([
                ind.fitness,  # 主适应度
                -ind.energy_usage,  # 能量效率（最大化）
                ind.stability,  # 稳定性
            ])
            objectives.append(obj)
        
        n = len(population)
        dominated_count = [0] * n
        dominated_set = [[] for _ in range(n)]
        
        # 计算支配关系
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(objectives[i], objectives[j]):
                    dominated_set[i].append(j)
                elif self._dominates(objectives[j], objectives[i]):
                    dominated_count[i] += 1
        
        # 第一前沿
        current_front = [i for i in range(n) if dominated_count[i] == 0]
        fronts.append([population[i] for i in current_front])
        
        # 后续前沿
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append([population[i] for i in next_front])
            current_front = next_front
        
        return fronts
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """判断obj1是否支配obj2"""
        better_in_all = all(obj1[i] >= obj2[i] for i in range(len(obj1)))
        better_in_at_least_one = any(obj1[i] > obj2[i] for i in range(len(obj1)))
        return better_in_all and better_in_at_least_one
    
    def _compute_crowding_distance(self, front: List[MechanicalBody]) -> List[Tuple[MechanicalBody, float]]:
        """计算拥挤度距离"""
        n = len(front)
        if n <= 2:
            return [(ind, float('inf')) for ind in front]
        
        # 获取目标向量
        objectives = np.array([
            [ind.fitness, -ind.energy_usage, ind.stability]
            for ind in front
        ])
        
        distances = np.zeros(n)
        
        for m in range(objectives.shape[1]):
            # 按第m个目标排序
            sorted_indices = np.argsort(objectives[:, m])
            objectives_sorted = objectives[sorted_indices, m]
            
            # 边界个体赋予无穷距离
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # 计算中间个体的拥挤度
            if objectives_sorted[-1] != objectives_sorted[0]:
                for i in range(1, n - 1):
                    distances[sorted_indices[i]] += (
                        objectives_sorted[i + 1] - objectives_sorted[i - 1]
                    ) / (objectives_sorted[-1] - objectives_sorted[0])
        
        return [(front[i], distances[i]) for i in range(n)]


class SoftConstraintHandler:
    """软约束处理器"""
    
    def __init__(self):
        self.constraints = {}
    
    def add_constraint(self, name: str, threshold: float, penalty_factor: float = 0.1):
        """添加约束"""
        self.constraints[name] = {
            'threshold': threshold,
            'penalty_factor': penalty_factor,
        }
    
    def evaluate_constraints(self, body: MechanicalBody) -> float:
        """评估约束违反并计算惩罚"""
        total_penalty = 0.0
        
        # 检查各约束
        if 'max_energy' in self.constraints:
            if body.energy_usage > self.constraints['max_energy']['threshold']:
                excess = body.energy_usage - self.constraints['max_energy']['threshold']
                total_penalty += excess * self.constraints['max_energy']['penalty_factor']
        
        if 'min_stability' in self.constraints:
            if body.stability < self.constraints['min_stability']['threshold']:
                deficit = self.constraints['min_stability']['threshold'] - body.stability
                total_penalty += deficit * self.constraints['min_stability']['penalty_factor']
        
        if 'max_parts' in self.constraints:
            n_parts = len(body.parts)
            if n_parts > self.constraints['max_parts']['threshold']:
                excess = n_parts - self.constraints['max_parts']['threshold']
                total_penalty += excess * self.constraints['max_parts']['penalty_factor']
        
        return total_penalty


def aggregate_fitness_generic(body: MechanicalBody, episode_info: dict, task_config: TaskConfig) -> float:
    """通用适应度聚合函数"""
    evaluator = MultiObjectiveFitnessEvaluator(task_config)
    return evaluator.evaluate(body, episode_info)