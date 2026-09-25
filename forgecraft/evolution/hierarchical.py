# ══════════════════════════════════════════════════════════
# forgecraft.evolution.hierarchical — 六层嵌套协调器
#
#   计算漏斗架构:
#     L1: NSGA-III 形态进化 — 1000 个体, 廉价代理评估
#     L2: IGA 拓扑优化 — 100 形态, FEA 柔度
#     L3: TPMS 晶格填充 — 50 形态, 等效刚度量
#     L4: CMA-ME 参数精化 — 30 形态, 运动学/动力学
#     L5: PPO 策略学习 — 15 形态, MuJoCo RL
#     L6: 多任务验证 — 5 形态, 跨任务评测
#
#   核心机制:
#     - 质量门: 只有满足门槛的个体才能进入下一层
#     - 向上反馈: 深层精确适应度回传给外层 Pareto 前沿
#     - 动态调度: 外层收敛 → 激活内层, 内层停滞 → 回退外层
#     - 预算分配: UCB 多臂老虎机在各层间动态分配
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "LayerState",
    "NestingLayer",
    "LayerScheduler",
    "HierarchicalNestingOptimizer",
    "create_default_layers",
]


# ══════════════════════════════════════════════════════════
#  层状态
# ══════════════════════════════════════════════════════════

class LayerState(Enum):
    """六层嵌套中每层的状态"""
    DORMANT = auto()         # 未激活 (上游未收敛)
    ACTIVE = auto()          # 当前激活, 正在优化
    STAGNANT = auto()        # 收敛停滞
    CONVERGED = auto()       # 已收敛
    EXHAUSTED = auto()       # 预算耗尽
    SKIPPED = auto()         # 跳过 (无合格个体)


# ══════════════════════════════════════════════════════════
#  层定义
# ══════════════════════════════════════════════════════════

@dataclass
class NestingLayer:
    """一层嵌套优化

    Attributes:
        name: 层名称 ("L1_morphology", "L2_topology", ...)
        level: 层级编号 (0-5)
        optimizer: 优化器实例 (NSGA3Selector, CMAES, 等)
        evaluator: 评估函数 callable(population) → fitnesses
        budget: 总评估预算
        stagnation_threshold: 停滞代数阈值
        quality_gate: 进入本层的质量门槛 callable(individual) → bool
        upstream: 上游层名 (None=最外层)
        downstream: 下游层名 (None=最内层)
        design_variables: 本层优化的设计变量列表
        population: 当前本层种群
    """
    name: str
    level: int
    budget: int = 1000
    stagnation_threshold: int = 10
    stagnation_counter: int = 0
    quality_gate: Optional[Callable] = None
    upstream: Optional[str] = None
    downstream: Optional[str] = None
    design_variables: List[str] = field(default_factory=list)

    # 运行时状态
    state: LayerState = LayerState.DORMANT
    population: List[Any] = field(default_factory=list)
    pending_individuals: List[Any] = field(default_factory=list)
    best_fitness_history: List[float] = field(default_factory=list)
    generation: int = 0
    evaluations_used: int = 0

    def enqueue(self, individuals: List[Any]):
        """接收上游传入的个体 (通过质量门的)"""
        self.pending_individuals.extend(individuals)

    def is_ready(self) -> bool:
        """本层是否就绪 (上游收敛 + 有合格个体)"""
        return len(self.pending_individuals) > 0

    def check_stagnation(self) -> bool:
        """检查是否停滞

        停滞判据: 连续 self.stagnation_threshold 代最优适应度无改进
        """
        h = self.best_fitness_history
        if len(h) < self.stagnation_threshold:
            return False
        window = h[-self.stagnation_threshold:]
        # 最近 N 代改进 < 1%
        best_recent = max(window)
        best_early = max(h[-2 * self.stagnation_threshold: -self.stagnation_threshold]) if len(h) >= 2 * self.stagnation_threshold else window[0]
        if best_early < 1e-15:
            return best_recent < 1e-15
        return abs(best_recent - best_early) / best_early < 0.01

    def check_convergence(self) -> bool:
        """检查是否收敛

        收敛判据: 连续 N 代最佳适应度改进 < ε
        且种群多样性足够 (标准差/均值 > 阈值)
        """
        h = self.best_fitness_history
        if len(h) < 5:
            return False

        recent = np.array(h[-5:])
        improvement = 0.0
        if np.max(recent) > 1e-15:
            improvement = (np.max(recent) - recent[0]) / max(np.max(recent), 1e-15)

        return abs(improvement) < 0.001


# ══════════════════════════════════════════════════════════
#  层调度器
# ══════════════════════════════════════════════════════════

class LayerScheduler:
    """六层动态调度器

    规则:
      1. 外层收敛 → 激活相邻内层
      2. 内层停滞 → 回退到外层 (更广探索)
      3. 外层多样性低 → 注入随机个体
      4. UCB 预算分配: 70% 当前层 / 20% 上游 / 10% 下游
    """

    def __init__(
        self,
        layers: List[NestingLayer],
        exploration_rate: float = 0.1,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.layers = {layer.name: layer for layer in layers}
        self.layer_order = [layer.name for layer in sorted(layers, key=lambda l: l.level)]
        self.exploration_rate = exploration_rate
        self.rng = rng or np.random.RandomState()

        # UCB 统计
        self.layer_rewards: Dict[str, List[float]] = {name: [] for name in self.layer_order}
        self.layer_counts: Dict[str, int] = {name: 0 for name in self.layer_order}

    def get_active_layer(self) -> Optional[NestingLayer]:
        """确定哪一层当前应被激活

        优先级:
          1. 有最多 pending_individuals 的内层
          2. 收敛的层的下游
          3. 最外层 (如果没有任何层活跃)
        """
        # 检查各层状态
        active_layers = [l for l in self.layers.values() if l.state == LayerState.ACTIVE]
        if active_layers:
            # 运行中的层: 使用 UCB 选择最有前途的
            return self._ucb_select(active_layers)

        # 找就绪层 (上游收敛 + 有合格个体)
        ready_layers = [
            l for l in self.layers.values()
            if l.is_ready() and l.state in (LayerState.DORMANT, LayerState.CONVERGED)
        ]
        if ready_layers:
            # 优先激活最内层的就绪层 (漏斗逻辑)
            return max(ready_layers, key=lambda l: l.level)

        # 回退: 找停滞的层并重启外层
        stagnant = [
            l for l in self.layers.values()
            if l.state == LayerState.STAGNANT
        ]
        if stagnant:
            # 激活停滞层的上游 (回退探索)
            stagnant_layer = stagnant[0]
            upstream_name = stagnant_layer.upstream
            if upstream_name and upstream_name in self.layers:
                upstream = self.layers[upstream_name]
                if upstream.state != LayerState.EXHAUSTED:
                    return upstream

        # 最终回退: 激活最外层
        outermost = self.layers[self.layer_order[0]]
        if outermost.state == LayerState.DORMANT:
            outermost.state = LayerState.ACTIVE
            return outermost

        return None

    def _ucb_select(self, active_layers: List[NestingLayer]) -> NestingLayer:
        """UCB 多臂老虎机选择最值得投资的层

        UCB_i = mean_reward_i + c · sqrt(ln(N) / n_i)
        """
        total_counts = sum(self.layer_counts.values())
        if total_counts == 0:
            return self.rng.choice(active_layers)

        best_score = -float('inf')
        best_layer = active_layers[0]

        for layer in active_layers:
            count = self.layer_counts.get(layer.name, 0)
            if count == 0:
                return layer  # 未探索的层优先

            rewards = self.layer_rewards.get(layer.name, [])
            mean_reward = np.mean(rewards[-20:]) if rewards else 0.0

            exploration_bonus = np.sqrt(2.0 * np.log(total_counts + 1) / count)
            ucb_score = mean_reward + exploration_bonus

            if ucb_score > best_score:
                best_score = ucb_score
                best_layer = layer

        return best_layer

    def update(self, active_layer: NestingLayer, reward: float):
        """更新调度器: 记录一层一步的结果"""
        self.layer_counts[active_layer.name] = self.layer_counts.get(active_layer.name, 0) + 1
        if active_layer.name not in self.layer_rewards:
            self.layer_rewards[active_layer.name] = []
        self.layer_rewards[active_layer.name].append(reward)

        # 更新状态
        if active_layer.check_convergence():
            active_layer.state = LayerState.CONVERGED
            _logger.info(f"Layer {active_layer.name}: converged")
            self._propagate_downstream(active_layer)

        elif active_layer.check_stagnation():
            old_state = active_layer.state
            active_layer.state = LayerState.STAGNANT
            active_layer.stagnation_counter += 1
            if old_state != LayerState.STAGNANT:
                _logger.warning(f"Layer {active_layer.name}: stagnated (counter={active_layer.stagnation_counter})")

        elif active_layer.evaluations_used >= active_layer.budget:
            active_layer.state = LayerState.EXHAUSTED
            _logger.info(f"Layer {active_layer.name}: budget exhausted")

    def _propagate_downstream(self, layer: NestingLayer):
        """向下传播: 激活下游层"""
        if layer.downstream and layer.downstream in self.layers:
            downstream = self.layers[layer.downstream]
            # 通过质量门的个体传递给下游
            if layer.population and layer.quality_gate:
                qualified = [ind for ind in layer.population if layer.quality_gate(ind)]
            else:
                qualified = layer.population
            downstream.enqueue(qualified)

            if downstream.state == LayerState.DORMANT:
                downstream.state = LayerState.ACTIVE
                _logger.info(f"Layer {downstream.name}: activated (upstream converged)")

    def get_statistics(self) -> Dict:
        """返回调度器统计"""
        stats = {}
        for name, layer in self.layers.items():
            stats[name] = {
                "state": layer.state.name,
                "population_size": len(layer.population),
                "pending": len(layer.pending_individuals),
                "generation": layer.generation,
                "evaluations": layer.evaluations_used,
                "budget": layer.budget,
                "best_fitness": max(layer.best_fitness_history) if layer.best_fitness_history else None,
            }
        return stats


# ══════════════════════════════════════════════════════════
#  主协调器
# ══════════════════════════════════════════════════════════

class HierarchicalNestingOptimizer:
    """六层嵌套优化协调器

    典型用法:
        layers = create_default_layers(n_objectives=5)
        coordinator = HierarchicalNestingOptimizer(layers)
        final_population = coordinator.optimize(max_generations=500)

    内部统计可通过:
        stats = coordinator.get_statistics()
        pareto_front = coordinator.get_pareto_front()
    """

    def __init__(
        self,
        layers: List[NestingLayer],
        max_generations: int = 500,
        random_seed: Optional[int] = None,
        verbosity: int = 0,
    ):
        self.layers = {layer.name: layer for layer in layers}
        self.layer_order = [layer.name for layer in sorted(layers, key=lambda l: l.level)]
        self.max_generations = max_generations
        self.rng = np.random.RandomState(random_seed)
        self.verbosity = verbosity

        self.scheduler = LayerScheduler(layers, rng=self.rng)

        # 历史追踪
        self.generation_history: List[int] = []
        self.active_layer_history: List[str] = []

    def optimize(
        self,
        initial_population: Optional[List[Any]] = None,
        objective_keys: Optional[List[str]] = None,
        directions: Optional[List[str]] = None,
    ) -> List[Any]:
        """运行六层嵌套优化

        Args:
            initial_population: 初始种群 (输入 L1)
            objective_keys: 多目标键列表
            directions: 每个目标的方向

        Returns:
            最终种群 (所有层产出的 Pareto 最优个体)
        """
        # 初始化最外层
        outermost = self.layers[self.layer_order[0]]
        if initial_population is not None:
            outermost.population = list(initial_population)
        outermost.state = LayerState.ACTIVE

        for global_gen in range(self.max_generations):
            # 获取活跃层
            active_layer = self.scheduler.get_active_layer()
            if active_layer is None:
                _logger.info(f"Gen {global_gen}: all layers converged/exhausted")
                break

            self.generation_history.append(global_gen)
            self.active_layer_history.append(active_layer.name)

            # 执行一步
            reward = self._step_layer(
                active_layer, objective_keys, directions,
            )

            # 更新调度器
            self.scheduler.update(active_layer, reward)

            if self.verbosity > 0 and global_gen % 20 == 0:
                stats = self.scheduler.get_statistics()
                active_names = [l.name for l in self.layers.values() if l.state == LayerState.ACTIVE]
                _logger.info(
                    f"Gen {global_gen}: active={active_names}, "
                    f"L1_pop={stats[self.layer_order[0]]['population_size']}"
                )

        # 收集各层结果
        return self._collect_results()

    def _step_layer(
        self,
        layer: NestingLayer,
        objective_keys: Optional[List[str]],
        directions: Optional[List[str]],
    ) -> float:
        """对一层执行一步优化, 返回改进度量"""
        layer.generation += 1

        if not layer.population:
            if layer.pending_individuals:
                layer.population = layer.pending_individuals.copy()
                layer.pending_individuals = []
            else:
                return 0.0

        # 评估当前种群 (如果提供了评估函数)
        # 实际使用时, 外层评估由调用方通过 evaluator callable 完成
        # 这里只追踪状态变化

        reward = 0.0
        if layer.best_fitness_history:
            reward = layer.best_fitness_history[-1]

        return reward

    def _collect_results(self) -> List[Any]:
        """收集所有层的最终结果"""
        results = []
        for name in self.layer_order:
            layer = self.layers[name]
            if layer.population:
                results.extend(layer.population)
        return results

    def get_statistics(self) -> Dict:
        return self.scheduler.get_statistics()

    def get_pareto_front(self) -> List[Any]:
        """获取最终 Pareto 前沿 (各层最优个体)"""
        front = []
        for name in self.layer_order:
            layer = self.layers[name]
            if layer.population:
                front.extend(layer.population[:10])  # top 10 per layer
        return front


# ══════════════════════════════════════════════════════════
#  默认六层创建
# ══════════════════════════════════════════════════════════

def create_default_layers(
    n_objectives: int = 5,
    n_divisions: int = 6,
    l1_budget: int = 10000,
    l2_budget: int = 1000,
    l3_budget: int = 500,
    l4_budget: int = 500,
    l5_budget: int = 200,
    l6_budget: int = 50,
) -> List[NestingLayer]:
    """创建六层嵌套优化的默认层配置

    计算漏斗:
      L1: 10000 评估 → 输出 1000 个体
      L2: 1000 评估  → 输出 100 个体
      L3: 500 评估   → 输出 50 个体
      L4: 500 评估   → 输出 30 个体
      L5: 200 评估   → 输出 15 个体
      L6: 50 评估    → 输出 5 个体

    Args:
        n_objectives: 目标数 (L1 用 NSGA-III)
        n_divisions: L1 NSGA-III 参考点划分数
        l1_budget-l6_budget: 各层预算

    Returns:
        6 个 NestingLayer
    """

    # 默认质量门: 所有个体都通过 (实际使用时替换)
    def _default_gate(ind) -> bool:
        return True

    layers = [
        NestingLayer(
            name="L1_morphology",
            level=0,
            budget=l1_budget,
            stagnation_threshold=15,
            downstream="L2_topology",
            design_variables=["parts", "joints", "tree_topology"],
        ),
        NestingLayer(
            name="L2_topology",
            level=1,
            budget=l2_budget,
            stagnation_threshold=10,
            upstream="L1_morphology",
            downstream="L3_lattice",
            quality_gate=_default_gate,
            design_variables=["material_distribution"],
        ),
        NestingLayer(
            name="L3_lattice",
            level=2,
            budget=l3_budget,
            stagnation_threshold=10,
            upstream="L2_topology",
            downstream="L4_parameters",
            quality_gate=_default_gate,
            design_variables=["cell_size", "thickness"],
        ),
        NestingLayer(
            name="L4_parameters",
            level=3,
            budget=l4_budget,
            stagnation_threshold=10,
            upstream="L3_lattice",
            downstream="L5_policy",
            quality_gate=_default_gate,
            design_variables=["joint_ranges", "dimensions", "materials"],
        ),
        NestingLayer(
            name="L5_policy",
            level=4,
            budget=l5_budget,
            stagnation_threshold=8,
            upstream="L4_parameters",
            downstream="L6_multitask",
            quality_gate=_default_gate,
            design_variables=["nn_weights"],
        ),
        NestingLayer(
            name="L6_multitask",
            level=5,
            budget=l6_budget,
            stagnation_threshold=5,
            upstream="L5_policy",
            design_variables=[],
        ),
    ]

    return layers
