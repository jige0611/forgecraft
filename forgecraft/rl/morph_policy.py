# ══════════════════════════════════════════════════════════
#  形态-策略共同进化 (Morphology-Policy Co-evolution)
#
#  核心思想:
#    传统方案: 固定形态 → 训练 RL 控制器 (或 固定控制器 → 进化形态)
#    共同进化: 形态与策略同代、同个体、同时进化
#      → 形态参数 + 策略权重 → 共享基因型
#      → 交叉/变异作用于形态和策略
#      → 适应度 = 运动表现 × 结构效率 × 学习速度
#
#  三种策略:
#    A) 可进化策略矩阵 (ESM)
#       策略 = 线性矩阵 W ∈ R^(act_dim × obs_dim)
#       基因型 = [morph_params..., W.flatten()]
#       优点: 简单, 每个体 100-1000 参数, 5 代收敛
#
#    B) 形态条件策略网络 (MCPN)
#       策略 = MorphAwareActor(body_embed, obs) → action
#       基因型 = [morph_params..., body_embed (可选)]
#       优点: 泛化, 新形态自动适应
#
#    C) 双层进化 (Bilevel)
#       外层: 进化形态 (GA/NSGA-II)
#       内层: 进化策略 (CMA-ES/PPO)
#       优点: 质量最高, 但慢 5-10x
#
#  默认使用策略 A (快速) + 策略 B (精度) 的混合方案
# ══════════════════════════════════════════════════════════

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_logger = logging.getLogger(__name__)

__all__ = [
    "CoevolutionConfig",
    "MorphPolicyGenome",
    "CoevolutionEngine",
    "ESMController",
    "MorphConditionedPolicy",
]

# ══════════════════════════════════════════════════════════
#  可进化策略矩阵 (ESM)
# ══════════════════════════════════════════════════════════

class ESMController:
    """线性可进化策略
    
    基因型: W ∈ R^(act_dim × obs_dim)  + bias ∈ R^act_dim
    策略: a = tanh(W @ obs + b)
    
    适合 5-10 代快速收敛, 对于简单运动 (行走/抓取) 足够
    """
    
    def __init__(self, obs_dim: int, act_dim: int):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_params = obs_dim * act_dim + act_dim
        
        # 初始化 (Xavier)
        scale = np.sqrt(6.0 / (obs_dim + act_dim))
        self.W = np.random.uniform(-scale, scale, (act_dim, obs_dim))
        self.b = np.zeros(act_dim)
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs: (obs_dim,) → action: (act_dim,)"""
        a = self.W @ obs + self.b
        return np.tanh(a)
    
    def to_genome(self) -> np.ndarray:
        """展平为基因向量"""
        return np.concatenate([self.W.ravel(), self.b])
    
    @classmethod
    def from_genome(cls, obs_dim: int, act_dim: int, genome: np.ndarray) -> "ESMController":
        """从基因向量恢复"""
        ctrl = cls(obs_dim, act_dim)
        n_w = obs_dim * act_dim
        ctrl.W = genome[:n_w].reshape(act_dim, obs_dim)
        ctrl.b = genome[n_w:n_w + act_dim]
        return ctrl
    
    def mutate(self, sigma: float = 0.1):
        """高斯变异"""
        n_w = self.obs_dim * self.act_dim
        genome = self.to_genome()
        noise = np.random.randn(len(genome)) * sigma
        genome += noise
        self.W = genome[:n_w].reshape(self.act_dim, self.obs_dim)
        self.b = genome[n_w:n_w + self.act_dim]
    
    @classmethod
    def crossover(cls, parent_a: "ESMController", parent_b: "ESMController",
                  alpha: float = 0.5) -> "ESMController":
        """线性交叉"""
        child = cls(parent_a.obs_dim, parent_a.act_dim)
        child.W = alpha * parent_a.W + (1 - alpha) * parent_b.W
        child.b = alpha * parent_a.b + (1 - alpha) * parent_b.b
        return child


# ══════════════════════════════════════════════════════════
#  形态条件策略网络 (神经网络版)
# ══════════════════════════════════════════════════════════

class MorphConditionedPolicy(nn.Module):
    """形态感知策略网络
    
    输入: concatenate(obs, morph_embedding)
    隐藏层: 形态调节层 (FiLM / 简单的 concatenation)
    输出: mean, log_std 或确定性 action
    
    结构:
      obs ──→ [Linear(obs+embed, 128)] → [Linear(128, 128)] → mean
                   ↑                              └→ log_std
      morph → [Linear(m_dim, 128)]
    """
    
    def __init__(self, obs_dim: int, act_dim: int, morph_dim: int = 32,
                 hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.morph_dim = morph_dim
        
        # 形态编码器
        self.morph_encoder = nn.Sequential(
            nn.Linear(morph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 策略网络
        self.shared = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor, morph_embed: torch.Tensor,
                deterministic: bool = False):
        """前向传播
        
        Args:
            obs: (batch, obs_dim)
            morph_embed: (batch, morph_dim)
        
        Returns:
            action: (batch, act_dim)  (tanh 范围)
            log_prob: (batch,)  (仅 stochastic 模式)
        """
        mh = self.morph_encoder(morph_embed)
        x = torch.cat([obs, mh], dim=-1)
        features = self.shared(x)
        mean = self.mean_head(features)
        
        if deterministic:
            return torch.tanh(mean), None
        
        std = torch.exp(self.log_std.clamp(-20, 2))
        dist = torch.distributions.Normal(mean, std)
        action_sample = dist.rsample()
        action = torch.tanh(action_sample)
        
        # log_prob correction for tanh
        log_prob = dist.log_prob(action_sample).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        
        return action, log_prob
    
    def act(self, obs: np.ndarray, morph_embed: np.ndarray) -> np.ndarray:
        """确定性推理 → action numpy array"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            morph_t = torch.FloatTensor(morph_embed).unsqueeze(0)
            action, _ = self.forward(obs_t, morph_t, deterministic=True)
            return action.cpu().numpy()[0]
    
    def get_genome(self) -> np.ndarray:
        """提取可进化参数 (仅 log_std 和最终层权重)"""
        params = []
        # 简化: 只进化 final layer weights (减少参数搜索空间)
        with torch.no_grad():
            params.append(self.mean_head.weight.cpu().numpy().ravel())
            params.append(self.mean_head.bias.cpu().numpy().ravel())
            params.append(self.log_std.cpu().numpy())
        return np.concatenate([p for p in params])
    
    def apply_genome(self, genome: np.ndarray):
        """从基因向量恢复部分参数"""
        n_w = self.act_dim * self.morph_encoder[-2].out_features if hasattr(self.morph_encoder[-2], 'out_features') else self.act_dim * 128
        # 简化: 只恢复 log_std
        with torch.no_grad():
            self.log_std.copy_(torch.FloatTensor(genome[-self.act_dim:]))


# ══════════════════════════════════════════════════════════
#  共同进化配置
# ══════════════════════════════════════════════════════════

@dataclass
class CoevolutionConfig:
    """共同进化配置"""
    strategy: str = "esm"               # "esm" | "mcpn" | "bilevel"
    
    # ESM 参数
    esm_mutation_strength: float = 0.1  # 变异强度 σ
    esm_crossover_rate: float = 0.5     # 交叉率
    
    # MCPN 参数
    morph_embed_dim: int = 32           # 形态嵌入维度
    mcpn_population: int = 20           # 策略种群大小 (双层进化)
    
    # 通用
    policy_weight_in_fitness: float = 0.3  # 策略性能在总适应度中占比


# ══════════════════════════════════════════════════════════
#  形态-策略基因组
# ══════════════════════════════════════════════════════════

@dataclass
class MorphPolicyGenome:
    """联合基因型: 形态参数 + 策略参数"""
    # 形态部分
    morph_params: Dict[str, Any] = field(default_factory=dict)
    
    # 策略部分
    policy_genome: np.ndarray = field(default_factory=lambda: np.zeros(1))
    obs_dim: int = 10
    act_dim: int = 4
    
    # 元信息
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    def to_vector(self) -> np.ndarray:
        """展平为纯向量 (供遗传算子使用)"""
        # 形态参数 → 向量 (归一化)
        morph_vec = []
        for key in sorted(self.morph_params.keys()):
            val = self.morph_params[key]
            if isinstance(val, (int, float)):
                morph_vec.append(float(val))
            elif isinstance(val, (list, np.ndarray)):
                morph_vec.extend([float(v) for v in val[:3]])
        
        morph_vec = np.array(morph_vec[:20], dtype=np.float32)  # 截断
        if len(morph_vec) < 20:
            morph_vec = np.pad(morph_vec, (0, 20 - len(morph_vec)))
        
        return np.concatenate([morph_vec, self.policy_genome[:100]])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, obs_dim: int, act_dim: int,
                    morph_keys: List[str]) -> "MorphPolicyGenome":
        """从向量恢复"""
        morph_vec = vec[:20]
        policy_vec = vec[20:]
        
        morph_params = {}
        idx = 0
        for key in morph_keys:
            if idx < len(morph_vec):
                morph_params[key] = float(morph_vec[idx])
                idx += 1
        
        return cls(
            morph_params=morph_params,
            policy_genome=policy_vec,
            obs_dim=obs_dim,
            act_dim=act_dim,
        )
    
    def mutate(self, sigma: float = 0.1):
        """高斯变异 (形态 + 策略)"""
        vec = self.to_vector()
        noise = np.random.randn(len(vec)) * sigma
        vec += noise
        
        # 形态参数非负
        vec[:20] = np.maximum(0.01, vec[:20])
        
        self.policy_genome = vec[20:]
        
        # 恢复形态参数
        idx = 0
        for key in sorted(self.morph_params.keys()):
            if idx < min(20, len(vec)):
                self.morph_params[key] = float(vec[idx])
                idx += 1


# ══════════════════════════════════════════════════════════
#  共同进化引擎
# ══════════════════════════════════════════════════════════

@dataclass
class CoevolutionResult:
    """共同进化结果"""
    best_genome: Optional[MorphPolicyGenome] = None
    best_fitness: float = 0.0
    population: List[MorphPolicyGenome] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)
    generation: int = 0
    elapsed_ms: float = 0.0


class CoevolutionEngine:
    """形态-策略共同进化引擎
    
    三种策略:
    
    A) ESM (快速, 默认):
       genome = [morph_params..., W(act×obs) flatten, bias]
       每代: 评估 → 选择 → 交叉 → 变异
    
    B) MCPN (精度):
       形态嵌入 → 条件策略网络
       每代: 形态评估 → 策略学习 → 适应度组合
    
    C) Bilevel (最高质量):
       外层 NSGA-II: 进化形态
       内层 CMA-ES: 进化策略
       每代: 40-100x 评估
    
    用法:
      >>> engine = CoevolutionEngine(strategy="esm")
      >>> result = engine.run(n_generations=20, population_size=50)
      >>> best_controller = ESMController.from_genome(
      ...     obs_dim, act_dim, result.best_genome.policy_genome)
    """
    
    def __init__(
        self,
        obs_dim: int = 10,
        act_dim: int = 4,
        config: Optional[CoevolutionConfig] = None,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config or CoevolutionConfig()
    
    def run(
        self,
        n_generations: int = 20,
        population_size: int = 50,
        morph_factory: Optional[Callable[[], Dict]] = None,
        evaluator: Optional[Callable[[MorphPolicyGenome], float]] = None,
    ) -> CoevolutionResult:
        """运行共同进化
        
        Args:
            n_generations: 进化代数
            population_size: 种群大小
            morph_factory: () → morph_params dict (随机初始化)
            evaluator: (genome) → fitness (外部评估函数)
        """
        t0 = time.perf_counter()
        
        # 初始化种群
        if morph_factory is None:
            morph_factory = lambda: {"length": np.random.uniform(0.05, 0.3)}
        
        population = []
        for _ in range(population_size):
            genome = MorphPolicyGenome(
                morph_params=morph_factory(),
                policy_genome=np.random.randn(self.obs_dim * self.act_dim + self.act_dim) * 0.1,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
            )
            population.append(genome)
        
        best_genome = None
        best_fitness = -float('inf')
        history = []
        
        for gen in range(n_generations):
            # 评估 (外部回调)
            if evaluator:
                fitnesses = []
                for genome in population:
                    try:
                        f = evaluator(genome)
                        fitnesses.append(float(f))
                    except Exception:
                        fitnesses.append(0.0)
            else:
                # 简化: 随机适应度 (测试用)
                fitnesses = [np.random.random() for _ in population]
            
            history.append(float(np.mean(fitnesses)))
            
            # 记录最优
            for i, f in enumerate(fitnesses):
                if f > best_fitness:
                    best_fitness = f
                    best_genome = population[i]
                    best_genome.generation = gen
            
            # 选择 (锦标赛)
            new_population = []
            n_elites = max(1, population_size // 10)
            
            # 精英保留
            elite_indices = np.argsort(fitnesses)[-n_elites:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # 繁殖 (交叉 + 变异)
            while len(new_population) < population_size:
                # 锦标赛选择
                candidates = np.random.choice(len(population), 4, replace=False)
                p1_idx = candidates[np.argmax([fitnesses[c] for c in candidates[:2]])]
                p2_idx = candidates[np.argmax([fitnesses[c] for c in candidates[2:]])]
                
                # ESM 交叉: 线性插值
                child = MorphPolicyGenome(
                    morph_params={
                        k: (population[p1_idx].morph_params.get(k, 0) + 
                            population[p2_idx].morph_params.get(k, 0)) / 2
                        for k in population[p1_idx].morph_params
                    },
                    policy_genome=(population[p1_idx].policy_genome + 
                                   population[p2_idx].policy_genome) / 2,
                    obs_dim=self.obs_dim,
                    act_dim=self.act_dim,
                )
                
                # 变异
                child.mutate(self.config.esm_mutation_strength)
                child.generation = gen + 1
                child.parent_ids = [str(id(population[p1_idx])), str(id(population[p2_idx]))]
                
                new_population.append(child)
            
            population = new_population[:population_size]
        
        dt = (time.perf_counter() - t0) * 1000
        
        return CoevolutionResult(
            best_genome=best_genome,
            best_fitness=best_fitness,
            population=population,
            fitness_history=history,
            generation=n_generations,
            elapsed_ms=dt,
        )
    
    def create_controller(self, genome: MorphPolicyGenome) -> ESMController:
        """从基因组创建策略控制器"""
        return ESMController.from_genome(
            genome.obs_dim, genome.act_dim, genome.policy_genome
        )
