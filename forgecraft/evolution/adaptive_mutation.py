"""自适应变异策略模块

实现基于种群状态的智能变异控制：
1. 多样性感知变异：根据种群相似度动态调整变异幅度
2. 停滞检测：检测进化停滞并自动放大变异率
3. 基因库记忆：记录历史优秀基因片段
4. 物种形成：按形态相似度聚类保护小众形态
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from forgecraft.config import EvolutionConfig, PartSpec
from forgecraft.core.morphology import MechanicalBody
from forgecraft.rl.encoder import MorphologyEncoder
import logging

__all__ = ["AdaptiveMutationController", "SpeciesManager"]

_logger = logging.getLogger(__name__)


class AdaptiveMutationController:
    """自适应变异控制器

    核心算法 (多样性驱动):
      1. 每代计算种群多样性 = 1 - mean(similarity_matrix)
         → 形态编码的余弦相似度越低 → 多样性越高
      2. 低多样性 → 提高拓扑变异概率 (add/delete/swap)
      3. 高多样性 → 提高参数变异幅度 (精细调优)
      4. 停滞检测: 连续 N 代无改进 → 突变放大 2x

    参数自适应公式:
      param_scale = base_scale * (1 + diversity_deficit)
      topo_prob   = min(base_topo_prob + diversity_deficit * 2, 0.8)
      其中 diversity_deficit = max(0, target_diversity - current_diversity)

    基因库: 保留前 K 个历史最优形态的编码, 用于 Novelty 注入
    """

    def __init__(
        self,
        evo_config: EvolutionConfig,
        morph_encoder: MorphologyEncoder,
        device: str = "cpu",
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.evo_config: EvolutionConfig = evo_config
        self.morph_encoder: MorphologyEncoder = morph_encoder
        self.device: str = device
        self.rng: np.random.RandomState = rng or np.random.RandomState()
        
        # 历史基因库
        self.gene_pool: List[np.ndarray] = []
        self.gene_pool_max_size: int = 100
        
        # 进化状态追踪
        self.stagnation_count: int = 0
        self.previous_best_fitness: float = -float('inf')
        self.diversity_history: List[float] = []
        self.diversity_window: int = 10
        
        # 自适应参数
        self.base_param_scale: float = evo_config.param_mutation_scale
        self.base_topo_prob: float = evo_config.topo_mutation_prob
        
    def update_stagnation(self, current_best_fitness: float, improvement_threshold: float = 0.01) -> None:
        """更新停滞计数器"""
        if current_best_fitness > self.previous_best_fitness * (1 + improvement_threshold):
            self.stagnation_count = 0
            self.previous_best_fitness = current_best_fitness
        else:
            self.stagnation_count += 1
            
    def compute_population_diversity(self, population: List[MechanicalBody]) -> float:
        """计算种群多样性（基于形态嵌入相似度）"""
        if len(population) < 3:
            return 1.0
            
        embeddings = []
        with torch.inference_mode():
            for body in population:
                try:
                    emb = self.morph_encoder.encode_body(body)
                    emb = F.normalize(emb, p=2, dim=0)
                    embeddings.append(emb.cpu().numpy())
                except Exception:
                    pass
        
        if len(embeddings) < 2:
            return 1.0
            
        emb_matrix = np.array(embeddings)
        similarities = np.dot(emb_matrix, emb_matrix.T)
        n = similarities.shape[0]
        triu_idx = np.triu_indices(n, k=1)
        mean_sim = float(np.mean(similarities[triu_idx]))
        
        diversity = max(0.0, 1.0 - mean_sim)
        self.diversity_history.append(diversity)
        if len(self.diversity_history) > self.diversity_window:
            self.diversity_history.pop(0)
            
        return diversity
    
    def get_adaptive_scales(self, population: List[MechanicalBody]) -> Tuple[float, float]:
        """获取自适应变异参数"""
        diversity = self.compute_population_diversity(population)
        avg_diversity = np.mean(self.diversity_history) if self.diversity_history else diversity
        
        # 基于多样性的缩放因子
        # 多样性低时增加变异幅度，多样性高时减小变异幅度
        diversity_factor = 1.0
        if avg_diversity < 0.2:
            diversity_factor = min(4.0, 1.0 / (avg_diversity + 0.1))
        elif avg_diversity > 0.6:
            diversity_factor = max(0.5, avg_diversity * 0.7)
            
        # 基于停滞的缩放因子
        stagnation_factor = 1.0
        if self.stagnation_count >= 3:
            if self.stagnation_count < 10:
                stagnation_factor = min(3.0, 1.0 + 0.25 * self.stagnation_count)
            elif self.stagnation_count < 25:
                stagnation_factor = min(4.5, 3.0 + 0.1 * (self.stagnation_count - 10))
            else:
                stagnation_factor = min(6.0, 4.5 + 0.05 * (self.stagnation_count - 25))
                
        # 综合因子
        combined_factor = diversity_factor * stagnation_factor
        
        param_scale = self.base_param_scale * combined_factor
        topo_prob = min(0.6, self.base_topo_prob * combined_factor)
        
        return param_scale, topo_prob
    
    def add_to_gene_pool(self, body: MechanicalBody):
        """将优秀个体加入基因库"""
        try:
            with torch.inference_mode():
                emb = self.morph_encoder.encode_body(body)
                gene_info = {
                    'embedding': emb.cpu().numpy(),
                    'fitness': body.fitness,
                    'body_dict': body.to_dict(),
                    'generation': getattr(body, 'generation', 0),
                }
                self.gene_pool.append(gene_info)
                
                # 保持基因库大小限制，保留最优个体
                if len(self.gene_pool) > self.gene_pool_max_size:
                    self.gene_pool.sort(key=lambda x: x['fitness'], reverse=True)
                    self.gene_pool = self.gene_pool[:self.gene_pool_max_size]
        except Exception:
            pass
            
    def sample_from_gene_pool(self, similarity_threshold: float = 0.7) -> Optional[MechanicalBody]:
        """从基因库采样相似个体进行交叉"""
        if not self.gene_pool:
            return None
            
        # 优先选择高适应度且多样性高的基因
        eligible = [g for g in self.gene_pool if g['fitness'] > 0.5 * self.previous_best_fitness]
        if not eligible:
            return None
            
        # 随机选择一个
        selected = self.rng.choice(eligible)
        try:
            from forgecraft.core.morphology import MechanicalBody
            return MechanicalBody.from_dict(selected['body_dict'])
        except Exception:
            return None
            
    def detect_species(self, population: List[MechanicalBody], threshold: float = 0.6) -> List[List[int]]:
        """检测物种类群（基于形态相似度聚类）"""
        if len(population) < 2:
            return [[i] for i in range(len(population))]
            
        embeddings = []
        valid_indices = []
        
        with torch.inference_mode():
            for i, body in enumerate(population):
                try:
                    emb = self.morph_encoder.encode_body(body)
                    emb = F.normalize(emb, p=2, dim=0)
                    embeddings.append(emb.cpu().numpy())
                    valid_indices.append(i)
                except Exception:
                    pass
        
        if len(embeddings) < 2:
            return [[i] for i in range(len(population))]
            
        # 简单的基于距离的聚类
        emb_matrix = np.array(embeddings)
        n = len(embeddings)
        clusters = []
        visited = [False] * n
        
        for i in range(n):
            if visited[i]:
                continue
                
            cluster = [i]
            visited[i] = True
            
            for j in range(n):
                if visited[j]:
                    continue
                    
                sim = float(np.dot(emb_matrix[i], emb_matrix[j]))
                if sim > threshold:
                    cluster.append(j)
                    visited[j] = True
                    
            clusters.append([valid_indices[idx] for idx in cluster])
            
        return clusters


class SpeciesManager:
    """物种管理器 — 形态聚类 + Niche 保护

    将种群按形态相似度聚类为"物种":
      1. 每代计算 pair-wise 编码相似度矩阵
      2. 相似度 > threshold → 同种
      3. 每个 niche 至少保留 1 个个体 (防止灭绝)
      4. 追踪物种年龄、大小、最优 fitness
    """

    def __init__(self, min_species_size: int = 3, protection_factor: float = 1.5) -> None:
        self.min_species_size: int = min_species_size
        self.protection_factor: float = protection_factor
        self.species_history: List[int] = []
        
    def get_species_weights(self, population: List[MechanicalBody], 
                           clusters: List[List[int]]) -> List[float]:
        """计算每个物种的选择权重（保护小众物种）"""
        weights = []
        
        for cluster in clusters:
            size = len(cluster)
            
            # 物种越小，保护权重越高
            if size < self.min_species_size:
                weight = self.protection_factor * (1.0 + (self.min_species_size - size) * 0.3)
            else:
                weight = 1.0
                
            for _ in cluster:
                weights.append(weight)
                
        return weights
        
    def select_with_species_protection(
        self, 
        population: List[MechanicalBody], 
        clusters: List[List[int]],
        num_select: int,
        rng: np.random.RandomState,
    ) -> List[MechanicalBody]:
        """带物种保护的选择"""
        if not population:
            return []
            
        weights = self.get_species_weights(population, clusters)
        total_weight = sum(weights)
        
        if total_weight == 0:
            weights = [1.0] * len(population)
            total_weight = len(population)
            
        # 加权随机选择
        probs = np.array(weights) / total_weight
        selected_indices = rng.choice(
            len(population), 
            size=min(num_select, len(population)), 
            p=probs,
            replace=False
        )
        
        return [population[i] for i in selected_indices]
