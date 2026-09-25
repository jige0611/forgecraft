"""形态迁移学习模块

实现跨形态知识迁移：
1. 条件归一化层：用形态嵌入调制网络参数
2. 迁移学习初始化：相似形态共享策略参数
3. 课程学习：从简单到复杂任务递进
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from forgecraft.config import RLConfig
from forgecraft.core.morphology import MechanicalBody
from forgecraft.rl.encoder import MorphologyEncoder
import logging

__all__ = [
    "ConditionalLayerNorm",
    "MorphConditionalActor",
    "MorphConditionalCritic",
    "TransferLearningManager",
    "CurriculumScheduler",
    "AdaptiveEntropyScheduler",
]

_logger = logging.getLogger(__name__)



class ConditionalLayerNorm(nn.Module):
    """条件归一化层：用形态嵌入调制归一化参数"""
    
    def __init__(self, normalized_shape: int, morph_dim: int):
        super().__init__()
        self.normalized_shape = normalized_shape
        
        # 形态嵌入到归一化参数的映射
        self.gamma_mlp = nn.Sequential(
            nn.Linear(morph_dim, morph_dim // 2),
            nn.ReLU(),
            nn.Linear(morph_dim // 2, normalized_shape),
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(morph_dim, morph_dim // 2),
            nn.ReLU(),
            nn.Linear(morph_dim // 2, normalized_shape),
        )
        
        # 默认参数（用于形态无关的基线）
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x: torch.Tensor, morph_embed: torch.Tensor) -> torch.Tensor:
        # 计算条件参数
        cond_gamma = self.gamma_mlp(morph_embed) + self.gamma
        cond_beta = self.beta_mlp(morph_embed) + self.beta
        
        # 归一化
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        
        # 应用条件缩放和平移
        return cond_gamma * x_norm + cond_beta


class MorphConditionalActor(nn.Module):
    """形态条件策略网络

    输入: observation + morph_embedding (形态编码)
    架构: [obs | morph] → Shared MLP → mean_head / log_std_head
    输出: (mean, std) → 高斯策略 π(a|s,m) ~ N(μ(s,m), σ(s,m))

    与普通 Actor 的区别: 形态编码作为条件输入, 使不同形态
    共享同一网络但产生不同策略 → 实现跨形态迁移学习
    """
    
    def __init__(self, obs_dim: int, act_dim: int, morph_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.morph_dim = morph_dim
        
        # 形态嵌入处理
        self.morph_proj = nn.Sequential(
            nn.Linear(morph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 观测处理
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 条件归一化层
        self.layer_norm1 = ConditionalLayerNorm(hidden_dim, morph_dim)
        self.layer_norm2 = ConditionalLayerNorm(hidden_dim, morph_dim)
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        
        # 输出头
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )
        
        self.log_std_min = -5.0
        self.log_std_max = 2.0
        
    def forward(self, obs: torch.Tensor, morph_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 处理观测
        obs_h = self.obs_proj(obs)
        
        # 处理形态嵌入
        morph_h = self.morph_proj(morph_embed)
        
        # 融合
        x = torch.cat([obs_h, morph_h], dim=-1)
        x = self.shared(x)
        
        # 条件归一化
        x = self.layer_norm1(x, morph_embed)
        x = F.relu(x)
        
        # 输出
        mean = self.mean_head(x)
        
        x2 = self.layer_norm2(x, morph_embed)
        x2 = F.relu(x2)
        log_std = self.log_std_head(x2)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, obs: torch.Tensor, morph_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.forward(obs, morph_embed)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, mean


class MorphConditionalCritic(nn.Module):
    """形态条件价值网络"""
    
    def __init__(self, obs_dim: int, morph_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.morph_dim = morph_dim
        
        self.morph_proj = nn.Sequential(
            nn.Linear(morph_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.layer_norm = ConditionalLayerNorm(hidden_dim, morph_dim)
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, obs: torch.Tensor, morph_embed: torch.Tensor) -> torch.Tensor:
        obs_h = self.obs_proj(obs)
        morph_h = self.morph_proj(morph_embed)
        
        x = torch.cat([obs_h, morph_h], dim=-1)
        x = self.net(x)
        
        return x.squeeze(-1)


class TransferLearningManager:
    """跨形态迁移学习管理器"""
    
    def __init__(
        self,
        morph_encoder: MorphologyEncoder,
        rl_config: RLConfig,
        device: str = "cpu",
    ):
        self.morph_encoder = morph_encoder
        self.rl_config = rl_config
        self.device = device
        
        # 存储历史优秀策略
        self.strategy_memory = []
        self.max_memory_size = 20
        
        # 相似度阈值
        self.similarity_threshold = 0.6
        
    def add_strategy(self, body: MechanicalBody, trainer_state: Dict):
        """添加优秀策略到记忆库"""
        try:
            with torch.inference_mode():
                embed = self.morph_encoder.encode_body(body)
                strategy_info = {
                    'morph_embed': embed.cpu().numpy(),
                    'trainer_state': trainer_state,
                    'fitness': body.fitness,
                    'body_hash': body.hash(),
                }
                self.strategy_memory.append(strategy_info)
                
                # 保持记忆库大小限制
                if len(self.strategy_memory) > self.max_memory_size:
                    self.strategy_memory.sort(key=lambda x: x['fitness'], reverse=True)
                    self.strategy_memory = self.strategy_memory[:self.max_memory_size]
        except Exception:
            pass
            
    def find_similar_strategy(self, target_body: MechanicalBody) -> Optional[Dict]:
        """查找相似形态的策略"""
        if not self.strategy_memory:
            return None
            
        try:
            with torch.inference_mode():
                target_embed = self.morph_encoder.encode_body(target_body)
                target_embed_np = target_embed.cpu().numpy()
                
                best_match = None
                best_similarity = 0.0
                
                for strategy in self.strategy_memory:
                    stored_embed = strategy['morph_embed']
                    similarity = float(np.dot(target_embed_np, stored_embed) / 
                                     (np.linalg.norm(target_embed_np) * np.linalg.norm(stored_embed)))
                    
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = strategy
                        
                return best_match
        except Exception:
            return None
            
    def transfer_strategy(
        self,
        target_trainer,
        source_strategy: Dict,
        noise_scale: Optional[float] = None,
    ):
        """迁移策略参数"""
        if noise_scale is None:
            noise_scale = self.rl_config.entropy_coef * 0.5
            
        source_state = source_strategy['trainer_state']
        target_state = target_trainer.state_dict()
        
        for key in target_state:
            if key in source_state:
                source_params = source_state[key]
                target_params = target_state[key]
                
                for name in target_params:
                    if name in source_params and target_params[name].shape == source_params[name].shape:
                        # 迁移参数并添加噪声
                        target_params[name] = source_params[name] + noise_scale * torch.randn_like(target_params[name])
        
        target_trainer.load_state_dict(target_state)


class CurriculumScheduler:
    """课程学习调度器"""
    
    def __init__(self, stages: List[Tuple[int, Dict]]):
        """
        Args:
            stages: List of (generation, config_dict) tuples
        """
        self.stages = sorted(stages, key=lambda x: x[0])
        self.current_stage = 0
        
    def get_stage(self, generation: int) -> Dict:
        """获取当前世代对应的课程配置"""
        for i, (gen, config) in enumerate(reversed(self.stages)):
            if generation >= gen:
                self.current_stage = len(self.stages) - 1 - i
                return config
                
        return self.stages[0][1] if self.stages else {}
    
    def is_stage_transition(self, generation: int) -> bool:
        """检查是否进入新阶段"""
        new_stage = None
        for i, (gen, _) in enumerate(reversed(self.stages)):
            if generation >= gen:
                new_stage = len(self.stages) - 1 - i
                break
                
        if new_stage is None:
            new_stage = 0
            
        transition = new_stage != self.current_stage
        self.current_stage = new_stage
        return transition


class AdaptiveEntropyScheduler:
    """自适应熵调度器"""
    
    def __init__(self, target_entropy: float = 0.5, initial_coef: float = 0.02):
        self.target_entropy = target_entropy
        self.current_coef = initial_coef
        self.history = []
        
    def update(self, current_entropy: float):
        """根据当前熵更新系数"""
        self.history.append(current_entropy)
        if len(self.history) > 10:
            self.history.pop(0)
            
        avg_entropy = np.mean(self.history)
        
        # PID-like调整
        error = self.target_entropy - avg_entropy
        self.current_coef += error * 0.1
        
        # 约束范围
        self.current_coef = max(0.001, min(0.1, self.current_coef))
        
        return self.current_coef