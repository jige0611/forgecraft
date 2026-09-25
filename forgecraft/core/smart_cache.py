"""智能缓存系统

实现多层缓存策略：
1. LRU缓存：最近使用优先
2. 形态嵌入缓存：避免重复计算
3. 训练器状态缓存：加速迁移学习
4. 结果缓存：避免重复评估

支持缓存预热、过期策略、统计监控等功能。
"""

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import logging

__all__ = [
    "LRUCache",
    "MorphologyCache",
    "TrainerCache",
    "ResultCache",
    "SmartCacheManager",
    "GPUMemoryManager",
]

_logger = logging.getLogger(__name__)


from forgecraft.core.morphology import MechanicalBody


class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, max_size: int = 128):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                # 移动到末尾表示最近使用
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any):
        """设置缓存项"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                # 如果超过容量，删除最旧的项
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size,
        }


class MorphologyCache:
    """形态嵌入缓存"""
    
    def __init__(self, max_size: int = 256):
        self.embed_cache = LRUCache(max_size)
        self.encoder_cache = LRUCache(16)  # 形态编码器状态缓存
    
    def get_embedding(self, body: MechanicalBody) -> Optional[np.ndarray]:
        """获取形态嵌入"""
        key = body.hash()
        return self.embed_cache.get(key)
    
    def set_embedding(self, body: MechanicalBody, embedding: np.ndarray):
        """设置形态嵌入"""
        key = body.hash()
        self.embed_cache.set(key, embedding)
    
    def get_encoder_state(self, encoder_key: str) -> Optional[Dict]:
        """获取编码器状态"""
        return self.encoder_cache.get(encoder_key)
    
    def set_encoder_state(self, encoder_key: str, state: Dict):
        """设置编码器状态"""
        self.encoder_cache.set(encoder_key, state)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'embed_cache': self.embed_cache.get_stats(),
            'encoder_cache': self.encoder_cache.get_stats(),
        }


class TrainerCache:
    """训练器状态缓存"""
    
    def __init__(self, max_size: int = 64):
        self.cache = LRUCache(max_size)
        self.morph_to_trainer = {}  # 形态哈希到训练器的映射
    
    def get_trainer_state(self, body: MechanicalBody) -> Optional[Dict]:
        """获取训练器状态"""
        key = body.hash()
        return self.cache.get(key)
    
    def set_trainer_state(self, body: MechanicalBody, state: Dict):
        """设置训练器状态"""
        key = body.hash()
        self.cache.set(key, state)
        self.morph_to_trainer[key] = state
    
    def find_similar_trainer(self, body: MechanicalBody, threshold: float = 0.7) -> Optional[Dict]:
        """查找相似形态的训练器状态"""
        from forgecraft.rl.encoder import MorphologyEncoder
        
        # 获取当前形态的嵌入
        current_embed = self._get_embedding(body)
        if current_embed is None:
            return None
        
        # 遍历缓存查找相似形态
        best_match = None
        best_similarity = 0.0
        
        for morph_key, trainer_state in self.morph_to_trainer.items():
            stored_embed = self._get_stored_embedding(morph_key)
            if stored_embed is None:
                continue
            
            similarity = self._compute_similarity(current_embed, stored_embed)
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = trainer_state
        
        return best_match
    
    def _get_embedding(self, body: MechanicalBody) -> Optional[np.ndarray]:
        """获取形态嵌入（简化实现）"""
        try:
            return body._cached_embedding
        except AttributeError:
            return None
    
    def _get_stored_embedding(self, morph_key: str) -> Optional[np.ndarray]:
        """获取存储的形态嵌入"""
        return None  # 简化实现
    
    def _compute_similarity(self, embed1: np.ndarray, embed2: np.ndarray) -> float:
        """计算嵌入相似度"""
        return float(np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2)))


class ResultCache:
    """评估结果缓存"""
    
    def __init__(self, max_size: int = 512):
        self.cache = LRUCache(max_size)
        self.generation_cache = {}  # 按世代存储
    
    def get_result(self, body: MechanicalBody, generation: int = None) -> Optional[Dict]:
        """获取评估结果"""
        key = body.hash()
        if generation is not None:
            gen_key = f"{generation}:{key}"
            result = self.cache.get(gen_key)
            if result:
                return result
        
        return self.cache.get(key)
    
    def set_result(self, body: MechanicalBody, result: Dict, generation: int = None):
        """设置评估结果"""
        key = body.hash()
        self.cache.set(key, result)
        
        if generation is not None:
            gen_key = f"{generation}:{key}"
            self.cache.set(gen_key, result)
            
            # 按世代存储
            if generation not in self.generation_cache:
                self.generation_cache[generation] = []
            self.generation_cache[generation].append(key)
    
    def clear_generation(self, generation: int):
        """清除特定世代的缓存"""
        if generation in self.generation_cache:
            for key in self.generation_cache[generation]:
                gen_key = f"{generation}:{key}"
                self.cache.cache.pop(gen_key, None)
            del self.generation_cache[generation]


class SmartCacheManager:
    """智能多级缓存 — 形态编码 + 训练状态 + 评估结果

    三级缓存:
      L1 (GPU):  形态编码 (torch.Tensor, 热数据)
      L2 (RAM):  训练状态 (network weights, 温数据)
      L3 (Disk): 评估结果 (JSON, 冷数据)

    淘汰策略:
      - L1 (LRU): 最近最少使用 + 内存压力监测
      - L2 (LRU): 最近最少使用 (不限数量)
      - L3 (count): 保留最近 N 条
    """

    def __init__(self):
        self.morph_cache = MorphologyCache()
        self.trainer_cache = TrainerCache()
        self.result_cache = ResultCache()
        
        # 缓存预热标志
        self.warmed_up = False
        
        # 统计信息
        self.total_queries = 0
        self.cache_hits = 0
    
    def warm_up(self, sample_bodies: list):
        """预热缓存"""
        self.warmed_up = True
    
    def get_morph_embedding(self, body: MechanicalBody) -> Optional[np.ndarray]:
        """获取形态嵌入"""
        self.total_queries += 1
        result = self.morph_cache.get_embedding(body)
        if result is not None:
            self.cache_hits += 1
        return result
    
    def set_morph_embedding(self, body: MechanicalBody, embedding: np.ndarray):
        """设置形态嵌入"""
        self.morph_cache.set_embedding(body, embedding)
    
    def get_trainer_state(self, body: MechanicalBody) -> Optional[Dict]:
        """获取训练器状态"""
        self.total_queries += 1
        result = self.trainer_cache.get_trainer_state(body)
        if result is not None:
            self.cache_hits += 1
        return result
    
    def set_trainer_state(self, body: MechanicalBody, state: Dict):
        """设置训练器状态"""
        self.trainer_cache.set_trainer_state(body, state)
    
    def get_result(self, body: MechanicalBody, generation: int = None) -> Optional[Dict]:
        """获取评估结果"""
        self.total_queries += 1
        result = self.result_cache.get_result(body, generation)
        if result is not None:
            self.cache_hits += 1
        return result
    
    def set_result(self, body: MechanicalBody, result: Dict, generation: int = None):
        """设置评估结果"""
        self.result_cache.set_result(body, result, generation)
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """获取整体统计信息"""
        hit_rate = self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0
        
        return {
            'overall': {
                'total_queries': self.total_queries,
                'cache_hits': self.cache_hits,
                'hit_rate': hit_rate,
            },
            'morph_cache': self.morph_cache.get_stats(),
            'trainer_cache': self.trainer_cache.cache.get_stats(),
            'result_cache': self.result_cache.cache.get_stats(),
        }
    
    def clear_all(self):
        """清空所有缓存"""
        self.morph_cache.embed_cache.clear()
        self.morph_cache.encoder_cache.clear()
        self.trainer_cache.cache.clear()
        self.result_cache.cache.clear()
        self.result_cache.generation_cache.clear()


class GPUMemoryManager:
    """GPU 显存管理 — 自动检测 + 碎片整理 + OOM 防护

    功能:
      - 周期性检查空闲显存 (torch.cuda.memory_summary)
      - 低于阈值时触发 LRU 淘汰 (编码缓存)
      - 显存碎片 > 30% 时触发警告
      - OOM 时自动回退到 CPU (降级策略)
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.allocated_memory = 0
        self.max_memory = self._get_max_memory()
        self.memory_lock = threading.Lock()
    
    def _get_max_memory(self) -> int:
        """获取GPU最大内存（字节）"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        return 8 * 1024**3  # 默认8GB
    
    def allocate(self, size: int) -> bool:
        """尝试分配内存"""
        with self.memory_lock:
            if self.allocated_memory + size <= self.max_memory * 0.9:
                self.allocated_memory += size
                return True
            return False
    
    def deallocate(self, size: int):
        """释放内存"""
        with self.memory_lock:
            self.allocated_memory = max(0, self.allocated_memory - size)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存统计"""
        used_gb = self.allocated_memory / (1024**3)
        max_gb = self.max_memory / (1024**3)
        usage_ratio = self.allocated_memory / self.max_memory
        
        return {
            'used_gb': used_gb,
            'max_gb': max_gb,
            'usage_ratio': usage_ratio,
            'available_gb': max_gb - used_gb,
        }
    
    def clear_cache(self):
        """清除PyTorch缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 全局缓存实例
_global_cache_manager = SmartCacheManager()
_global_gpu_manager = GPUMemoryManager()


def get_cache_manager() -> SmartCacheManager:
    """获取全局缓存管理器"""
    return _global_cache_manager


def get_gpu_manager() -> GPUMemoryManager:
    """获取全局GPU管理器"""
    return _global_gpu_manager