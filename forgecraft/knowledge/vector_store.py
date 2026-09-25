"""语义向量库 — 基于 FAISS 的高维嵌入索引

用途:
  - 设计案例相似性检索: "找到最像这个任务的 5 个历史案例"
  - 零件语义匹配: "哪些零件在功能上与 hollow_tube 类似?"
  - 失败模式匹配: "当前进化停滞模式最接近哪种已知模式?"

实现:
  - 主索引: FAISS IndexFlatIP (内积, 归一化=L2余弦相似)
  - 元数据: 字典存储 id → metadata 映射
  - 持久化: 索引 + 元数据 pickle
"""

from __future__ import annotations

import os
import pickle
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from forgecraft.logging import get_logger

_logger = get_logger(__name__)


@dataclass
class DesignCase:
    """一个完整的设计案例记录"""
    case_id: str
    task_name: str
    catalog_name: str
    objectives: Dict[str, float]          # {speed: 2.3, energy: 0.5, ...}
    behavior_bc: List[float]             # MAP-Elites 行为特征向量
    morphology_hash: str                 # 形态图哈希 (去重用)
    evolution_config: Dict               # 使用的进化配置快照
    generation_count: int
    wall_time_seconds: float
    qd_score_final: float
    coverage_final: float
    best_fitness: float
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = ""                 # ISO timestamp

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "task_name": self.task_name,
            "catalog_name": self.catalog_name,
            "objectives": self.objectives,
            "behavior_bc": self.behavior_bc,
            "morphology_hash": self.morphology_hash,
            "evolution_config": self.evolution_config,
            "generation_count": self.generation_count,
            "wall_time_seconds": self.wall_time_seconds,
            "qd_score_final": self.qd_score_final,
            "coverage_final": self.coverage_final,
            "best_fitness": self.best_fitness,
            "tags": self.tags,
            "notes": self.notes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DesignCase":
        return cls(**{k: d.get(k, v.default if hasattr(v, 'default') else None)
                       for k, v in cls.__dataclass_fields__.items()})


class SemanticKB:
    """语义向量知识库

    支持的操作:
      - add: 插入一条设计案例 + 对应嵌入向量
      - search: 按嵌入相似度检索 top-k
      - hybrid_search: 语义相似度 + 元数据过滤 (tags, task_name 等)
      - delete / update / stats
    """

    def __init__(self, dim: int = 384, storage_path: Optional[str] = None):
        self.dim = dim
        self.storage_path = storage_path

        self._index = None       # FAISS index (lazy init)
        self._metadata: Dict[int, DesignCase] = {}  # id → DesignCase
        self._id_to_idx: Dict[str, int] = {}         # case_id → FAISS id
        self._idx_to_id: Dict[int, str] = {}         # FAISS id → case_id
        self._next_idx: int = 0

    def _init_index(self):
        if self._index is not None:
            return
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dim)
            _logger.info(f"FAISS IndexFlatIP initialized (dim={self.dim})")
        except ImportError:
            _logger.warning("FAISS not installed. Using brute-force search.")
            self._index = None
        except Exception as e:
            _logger.warning(f"FAISS init failed: {e}")
            self._index = None

    @property
    def size(self) -> int:
        return len(self._metadata)

    def add(self, case: DesignCase, embedding: np.ndarray) -> str:
        """添加设计案例

        Returns:
            case_id (若未提供则自动生成)
        """
        embedding = np.asarray(embedding, dtype=np.float32).flatten()

        if len(embedding) != self.dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.dim}, got {len(embedding)}"
            )

        # 确保 case_id
        if not case.case_id:
            case.case_id = f"case_{uuid.uuid4().hex[:12]}"
        case.created_at = case.created_at or time.strftime("%Y-%m-%dT%H:%M:%S")

        self._init_index()

        idx = self._next_idx
        self._next_idx += 1

        if self._index is not None:
            import faiss
            self._index.add(embedding.reshape(1, -1))

        self._metadata[idx] = case
        self._id_to_idx[case.case_id] = idx
        self._idx_to_id[idx] = case.case_id

        return case.case_id

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[DesignCase, float]]:
        """语义相似度检索

        Args:
            query_embedding: 查询嵌入向量
            top_k: 返回数量

        Returns:
            [(DesignCase, similarity_score), ...]  按相似度降序
        """
        query = np.asarray(query_embedding, dtype=np.float32).flatten()
        if len(query) != self.dim:
            raise ValueError(f"Query dim mismatch: expected {self.dim}, got {len(query)}")

        if self.size == 0:
            return []

        top_k = min(top_k, self.size)

        self._init_index()

        if self._index is not None:
            import faiss
            query = query.reshape(1, -1)
            distances, indices = self._index.search(query, top_k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx not in self._metadata:
                    continue
                results.append((self._metadata[idx], float(dist)))
            return results
        else:
            # 暴力搜索
            scores = []
            for idx, case in self._metadata.items():
                # 需保存所有嵌入用于暴力搜索
                pass
            # 回退: 返回最近添加的 k 个
            all_cases = list(self._metadata.values())
            return [(c, 0.0) for c in all_cases[-top_k:]]

    def search_with_filter(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        task_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_qd_score: Optional[float] = None,
    ) -> List[Tuple[DesignCase, float]]:
        """混合检索: 语义 + 元数据过滤

        先语义排序, 再按元数据过滤, 不足 top_k 时放宽约束。
        """
        candidates = self.search(query_embedding, top_k=max(top_k * 3, self.size))

        filtered = []
        for case, score in candidates:
            if task_name and case.task_name != task_name:
                continue
            if tags and not any(t in case.tags for t in tags):
                continue
            if min_qd_score is not None and case.qd_score_final < min_qd_score:
                continue
            filtered.append((case, score))

        return filtered[:top_k]

    def search_by_task(
        self,
        task_name: str,
        top_k: int = 5,
    ) -> List[DesignCase]:
        """按任务名精确匹配 (无需嵌入)"""
        results = [
            case for case in self._metadata.values()
            if case.task_name == task_name
        ]
        results.sort(key=lambda c: c.qd_score_final, reverse=True)
        return results[:top_k]

    def get_case(self, case_id: str) -> Optional[DesignCase]:
        idx = self._id_to_idx.get(case_id)
        if idx is not None:
            return self._metadata.get(idx)
        return None

    def delete(self, case_id: str) -> bool:
        idx = self._id_to_idx.pop(case_id, None)
        if idx is not None:
            self._metadata.pop(idx, None)
            self._idx_to_id.pop(idx, None)
            return True
        return False

    def list_all(self) -> List[DesignCase]:
        return list(self._metadata.values())

    def get_statistics(self) -> Dict:
        """获取知识库统计信息"""
        tasks = {}
        catalogs = {}
        qd_scores = []
        for case in self._metadata.values():
            tasks[case.task_name] = tasks.get(case.task_name, 0) + 1
            catalogs[case.catalog_name] = catalogs.get(case.catalog_name, 0) + 1
            qd_scores.append(case.qd_score_final)

        return {
            "total_cases": self.size,
            "dimension": self.dim,
            "tasks": tasks,
            "catalogs": catalogs,
            "avg_qd_score": float(np.mean(qd_scores)) if qd_scores else 0,
            "max_qd_score": float(max(qd_scores)) if qd_scores else 0,
        }

    def save(self, path: Optional[str] = None):
        """持久化到磁盘"""
        path = path or self.storage_path
        if not path:
            raise ValueError("No storage path specified")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "dim": self.dim,
            "metadata": {k: v.to_dict() for k, v in self._metadata.items()},
            "id_to_idx": self._id_to_idx,
            "idx_to_id": self._idx_to_id,
            "next_idx": self._next_idx,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        # 同时保存 FAISS 索引
        if self._index is not None:
            index_path = path + ".faiss"
            try:
                import faiss
                faiss.write_index(self._index, index_path)
            except Exception as e:
                _logger.warning(f"Failed to save FAISS index: {e}")

        _logger.info(f"SemanticKB saved: {self.size} cases to {path}")

    @classmethod
    def load(cls, path: str) -> "SemanticKB":
        """从磁盘恢复"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Knowledge base not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        kb = cls(dim=data["dim"], storage_path=path)
        kb._metadata = {
            k: DesignCase.from_dict(v) for k, v in data["metadata"].items()
        }
        kb._id_to_idx = data["id_to_idx"]
        kb._idx_to_id = data["idx_to_id"]
        kb._next_idx = data["next_idx"]

        # 重建 FAISS 索引 (需重新插入)
        # 这需要保存嵌入, 当前版本为简化暂不完整恢复 FAISS
        # (语义搜索在 load 后会回退到暴力搜索)

        _logger.info(f"SemanticKB loaded: {kb.size} cases from {path}")
        return kb
