"""嵌入计算 — 将零件/任务/形态/性能映射到统一语义空间

设计选择:
  - 零件嵌入: PartSpec → 标准化文本描述 → sentence-transformer
  - 任务嵌入: TaskConfig → 自然语言描述 → sentence-transformer
  - 形态嵌入: MechanicalBody.graph → 复用现有 MorphologyEncoder (PyTorch GNN)
  - 性能嵌入: 多目标向量 → 轻量 MLP projection

统一维度: 768 (all-MiniLM-L6-v2 / all-mpnet-base-v2 兼容)
零外部 API 依赖 — 全部本地嵌入模型, 不联网。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from forgecraft.logging import get_logger

_logger = get_logger(__name__)

# ── 可配置常量 ──────────────────────────────────────────────

DEFAULT_EMBED_DIM = 384  # all-MiniLM-L6-v2 输出维度
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cpu"
USE_CACHE = True

# ── 嵌入缓存 ────────────────────────────────────────────────

_embed_cache: Dict[str, np.ndarray] = {}


def _cache_key(prefix: str, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{h}"


# ── 标准化文本描述 ──────────────────────────────────────────


def part_spec_to_text(part_spec: Dict) -> str:
    """将 PartSpec 字典转换为结构化自然语言描述"""
    import yaml

    # 提取关键字段, 生成 LLM 友好的描述
    fields = {}
    for key in ("name", "type", "geometry", "physics", "size"):
        if key in part_spec:
            fields[key] = part_spec[key]

    # 动作信息
    actuated = part_spec.get("actuated", False)
    joint_info = ""
    if actuated and "joint" in part_spec:
        j = part_spec["joint"]
        joint_info = (
            f" joint_type={j.get('type', 'hinge')}"
            f" axis={j.get('axis', [0, 0, 0])}"
            f" range={j.get('range', [0, 0])}"
            f" torque={j.get('torque', [0, 0])}"
        )

    # 功能标签
    functions = part_spec.get("functions", [])
    func_str = " functions=" + ",".join(functions) if functions else ""

    return (
        f"Part name={fields.get('name', 'unknown')}"
        f" geometry_type={fields.get('geometry', {}).get('type', 'box')}"
        f" actuated={actuated}{joint_info}"
        f" physics={fields.get('physics', {})}"
        f" size={fields.get('size', {})}"
        f"{func_str}"
    )


def task_config_to_text(task_config: Dict) -> str:
    """将任务配置转换为自然语言描述"""
    name = task_config.get("name", "unknown")
    desc = task_config.get("description", "").strip()

    # 提取奖励组件
    reward = task_config.get("reward", {})
    comps = reward.get("components", [])
    comp_strs = []
    for c in comps:
        comp_strs.append(f"{c.get('name', '?')}(w={c.get('weight', 1.0)}): {c.get('description', '')[:80]}")

    # 提取适应度组件
    fc = reward.get("fitness_components", [])
    ff = reward.get("fitness_formula", "")

    # 仿真环境
    sim = task_config.get("simulation", {})
    terrain = task_config.get("terrain", {})

    return (
        f"Task: {name}. "
        f"Description: {desc}. "
        f"Reward components: {'; '.join(comp_strs)}. "
        f"Fitness components: {fc}. "
        f"Fitness formula: {ff}. "
        f"Terrain: {terrain.get('type', 'flat')}. "
        f"Gravity: {sim.get('gravity', [0, 0, -9.81])}. "
        f"Max steps: {task_config.get('termination', {}).get('max_steps', 500)}."
    )


def catalog_to_text(catalog_name: str, parts: Dict) -> str:
    """将整个零件目录转换为文本描述"""
    descriptions = []
    for pname, pspec in parts.items():
        if isinstance(pspec, dict):
            descriptions.append(f"  - {pname}: {part_spec_to_text({**pspec, 'name': pname})}")
    return f"Catalog '{catalog_name}' with {len(parts)} parts:\n" + "\n".join(descriptions)


def morphology_to_text(body) -> str:
    """将 MechanicalBody 转换为文本描述"""
    from forgecraft.core.morphology import MechanicalBody

    if not isinstance(body, MechanicalBody):
        return str(body)

    parts_info = []
    for node_id in body.graph.nodes:
        part = body.get_part(node_id)
        if part is None:
            continue
        joint = body.get_joint(node_id)
        joint_str = ""
        if joint is not None:
            joint_str = f" joint={joint.type} axis={joint.axis}"
        parts_info.append(
            f"  part_id={node_id[:8]} name={part.name}"
            f" params={dict(part.params)}{joint_str}"
        )

    depth = body.max_depth() if hasattr(body, 'max_depth') else 0
    leaf_count = len(body.leaf_parts()) if hasattr(body, 'leaf_parts') else 0

    return (
        f"Morphology: {len(body.graph.nodes)} nodes, "
        f"max_depth={depth}, leaf_count={leaf_count}\n"
        + "\n".join(parts_info)
    )


# ── 嵌入类 ──────────────────────────────────────────────────


@dataclass
class PartEmbedder:
    """零件嵌入器

    将 PartSpec → 文本 → sentence-transformer 嵌入
    """

    model_name: str = EMBED_MODEL_NAME
    device: str = DEVICE
    _model: object = field(default=None, repr=False)

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            _logger.info(f"PartEmbedder loaded: {self.model_name}")
        except ImportError:
            _logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self._model = None
        except Exception as e:
            _logger.warning(f"Failed to load embedding model: {e}")
            self._model = None

    def embed(self, part_spec: Dict, use_cache: bool = USE_CACHE) -> np.ndarray:
        text = part_spec_to_text(part_spec)
        key = _cache_key("part", text)

        if use_cache and key in _embed_cache:
            return _embed_cache[key]

        self._load_model()
        if self._model is None:
            # 回退: 基于文本哈希生成确定性伪嵌入
            vec = _pseudo_embed(text, DEFAULT_EMBED_DIM)
        else:
            vec = self._model.encode([text], show_progress_bar=False)[0]
            vec = np.asarray(vec, dtype=np.float32)

        if use_cache:
            _embed_cache[key] = vec
        return vec

    def embed_batch(self, parts: List[Dict]) -> np.ndarray:
        self._load_model()
        texts = [part_spec_to_text(p) for p in parts]
        if self._model is None:
            return np.stack([_pseudo_embed(t, DEFAULT_EMBED_DIM) for t in texts])
        return np.asarray(
            self._model.encode(texts, show_progress_bar=False),
            dtype=np.float32,
        )


@dataclass
class TaskEmbedder:
    """任务嵌入器"""

    model_name: str = EMBED_MODEL_NAME
    device: str = DEVICE
    _model: object = field(default=None, repr=False)

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            self._model = None
        except Exception as e:
            _logger.warning(f"Failed to load embedding model: {e}")
            self._model = None

    def embed(self, task_config: Dict, use_cache: bool = USE_CACHE) -> np.ndarray:
        text = task_config_to_text(task_config)
        key = _cache_key("task", text)

        if use_cache and key in _embed_cache:
            return _embed_cache[key]

        self._load_model()
        if self._model is None:
            vec = _pseudo_embed(text, DEFAULT_EMBED_DIM)
        else:
            vec = self._model.encode([text], show_progress_bar=False)[0]
            vec = np.asarray(vec, dtype=np.float32)

        if use_cache:
            _embed_cache[key] = vec
        return vec

    def embed_text(self, task_description: str) -> np.ndarray:
        """直接嵌入任务描述文本 (无需 config)"""
        self._load_model()
        key = _cache_key("task_desc", task_description)
        if USE_CACHE and key in _embed_cache:
            return _embed_cache[key]

        if self._model is None:
            vec = _pseudo_embed(task_description, DEFAULT_EMBED_DIM)
        else:
            vec = self._model.encode([task_description], show_progress_bar=False)[0]
            vec = np.asarray(vec, dtype=np.float32)

        if USE_CACHE:
            _embed_cache[key] = vec
        return vec


@dataclass
class MorphologyEmbedder:
    """形态嵌入器

    复用现有 MorphologyEncoder (GNN) → 投影到统一维度。
    若 PyTorch/GNN 不可用, 回退到图拓扑特征编码。
    """

    embed_dim: int = 64
    output_dim: int = DEFAULT_EMBED_DIM
    _encoder: object = field(default=None, repr=False)
    _projection: object = field(default=None, repr=False)

    def _init_encoder(self):
        if self._encoder is not None:
            return
        try:
            import torch
            from forgecraft.rl.encoder import MorphologyEncoder, _build_type_registry

            type_registry = _build_type_registry()
            self._encoder = MorphologyEncoder(
                node_input_dim=16,
                hidden_dim=self.embed_dim,
                output_dim=self.embed_dim,
                type_registry=type_registry,
            )

            # 轻量投影层
            class Projection(torch.nn.Module):
                def __init__(self, in_dim, out_dim):
                    super().__init__()
                    self.net = torch.nn.Sequential(
                        torch.nn.Linear(in_dim, out_dim),
                        torch.nn.LayerNorm(out_dim),
                    )

                def forward(self, x):
                    return self.net(x)

            self._projection = Projection(self.embed_dim, self.output_dim)
            self._projection.eval()
            _logger.info("MorphologyEmbedder initialized (GNN encoder)")
        except ImportError as e:
            _logger.warning(f"Cannot init GNN encoder: {e}")
            self._encoder = None
            self._projection = None
        except Exception as e:
            _logger.warning(f"Failed to init morphology encoder: {e}")
            self._encoder = None
            self._projection = None

    def embed(self, body) -> np.ndarray:
        """嵌入 MechanicalBody 形态图"""
        import torch

        key = _cache_key("morph", str(id(body)))
        if USE_CACHE and key in _embed_cache:
            return _embed_cache[key]

        self._init_encoder()

        if self._encoder is not None and self._projection is not None:
            try:
                from forgecraft.rl.encoder import morphology_to_graph_data
                data = morphology_to_graph_data(body)
                with torch.no_grad():
                    gnn_out = self._encoder(data)
                    vec = self._projection(gnn_out.unsqueeze(0)).squeeze(0)
                    vec = vec.cpu().numpy().astype(np.float32)
                if USE_CACHE:
                    _embed_cache[key] = vec
                return vec
            except Exception as e:
                _logger.warning(f"GNN encoding failed, using fallback: {e}")

        # 回退: 图拓扑特征
        vec = _graph_topology_features(body)
        if USE_CACHE:
            _embed_cache[key] = vec
        return vec


@dataclass
class PerformanceEmbedder:
    """性能剖面嵌入器

    将多目标向量 → 轻量 MLP 投影到统一维度。
    支持在线学习 (每次新的 Pareto 前沿更新投影)。
    """

    output_dim: int = DEFAULT_EMBED_DIM
    _projection: Optional[np.ndarray] = field(default=None, repr=False)

    def embed(self, objectives: np.ndarray) -> np.ndarray:
        """嵌入性能向量 [n_objectives] → [output_dim]

        objectives: 1D array of objective values
        """
        objectives = np.asarray(objectives, dtype=np.float32).flatten()

        key = _cache_key("perf", objectives.tobytes().hex())
        if USE_CACHE and key in _embed_cache:
            return _embed_cache[key]

        n_obj = len(objectives)
        if n_obj == 0:
            return np.zeros(self.output_dim, dtype=np.float32)

        # 使用随机投影 (固定种子以保证确定性)
        if self._projection is None or self._projection.shape[0] != n_obj:
            rng = np.random.RandomState(42)
            raw = rng.randn(n_obj, self.output_dim) / np.sqrt(n_obj)
            self._projection = raw.astype(np.float32)

        vec = objectives @ self._projection
        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        vec = vec.astype(np.float32)
        if USE_CACHE:
            _embed_cache[key] = vec
        return vec

    def embed_batch(self, objectives: np.ndarray) -> np.ndarray:
        """批量嵌入 [batch, n_objectives]"""
        objectives = np.asarray(objectives, dtype=np.float32)
        if objectives.ndim == 1:
            objectives = objectives.reshape(1, -1)

        results = [self.embed(obj) for obj in objectives]
        return np.stack(results)


# ── 便捷函数 ────────────────────────────────────────────────

_part_embedder = None
_task_embedder = None
_morph_embedder = None
_perf_embedder = None


def _get_part_embedder() -> PartEmbedder:
    global _part_embedder
    if _part_embedder is None:
        _part_embedder = PartEmbedder()
    return _part_embedder


def _get_task_embedder() -> TaskEmbedder:
    global _task_embedder
    if _task_embedder is None:
        _task_embedder = TaskEmbedder()
    return _task_embedder


def _get_morph_embedder() -> MorphologyEmbedder:
    global _morph_embedder
    if _morph_embedder is None:
        _morph_embedder = MorphologyEmbedder()
    return _morph_embedder


def _get_perf_embedder() -> PerformanceEmbedder:
    global _perf_embedder
    if _perf_embedder is None:
        _perf_embedder = PerformanceEmbedder()
    return _perf_embedder


def embed_part(part_spec: Dict) -> np.ndarray:
    return _get_part_embedder().embed(part_spec)


def embed_task(task_config: Dict) -> np.ndarray:
    return _get_task_embedder().embed(task_config)


def embed_task_text(description: str) -> np.ndarray:
    return _get_task_embedder().embed_text(description)


def embed_morphology(body) -> np.ndarray:
    return _get_morph_embedder().embed(body)


def embed_performance(objectives: np.ndarray) -> np.ndarray:
    return _get_perf_embedder().embed(objectives)


# ── 回退: 伪嵌入 ────────────────────────────────────────────


def _pseudo_embed(text: str, dim: int) -> np.ndarray:
    """基于文本哈希的确定性伪嵌入 (无模型退化方案)"""
    import hashlib

    vec = np.zeros(dim, dtype=np.float32)
    for i in range(dim):
        h = hashlib.sha256(f"{text}:{i}".encode("utf-8")).digest()
        # 将 32 字节映射到一个 float
        val = sum(h[j] * (256 ** j) for j in range(4)) / (256 ** 4)
        vec[i] = (val - 0.5) * 2  # 归一化到 [-1, 1]

    # L2 归一化
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _graph_topology_features(body) -> np.ndarray:
    """从 MechanicalBody 图提取拓扑特征"""
    from forgecraft.core.morphology import MechanicalBody

    if not isinstance(body, MechanicalBody):
        return np.zeros(DEFAULT_EMBED_DIM, dtype=np.float32)

    g = body.graph
    n_nodes = len(g.nodes)
    n_edges = len(g.edges)
    depth = body.max_depth() if hasattr(body, 'max_depth') else 0
    leaf_count = len(body.leaf_parts()) if hasattr(body, 'leaf_parts') else 0
    actuated = len(body.actuated_joints()) if hasattr(body, 'actuated_joints') else 0

    # 度数分布
    degrees = [d for _, d in g.degree()]
    if degrees:
        deg_mean = np.mean(degrees)
        deg_std = np.std(degrees)
        deg_max = max(degrees)
    else:
        deg_mean = deg_std = deg_max = 0

    features = np.array([
        n_nodes, n_edges, depth, leaf_count, actuated,
        deg_mean, deg_std, deg_max,
        np.log1p(n_nodes), np.log1p(n_edges),
        float(actuated) / max(n_nodes, 1),
        float(leaf_count) / max(n_nodes, 1),
    ], dtype=np.float32)

    # 填充/截断到目标维度
    vec = np.zeros(DEFAULT_EMBED_DIM, dtype=np.float32)
    n_feat = min(len(features), DEFAULT_EMBED_DIM)
    vec[:n_feat] = features[:n_feat]

    # L2 归一化
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec
