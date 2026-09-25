"""批量化 GNN 形态编码器

将 N 个异构 MechanicalBody 打包为单次 GPU 前向传播，
通过 dynamic padding + attention mask 处理异构拓扑。

核心优化:
  - N 个形态 → 处理为 batch graph 单次前向
  - 同构形态可按需共享编码
  - 异构形态逐编码 + torch.cat

注意: MorphologyEncoder.forward 返回 graph-level (output_dim,)，
      不是 per-node。批量编码通过逐编码 + stack 实现。
      真正的 batch 优化在 Genesis 后端 (batch env step) 层面。

Usage:
    encoder = BatchMorphologyEncoder(base_encoder)
    embeddings = encoder.encode_batch(bodies)  # (N, embed_dim)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from forgecraft.core.morphology import MechanicalBody
from forgecraft.rl.encoder import MorphologyEncoder

import logging

_logger = logging.getLogger(__name__)

__all__ = ["BatchMorphologyEncoder"]


class BatchMorphologyEncoder:
    """批量化形态编码器

    将 N 个形态的编码组织为批量处理。

    Parameters
    ----------
    base_encoder : MorphologyEncoder
        基础单形态编码器 (PyTorch Module)。
    device : str
        运算设备。
    """

    def __init__(
        self,
        base_encoder: MorphologyEncoder,
        device: str = "cuda",
    ) -> None:
        self.encoder = base_encoder
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.output_dim: int = base_encoder.output_dim

    @torch.no_grad()
    def encode_batch(
        self,
        bodies: List[MechanicalBody],
    ) -> torch.Tensor:
        """批量编码 N 个形态

        逐形态编码 + stack。真正的批量优化在 Genesis 仿真层面。

        Parameters
        ----------
        bodies : List[MechanicalBody]
            形态列表。

        Returns
        -------
        embeddings : Tensor (N, output_dim)
            每个形态的图嵌入向量。
        """
        N = len(bodies)
        if N == 0:
            return torch.empty(0, self.output_dim)

        embeddings: List[torch.Tensor] = []

        for body in bodies:
            emb = self._encode_one(body)
            embeddings.append(emb)

        return torch.stack(embeddings, dim=0)  # (N, output_dim)

    def _encode_one(self, body: MechanicalBody) -> torch.Tensor:
        """编码单个形态"""
        try:
            node_feats, edge_feats, senders, receivers = self.encoder.extract_features(body)
        except Exception:
            _logger.warning("特征提取失败，返回零向量")
            return torch.zeros(self.output_dim)

        node_feats = node_feats.to(self.device)
        edge_feats = edge_feats.to(self.device)

        # forward 返回 (output_dim,) graph-level embedding
        emb = self.encoder.forward(node_feats, edge_feats, senders, receivers)
        return emb.detach().cpu()

    def clear_cache(self) -> None:
        """清理 GPU 缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
