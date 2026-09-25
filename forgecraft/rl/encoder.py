from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from forgecraft.core.morphology import MechanicalBody, Part

import logging

_logger = logging.getLogger(__name__)


__all__ = [
    "MorphologyEncoder",
    "MorphAwareActor",
    "MorphAwareCritic",
    "JOINT_TYPE_TO_IDX",
    "JOINT_TYPE_COUNT",
    "_build_type_registry",
]

JOINT_TYPE_TO_IDX = {
    "fixed": 0,
    "hinge": 1,
    "ball": 2,
    "slide": 3,
    "free": 4,
}

_JOINT_TYPE_COUNT = len(JOINT_TYPE_TO_IDX)


def _one_hot(idx: int, n_classes: int) -> np.ndarray:
    if n_classes <= 0:
        return np.zeros(1, dtype=np.float32)
    v = np.zeros(n_classes, dtype=np.float32)
    if 0 <= idx < n_classes:
        v[idx] = 1.0
    return v


def _build_type_registry(catalog = None):
    if catalog:
        return {t: i for i, t in enumerate(sorted(catalog.keys()))}
    return {
        "base": 0, "segment": 1, "motor": 2, "foot": 3, "wheel": 4,
    }


class MorphologyEncoder(nn.Module):
    """形态编码器 — GNN + 注意力池化

    将 MechanicalBody 的图结构编码为固定维度向量 (output_dim,):
      1. 节点特征: 零件类型 one-hot + 参数 (长度/半径/质量/扭矩)
      2. 边特征: 连接类型 one-hot + 相对位置 + 关节参数
      3. 消息传递 (K 层):
         - message: MLP([sender_h, receiver_h, edge_emb])
         - attention: 同一接收者的消息做 softmax 加权
         - update: MLP(messages_aggregated, h)
      4. 读出: 注意力池化 → MLP → (output_dim,)

    支持缓存: encode_body() 缓存特征到 body._cached_features
    """
    def __init__(
        self,
        node_feat_dim: int = 32,
        edge_feat_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 3,
        part_type_registry: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.part_type_registry = part_type_registry or {"base": 0, "segment": 1, "motor": 2, "foot": 3, "wheel": 4}

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.attn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + hidden_dim, 1),
                nn.LeakyReLU(0.2),
            )
            for _ in range(num_layers)
        ])
        self.update_layers = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def extract_features(self, body: MechanicalBody) -> Tuple[torch.Tensor, torch.Tensor, list, list]:
        """从 MechanicalBody 提取 GNN 输入特征

        Returns:
            node_feats: (N, node_feat_dim) 每个零件的嵌入
            edge_feats: (E, edge_feat_dim) 每条关节边的嵌入
            senders:    [E] 有向边起点索引列表
            receivers:  [E] 有向边终点索引列表
        """
        # 1. 按 DFS 顺序建立 node→index 映射
        nodes = list(body.graph.nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n_nodes = len(nodes)

        n_part_types = max(len(self.part_type_registry), 5)

        # 2. 构建节点特征: [onehot(类型) | 参数 | 位置]
        node_feats = np.zeros((n_nodes, self.node_feat_dim), dtype=np.float32)
        for i, n in enumerate(nodes):
            part = body.get_part(n)
            type_idx = self.part_type_registry.get(part.part_type, 0)
            type_onehot = _one_hot(type_idx, n_part_types)
            params_arr = []
            for k in ["length", "radius", "thickness", "width", "height", "mass", "max_torque", "max_velocity", "density", "friction_primary"]:
                params_arr.append(part.params.get(k, 0.0))
            pos_arr = [part.position[0], part.position[1], part.position[2]]
            feat = np.concatenate([type_onehot, params_arr, pos_arr]).astype(np.float32)
            feat = feat[:self.node_feat_dim]
            if len(feat) < self.node_feat_dim:
                feat = np.pad(feat, (0, self.node_feat_dim - len(feat)))
            node_feats[i] = feat

        # 3. 构建边特征: [onehot(关节类型) | 参数 | 锚点 | 轴]
        senders = []
        receivers = []
        edge_feats_list = []
        for u, v in body.graph.edges:
            joint = body.get_joint(u, v)
            if u in node_to_idx and v in node_to_idx:
                senders.append(node_to_idx[u])
                receivers.append(node_to_idx[v])
                type_idx = JOINT_TYPE_TO_IDX.get(joint.joint_type, 0)
                type_onehot = _one_hot(type_idx, len(JOINT_TYPE_TO_IDX))
                params_arr = []
                for k in ["damping", "range_min", "range_max", "armature"]:
                    params_arr.append(joint.params.get(k, 0.0))
                anchor = [joint.anchor[0], joint.anchor[1], joint.anchor[2]]
                axis = [joint.axis[0], joint.axis[1], joint.axis[2]]
                ef = np.concatenate([type_onehot, params_arr, anchor, axis]).astype(np.float32)
                ef = ef[:self.edge_feat_dim]
                if len(ef) < self.edge_feat_dim:
                    ef = np.pad(ef, (0, self.edge_feat_dim - len(ef)))
                edge_feats_list.append(ef)

        edge_index = torch.tensor([senders, receivers], dtype=torch.long) if senders else torch.zeros((2, 0), dtype=torch.long)
        edge_feats = torch.tensor(np.array(edge_feats_list), dtype=torch.float32) if edge_feats_list else torch.zeros((0, self.edge_feat_dim), dtype=torch.float32)
        node_feats = torch.tensor(node_feats, dtype=torch.float32)

        return node_feats, edge_feats, edge_index[0].tolist(), edge_index[1].tolist()

    def forward(self, node_feats: torch.Tensor, edge_feats: torch.Tensor, senders: List[int], receivers: List[int]) -> torch.Tensor:
        """GNN 前向传播: 消息传递 → 更新 → 注意力池化

        Pipeline: node_encode → L×[message + attention + update] → attn_pool → readout
        输出: (output_dim,) 图级别嵌入向量
        """
        # 节点/边初始编码
        h = self.node_encoder(node_feats)
        e = self.edge_encoder(edge_feats) if edge_feats.shape[0] > 0 else torch.zeros((0, self.hidden_dim), device=node_feats.device)

        n_nodes = h.shape[0]
        device = h.device

        # 多层消息传递 (默认 3 层)
        for layer_idx in range(self.num_layers):
            if len(senders) > 0:
                # 4a. 消息构建: concat(sender, receiver, edge) → MLP
                sender_h = h[senders]
                receiver_h = h[receivers]
                edge_emb = e
                msg_input = torch.cat([sender_h, receiver_h, edge_emb], dim=-1)
                msg = self.message_layers[layer_idx](msg_input)

                # 4b. 注意力聚合: per-receiver softmax
                attn_scores = self.attn_layers[layer_idx](msg_input).squeeze(-1)
                attn_weights = torch.zeros_like(attn_scores)
                receiver_tensor = torch.tensor(receivers, device=device)
                for rcvr in receiver_tensor.unique():
                    mask = receiver_tensor == rcvr
                    sub_scores = attn_scores[mask]
                    attn_weights[mask] = F.softmax(sub_scores, dim=0)

                msg = msg * attn_weights.unsqueeze(-1)
                # 4c. scatter-add 聚合: 所有指向同一节点的消息求和
                messages = torch.zeros(n_nodes, self.hidden_dim, device=device)
                messages = messages.index_add(0, receiver_tensor, msg)
            else:
                messages = torch.zeros(n_nodes, self.hidden_dim, device=device)

            # 4d. 更新节点状态: update(messages, prev_h)
            h = self.update_layers[layer_idx](messages, h)

        # 5. 注意力池化 → 图级别嵌入
        attn_scores = self.attn_pool(h)
        attn_weights = F.softmax(attn_scores, dim=0)
        global_feat = (h * attn_weights).sum(dim=0)
        return self.readout(global_feat)

    def encode_body(self, body: MechanicalBody) -> torch.Tensor:
        if body._cached_features is not None:
            node_feats, edge_feats, senders, receivers = body._cached_features
            node_feats = node_feats.clone()
            edge_feats = edge_feats.clone()
        else:
            node_feats, edge_feats, senders, receivers = self.extract_features(body)
            body._cached_features = (
                node_feats.detach().clone(),
                edge_feats.detach().clone(),
                senders,
                receivers,
            )
        device = next(self.parameters()).device
        node_feats = node_feats.to(device)
        edge_feats = edge_feats.to(device)
        return self.forward(node_feats, edge_feats, senders, receivers)


class MorphAwareActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, morph_embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = obs_dim + morph_embed_dim
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.log_std_min = -5.0
        self.log_std_max = 2.0

    def forward(self, obs: torch.Tensor, morph_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, morph_embed], dim=-1)
        h = self.shared(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, obs: torch.Tensor, morph_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.forward(obs, morph_embed)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, mean


class MorphAwareCritic(nn.Module):
    def __init__(self, obs_dim: int, morph_embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + morph_embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, morph_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, morph_embed], dim=-1)
        return self.net(x).squeeze(-1)
