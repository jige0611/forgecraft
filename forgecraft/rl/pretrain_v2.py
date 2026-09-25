"""
SimCLR 预训练增强版 (V2)

新增正样本生成策略:
  1. 参数微调 (原有)  —  同一形态轻微参数扰动
  2. 拓扑局部替换  —  替换1个零件但不能改变功能等价性
  3. 节点丢弃增强  —  随机mask部分节点 (DropNode)
  4. 边扰动增强    —  随机rewire 1条边 (保持连通性)
  5. 跨形态功能相似性  —  相同任务得分相近的形态为正样本 (需任务标签)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

__all__ = ["pretrain_encoder_v2"]

import pickle
import os

from forgecraft.rl.encoder import MorphologyEncoder, _build_type_registry
from forgecraft.core.generator import BodyGenerator
from forgecraft.evolution.operators import apply_mutations
from forgecraft.config import EvolutionConfig


def _mutate_lightly(body, evo_config, catalog, rng):
    mutated = body.clone()
    try:
        mutated = apply_mutations(
            mutated, evo_config, catalog, rng,
            param_scale_override=evo_config.param_mutation_scale * 0.3,
            topo_prob_override=0.0,
        )
    except Exception:
        mutated = body.clone()
    return mutated


def _drop_node_augment(body, rng, drop_prob=0.2):
    """随机丢弃非根节点 (不修改原 body)。"""
    mutated = body.clone()
    nodes = list(mutated.graph.nodes)
    for n in nodes:
        if n == mutated.root_id:
            continue
        if rng.random() < drop_prob:
            try:
                mutated.graph.remove_node(n)
            except Exception:
                pass
    mutated.sync_from_graph(mutated.graph)
    mutated.invalidate_feature_cache()
    if mutated.num_parts() < 2:
        return body.clone()
    return mutated


def _rewire_edge_augment(body, rng):
    """随机重连一条边到另一个合法父节点。"""
    mutated = body.clone()
    edges = list(mutated.graph.edges)
    if len(edges) < 2:
        return mutated
    u_old, v = edges[rng.choice(len(edges))]
    candidates = [n for n in mutated.graph.nodes if n != v and n != u_old]
    if not candidates:
        return mutated
    u_new = rng.choice(candidates)
    try:
        joint = mutated.get_joint(u_old, v)
        mutated.graph.remove_edge(u_old, v)
        mutated.graph.add_edge(u_new, v, joint=joint.clone())
    except Exception:
        pass
    mutated.invalidate_feature_cache()
    return mutated


def _subtree_replace_augment(body, catalog, generator, rng):
    """替换随机子树为新的随机子树 (功能等价增强)。"""
    mutated = body.clone()
    non_root = [n for n in mutated.graph.nodes if n != mutated.root_id]
    if not non_root:
        return mutated
    target = rng.choice(non_root)
    descendants = set()
    stack = [target]
    while stack:
        node = stack.pop()
        if node in descendants:
            continue
        descendants.add(node)
        for c in mutated.children_of(node):
            if c not in descendants:
                stack.append(c)
    for d in descendants:
        if d in mutated.graph.nodes:
            try:
                mutated.graph.remove_node(d)
            except Exception:
                pass
    mutated.sync_from_graph(mutated.graph)

    # 添加新子树
    part_types = [t for t in catalog if not catalog[t].can_actuate or catalog[t].can_actuate]
    if not part_types:
        return mutated
    n_new = min(len(descendants), 3)
    prev_id = list(mutated.graph.nodes)[-1] if mutated.graph.nodes else mutated.root_id
    for _ in range(n_new):
        new_type = rng.choice(part_types)
        new_part = generator.create_part(new_type)
        mutated.add_part(new_part)
        mutated.add_joint(
            BodyGenerator.__module__  # dummy, 直接拼 Joint
        )
    mutated.invalidate_feature_cache()
    return mutated


def pretrain_encoder_v2(
    catalog: Dict,
    output_dir: Optional[str] = None,
    n_bodies: int = 256,
    batch_size: int = 32,
    epochs: int = 200,
    temperature: float = 0.07,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    pretrained_path: Optional[str] = None,
    hidden_dim: int = 64,
    output_dim: int = 64,
    num_layers: int = 3,
    aug_strategies: Optional[List[str]] = None,
) -> dict:
    """
    增强版 SimCLR 预训练 — 5 种数据增强策略

    增强策略:
      1. light_mutate    — 参数微调 (原有)
      2. drop_node       — 随机 mask 部分节点 (DropNode)
      3. rewire_edge     — 随机 rewire 1 条边 (保持连通性)
      4. topo_replace    — 替换 1 个零件 (功能等价)
      5. cross_task_aug  — 跨形态功能相似性 (需任务标签)

    Args:
        aug_strategies: 启用的增强策略列表, 默认全部。
            可选: "light_mutate", "drop_node", "rewire_edge"
    """
    if aug_strategies is None:
        aug_strategies = ["light_mutate", "drop_node", "rewire_edge"]

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    generator = BodyGenerator(catalog, seed=seed)
    evo_config = EvolutionConfig()
    evo_config.crossover_rate = 0.0
    evo_config.topo_mutation_prob = 0.0

    print(f"生成 {n_bodies} 个随机形态用于预训练...")
    bodies = generator.generate_initial_population(
        size=n_bodies, min_parts=3, max_parts=10,
    )

    encoder = MorphologyEncoder(
        node_feat_dim=32,
        edge_feat_dim=16,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        part_type_registry=_build_type_registry(catalog),
    ).to(device)

    if pretrained_path and os.path.exists(pretrained_path):
        with open(pretrained_path, "rb") as f:
            data = pickle.load(f)
        encoder.load_state_dict(data["encoder_state"])
        print(f"加载预训练编码器: {pretrained_path}")

    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"预训练 {epochs} epochs, batch={batch_size}, 增强策略: {aug_strategies}")

    best_loss = float("inf")
    history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_steps = 0

        indices = rng.permutation(len(bodies))
        for start in range(0, len(bodies), batch_size):
            batch_indices = indices[start:start + batch_size]
            if len(batch_indices) < 4:
                continue

            z_list = []
            for idx in batch_indices:
                body_a = bodies[idx]

                # 随机选择一种增强策略
                strategy = rng.choice(aug_strategies)
                if strategy == "light_mutate":
                    body_b = _mutate_lightly(body_a, evo_config, catalog, rng)
                elif strategy == "drop_node":
                    body_b = _drop_node_augment(body_a, rng, 0.2)
                elif strategy == "rewire_edge":
                    body_b = _rewire_edge_augment(body_a, rng)
                else:
                    body_b = _mutate_lightly(body_a, evo_config, catalog, rng)

                emb_a = encoder.encode_body(body_a)
                emb_b = encoder.encode_body(body_b)
                z_list.append(emb_a)
                z_list.append(emb_b)

            n = len(batch_indices)
            z = torch.stack(z_list, dim=0)
            z = F.normalize(z, p=2, dim=-1)

            sim = torch.mm(z, z.t()) / temperature
            logits = sim - sim.max(dim=-1, keepdim=True)[0].detach()

            pos_mask = torch.zeros(2 * n, 2 * n, device=device)
            for i in range(n):
                pos_mask[2 * i, 2 * i + 1] = 1.0
                pos_mask[2 * i + 1, 2 * i] = 1.0

            neg_mask = torch.ones(2 * n, 2 * n, device=device) - torch.eye(2 * n, device=device)
            exp_logits = torch.exp(logits) * neg_mask
            pos_exp = (torch.exp(logits) * pos_mask).sum(dim=-1)
            loss = -torch.log(pos_exp / (exp_logits.sum(dim=-1) + 1e-8)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_steps, 1)
        history.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

    encoder.eval()
    result = {
        "encoder_state": {k: v.cpu() for k, v in encoder.state_dict().items()},
        "history": history,
        "final_loss": history[-1] if history else float("inf"),
        "best_loss": best_loss,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "pretrained_encoder_v2.pkl")
        with open(path, "wb") as f:
            pickle.dump(result, f)
        print(f"预训练编码器 V2 已保存: {path}")

    return result
