"""
SimCLR 形态编码器预训练

用对比学习预训练 GNN 编码器:
  1. 生成 N 个随机形态
  2. 对每个形态做轻量变异作为正样本对
  3. InfoNCE loss + temperature=0.07 + cosine annealing
  4. 保存编码器权重到 pretrained_encoder.pkl

用户入口:
  pretrain_encoder(...)          → dict
  load_pretrained_encoder(path)  → dict
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
import pickle
import os

__all__ = ["pretrain_encoder", "load_pretrained_encoder"]

# ── SimCLR 形态编码器预训练 ──
# 用对比学习预训练 GNN 编码器:
#   1. 生成 N 个随机形态
#   2. 对每个形态做轻量变异 (参数微调) 作为正样本对
#   3. InfoNCE loss: 同形态变异体 → 相似, 不同形态 → 远离
#   4. temperature=0.07 + cosine annealing LR

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


def pretrain_encoder(
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
) -> dict:
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    generator = BodyGenerator(catalog, seed=seed)
    evo_config = EvolutionConfig()
    evo_config.crossover_rate = 0.0
    evo_config.topo_mutation_prob = 0.0

    print(f"生成 {n_bodies} 个随机形态用于预训练...")
    # ── Phase 1: Generate random body population ──
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

    best_loss = float("inf")
    history = []

    print(f"预训练 {epochs} epochs, batch_size={batch_size}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_steps = 0

        # ── Phase 2: InfoNCE training loop ──
        indices = rng.permutation(len(bodies))
        for start in range(0, len(bodies), batch_size):
            batch_indices = indices[start:start + batch_size]
            if len(batch_indices) < 4:
                continue

            # 为每个 body 构造正样本对 (原体 + 轻量变异体)
            z_list = []
            for idx in batch_indices:
                body_a = bodies[idx]
                body_b = _mutate_lightly(body_a, evo_config, catalog, rng)
                emb_a = encoder.encode_body(body_a)
                emb_b = encoder.encode_body(body_b)
                z_list.append(emb_a)
                z_list.append(emb_b)

            # 归一化 + 相似度矩阵
            n = len(batch_indices)
            z = torch.stack(z_list, dim=0)
            z = F.normalize(z, p=2, dim=-1)

            sim = torch.mm(z, z.t()) / temperature  # (2n, 2n) 余弦相似度 / τ
            logits = sim - sim.max(dim=-1, keepdim=True)[0].detach()  # 数值稳定

            # 正样本 mask: (2i, 2i+1) 和 (2i+1, 2i) 是正例
            pos_mask = torch.zeros(2 * n, 2 * n, device=device)
            for i in range(n):
                pos_mask[2 * i, 2 * i + 1] = 1.0
                pos_mask[2 * i + 1, 2 * i] = 1.0

            # InfoNCE: -log(exp(pos) / Σexp(all_neg))
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

        if (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f} lr={lr_now:.6f}")

    # ── Finalize: serialize to pickle ──
    encoder.eval()
    result = {
        "encoder_state": {k: v.cpu() for k, v in encoder.state_dict().items()},
        "history": history,
        "final_loss": history[-1] if history else float("inf"),
        "best_loss": best_loss,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "pretrained_encoder.pkl")
        with open(path, "wb") as f:
            pickle.dump(result, f)
        print(f"预训练编码器已保存: {path}")

    return result


def load_pretrained_encoder(path: str, device: str = "cpu") -> MorphologyEncoder:
    with open(path, "rb") as f:
        data = pickle.load(f)

    encoder = MorphologyEncoder(
        node_feat_dim=32,
        edge_feat_dim=16,
        hidden_dim=64,
        output_dim=64,
        num_layers=3,
        part_type_registry=_build_type_registry(None),
    ).to(device)
    encoder.load_state_dict(data["encoder_state"])
    encoder.eval()
    return encoder
