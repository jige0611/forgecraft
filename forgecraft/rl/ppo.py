import copy
import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from forgecraft.config import RLConfig
from forgecraft.rl.encoder import MorphologyEncoder
from forgecraft.rl.morph_transfer import (
    MorphConditionalActor,
    MorphConditionalCritic,
    TransferLearningManager,
)

_logger = logging.getLogger(__name__)

__all__ = ["PPOBuffer", "PPOTrainer"]


class PPOBuffer:
    def __init__(self, obs_dim: int, act_dim: int, morph_dim: int, max_size: int):
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.morph = np.zeros((max_size, morph_dim), dtype=np.float32)
        self.act = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew = np.zeros(max_size, dtype=np.float32)
        self.val = np.zeros(max_size, dtype=np.float32)
        self.logp = np.zeros(max_size, dtype=np.float32)
        self.adv = np.zeros(max_size, dtype=np.float32)
        self.ret = np.zeros(max_size, dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.float32)
        self.ptr = 0
        self.max_size = max_size
        self.full = False
        self._overwrites = 0

    def store(self, obs, morph, act, rew, val, logp, done):
        if self.full and self.ptr == 0:
            self._overwrites += 1
            if self._overwrites <= 1:
                _logger.warning(
                    "PPOBuffer 已满 (%d 条)，开始覆盖旧数据。请在 store() 前调用 compute_gae() + clear()。",
                    self.max_size,
                )
        idx = self.ptr
        self.obs[idx] = obs
        self.morph[idx] = morph
        self.act[idx] = act
        self.rew[idx] = rew
        self.val[idx] = val
        self.logp[idx] = logp
        self.done[idx] = done
        self.ptr += 1
        if self.ptr >= self.max_size:
            self.full = True
            self.ptr = 0

    def get_all(self) -> Dict[str, torch.Tensor]:
        n = self.max_size if self.full else self.ptr
        if n == 0:
            return {}
        return {
            "obs": torch.tensor(self.obs[:n]),
            "morph": torch.tensor(self.morph[:n]),
            "act": torch.tensor(self.act[:n]),
            "rew": torch.tensor(self.rew[:n]),
            "val": torch.tensor(self.val[:n]),
            "logp": torch.tensor(self.logp[:n]),
            "adv": torch.tensor(self.adv[:n]),
            "ret": torch.tensor(self.ret[:n]),
            "done": torch.tensor(self.done[:n]),
        }

    def compute_gae(self, last_val: float, gamma: float, lam: float):
        """GAE (Generalized Advantage Estimation, Schulman et al. 2016)

        递归计算广义优势 A_t^GAE = sum_{k=0}^∞ (γλ)^k δ_{t+k}
        其中 δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)

        λ=0 → 一步 TD 误差 (低方差, 高偏差)
        λ=1 → Monte Carlo 回报 (高方差, 无偏差)
        通常 λ=0.95 取折中

        同时计算 return = advantage + V(s_t) 用于 Value 网络回归目标
        """
        n = self.max_size if self.full else self.ptr
        if n == 0:
            return
        gae = 0.0
        for i in reversed(range(n)):
            if i == n - 1:
                next_val = last_val
            else:
                next_val = self.val[i + 1]
            delta = self.rew[i] + gamma * next_val * (1 - self.done[i]) - self.val[i]
            gae = delta + gamma * lam * (1 - self.done[i]) * gae
            self.adv[i] = gae
            self.ret[i] = self.adv[i] + self.val[i]

    def clear(self):
        self.ptr = 0
        self.full = False
        self._overwrites = 0


class PPOTrainer:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        morph_dim: int = 64,
        config: Optional[RLConfig] = None,
        device: str = "cpu",
        compile_nets: bool = False,
        use_conditional_nets: bool = True,
    ):
        self.config = config or RLConfig()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.morph_dim = morph_dim
        self.use_conditional_nets = use_conditional_nets

        if use_conditional_nets:
            # 使用形态条件网络
            self.actor = MorphConditionalActor(
                self.obs_dim, self.act_dim, self.morph_dim, self.config.hidden_dim
            ).to(self.device)

            self.critic = MorphConditionalCritic(
                self.obs_dim, self.morph_dim, self.config.hidden_dim
            ).to(self.device)
        else:
            # 使用传统网络
            from forgecraft.rl.encoder import MorphAwareActor, MorphAwareCritic
            self.actor = MorphAwareActor(
                self.obs_dim, self.act_dim, self.morph_dim, self.config.hidden_dim
            ).to(self.device)

            self.critic = MorphAwareCritic(
                self.obs_dim, self.morph_dim, self.config.hidden_dim
            ).to(self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.config.critic_lr)

        # 余弦退火学习率调度器
        self.actor_scheduler = CosineAnnealingLR(
            self.actor_optimizer,
            T_max=self.config.train_iters * self.config.ppo_epochs,
            eta_min=self.config.actor_lr * self.config.lr_final_factor,
        )
        self.critic_scheduler = CosineAnnealingLR(
            self.critic_optimizer,
            T_max=self.config.train_iters * self.config.ppo_epochs,
            eta_min=self.config.critic_lr * self.config.lr_final_factor,
        )

        self._compiled = False
        if compile_nets and hasattr(torch, "compile"):
            try:
                self.actor = torch.compile(self.actor, mode="reduce-overhead")
                self.critic = torch.compile(self.critic, mode="reduce-overhead")
                self._compiled = True
            except Exception:
                pass

        self._update_count = 0
        self._total_updates = 0
        self._initial_actor_lr = self.config.actor_lr
        self._initial_critic_lr = self.config.critic_lr
        
        # 自适应熵调度器
        self._entropy_scheduler = _AdaptiveEntropyScheduler(
            target_entropy=self.config.entropy_coef * 10,
            initial_coef=self.config.entropy_coef,
        )

    def get_action(
        self, obs: np.ndarray, morph_embed: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        morph_t = torch.tensor(morph_embed, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.inference_mode():
            if deterministic:
                mean, _ = self.actor.forward(obs_t, morph_t)
                action = mean.squeeze(0).cpu().numpy()
                log_prob = 0.0
            else:
                action, log_prob, _ = self.actor.sample(obs_t, morph_t)
                action = action.squeeze(0).cpu().numpy()
                log_prob = log_prob.item()
            value = self.critic(obs_t, morph_t).item()

        return action, log_prob, value

    def update(self, buffer: PPOBuffer) -> Dict[str, float]:
        data = buffer.get_all()
        if not data:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "approx_kl": 0.0}

        obs = data["obs"].to(self.device)
        morph = data["morph"].to(self.device)
        act = data["act"].to(self.device)
        adv = data["adv"].to(self.device)
        ret = data["ret"].to(self.device)
        logp_old = data["logp"].to(self.device)

        # 优势函数归一化（带梯度裁剪）
        # 注意: 用 unbiased=False 的方差, 否则 N=1 时 std() 为 NaN 会污染整个更新
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        adv = torch.clamp(adv, -10, 10)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.config.ppo_epochs):
            indices = torch.randperm(obs.shape[0], device=self.device)
            for start in range(0, obs.shape[0], self.config.batch_size):
                end = min(start + self.config.batch_size, obs.shape[0])
                batch_idx = indices[start:end]

                batch_obs = obs[batch_idx]
                batch_morph = morph[batch_idx]
                batch_act = act[batch_idx]
                batch_adv = adv[batch_idx]
                batch_ret = ret[batch_idx]
                batch_logp_old = logp_old[batch_idx]

                mean, std = self.actor.forward(batch_obs, batch_morph)
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(batch_act).sum(dim=-1)

                ratio = torch.exp(logp - batch_logp_old)
                
                # PPO-Clip 目标 (Schulman et al. 2017):
                #   L^CLIP = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]
                #   ratio = π_new(a|s) / π_old(a|s) — 新旧策略概率比
                #   clip 防止策略更新步长过大 (信任域约束)
                #   min(ratio*A, clipped_ratio*A) = pessimism: 保守取更小目标
                clip_adv = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch_adv
                actor_loss = -torch.min(ratio * batch_adv, clip_adv).mean()

                # 自适应熵正则化
                entropy = dist.entropy().sum(dim=-1).mean()
                total_entropy += entropy.item()
                
                if self.config.adaptive_entropy:
                    entropy_coef = self._entropy_scheduler.update(entropy.item())
                else:
                    entropy_coef = self.config.entropy_coef

                actor_loss = actor_loss - entropy_coef * entropy

                # 演员网络更新
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()

                # 评论家网络更新
                values = self.critic(batch_obs, batch_morph)
                critic_loss = F.mse_loss(values, batch_ret) * self.config.value_loss_coef

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                # KL散度估计
                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - batch_logp_old)).mean().item()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_kl += kl
                n_updates += 1
                self._total_updates += 1

                # KL散度早期停止
                if kl > self.config.target_kl * 1.5:
                    break

            # 学习率调度
            if self.config.lr_decay:
                self.actor_scheduler.step()
                self.critic_scheduler.step()

        return {
            "actor_loss": total_actor_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "approx_kl": total_kl / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "current_lr": self.actor_optimizer.param_groups[0]['lr'],
            "entropy_coef": entropy_coef,
        }

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_opt": self.actor_optimizer.state_dict(),
                "critic_opt": self.critic_optimizer.state_dict(),
                "actor_scheduler": self.actor_scheduler.state_dict(),
                "critic_scheduler": self.critic_scheduler.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        self.critic_optimizer.load_state_dict(ckpt["critic_opt"])
        if "actor_scheduler" in ckpt:
            self.actor_scheduler.load_state_dict(ckpt["actor_scheduler"])
            self.critic_scheduler.load_state_dict(ckpt["critic_scheduler"])

    def state_dict(self) -> Dict:
        return {
            "actor": copy.deepcopy(self.actor.state_dict()),
            "critic": copy.deepcopy(self.critic.state_dict()),
        }

    def load_state_dict(self, state: Dict):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])

    def inherit_from(self, parent_trainer: "PPOTrainer", noise_scale: float = 0.01):
        """从父代训练器继承参数"""
        parent_state = parent_trainer.state_dict()
        own_state = self.state_dict()

        for key in own_state:
            if key in parent_state:
                parent_params = parent_state[key]
                own_params = own_state[key]
                for name in own_params:
                    if name in parent_params and own_params[name].shape == parent_params[name].shape:
                        own_params[name] = parent_params[name] + noise_scale * torch.randn_like(own_params[name])

        self.load_state_dict(own_state)

    def smart_init_from(self, morph_embed: np.ndarray, transfer_manager: TransferLearningManager):
        """智能初始化：尝试从相似形态迁移策略"""
        source_strategy = transfer_manager.find_similar_strategy_by_embed(morph_embed)
        if source_strategy:
            similarity = transfer_manager.compute_similarity(morph_embed, source_strategy['morph_embed'])
            noise_scale = 0.02 / (similarity + 0.1)
            self.inherit_from(source_strategy['trainer'], noise_scale)
            return True
        return False


class _AdaptiveEntropyScheduler:
    """自适应熵调度器"""
    
    def __init__(self, target_entropy: float = 5.0, initial_coef: float = 0.02):
        self.target_entropy = target_entropy
        self.current_coef = initial_coef
        self.history = []
        self.window_size = 5
        
    def update(self, current_entropy: float) -> float:
        """根据当前熵更新系数"""
        self.history.append(current_entropy)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        avg_entropy = np.mean(self.history)
        
        # 比例控制调整
        error = self.target_entropy - avg_entropy
        self.current_coef += error * 0.05
        
        # 约束范围
        self.current_coef = max(self.config.entropy_min if hasattr(self, 'config') else 0.001, 
                                min(self.config.entropy_max if hasattr(self, 'config') else 0.1, 
                                    self.current_coef))
        
        return self.current_coef