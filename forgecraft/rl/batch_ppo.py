"""批量化 PPO Rollout 引擎

借鉴 CleanRL (https://github.com/vwxyzjn/cleanrl) 的单文件 PPO 实现风格，
提供批量化 rollout + advantage 计算，适用于 GPU 并行仿真后端。

核心优化:
  - N 个环境的 rollout → 1 次 policy forward (batch)
  - GAE 批量计算 (向量化)
  - 内存预分配 (零分配开销)

Usage:
    engine = BatchPPORollout(policy, env_fn, n_envs=50)
    batch = engine.rollout(num_steps=500, morph_embeddings=embeds)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

import logging

_logger = logging.getLogger(__name__)

__all__ = ["BatchPPORollout"]


class BatchPPORollout:
    """批量化 PPO Rollout 引擎

    特点:
      - 单次 policy forward 处理所有环境
      - 预分配 buffer (零动态分配)
      - 借鉴 CleanRL 的计算图组织方式

    Parameters
    ----------
    policy_fn : Callable
        创建策略网络的工厂函数: (obs_dim, act_dim, morph_dim) → Module
    env_fn : Callable
        创建环境的工厂函数: () → env (需要有 step/reset 接口)
    n_envs : int
        并行环境数。
    device : str
        运算设备。
    """

    def __init__(
        self,
        policy_fn: Callable,
        env_fn: Callable,
        n_envs: int = 50,
        device: str = "cuda",
    ) -> None:
        self.policy_fn = policy_fn
        self.env_fn = env_fn
        self.n_envs = n_envs
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

        self.policy: Optional[Any] = None
        self._built = False

    def build(
        self,
        obs_dim: int,
        act_dim: int,
        morph_dim: int,
        compile_nets: bool = False,
    ) -> None:
        """构建策略网络和 buffer

        Parameters
        ----------
        obs_dim : int
            观测维度。
        act_dim : int
            动作维度。
        morph_dim : int
            形态嵌入维度。
        compile_nets : bool
            是否使用 torch.compile 加速。
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.morph_dim = morph_dim

        self.policy = self.policy_fn(obs_dim, act_dim, morph_dim)
        self.policy = self.policy.to(self.device)

        if compile_nets and hasattr(torch, "compile"):
            try:
                self.policy = torch.compile(self.policy, mode="reduce-overhead")
            except Exception:
                pass

        self._built = True

    def set_policy_weights(self, state_dict: dict) -> None:
        """加载策略权重"""
        if self.policy is None:
            raise RuntimeError("请先调用 build()")
        self.policy.load_state_dict(state_dict)

    @torch.no_grad()
    def rollout(
        self,
        num_steps: int,
        morph_embeddings: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """批量 rollout N 个环境

        Parameters
        ----------
        num_steps : int
            rollout 步数。
        morph_embeddings : Tensor (n_envs, morph_dim)
            每个环境的形态嵌入 (预先批量计算)。
        deterministic : bool
            True = 确定性动作 (评估模式), False = 随机采样 (训练模式)。

        Returns
        -------
        batch : dict
            {obs, actions, rewards, dones, values, log_probs}
            所有 tensor 维度: (num_steps, n_envs, ...)
        """
        if not self._built:
            raise RuntimeError("请先调用 build()")

        self.policy.eval()

        # 预分配
        obs_shape = (num_steps, self.n_envs, self.obs_dim)
        act_shape = (num_steps, self.n_envs, self.act_dim)

        obs_buffer = torch.zeros(obs_shape, device=self.device)
        actions_buffer = torch.zeros(act_shape, device=self.device)
        rewards_buffer = torch.zeros(num_steps, self.n_envs, device=self.device)
        dones_buffer = torch.zeros(num_steps, self.n_envs, device=self.device)
        values_buffer = torch.zeros(num_steps, self.n_envs, device=self.device)
        log_probs_buffer = torch.zeros(num_steps, self.n_envs, device=self.device)

        # 初始观测 (简化: 零观测)
        obs = torch.zeros(self.n_envs, self.obs_dim, device=self.device)

        for step in range(num_steps):
            # 批量推理
            action, log_prob, _, value = self.policy.get_action_and_value(
                obs, morph_embeddings, deterministic=deterministic
            )

            # 存储
            obs_buffer[step] = obs
            actions_buffer[step] = action
            values_buffer[step] = value.squeeze(-1)
            log_probs_buffer[step] = log_prob

            # 模拟环境步进 (简化: 随机 reward)
            # 实际使用时替换为 Genesis batch env step
            reward = torch.randn(self.n_envs, device=self.device) * 0.01
            done = torch.zeros(self.n_envs, device=self.device)

            rewards_buffer[step] = reward
            dones_buffer[step] = done

            # 下一个观测 (简化: 保持当前)
            obs = obs + action * 0.01

        return {
            "obs": obs_buffer,
            "actions": actions_buffer,
            "rewards": rewards_buffer,
            "dones": dones_buffer,
            "values": values_buffer,
            "log_probs": log_probs_buffer,
        }

    @torch.no_grad()
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量 GAE 计算

        借鉴 CleanRL 的向量化实现，一次计算所有环境的 advantage + return。

        Parameters
        ----------
        rewards : Tensor (num_steps, n_envs)
        values : Tensor (num_steps, n_envs)
        dones : Tensor (num_steps, n_envs)
        gamma : float
            折扣因子。
        lam : float
            GAE lambda。

        Returns
        -------
        advantages : Tensor (num_steps, n_envs)
        returns : Tensor (num_steps, n_envs)
        """
        num_steps, n_envs = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        last_gae = torch.zeros(n_envs, device=rewards.device)
        last_value = torch.zeros(n_envs, device=rewards.device)

        for t in reversed(range(num_steps)):
            next_value = last_value if t == num_steps - 1 else values[t + 1]
            next_done = dones[t]
            delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
            last_gae = delta + gamma * lam * (1 - next_done) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

        # 标准化 advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def clear_cache(self) -> None:
        """清理 GPU 缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
