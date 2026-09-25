"""GPU加速评估引擎

实现：
1. 批量环境并行化
2. GPU优化的形态编码
3. 批量策略推断
4. 内存高效的数据处理
"""

from typing import Dict, List, Optional
import logging
import numpy as np
import torch

from forgecraft.config import PartSpec, RLConfig, SimConfig, TaskConfig
from forgecraft.core.morphology import MechanicalBody
from forgecraft.rl.encoder import MorphologyEncoder
from forgecraft.rl.ppo import PPOTrainer

__all__ = ["GPUAcceleratedEvaluator", "ThreadedMuJoCoEvaluator"]

_logger = logging.getLogger(__name__)


class GPUAcceleratedEvaluator:
    """GPU加速评估器"""
    
    def __init__(
        self,
        sim_config: SimConfig,
        task_config: TaskConfig,
        rl_config: RLConfig,
        catalog: Dict[str, PartSpec],
        device: str = "cuda",
        compile_nets: bool = True,
    ):
        self.sim_config = sim_config
        self.task_config = task_config
        self.rl_config = rl_config
        self.catalog = catalog
        self.device = torch.device(device)
        
        # 检查CUDA可用性
        if not torch.cuda.is_available() and device == "cuda":
            raise RuntimeError("CUDA不可用，请使用CPU模式")
        
        self.morph_encoder = None
        self.compile_nets = compile_nets
        
        # 缓存
        self.embedding_cache = {}
    
    def set_morph_encoder(self, encoder_state: dict):
        """设置形态编码器"""
        self.morph_encoder = MorphologyEncoder(
            node_feat_dim=encoder_state.get("node_feat_dim", 32),
            edge_feat_dim=encoder_state.get("edge_feat_dim", 16),
            hidden_dim=encoder_state["hidden_dim"],
            output_dim=encoder_state["output_dim"],
            num_layers=encoder_state["num_layers"],
        )
        self.morph_encoder.load_state_dict(encoder_state["weights"])
        self.morph_encoder = self.morph_encoder.to(self.device).eval()
        
        if self.compile_nets and hasattr(torch, "compile"):
            try:
                self.morph_encoder = torch.compile(self.morph_encoder, mode="reduce-overhead")
            except Exception:
                pass
    
    def encode_bodies(self, bodies: List[MechanicalBody]) -> torch.Tensor:
        """批量编码形态"""
        if self.morph_encoder is None:
            raise RuntimeError("形态编码器未设置")
        
        embeddings = []
        
        with torch.inference_mode():
            for body in bodies:
                # 检查缓存
                body_hash = body.hash()
                if body_hash in self.embedding_cache:
                    embed = self.embedding_cache[body_hash]
                else:
                    # 提取特征
                    node_feats, edge_feats, senders, receivers = self.morph_encoder.extract_features(body)
                    node_feats = node_feats.to(self.device)
                    edge_feats = edge_feats.to(self.device)
                    
                    # 编码
                    embed = self.morph_encoder.forward(node_feats, edge_feats, senders, receivers)
                    self.embedding_cache[body_hash] = embed.detach().cpu()
                
                embeddings.append(embed)
        
        return torch.stack(embeddings)
    
    def evaluate_bodies(
        self,
        bodies: List[MechanicalBody],
        budgets: List[dict],
        inherit_state: Optional[dict] = None,
    ) -> List[dict]:
        """并行评估多个形态"""
        n_bodies = len(bodies)
        results = []
        
        for i in range(n_bodies):
            body = bodies[i]
            budget = budgets[i]
            
            result = self._evaluate_single_body(body, budget, inherit_state)
            results.append(result)
        
        return results
    
    def _evaluate_single_body(
        self,
        body: MechanicalBody,
        budget: dict,
        inherit_state: Optional[dict] = None,
    ) -> dict:
        """评估单个形态（GPU加速版本）"""
        from forgecraft.rl.env import ForgeCraftEnv
        from forgecraft.evaluation.metrics import aggregate_fitness_generic, compute_manufacturability
        
        # 创建环境
        env = ForgeCraftEnv(body, self.sim_config, self.task_config, catalog=self.catalog)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        manufacturability = compute_manufacturability(body)
        
        # 获取形态嵌入
        body_hash = body.hash()
        if body_hash in self.embedding_cache:
            morph_embed = self.embedding_cache[body_hash].to(self.device)
        else:
            node_feats, edge_feats, senders, receivers = self.morph_encoder.extract_features(body)
            node_feats = node_feats.to(self.device)
            edge_feats = edge_feats.to(self.device)
            
            with torch.inference_mode():
                morph_embed = self.morph_encoder.forward(node_feats, edge_feats, senders, receivers)
            self.embedding_cache[body_hash] = morph_embed.detach().cpu()
        
        # 创建训练器
        trainer = PPOTrainer(
            obs_dim, act_dim, self.morph_encoder.output_dim,
            self.rl_config, str(self.device), compile_nets=self.compile_nets
        )
        
        if inherit_state is not None:
            try:
                trainer.load_state_dict(inherit_state)
            except Exception:
                pass
        
        # 评估循环
        episode_fitnesses = []
        total_speed = 0.0
        total_energy = 0.0
        total_upright = 0.0
        total_displacement = 0.0
        total_survival_ratio = 0.0
        total_fell = 0
        n_completed = 0
        
        morph_embed_np = morph_embed.detach().cpu().numpy()
        
        for episode in range(budget["episodes"]):
            obs, _ = env.reset()
            
            max_speed = 0.0
            max_upright = 0.0
            final_displacement = 0.0
            total_e = 0.0
            
            for step in range(budget["steps_per_ep"]):
                action, _, _ = trainer.get_action(obs, morph_embed_np)
                obs, reward, terminated, truncated, info = env.step(action)
                
                max_speed = max(max_speed, info.get("speed", 0.0))
                max_upright = max(max_upright, info.get("upright", 0.0))
                total_e = info.get("total_energy", 0.0)
                
                if terminated:
                    final_displacement = info.get("final_displacement", 0.0)
                    break
                if truncated:
                    break
            
            episode_info = {
                "speed": max_speed,
                "energy": total_e,
                "upright": max_upright,
                "displacement": final_displacement,
                "manufacturability": manufacturability,
                "survival_ratio": float(info.get("episode_length", budget["steps_per_ep"])) / max(budget["steps_per_ep"], 1),
                "fell": info.get("fell", False),
            }
            ep_fitness = aggregate_fitness_generic(body, episode_info, self.task_config)
            episode_fitnesses.append(ep_fitness)
            
            total_speed += max_speed
            total_energy += total_e
            total_upright += max_upright
            total_displacement += final_displacement
            ep_len = info.get("episode_length", budget["steps_per_ep"])
            total_survival_ratio += ep_len / max(budget["steps_per_ep"], 1)
            if info.get("fell", False):
                total_fell += 1
            n_completed += 1
        
        env.close()
        
        return {
            "fitness": np.mean(episode_fitnesses) if episode_fitnesses else 0.0,
            "speed": total_speed / max(n_completed, 1),
            "energy": total_energy / max(n_completed, 1),
            "upright": total_upright / max(n_completed, 1),
            "displacement": total_displacement / max(n_completed, 1),
            "manufacturability": manufacturability,
            "survival_ratio": total_survival_ratio / max(n_completed, 1),
            "fell": total_fell > 0,
            "trainer_state": trainer.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
        }


class ThreadedMuJoCoEvaluator:
    """多线程MuJoCo评估器"""
    
    def __init__(
        self,
        sim_config: SimConfig,
        task_config: TaskConfig,
        catalog: Dict[str, PartSpec],
        n_threads: int = 4,
    ):
        self.sim_config = sim_config
        self.task_config = task_config
        self.catalog = catalog
        self.n_threads = n_threads
    
    def evaluate_bodies_simple(self, bodies: List[MechanicalBody], steps_per_body: int) -> List[dict]:
        """简单评估多个形态"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def _evaluate_single(body):
            return _evaluate_body_simple(body, self.sim_config, self.task_config, self.catalog, steps_per_body)
        
        results = [None] * len(bodies)
        
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = {
                executor.submit(_evaluate_single, body): idx
                for idx, body in enumerate(bodies)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"error": str(e)}
        
        return results


def _evaluate_body_simple(body, sim_config, task_config, catalog, steps_per_body) -> dict:
    """简单评估单个形态"""
    from forgecraft.rl.env import ForgeCraftEnv
    from forgecraft.evaluation.metrics import aggregate_fitness_generic, compute_manufacturability
    
    env = ForgeCraftEnv(body, sim_config, task_config, catalog=catalog)
    
    manufacturability = compute_manufacturability(body)
    
    max_speed = 0.0
    max_upright = 0.0
    total_energy = 0.0
    final_displacement = 0.0
    
    obs, _ = env.reset()
    
    for step in range(steps_per_body):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        max_speed = max(max_speed, info.get("speed", 0.0))
        max_upright = max(max_upright, info.get("upright", 0.0))
        total_energy = info.get("total_energy", 0.0)
        
        if terminated:
            final_displacement = info.get("final_displacement", 0.0)
            break
        if truncated:
            break
    
    env.close()
    
    episode_info = {
        "speed": max_speed,
        "energy": total_energy,
        "upright": max_upright,
        "displacement": final_displacement,
        "manufacturability": manufacturability,
        "survival_ratio": float(info.get("episode_length", steps_per_body)) / max(steps_per_body, 1),
        "fell": info.get("fell", False),
    }
    fitness = aggregate_fitness_generic(body, episode_info, task_config)
    
    return {
        "fitness": fitness,
        "speed": max_speed,
        "energy": total_energy,
        "upright": max_upright,
        "displacement": final_displacement,
        "manufacturability": manufacturability,
        "fell": info.get("fell", False),
    }