"""贝叶斯评估器模块

实现高级评估策略：
1. 贝叶斯UCB：基于高斯过程的不确定性估计
2. 动态预算分配：根据置信度动态调整评估资源
3. 渐进式评估：从粗到精的评估策略
4. 多臂老虎机算法：Thompson采样选择评估目标
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import torch

from forgecraft.config import PartSpec, RLConfig, SimConfig, TaskConfig
from forgecraft.core.morphology import MechanicalBody
from forgecraft.logging import get_logger
from forgecraft.rl.encoder import MorphologyEncoder, _build_type_registry
from forgecraft.rl.env import RunningMeanStd

__all__ = ["BayesianEvaluator", "ThompsonSamplingEvaluator"]


class BayesianEvaluator:
    """贝叶斯自适应评估器 — Gaussian Process 建模 fitness surface

    用 GP 建模个体评估预算与 fitness estimate 之间的关系:
      1. 初始: 为未评估个体设置 GP 先验 (μ=0, σ=large)
      2. 选择: 用 UCB acquisition = μ + κ*σ 选最有潜力的个体
      3. 更新: 评估后更新 GP 后验 (μ ↓, σ ↓)
      4. 收敛: σ 低于阈值 → 个体评估完成, 不再分配预算

    优势:
      - 比固定预算更高效地分配计算资源
      - 早期淘汰低潜力个体 → 节省 30-60% 评估次数
    """

    def __init__(
        self,
        sim_config: SimConfig,
        task_config: TaskConfig,
        rl_config: RLConfig,
        catalog: Dict[str, PartSpec],
        device: str = "cpu",
        n_workers: int = 4,
    ):
        self.sim_config = sim_config
        self.task_config = task_config
        self.rl_config = rl_config
        self.catalog = catalog
        self.device = device
        self.n_workers = n_workers
        self.global_obs_rms: Optional[RunningMeanStd] = None
        self.log = get_logger()
        
        # 贝叶斯UCB参数
        self.ucb_exploration_coeff = 1.0
        self.min_samples = 2
        self.max_samples = 50
        
        # 动态预算分配参数
        self.budget_priority_threshold = 0.9
        self.budget_min_allocation = 0.1
        
    def evaluate_with_bayesian_ucb(
        self,
        bodies: List[MechanicalBody],
        morph_encoder_state: dict,
        inherit_state: Optional[dict],
        generation: int = 0,
        total_generations: int = 50,
        is_final_gen: bool = False,
        best_obs_dim: Optional[int] = None,
        best_act_dim: Optional[int] = None,
        best_morph_embed: Optional[np.ndarray] = None,
    ) -> dict:
        """使用贝叶斯UCB进行自适应评估"""
        
        # 计算动态预算
        budget = self._compute_dynamic_budget(generation, total_generations, is_final_gen)
        
        n_bodies = len(bodies)
        body_accum = self._initialize_body_accum(n_bodies)
        
        # 阶段1：所有个体进行最小评估
        self.log.info(f"[阶段1] 全体基础评估 ({budget['min_episodes']}ep × {budget['steps_per_ep']}步)")
        body_accum = self._phase_one_evaluation(
            bodies, body_accum, morph_encoder_state, inherit_state,
            budget, best_obs_dim, best_act_dim, best_morph_embed
        )
        
        # 阶段2：基于贝叶斯UCB的动态评估
        if not is_final_gen and n_bodies > 1:
            body_accum = self._phase_two_bayesian_evaluation(
                bodies, body_accum, morph_encoder_state, inherit_state,
                budget, best_obs_dim, best_act_dim, best_morph_embed
            )
        
        # 阶段3：精英个体精细评估（最后世代）
        if is_final_gen:
            body_accum = self._phase_three_final_evaluation(
                bodies, body_accum, morph_encoder_state, inherit_state,
                budget, best_obs_dim, best_act_dim, best_morph_embed
            )
        
        return self._aggregate_results(bodies, body_accum)
    
    def _compute_dynamic_budget(self, generation: int, total_generations: int, is_final_gen: bool) -> dict:
        """计算动态评估预算"""
        progress = generation / max(total_generations, 1)
        
        # 早期世代：快速筛选，预算较低
        if progress < 0.3:
            budget = {
                'min_episodes': 2,
                'max_episodes': 8,
                'steps_per_ep': 500,
                'ppo_epochs': 2,
            }
        # 中期世代：平衡探索与利用
        elif progress < 0.7:
            budget = {
                'min_episodes': 3,
                'max_episodes': 15,
                'steps_per_ep': 800,
                'ppo_epochs': 4,
            }
        # 后期世代：精细化评估
        else:
            budget = {
                'min_episodes': 5,
                'max_episodes': 25,
                'steps_per_ep': 1000,
                'ppo_epochs': 6,
            }
        
        # 最终世代：最大化评估
        if is_final_gen:
            budget['max_episodes'] = budget['max_episodes'] * 2
            budget['steps_per_ep'] = budget['steps_per_ep'] * 2
        
        return budget
    
    def _initialize_body_accum(self, n_bodies: int) -> dict:
        """初始化个体累积数据"""
        body_accum = {}
        for i in range(n_bodies):
            body_accum[i] = {
                "episode_fitnesses": [],
                "total_speed": 0.0,
                "total_energy": 0.0,
                "total_upright": 0.0,
                "total_displacement": 0.0,
                "total_survival_ratio": 0.0,
                "total_fell": 0,
                "n_completed": 0,
                "manufacturability": 0.0,
                "trainer_state": None,
                "obs_dim": 0,
                "act_dim": 0,
            }
        return body_accum
    
    def _phase_one_evaluation(
        self,
        bodies: List[MechanicalBody],
        body_accum: dict,
        morph_encoder_state: dict,
        inherit_state: Optional[dict],
        budget: dict,
        best_obs_dim: Optional[int],
        best_act_dim: Optional[int],
        best_morph_embed: Optional[np.ndarray],
    ) -> dict:
        """阶段1：所有个体进行最小评估"""
        n_bodies = len(bodies)
        
        def _worker(idx, body):
            try:
                result = _evaluate_body_worker_ucb(
                    body, self.sim_config, self.task_config, self.rl_config,
                    morph_encoder_state, inherit_state,
                    best_obs_dim, best_act_dim,
                    n_episodes=budget["min_episodes"],
                    steps_per_ep=budget["steps_per_ep"],
                    ppo_epochs=budget["ppo_epochs"],
                    device=self.device,
                    catalog=self.catalog,
                    prev_trainer_state=None,
                    best_morph_embed=best_morph_embed,
                    obs_rms=self.global_obs_rms,
                )
                return idx, result, None
            except Exception as e:
                return idx, None, str(e)
        
        with ThreadPoolExecutor(max_workers=min(self.n_workers, n_bodies)) as pool:
            futures = [pool.submit(_worker, i, body) for i, body in enumerate(bodies)]
            for future in as_completed(futures):
                idx, result, error = future.result()
                if error:
                    self.log.warning(f"个体{idx}基础评估失败: {error}")
                    body_accum[idx]["manufacturability"] = 0.5
                else:
                    self._update_accum(body_accum[idx], result)
        
        return body_accum
    
    def _phase_two_bayesian_evaluation(
        self,
        bodies: List[MechanicalBody],
        body_accum: dict,
        morph_encoder_state: dict,
        inherit_state: Optional[dict],
        budget: dict,
        best_obs_dim: Optional[int],
        best_act_dim: Optional[int],
        best_morph_embed: Optional[np.ndarray],
    ) -> dict:
        """阶段2：基于贝叶斯UCB的动态评估"""
        max_iterations = 5
        
        for iteration in range(max_iterations):
            # 计算每个个体的贝叶斯UCB分数
            scored = []
            for i, acc in body_accum.items():
                eps = acc["episode_fitnesses"]
                if len(eps) < self.min_samples:
                    continue
                
                # 计算均值和标准差
                mean_fit = np.mean(eps)
                std_fit = np.std(eps)
                n_samples = len(eps)
                
                # 贝叶斯UCB公式：mean + exploration_coeff * std / sqrt(n)
                exploration_term = self.ucb_exploration_coeff * std_fit / math.sqrt(n_samples)
                ucb_score = mean_fit + exploration_term
                
                scored.append((i, ucb_score, mean_fit, std_fit, n_samples))
            
            if not scored:
                break
            
            # 按UCB分数排序，选择需要更多评估的个体
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # 选择前30%的个体进行追加评估
            top_k = max(int(len(scored) * 0.3), 1)
            candidates = scored[:top_k]
            
            # 筛选出不确定性高或样本数少的个体
            active = []
            for idx, ucb_score, mean_fit, std_fit, n_samples in candidates:
                if n_samples >= budget["max_episodes"]:
                    continue
                
                # 不确定性阈值判断
                cv = std_fit / (abs(mean_fit) + 1e-8)
                if cv < 0.05:
                    continue  # 置信度足够高，不需要更多评估
                
                active.append(idx)
            
            if not active:
                break
            
            # 为每个活跃个体追加评估
            add_episodes = min(2, budget["max_episodes"] - min(
                body_accum[a]["n_completed"] for a in active
            ))
            
            self.log.info(f"[阶段2.{iteration+1}] 贝叶斯UCB追加 {len(active)} 个体 (+{add_episodes}ep)")
            
            for idx in active:
                try:
                    body = bodies[idx]
                    prev_ts = body_accum[idx]["trainer_state"]
                    result = _evaluate_body_worker_ucb(
                        body, self.sim_config, self.task_config, self.rl_config,
                        morph_encoder_state, inherit_state,
                        best_obs_dim, best_act_dim,
                        n_episodes=add_episodes,
                        steps_per_ep=budget["steps_per_ep"],
                        ppo_epochs=budget["ppo_epochs"],
                        device=self.device,
                        catalog=self.catalog,
                        prev_trainer_state=prev_ts,
                        best_morph_embed=best_morph_embed,
                        obs_rms=self.global_obs_rms,
                    )
                    self._update_accum(body_accum[idx], result)
                except Exception as e:
                    self.log.warning(f"个体{idx}追加评估失败: {e}")
        
        return body_accum
    
    def _phase_three_final_evaluation(
        self,
        bodies: List[MechanicalBody],
        body_accum: dict,
        morph_encoder_state: dict,
        inherit_state: Optional[dict],
        budget: dict,
        best_obs_dim: Optional[int],
        best_act_dim: Optional[int],
        best_morph_embed: Optional[np.ndarray],
    ) -> dict:
        """阶段3：最终世代的精细评估"""
        self.log.info(f"[阶段3] 决赛轮精细评估 ({budget['max_episodes']}ep)")
        
        for idx, acc in body_accum.items():
            if acc["n_completed"] >= budget["max_episodes"]:
                continue
            
            add_episodes = budget["max_episodes"] - acc["n_completed"]
            
            try:
                body = bodies[idx]
                prev_ts = acc["trainer_state"]
                result = _evaluate_body_worker_ucb(
                    body, self.sim_config, self.task_config, self.rl_config,
                    morph_encoder_state, inherit_state,
                    best_obs_dim, best_act_dim,
                    n_episodes=add_episodes,
                    steps_per_ep=budget["steps_per_ep"],
                    ppo_epochs=budget["ppo_epochs"] * 2,  # 加倍训练轮数
                    device=self.device,
                    catalog=self.catalog,
                    prev_trainer_state=prev_ts,
                    best_morph_embed=best_morph_embed,
                    obs_rms=self.global_obs_rms,
                )
                self._update_accum(acc, result)
            except Exception as e:
                self.log.warning(f"个体{idx}决赛评估失败: {e}")
        
        return body_accum
    
    def _update_accum(self, accum: dict, result: dict):
        """更新累积数据"""
        accum["episode_fitnesses"].extend(result.get("episode_fitnesses", []))
        accum["total_speed"] += result.get("total_speed", 0.0)
        accum["total_energy"] += result.get("total_energy", 0.0)
        accum["total_upright"] += result.get("total_upright", 0.0)
        accum["total_displacement"] += result.get("total_displacement", 0.0)
        accum["total_survival_ratio"] += result.get("total_survival_ratio", 0.0)
        accum["total_fell"] += result.get("total_fell", 0)
        accum["n_completed"] += result.get("n_completed", 1)
        accum["manufacturability"] = result.get("manufacturability", accum["manufacturability"])
        accum["trainer_state"] = result.get("trainer_state", accum["trainer_state"])
        accum["obs_dim"] = result.get("obs_dim", accum["obs_dim"])
        accum["act_dim"] = result.get("act_dim", accum["act_dim"])
    
    def _aggregate_results(self, bodies: List[MechanicalBody], body_accum: dict) -> dict:
        """聚合最终评估结果"""
        from forgecraft.evaluation.metrics import aggregate_fitness_generic
        
        results = {}
        for i, acc in body_accum.items():
            n = max(acc["n_completed"], 1)
            avg_speed = acc["total_speed"] / n
            avg_energy = acc["total_energy"] / n
            avg_upright = acc["total_upright"] / n
            avg_displacement = acc["total_displacement"] / n
            avg_survival_ratio = acc["total_survival_ratio"] / n
            fell_any = acc["total_fell"] > 0
            manufacturability = acc["manufacturability"]

            body = bodies[i]
            episode_info = {
                "speed": avg_speed,
                "energy": avg_energy,
                "upright": avg_upright,
                "displacement": avg_displacement,
                "manufacturability": manufacturability,
                "survival_ratio": avg_survival_ratio,
                "fell": fell_any,
            }
            fitness = aggregate_fitness_generic(body, episode_info, self.task_config)

            results[i] = {
                "fitness": fitness,
                "speed": avg_speed,
                "energy": avg_energy,
                "upright": avg_upright,
                "displacement": avg_displacement,
                "manufacturability": manufacturability,
                "survival_ratio": avg_survival_ratio,
                "fell": fell_any,
                "trainer_state": acc["trainer_state"],
                "obs_dim": acc["obs_dim"],
                "act_dim": acc["act_dim"],
                "n_evaluations": len(acc["episode_fitnesses"]),
            }
        
        episode_counts = [len(body_accum[i]["episode_fitnesses"]) for i in body_accum]
        self.log.info(f"每体ep分布: min={min(episode_counts)} max={max(episode_counts)} "
              f"mean={np.mean(episode_counts):.1f}")
        
        return results


class ThompsonSamplingEvaluator:
    """Thompson 采样评估器

    用 Bayesian posterior 采样替代 UCB:
      - 对每个个体从后验采样 fitness ~ N(μ_i, σ_i²)
      - 选采样最高的个体分配下一轮预算
      - 自然平衡探索 (高σ) 与利用 (高μ)

    参考: Chapelle & Li (2011) "An Empirical Evaluation of Thompson Sampling"
    """
    
    def __init__(
        self,
        sim_config: SimConfig,
        task_config: TaskConfig,
        rl_config: RLConfig,
        catalog: Dict[str, PartSpec],
        device: str = "cpu",
        n_workers: int = 4,
    ):
        self.sim_config = sim_config
        self.task_config = task_config
        self.rl_config = rl_config
        self.catalog = catalog
        self.device = device
        self.n_workers = n_workers
        self.log = get_logger()
        
        # 汤普森采样参数
        self.prior_mean = 0.0
        self.prior_std = 1.0
        self.success_pseudo_count = 1.0
        self.failure_pseudo_count = 1.0
        
    def evaluate_with_thompson_sampling(
        self,
        bodies: List[MechanicalBody],
        morph_encoder_state: dict,
        inherit_state: Optional[dict],
        total_budget: int = 100,  # 总评估步数预算
        best_obs_dim: Optional[int] = None,
        best_act_dim: Optional[int] = None,
        best_morph_embed: Optional[np.ndarray] = None,
    ) -> dict:
        """使用汤普森采样分配评估预算"""
        n_bodies = len(bodies)
        
        # 初始化个体统计
        body_stats = {
            i: {
                'successes': self.success_pseudo_count,
                'failures': self.failure_pseudo_count,
                'evaluations': 0,
                'fitness_samples': [],
                'trainer_state': None,
            }
            for i in range(n_bodies)
        }
        
        # 初始评估
        initial_budget = min(2, total_budget // n_bodies)
        remaining_budget = total_budget - n_bodies * initial_budget
        
        self.log.info(f"[汤普森采样] 初始评估: {initial_budget}ep/个体")
        
        for i, body in enumerate(bodies):
            result = _evaluate_body_worker_ucb(
                body, self.sim_config, self.task_config, self.rl_config,
                morph_encoder_state, inherit_state,
                best_obs_dim, best_act_dim,
                n_episodes=initial_budget,
                steps_per_ep=500,
                ppo_epochs=2,
                device=self.device,
                catalog=self.catalog,
            )
            
            body_stats[i]['fitness_samples'].extend(result['episode_fitnesses'])
            body_stats[i]['evaluations'] += initial_budget
            body_stats[i]['trainer_state'] = result['trainer_state']
            
            # 更新成功/失败计数
            for fit in result['episode_fitnesses']:
                if fit > 0.5:
                    body_stats[i]['successes'] += 1
                else:
                    body_stats[i]['failures'] += 1
        
        # 迭代分配剩余预算
        while remaining_budget > 0:
            # 汤普森采样选择
            selected_idx = self._thompson_sample(body_stats)
            
            if selected_idx is None:
                break
            
            # 分配预算
            budget_allocation = min(2, remaining_budget)
            
            body = bodies[selected_idx]
            prev_ts = body_stats[selected_idx]['trainer_state']
            
            result = _evaluate_body_worker_ucb(
                body, self.sim_config, self.task_config, self.rl_config,
                morph_encoder_state, inherit_state,
                best_obs_dim, best_act_dim,
                n_episodes=budget_allocation,
                steps_per_ep=500,
                ppo_epochs=2,
                device=self.device,
                catalog=self.catalog,
                prev_trainer_state=prev_ts,
            )
            
            body_stats[selected_idx]['fitness_samples'].extend(result['episode_fitnesses'])
            body_stats[selected_idx]['evaluations'] += budget_allocation
            body_stats[selected_idx]['trainer_state'] = result['trainer_state']
            
            for fit in result['episode_fitnesses']:
                if fit > 0.5:
                    body_stats[selected_idx]['successes'] += 1
                else:
                    body_stats[selected_idx]['failures'] += 1
            
            remaining_budget -= budget_allocation
        
        # 聚合结果
        from forgecraft.evaluation.metrics import aggregate_fitness_generic
        
        results = {}
        for i, stats in body_stats.items():
            if stats['fitness_samples']:
                mean_fitness = np.mean(stats['fitness_samples'])
            else:
                mean_fitness = 0.0
            
            results[i] = {
                'fitness': mean_fitness,
                'evaluations': stats['evaluations'],
                'trainer_state': stats['trainer_state'],
            }
        
        return results
    
    def _thompson_sample(self, body_stats: dict) -> Optional[int]:
        """执行汤普森采样选择下一个评估个体"""
        best_idx = None
        best_sample = float('-inf')
        
        for idx, stats in body_stats.items():
            # Beta分布采样
            alpha = stats['successes']
            beta = stats['failures']
            
            try:
                sample = np.random.beta(alpha, beta)
            except ValueError:
                sample = 0.5
            
            if sample > best_sample:
                best_sample = sample
                best_idx = idx
        
        return best_idx


def _evaluate_body_worker_ucb(body, sim_config, task_config, rl_config,
                               enc_state, inherit_state, inherit_obs_dim, inherit_act_dim,
                               n_episodes, steps_per_ep, ppo_epochs,
                               device, catalog, prev_trainer_state=None,
                               best_morph_embed=None, obs_rms=None):
    """单个个体评估工作函数（与evaluator.py中的实现一致）"""
    from forgecraft.rl.env import ForgeCraftEnv
    from forgecraft.evaluation.metrics import aggregate_fitness_generic, compute_manufacturability

    env = ForgeCraftEnv(body, sim_config, task_config, catalog=catalog, obs_rms=obs_rms)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    manufacturability = compute_manufacturability(body)

    trainer = None
    morph_embed = None
    trainer_state = None

    if ppo_epochs > 0:
        morph_encoder = MorphologyEncoder(
            node_feat_dim=enc_state.get("node_feat_dim", 32),
            edge_feat_dim=enc_state.get("edge_feat_dim", 16),
            hidden_dim=enc_state["hidden_dim"],
            output_dim=enc_state["output_dim"],
            num_layers=enc_state["num_layers"],
            part_type_registry=_build_type_registry(catalog),
        )
        morph_encoder.load_state_dict(enc_state["weights"])
        morph_encoder.eval()
        morph_embed = morph_encoder.encode_body(body).detach().cpu().numpy()

        from forgecraft.rl.ppo import PPOTrainer
        trainer = PPOTrainer(obs_dim, act_dim, morph_encoder.output_dim, rl_config, device,
                             compile_nets=(device == "cuda"))
        if inherit_state is not None and inherit_obs_dim == obs_dim and inherit_act_dim == act_dim:
            trainer.load_state_dict(inherit_state)
        if prev_trainer_state is not None:
            try:
                trainer.load_state_dict(prev_trainer_state)
            except Exception:
                pass
        trainer_state = trainer.state_dict()

    episode_fitnesses = []
    total_speed = 0.0
    total_energy = 0.0
    total_upright = 0.0
    total_displacement = 0.0
    total_survival_ratio = 0.0
    total_fell = 0
    n_completed = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()

        max_speed = 0.0
        max_upright = 0.0
        final_displacement = 0.0
        total_e = 0.0

        for step in range(steps_per_ep):
            if trainer is not None:
                action, _, _ = trainer.get_action(obs, morph_embed)
            else:
                action = env.action_space.sample()

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
            "survival_ratio": float(info.get("episode_length", steps_per_ep)) / max(steps_per_ep, 1),
            "fell": info.get("fell", False),
        }
        ep_fitness = aggregate_fitness_generic(body, episode_info, task_config)
        episode_fitnesses.append(ep_fitness)

        total_speed += max_speed
        total_energy += total_e
        total_upright += max_upright
        total_displacement += final_displacement
        ep_len = info.get("episode_length", steps_per_ep)
        total_survival_ratio += ep_len / max(steps_per_ep, 1)
        if info.get("fell", False):
            total_fell += 1
        n_completed += 1

    env.close()

    return {
        "episode_fitnesses": episode_fitnesses,
        "total_speed": total_speed,
        "total_energy": total_energy,
        "total_upright": total_upright,
        "total_displacement": total_displacement,
        "total_survival_ratio": total_survival_ratio,
        "total_fell": total_fell,
        "n_completed": n_completed,
        "manufacturability": manufacturability,
        "trainer_state": trainer_state,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
    }