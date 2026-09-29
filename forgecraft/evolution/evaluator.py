from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import sys

import numpy as np
import torch

# 子进程递归深度保护
if sys.getrecursionlimit() < 5000:
    sys.setrecursionlimit(10000)

from forgecraft.config import PartSpec, RLConfig, SimConfig, TaskConfig
from forgecraft.core.morphology import MechanicalBody
from forgecraft.logging import get_logger
from forgecraft.rl.encoder import MorphologyEncoder, _build_type_registry
from forgecraft.rl.env import RunningMeanStd

__all__ = ["PopulationEvaluator"]


class PopulationEvaluator:
    """种群评估器 — UCB 渐进式预算 (Upper Confidence Bound)

    不一次性给所有个体相同预算, 而是根据每体的"潜力"动态分配:
      1. 基础阶段: 所有个体获得 min_episodes 轮评估
      2. 选择阶段: 每个体的 UCB = mean(fitness) + c*sqrt(log(total)/count)
         → 均值高 + 评估次数少 → UCB 高 → 获得追加预算
      3. 每轮从 top-K 中再加一轮, 直到 budget 耗尽

    参考:
      - Auer et al. (2002) "Finite-time Analysis of the Multi-armed Bandit"
      - 适用于进化场景的渐进式评估策略

    决赛世代: is_final_gen=True → 所有个体强制满预算评估
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

    # ── UCB Progressive Budget — 高 fitness 个体多跑 episodes, 低 fitness 早淘汰
    def evaluate_population_ucb(
        self,
        bodies: List[MechanicalBody],
        budget: dict,
        morph_encoder_state: dict,
        inherit_state: Optional[dict],
        generation: int = 0,
        total_generations: int = 50,
        is_final_gen: bool = False,
        best_obs_dim: Optional[int] = None,
        best_act_dim: Optional[int] = None,
        best_morph_embed: Optional[np.ndarray] = None,
    ) -> dict:
        steps_per_ep = budget["steps_per_ep"]
        ppo_epochs = budget["ppo_epochs"]
        max_episodes = budget["episodes"]
        min_episodes = min(2, max_episodes)
        # UCB exploration constant: fitness_mean + c * sqrt(log(N) / n)
        ucb_c = 1.0
        # top 30% 个体多评估, 其余只跑 min_episodes
        top_k_ratio = 0.3

        if is_final_gen:
            min_episodes = max_episodes
            top_k_ratio = 1.0
            self.log.info(f"[决赛轮] 所有个体强制满预算评估 ({max_episodes}ep)")

        obs_rms = self.global_obs_rms
        if obs_rms is None:
            dummy_env = None
            try:
                from forgecraft.rl.env import ForgeCraftEnv
                dummy_env = ForgeCraftEnv(bodies[0], self.sim_config, self.task_config, catalog=self.catalog)
                obs_rms = RunningMeanStd(shape=(dummy_env.observation_space.shape[0],))
                dummy_env.close()
            except Exception:
                obs_rms = RunningMeanStd(shape=(10,))
            finally:
                if dummy_env is not None:
                    try:
                        dummy_env.close()
                    except Exception:
                        pass

        n_bodies = len(bodies)
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

        self.log.info(f"[阶段1] 全体基础评估 ({min_episodes}ep×{steps_per_ep}步, {min(self.n_workers, n_bodies)} 进程并行)")

        # 根据 worker 数选择 ThreadPool(小规模) 或 ProcessPool(大规模)
        if n_bodies >= 8 and self.n_workers >= 4:
            # 大规模评估 → ProcessPool 避开 GIL
            results_list = self.evaluate_process_pool(
                bodies, budget={"episodes": min_episodes, "steps_per_ep": steps_per_ep, "ppo_epochs": ppo_epochs},
                encoder_state=morph_encoder_state, inherit_state=inherit_state,
                best_obs_dim=best_obs_dim, best_act_dim=best_act_dim,
            )
            for idx, result in results_list.items():
                if result is None:
                    self.log.warning(f"个体{idx}基础评估失败 (ProcessPool)")
                    acc = body_accum[idx]
                    acc["manufacturability"] = 0.5
                else:
                    acc = body_accum[idx]
                    acc["episode_fitnesses"].extend(result.get("episode_fitnesses", []))
                    acc["total_speed"] += result.get("total_speed", 0.0)
                    acc["total_energy"] += result.get("total_energy", 0.0)
                    acc["total_upright"] += result.get("total_upright", 0.0)
                    acc["total_displacement"] += result.get("total_displacement", 0.0)
                    acc["total_survival_ratio"] += result.get("total_survival_ratio", 0.0)
                    acc["total_fell"] += result.get("total_fell", 0)
                    acc["n_completed"] += result.get("n_completed", 1)
                    acc["manufacturability"] = result.get("manufacturability", 0.5)
                    acc["trainer_state"] = result.get("trainer_state")
                    acc["obs_dim"] = result.get("obs_dim", 0)
                    acc["act_dim"] = result.get("act_dim", 0)
        else:
            # 小规模 → ThreadPool (低开销)
            def _phase1_worker(idx, body):
                try:
                    result = _evaluate_body_worker_ucb(
                        body, self.sim_config, self.task_config, self.rl_config,
                        morph_encoder_state, inherit_state,
                        best_obs_dim, best_act_dim,
                        n_episodes=min_episodes,
                        steps_per_ep=steps_per_ep,
                        ppo_epochs=ppo_epochs,
                        device="cpu",
                        catalog=self.catalog,
                        prev_trainer_state=None,
                        best_morph_embed=best_morph_embed,
                        obs_rms=obs_rms,
                    )
                    return idx, result, None
                except Exception as e:
                    return idx, None, str(e)

            with ThreadPoolExecutor(max_workers=min(4, n_bodies)) as pool:
                futures = [pool.submit(_phase1_worker, i, body) for i, body in enumerate(bodies)]
                for future in as_completed(futures):
                    idx, result, error = future.result()
                    if error:
                        self.log.warning(f"个体{idx}基础评估失败: {error}")
                        acc = body_accum[idx]
                        acc["manufacturability"] = 0.5
                    else:
                        acc = body_accum[idx]
                        acc["episode_fitnesses"].extend(result["episode_fitnesses"])
                        acc["total_speed"] += result["total_speed"]
                        acc["total_energy"] += result["total_energy"]
                        acc["total_upright"] += result["total_upright"]
                        acc["total_displacement"] += result["total_displacement"]
                        acc["total_survival_ratio"] += result.get("total_survival_ratio", 0.0)
                        acc["total_fell"] += result.get("total_fell", 0)
                    acc["n_completed"] += result["n_completed"]
                    acc["manufacturability"] = result["manufacturability"]
                    acc["trainer_state"] = result["trainer_state"]
                    acc["obs_dim"] = result["obs_dim"]
                    acc["act_dim"] = result["act_dim"]

        for round_num in range(1, 10):
            if is_final_gen:
                break

            scored = []
            for i, acc in body_accum.items():
                eps = acc["episode_fitnesses"]
                if len(eps) < 2:
                    continue
                mean_fit = np.mean(eps)
                std_fit = np.std(eps)
                ucb = mean_fit + ucb_c * std_fit
                scored.append((i, ucb, mean_fit, std_fit))

            if not scored:
                break

            scored.sort(key=lambda x: x[1], reverse=True)
            top_k = max(int(n_bodies * top_k_ratio), 1)
            candidates = scored[:top_k]

            active = []
            for idx, ucb, mean_fit, std_fit in candidates:
                n_eps = len(body_accum[idx]["episode_fitnesses"])
                if n_eps >= max_episodes:
                    continue
                if std_fit < 0.02 * abs(mean_fit) + 0.001:
                    continue
                active.append(idx)

            if not active:
                break

            add_episodes = min(2, max_episodes - min(
                len(body_accum[a]["episode_fitnesses"]) for a in active
            ))
            if add_episodes <= 0:
                break

            self.log.info(f"[阶段2.{round_num}] UCB追加 {len(active)} 个精英个体 (+{add_episodes}ep)")

            for idx in active:
                try:
                    body = bodies[idx]
                    prev_ts = body_accum[idx]["trainer_state"]
                    result = _evaluate_body_worker_ucb(
                        body, self.sim_config, self.task_config, self.rl_config,
                        morph_encoder_state, inherit_state,
                        best_obs_dim, best_act_dim,
                        n_episodes=add_episodes,
                        steps_per_ep=steps_per_ep,
                        ppo_epochs=ppo_epochs,
                        device="cpu",
                        catalog=self.catalog,
                        prev_trainer_state=prev_ts,
                        best_morph_embed=best_morph_embed,
                        obs_rms=obs_rms,
                    )
                    acc = body_accum[idx]
                    acc["episode_fitnesses"].extend(result["episode_fitnesses"])
                    acc["total_speed"] += result["total_speed"]
                    acc["total_energy"] += result["total_energy"]
                    acc["total_upright"] += result["total_upright"]
                    acc["total_displacement"] += result["total_displacement"]
                    acc["total_survival_ratio"] += result.get("total_survival_ratio", 0.0)
                    acc["total_fell"] += result.get("total_fell", 0)
                    acc["n_completed"] += result["n_completed"]
                    acc["trainer_state"] = result["trainer_state"]
                except Exception as e:
                    self.log.warning(f"个体{idx}追加评估失败: {e}")

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
            }

        episode_counts = [len(body_accum[i]["episode_fitnesses"]) for i in body_accum]
        self.log.info(f"每体ep分布: min={min(episode_counts)} max={max(episode_counts)} "
              f"mean={np.mean(episode_counts):.1f}")

        self.global_obs_rms = obs_rms

        return results

    def evaluate_threaded(self, bodies: List[MechanicalBody], budget: dict) -> dict:
        from forgecraft.evaluation.gpu_engine import ThreadedMuJoCoEvaluator
        evaluator = ThreadedMuJoCoEvaluator(
            self.sim_config, self.task_config, catalog=self.catalog,
            n_threads=min(self.n_workers, len(bodies)),
        )
        results_list = evaluator.evaluate_bodies_simple(
            bodies, steps_per_body=budget["steps_per_ep"],
        )
        return {i: r for i, r in enumerate(results_list)}

    # ── GPU batch evaluation (torch batch inference) ──
    def evaluate_gpu(
        self, bodies: List[MechanicalBody], budget: dict,
        encoder_state: dict, inherit_state: Optional[dict],
    ) -> dict:
        from forgecraft.evaluation.gpu_engine import GPUAcceleratedEvaluator
        evaluator = GPUAcceleratedEvaluator(
            self.sim_config, self.task_config, self.rl_config,
            catalog=self.catalog, device="cuda", compile_nets=False,
        )
        evaluator.set_morph_encoder(encoder_state)
        if inherit_state is not None:
            evaluator.set_inherit_state(inherit_state)
        budgets = [budget] * len(bodies)
        results_list = evaluator.evaluate_bodies(bodies, budgets)
        return {i: r for i, r in enumerate(results_list)}

    def evaluate_process_pool(
        self, bodies: List[MechanicalBody], budget: dict,
        encoder_state: dict, inherit_state: Optional[dict],
        best_obs_dim: Optional[int] = None,
        best_act_dim: Optional[int] = None,
    ) -> dict:
        body_dicts = [body.to_dict() for body in bodies]

        n_workers = min(self.n_workers, len(bodies))
        results = {}

        if n_workers <= 1:
            for idx, body_dict in enumerate(body_dicts):
                body = MechanicalBody.from_dict(body_dict)
                result = _evaluate_body_worker(
                    body, self.sim_config, self.task_config, self.rl_config,
                    encoder_state, inherit_state, best_obs_dim, best_act_dim,
                    budget, "cpu", self.catalog,
                )
                results[idx] = result
        else:
            tasks = [
                (
                    idx, body_dict,
                    self.sim_config, self.task_config, self.rl_config,
                    encoder_state, inherit_state, best_obs_dim, best_act_dim,
                    budget, "cpu", self.catalog,
                )
                for idx, body_dict in enumerate(body_dicts)
            ]

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_evaluate_body_worker_from_dict, *t): t[0]
                    for t in tasks
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        self.log.warning(f"个体{idx}评估失败: {e}")

        return results


def _evaluate_body_worker_from_dict(
    idx: int, body_dict: dict,
    sim_config, task_config, rl_config,
    enc_state, inherit_state, inherit_obs_dim, inherit_act_dim,
    budget, device, catalog,
):
    from forgecraft.core.morphology import MechanicalBody
    body = MechanicalBody.from_dict(body_dict)
    return _evaluate_body_worker(
        body, sim_config, task_config, rl_config,
        enc_state, inherit_state, inherit_obs_dim, inherit_act_dim,
        budget, device, catalog,
    )


def _evaluate_body_worker(body, sim_config, task_config, rl_config,
                          enc_state, inherit_state, inherit_obs_dim, inherit_act_dim,
                          budget, device, catalog, obs_rms=None) -> dict:
    """薄封装: 委托给 _evaluate_body_worker_ucb 并聚合结果。"""
    result = _evaluate_body_worker_ucb(
        body, sim_config, task_config, rl_config,
        enc_state, inherit_state, inherit_obs_dim, inherit_act_dim,
        n_episodes=budget["episodes"],
        steps_per_ep=budget["steps_per_ep"],
        ppo_epochs=budget["ppo_epochs"],
        device=device, catalog=catalog,
        prev_trainer_state=None, best_morph_embed=None, obs_rms=obs_rms,
    )
    return result


def _body_has_cycle(body: MechanicalBody) -> bool:
    """检测形态体图是否包含环（可导致仿真崩溃）— 迭代DFS，避免递归溢出"""
    if body.root_id is None:
        return False
    try:
        visited = set()
        # 用栈模拟DFS: (node_id, iterator_index)
        children_cache = {}
        stack = [(body.root_id, 0)]
        path = [body.root_id]
        path_set = {body.root_id}

        while stack:
            node_id, child_idx = stack[-1]
            if node_id not in children_cache:
                children_cache[node_id] = list(body.children_of(node_id))
            children = children_cache[node_id]

            if child_idx >= len(children):
                # 回溯
                stack.pop()
                path.pop()
                path_set.discard(node_id)
                continue

            child = children[child_idx]
            stack[-1] = (node_id, child_idx + 1)  # 推进索引

            if child in path_set:
                return True  # 发现环

            visited.add(child)
            path.append(child)
            path_set.add(child)
            stack.append((child, 0))

        return False
    except Exception:
        return False  # 保守策略：无法检测则不拦截


def _evaluate_body_worker_ucb(body, sim_config, task_config, rl_config,
                               enc_state, inherit_state, inherit_obs_dim, inherit_act_dim,
                               n_episodes, steps_per_ep, ppo_epochs,
                               device, catalog, prev_trainer_state=None,
                               best_morph_embed=None, obs_rms=None):
    from forgecraft.rl.env import ForgeCraftEnv
    from forgecraft.evaluation.metrics import aggregate_fitness_generic, compute_manufacturability

    # 预检查: 检测环状形态，直接返回零适应度
    if _body_has_cycle(body):
        return {
            "episode_fitnesses": [0.0] * n_episodes,
            "total_speed": 0.0, "total_energy": 0.0, "total_upright": 0.0,
            "total_displacement": 0.0, "total_survival_ratio": 0.0,
            "total_fell": n_episodes, "n_completed": 0,
            "manufacturability": 0.0,
            "trainer_state": None, "obs_dim": 0, "act_dim": 0,
        }

    env = ForgeCraftEnv(body, sim_config, task_config, catalog=catalog, obs_rms=obs_rms)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    manufacturability = compute_manufacturability(body)

    trainer = None
    morph_embed = None
    trainer_state = None
    buffer = None

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

        from forgecraft.rl.ppo import PPOBuffer, PPOTrainer
        trainer = PPOTrainer(obs_dim, act_dim, morph_encoder.output_dim, rl_config, device,
                             compile_nets=(device == "cuda"))
        buffer = PPOBuffer(obs_dim, act_dim, morph_encoder.output_dim, max_size=steps_per_ep)
        if inherit_state is not None and inherit_obs_dim == obs_dim and inherit_act_dim == act_dim:
            trainer.load_state_dict(inherit_state)
            if best_morph_embed is not None and morph_embed is not None:
                parent_flat = best_morph_embed.flatten()
                child_flat = morph_embed.flatten()
                pn = np.linalg.norm(parent_flat)
                cn = np.linalg.norm(child_flat)
                if pn > 1e-8 and cn > 1e-8:
                    similarity = float(np.dot(parent_flat, child_flat) / (pn * cn))
                    similarity = max(0.0, min(1.0, similarity))
                    noise_scale = 0.005 / (similarity + 0.05)
                    noise_scale = min(noise_scale, 0.1)
                    for param in trainer.actor.parameters():
                        param.data += noise_scale * torch.randn_like(param.data)
                    for param in trainer.critic.parameters():
                        param.data += noise_scale * torch.randn_like(param.data)
        if prev_trainer_state is not None:
            try:
                trainer.load_state_dict(prev_trainer_state)
            except Exception:
                pass

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
        if buffer is not None:
            buffer.clear()

        max_speed = 0.0
        max_upright = 0.0
        final_displacement = 0.0
        total_e = 0.0
        terminated = False
        truncated = False

        for step in range(steps_per_ep):
            if trainer is not None:
                action, log_prob, value = trainer.get_action(obs, morph_embed)
            else:
                action = env.action_space.sample()
                log_prob, value = 0.0, 0.0

            next_obs, reward, terminated, truncated, info = env.step(action)

            if buffer is not None:
                buffer.store(obs, morph_embed, action, float(reward), value,
                             log_prob, float(terminated or truncated))
            obs = next_obs

            max_speed = max(max_speed, info.get("speed", 0.0))
            max_upright = max(max_upright, info.get("upright", 0.0))
            total_e = info.get("total_energy", 0.0)

            if terminated:
                final_displacement = info.get("final_displacement", 0.0)
                break
            if truncated:
                break

        # ── PPO 更新: GAE 优势估计 + 策略/价值网络更新 ──
        # 单步轨迹不足以估计优势, 直接跳过 (避免无意义更新 / NaN)
        n_stored = 0
        if buffer is not None:
            n_stored = buffer.max_size if buffer.full else buffer.ptr
        if trainer is not None and n_stored >= 2:
            if terminated:
                last_val = 0.0
            else:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32,
                                         device=trainer.device).unsqueeze(0)
                    morph_t = torch.tensor(morph_embed, dtype=torch.float32,
                                           device=trainer.device).unsqueeze(0)
                    last_val = trainer.critic(obs_t, morph_t).item()
            buffer.compute_gae(last_val, rl_config.gamma, rl_config.lam)
            trainer.update(buffer)

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

    # 训练后的策略参数 (供下一代/追加评估继承)
    if trainer is not None:
        trainer_state = trainer.state_dict()

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
