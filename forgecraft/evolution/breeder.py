from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from forgecraft.config import EvolutionConfig, PartSpec
from forgecraft.core.generator import BodyGenerator
from forgecraft.core.morphology import MechanicalBody
from forgecraft.evolution.operators import apply_mutations, crossover
from forgecraft.evolution.selection import (
    compute_population_stats,
    select_elites,
    tournament_select,
    pareto_elites,
    pareto_tournament_select,
)
from forgecraft.evolution.adaptive_mutation import AdaptiveMutationController, SpeciesManager
from forgecraft.logging import get_logger
from forgecraft.rl.encoder import MorphologyEncoder

__all__ = ["PopulationBreeder"]


class PopulationBreeder:
    """种群繁殖器

    每代执行选择 → 交叉 → 变异 流水线:
      1. 选择: NSGA-II Pareto (多目标) 或 精英 + 锦标赛 (单目标)
      2. 交叉: 精英保留 + tournament select 生成子代
      3. 变异: 自适应参数缩放 + 拓扑变异 (add/delete/swap parts)
      4. 填充: 补充新随机个体维持种群规模
      5. 注入: MAP-Elites 精英 + 多样性注入

    还管理:
      - AdaptiveMutationController: 根据多样性自适应调节变异率
      - SpeciesManager: 物种追踪与 niche 保护
    """
    def __init__(
        self,
        evo_config: EvolutionConfig,
        catalog: Dict[str, PartSpec],
        generator: BodyGenerator,
        morph_encoder: MorphologyEncoder,
        device: str = "cpu",
        rng: Optional[np.random.RandomState] = None,
    ):
        self.evo_config = evo_config
        self.catalog = catalog
        self.generator = generator
        self.morph_encoder = morph_encoder
        self.device = device
        self.rng = rng or np.random.RandomState()
        self.log = get_logger()
        
        # 自适应变异控制器
        self.adaptive_controller = AdaptiveMutationController(
            evo_config=evo_config,
            morph_encoder=morph_encoder,
            device=device,
            rng=rng,
        )
        
        # 物种管理器
        self.species_manager = SpeciesManager()

    def breed(
        self,
        population: List[MechanicalBody],
        fitness_results: dict,
        generation: int,
        stagnation_count: int,
        pareto_objectives: List[str],
        pareto_directions: List[str],
        historical_best: Optional[MechanicalBody] = None,
    ) -> Tuple[List[MechanicalBody], dict, MechanicalBody]:
        """执行一代繁殖: 选择 → 交叉 → 变异 → 填充

        返回: (new_population, stats_dict, best_body)
        """
        # ═══════════════════════════════════════════════
        # 阶段 1: 精英选择 — Pareto (多目标) 或 fitness (单目标)
        # ═══════════════════════════════════════════════
        # ═══ Phase 1: Elite Selection (Pareto or single-objective) ═══
        use_pareto = len(pareto_objectives) >= 2 and pareto_directions

        if use_pareto:
            elites = pareto_elites(population, self.evo_config.elite_count,
                                     pareto_objectives, pareto_directions)
            pareto_front = pareto_elites(population, len(population),
                                           pareto_objectives, pareto_directions)
        else:
            elites = select_elites(population, self.evo_config.elite_count)
            pareto_front = []

        stats = compute_population_stats(population, pareto_objectives, pareto_directions)
        stats["generation"] = generation

        best_body = elites[0] if elites else population[0] if population else None
        
        # 更新自适应控制器的停滞状态
        self.adaptive_controller.update_stagnation(best_body.fitness if best_body else -float('inf'))
        
        # 将最佳个体加入基因库
        if best_body:
            self.adaptive_controller.add_to_gene_pool(best_body)

        # ═══ Phase 2: Seed elites + historical best ═══
        new_population = [e.clone() for e in elites]

        if historical_best is not None and len(new_population) < self.evo_config.population_size:
            hist_clone = historical_best.clone()
            hist_clone.name = f"gen{generation + 1}_hist_best"
            if self._body_is_valid(hist_clone):
                new_population.append(hist_clone)

        num_parents = self.evo_config.population_size - len(new_population)
        
        # 物种检测和保护选择
        clusters = self.adaptive_controller.detect_species(population)
        stats["num_species"] = len(clusters)
        stats["species_sizes"] = [len(c) for c in clusters]
        
        if use_pareto:
            # ═══ Phase 3: Parent Selection (species-protected or tournament) ═══
            parents = pareto_tournament_select(
                population, num_parents,
                pareto_objectives, pareto_directions,
                tournament_size=3, rng=self.rng,
            )
        else:
            # 使用带物种保护的选择
            if len(clusters) > 1 and min(len(c) for c in clusters) < 5:
                parents = self.species_manager.select_with_species_protection(
                    population, clusters, num_parents, self.rng
                )
            else:
                parents = tournament_select(
                    population, num_parents,
                    tournament_size=3, rng=self.rng,
                )

        # 获取自适应变异参数
        adaptive_scale, adaptive_topo_prob = self.adaptive_controller.get_adaptive_scales(population)
        
        # 记录自适应参数到stats
        stats["adaptive_param_scale"] = adaptive_scale
        stats["adaptive_topo_prob"] = adaptive_topo_prob
        stats["population_diversity"] = self.adaptive_controller.compute_population_diversity(population)
        
        if adaptive_scale != self.evo_config.param_mutation_scale:
            self.log.adaptive(f"自适应变异: 参数缩放={adaptive_scale:.3f}, 拓扑概率={adaptive_topo_prob:.3f}")

        # ═══ Phase 4: Crossover + Mutation + Gene Pool Injection ═══
        for i in range(0, len(parents) - 1, 2):
            parent_a = parents[i]
            parent_b = parents[i + 1] if i + 1 < len(parents) else parents[0]

            # 尝试从基因库引入优秀基因
            if self.rng.random() < 0.1 and len(self.adaptive_controller.gene_pool) > 5:
                gene_pool_parent = self.adaptive_controller.sample_from_gene_pool()
                if gene_pool_parent:
                    if self.rng.random() < 0.5:
                        parent_b = gene_pool_parent
                    else:
                        parent_a = gene_pool_parent

            if self.evo_config.crossover_rate > 0 and self.rng.random() < self.evo_config.crossover_rate:
                try:
                    child = crossover(parent_a, parent_b, self.rng)
                except Exception:
                    child = parent_a.clone()
            else:
                child = parent_a.clone()

            child = apply_mutations(
                child, self.evo_config, self.catalog, self.rng,
                param_scale_override=adaptive_scale,
                topo_prob_override=adaptive_topo_prob,
            )
            child.name = f"gen{generation + 1}_ind{len(new_population):03d}"
            child.generation = generation + 1

            if self._body_is_valid(child):
                new_population.append(child)
            else:
                fallback = self.generator.generate_random_body(
                    name=child.name, min_parts=4, max_parts=10,
                )
                new_population.append(fallback)

        while len(new_population) < self.evo_config.population_size:
            extra = self.generator.generate_random_body(
                name=f"gen{generation + 1}_ind{len(new_population):03d}",
                min_parts=4, max_parts=10,
            )
            new_population.append(extra)

        # 多样性注入
        new_population = self._inject_novelty(new_population, generation)

        return new_population, stats, best_body

    def _body_is_valid(self, body: MechanicalBody) -> bool:
        try:
            from forgecraft.rl.env import ForgeCraftEnv
            env = ForgeCraftEnv(body, self.generator.sim_config if hasattr(self.generator, 'sim_config') else None,
                                catalog=self.catalog)
            env.close()
            return True
        except Exception:
            return False

    def _inject_novelty(
        self, population: List[MechanicalBody], generation: int
    ) -> List[MechanicalBody]:
        diversity_score = self.adaptive_controller.compute_population_diversity(population)
        
        # 多样性阈值自适应
        if generation < 10:
            min_diversity = 0.3
        elif generation < 50:
            min_diversity = 0.25
        else:
            min_diversity = 0.2
            
        if diversity_score >= min_diversity or len(population) < 6:
            return population

        n_inject = max(1, len(population) // 5)
        self.log.diversity(diversity_score, n_inject)

        embeddings = []
        valid_indices = []
        with torch.inference_mode():
            for i, body in enumerate(population):
                try:
                    emb = self.morph_encoder.encode_body(body)
                    emb = F.normalize(emb, p=2, dim=0)
                    embeddings.append(emb.cpu().numpy())
                    valid_indices.append(i)
                except Exception:
                    pass

        if len(embeddings) < 3:
            for k in range(n_inject):
                random_body = self.generator.generate_random_body(
                    name=f"gen{generation + 1}_diversity_{k:02d}",
                    min_parts=4, max_parts=10,
                )
                apply_mutations(
                    random_body, self.evo_config, self.catalog, self.rng,
                    param_scale_override=self.evo_config.param_mutation_scale * 2.0,
                    topo_prob_override=min(0.5, self.evo_config.topo_mutation_prob * 2.0),
                )
                population[-(k + 1)] = random_body
            return population

        emb_matrix = np.array(embeddings)
        similarities = np.dot(emb_matrix, emb_matrix.T)
        novelty = 1.0 - similarities.mean(axis=1)
        top_novel = np.argsort(novelty)[-n_inject:][::-1]

        for k, novel_idx in enumerate(top_novel):
            src_idx = valid_indices[novel_idx]
            novel_body = population[src_idx].clone()
            novel_body = apply_mutations(
                novel_body, self.evo_config, self.catalog, self.rng,
                param_scale_override=self.evo_config.param_mutation_scale * 2.0,
                topo_prob_override=min(0.5, self.evo_config.topo_mutation_prob * 2.0),
            )
            novel_body.name = f"gen{generation + 1}_novelty_{k:02d}"
            replace_pos = -(n_inject - k) if n_inject - k <= len(population) else -1
            if self._body_is_valid(novel_body):
                population[replace_pos] = novel_body
            else:
                population[replace_pos] = self.generator.generate_random_body(
                    name=f"gen{generation + 1}_novel_{k:02d}",
                    min_parts=4, max_parts=8,
                )

        return population