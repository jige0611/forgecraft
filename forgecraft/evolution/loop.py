import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from forgecraft.config import EvolutionConfig, RLConfig, SimConfig, TaskConfig
from forgecraft.core.catalog import DEFAULT_CATALOG, PartSpec
from forgecraft.core.generator import BodyGenerator
from forgecraft.core.morphology import MechanicalBody
from forgecraft.evolution.breeder import PopulationBreeder
from forgecraft.evolution.evaluator import PopulationEvaluator
from forgecraft.logging import get_logger
from forgecraft.rl.encoder import MorphologyEncoder, _build_type_registry
from forgecraft.rl.env import RunningMeanStd
from forgecraft.evolution.map_elites import (
    MAPElitesEngine, MAPElitesConfig, inject_elites,
    compute_behavior_characteristics,
)
from forgecraft.evolution.recovery import SafeCheckpointManager, TrainGuard

__all__ = ["EvolutionLoop", "_get_progressive_budget"]


def _get_progressive_budget(generation: int, total_generations: int) -> Dict[str, int]:
    progress: float = generation / max(total_generations, 1)

    if progress < 0.1:
        return {"episodes": 8, "steps_per_ep": 500, "ppo_epochs": 4}
    elif progress < 0.25:
        return {"episodes": 10, "steps_per_ep": 600, "ppo_epochs": 5}
    elif progress < 0.5:
        return {"episodes": 12, "steps_per_ep": 700, "ppo_epochs": 6}
    elif progress < 0.75:
        return {"episodes": 14, "steps_per_ep": 800, "ppo_epochs": 7}
    else:
        return {"episodes": 16, "steps_per_ep": 900, "ppo_epochs": 8}


class EvolutionLoop:
    """进化主循环

    每代流程:
      1. 评估: UCB 渐进式评估 → 适应度分配
      2. 繁殖: 选择 → 交叉 → 变异 → 填充 → 注入
      3. 管理: 自适应变异控制 + MAP-Elites 精英注入 + 物种追踪
      4. 保存: 原子断点 (write-then-rename-fsync)
      5. 监控: 日志 + tracker 记录 + 仪表盘推送

    课程学习 (Curriculum):
      - progress < 10%: 轻量预算 (8ep x 500步)
      - progress 10-25%: 中等预算
      - progress 50-75%: 中高预算
      - progress 75-100%: 全预算 (16ep x 900步)
      - 决赛世代: 强制满预算

    自适应变异:
      - 种群相似度 > 阈值 → 增大拓扑变异概率
      - 种群相似度 < 阈值 → 增大参数变异幅度
      - 连续停滞 → 变异放大 2x
    """
    def __init__(
        self,
        evo_config: Optional[EvolutionConfig] = None,
        sim_config: Optional[SimConfig] = None,
        rl_config: Optional[RLConfig] = None,
        task_config: Optional[TaskConfig] = None,
        catalog: Optional[Dict[str, PartSpec]] = None,
        device: str = "cpu",
        seed: int = 42,
        n_workers: int = 4,
        pretrained_encoder_weights: Optional[Dict[str, Any]] = None,
        enable_map_elites: bool = False,
        enable_fea: bool = False,           # 有限元结构验证
        fea_material: str = "aluminum_6061",
        fea_interval: int = 5,              # 每 N 代验一次
    ) -> None:
        self.evo_config: EvolutionConfig = evo_config or EvolutionConfig()
        self.sim_config: SimConfig = sim_config or SimConfig()
        self.rl_config: RLConfig = rl_config or RLConfig()
        self.task_config: TaskConfig = task_config or TaskConfig()
        self.catalog: Dict[str, PartSpec] = catalog or dict(DEFAULT_CATALOG)
        self.device: str = device
        self.seed: int = seed
        self.n_workers: int = n_workers
        
        # FEA 集成
        self.enable_fea: bool = enable_fea
        self.fea_material: str = fea_material
        self.fea_interval: int = fea_interval
        self._fea_engine = None   # lazy init

        self.rng: np.random.RandomState = np.random.RandomState(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)

        self.log = get_logger()

        self.generator: BodyGenerator = BodyGenerator(self.catalog, seed=seed)
        self.morph_encoder: MorphologyEncoder = MorphologyEncoder(
            hidden_dim=self.rl_config.gnn_hidden,
            output_dim=self.rl_config.morph_embed_dim,
            num_layers=self.rl_config.gnn_layers,
            part_type_registry=_build_type_registry(self.catalog),
        ).to(self.device)

        if pretrained_encoder_weights is not None:
            try:
                self.morph_encoder.load_state_dict(pretrained_encoder_weights)
                self.log.info("已加载预训练形态编码器 (SimCLR)")
            except Exception as e:
                self.log.warning(f"预训练编码器加载失败 ({e})，使用随机初始化")

        self.evaluator: PopulationEvaluator = PopulationEvaluator(
            sim_config=self.sim_config,
            task_config=self.task_config,
            rl_config=self.rl_config,
            catalog=self.catalog,
            device=self.device,
            n_workers=self.n_workers,
        )

        self.breeder: PopulationBreeder = PopulationBreeder(
            evo_config=self.evo_config,
            catalog=self.catalog,
            generator=self.generator,
            morph_encoder=self.morph_encoder,
            device=self.device,
            rng=self.rng,
        )

        # MAP-Elites 引擎
        self.enable_map_elites: bool = enable_map_elites
        self.map_elites_engine: Optional[MAPElitesEngine] = None
        if enable_map_elites:
            me_config = MAPElitesConfig(
                archive_size=min(1000, self.evo_config.population_size * 50),
                evaluations_per_generation=max(10, self.evo_config.population_size // 2),
                inject_every_n_gens=5,
                inject_count=3,
            )
            self.map_elites_engine = MAPElitesEngine(config=me_config, seed=seed)
            self.log.info(f"MAP-Elites 已启用: {me_config.bc_names}, "
                         f"存档大小={me_config.archive_size}")

        self.population: List[MechanicalBody] = []
        self.generation: int = 0
        self.history: List[Dict[str, Any]] = []
        self.best_body: Optional[MechanicalBody] = None
        self.best_trainer_state: Optional[Dict[str, Any]] = None
        self.best_obs_dim: Optional[int] = None
        self.best_act_dim: Optional[int] = None
        self.best_morph_embed: Optional[np.ndarray] = None
        self._stagnation_count: int = 0
        self._is_final_gen: bool = False
        self._curriculum: Optional[Dict[int, TaskConfig]] = None
        self._pareto_objectives: List[str] = []
        self._pareto_directions: List[str] = []
        self.pareto_front: List[MechanicalBody] = []

        if task_config and hasattr(task_config, 'fitness_components') and task_config.fitness_components:
            self._pareto_objectives = list(task_config.fitness_components)
            self._pareto_directions = ["maximize"] * len(self._pareto_objectives)
            if "energy" in self._pareto_objectives:
                idx = self._pareto_objectives.index("energy")
                self._pareto_directions[idx] = "minimize"

    def initialize_population(self) -> None:
        self.population = self.generator.generate_initial_population(
            size=self.evo_config.population_size,
            min_parts=4,
            max_parts=8,
        )
        self.generation = 0
        self.history = []
        self.best_body = None
        self.best_trainer_state = None

    def setup_curriculum(self, stages: List[Tuple[int, str]]) -> None:
        from forgecraft.core.loader import load_task
        self._curriculum = {}
        for gen, task_name in stages:
            task_spec = load_task(name=task_name)
            self._curriculum[gen] = task_spec.to_task_config()
        stage_desc = " → ".join(f"第{g}代={t}" for g, t in stages)
        self.log.info(f"课程学习已设置: {stage_desc}")

    def _apply_curriculum(self) -> None:
        """课程学习: 根据当前代数切换到对应难度的任务配置"""
        if self._curriculum is None:
            return
        for gen, task_cfg in sorted(self._curriculum.items(), reverse=True):
            if self.generation >= gen:
                if self.task_config is not task_cfg:
                    self.task_config = task_cfg
                    self.evaluator.task_config = task_cfg
                    self.evaluator.global_obs_rms = None
                    self.log.info(f"[课程] 切换到任务: {task_cfg.domain} (max_steps={task_cfg.max_episode_steps})")
                break

    def _get_body_dims(self, body: MechanicalBody) -> Tuple[int, int]:
        from forgecraft.rl.env import ForgeCraftEnv
        env = ForgeCraftEnv(body, self.sim_config, self.task_config, catalog=self.catalog)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        env.close()
        return obs_dim, act_dim

    def _validate_structures(self):
        """FEA 结构验证 — 对精英个体进行应力分析
        
        对 fitness 前 20% 的个体:
          1. 生成 trimesh 网格
          2. 运行线性弹性 FEM
          3. 若安全系数 < 1.5 → fitness *= 0.5 (惩罚)
          4. 记录 structural_safety 到 fitness_components
        """
        if self._fea_engine is None:
            try:
                from forgecraft.analysis.fea import FEAEngine
                self._fea_engine = FEAEngine(material=self.fea_material, backend="scipy")
            except ImportError:
                self.log.warning("FEA 模块不可用，跳过结构验证")
                self.enable_fea = False
                return
        
        if not self._fea_engine.available:
            return
        
        # 筛选精英个体 (fitness 前 20%)
        n_check = max(1, len(self.population) // 5)
        sorted_bodies = sorted(self.population, key=lambda b: b.fitness, reverse=True)[:n_check]
        
        try:
            from forgecraft.manufacturing.mesh_engine import MeshEngine
            mesh_eng = MeshEngine("medium")
        except ImportError:
            return
        
        n_checked = 0
        n_failed = 0
        
        for body in sorted_bodies:
            if body.fitness <= 0:
                continue
            
            try:
                # 为每个零件生成 mesh 并拼接
                meshes = []
                for part in body.parts():
                    spec = self.catalog.get(part.part_type)
                    if spec is None:
                        continue
                    try:
                        mesh = mesh_eng.build_part(spec, dict(part.params))
                        # 应用零件位置
                        mesh.apply_translation(part.position)
                        meshes.append(mesh)
                    except Exception:
                        pass
                
                if not meshes:
                    continue
                
                # 合并为整体
                if len(meshes) > 1:
                    try:
                        import trimesh
                        assembly = trimesh.util.concatenate(meshes)
                    except Exception:
                        assembly = meshes[0]
                else:
                    assembly = meshes[0]
                
                # 运行 FEA
                result = self._fea_engine.analyze(assembly, gravity=True)
                n_checked += 1
                
                # 记录安全系数
                body.fitness_components["structural_safety"] = result.safety_factor
                
                # 惩罚不合格个体
                if result.safety_factor < 1.0:
                    # 严重不合格 → 大幅惩罚
                    body.fitness *= 0.3
                    n_failed += 1
                elif result.safety_factor < 1.5:
                    # 边缘 → 轻度惩罚
                    body.fitness *= max(0.5, result.safety_factor / 3.0)
                    n_failed += 1
                
            except Exception as e:
                self.log.debug(f"FEA failed for body {body.name}: {e}")
        
        if n_checked > 0:
            self.log.info(
                f"[FEA] 已验证 {n_checked} 个精英个体, "
                f"{n_failed} 个结构不合格 (安全系数<1.5)"
            )

    def _body_is_valid(self, body: MechanicalBody) -> bool:
        try:
            from forgecraft.rl.env import ForgeCraftEnv
            env = ForgeCraftEnv(body, self.sim_config, self.task_config, catalog=self.catalog)
            env.close()
            return True
        except Exception:
            return False

    def evolve_one_generation(self) -> None:
        if not self.population:
            return

        valid_bodies = [b for b in self.population if self._body_is_valid(b)]
        self.population = valid_bodies

        if not self.population:
            self.population = self.generator.generate_initial_population(
                size=self.evo_config.population_size, min_parts=4, max_parts=8
            )
            return

        morph_encoder_state = {
            "node_feat_dim": self.morph_encoder.node_feat_dim,
            "edge_feat_dim": self.morph_encoder.edge_feat_dim,
            "hidden_dim": self.morph_encoder.hidden_dim,
            "output_dim": self.morph_encoder.output_dim,
            "num_layers": self.morph_encoder.num_layers,
            "weights": {k: v.cpu() for k, v in self.morph_encoder.state_dict().items()},
        }

        inherit_state = None
        if self.best_trainer_state is not None:
            inherit_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                           for k, v in self.best_trainer_state.items()
                           if k in ("actor", "critic")}

        total_gens = self.evo_config.generations
        eval_budget = _get_progressive_budget(self.generation, total_gens)

        results = self.evaluator.evaluate_population_ucb(
            self.population, eval_budget, morph_encoder_state, inherit_state,
            generation=self.generation,
            total_generations=total_gens,
            is_final_gen=self._is_final_gen,
            best_obs_dim=self.best_obs_dim,
            best_act_dim=self.best_act_dim,
            best_morph_embed=self.best_morph_embed,
        )

        sorted_results = sorted(results.items())
        for idx, result in sorted_results:
            if idx < len(self.population):
                body = self.population[idx]
                body.fitness = result["fitness"]
                body.fitness_components = {
                    k: result.get(k, 0.0)
                    for k in self.task_config.fitness_components
                } if self.task_config.fitness_components else {
                    "speed": result.get("speed", 0.0),
                    "energy": result.get("energy", 0.0),
                    "upright": result.get("upright", 0.0),
                    "displacement": result.get("displacement", 0.0),
                    "manufacturability": result.get("manufacturability", 0.0),
                }
                body.policy_state = result.get("trainer_state")

        # ── 结构验证: FEA 应力检查 (每 fea_interval 代) ──
        if self.enable_fea and self.generation % self.fea_interval == 0:
            self._validate_structures()

        new_population, stats, fittest = self.breeder.breed(
            self.population, results, self.generation,
            self._stagnation_count,
            self._pareto_objectives, self._pareto_directions,
            historical_best=self.best_body,
        )
        self.history.append(stats)

        use_pareto = len(self._pareto_objectives) >= 2 and self._pareto_directions
        if use_pareto:
            from forgecraft.evolution.selection import pareto_elites
            self.pareto_front = pareto_elites(
                self.population, len(self.population),
                self._pareto_objectives, self._pareto_directions,
            )

        if self.best_body is None or fittest.fitness > self.best_body.fitness:
            self.best_body = fittest.clone()
            if fittest.policy_state is not None:
                self.best_trainer_state = fittest.policy_state
                try:
                    self.best_obs_dim, self.best_act_dim = self._get_body_dims(fittest)
                except Exception:
                    pass
            with torch.inference_mode():
                self.best_morph_embed = self.morph_encoder.encode_body(
                    self.best_body
                ).cpu().numpy()
            self._stagnation_count = 0
            mfg = self.best_body.fitness_components.get("manufacturability", 0.0)
            n_parts = self.best_body.num_parts()
            n_motors = len(self.best_body.actuated_joints())
            pareto_info = ""
            if use_pareto:
                pareto_info = f"[帕累托] 前沿规模: {len(self.pareto_front)}/{len(self.population)} 前沿: {stats.get('front_sizes', '?')}"
            self.log.new_best(self.best_body.fitness, mfg, n_parts, n_motors, pareto_info)
        else:
            self._stagnation_count += 1

        self.population = new_population
        self.generation += 1
        
        # MAP-Elites 步骤
        if self.enable_map_elites and self.map_elites_engine is not None:
            self._step_map_elites()

    def _step_map_elites(self) -> None:
        """执行一步 MAP-Elites 进化"""
        engine = self.map_elites_engine
        
        # 初始化 (首代)
        if engine.generation == 0:
            def random_generator():
                bodies = self.generator.generate_initial_population(
                    size=1, min_parts=4, max_parts=8
                )
                return bodies[0] if bodies else None
            
            engine.initialize(generator_fn=random_generator)
        
        # 包装评估器
        def me_evaluator(body: MechanicalBody):
            try:
                # 构建 episode_info
                episode_info = {
                    "speed": body.fitness_components.get("speed", 0.0),
                    "energy": body.fitness_components.get("energy", 0.0),
                    "displacement": body.fitness_components.get("displacement", 0.0),
                    "stability": body.fitness_components.get("upright", 0.5),
                }
                bc = compute_behavior_characteristics(body, episode_info)
                return body.fitness, bc, episode_info
            except Exception:
                return -float("inf"), np.zeros(5), {}
        
        # 执行 MAP-Elites 步骤
        stats = engine.step(me_evaluator)
        
        # 周期性注入精英
        if engine.generation % engine.config.inject_every_n_gens == 0:
            self.population = inject_elites(
                self.population, engine,
                n=engine.config.inject_count,
                replace_worst=True,
            )
            self.log.info(
                f"MAP-Elites 注入: {engine.config.inject_count} 精英, "
                f"覆盖率={stats['coverage']:.1%}, "
                f"QD-score={stats['qd_score']:.1f}"
            )

    def save_checkpoint(self, path: str = "checkpoint.pkl"):
        obs_rms_state = None
        if self.evaluator.global_obs_rms is not None:
            obs_rms_state = {
                "mean": self.evaluator.global_obs_rms.mean,
                "var": self.evaluator.global_obs_rms.var,
                "count": self.evaluator.global_obs_rms.count,
            }
        data = {
            "generation": self.generation,
            "population": pickle.dumps(self.population),
            "history": self.history,
            "best_body": pickle.dumps(self.best_body) if self.best_body else None,
            "best_trainer_state": self.best_trainer_state,
            "best_obs_dim": self.best_obs_dim,
            "best_act_dim": self.best_act_dim,
            "best_morph_embed": self.best_morph_embed,
            "_stagnation_count": self._stagnation_count,
            "global_obs_rms": obs_rms_state,
            "morph_encoder_weights": {
                k: v.cpu() for k, v in self.morph_encoder.state_dict().items()
            },
            "morph_encoder_config": {
                "hidden_dim": self.morph_encoder.hidden_dim,
                "output_dim": self.morph_encoder.output_dim,
                "num_layers": self.morph_encoder.num_layers,
                "node_feat_dim": self.morph_encoder.node_feat_dim,
                "edge_feat_dim": self.morph_encoder.edge_feat_dim,
            },
            "seed": self.seed,
            "rng_state": self.rng.get_state(),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        self.log.info(f"断点已保存: {path} (第{self.generation}代)")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "EvolutionLoop":
        with open(path, "rb") as f:
            data = pickle.load(f)

        evo_config = EvolutionConfig()
        sim_config = SimConfig()
        rl_config = RLConfig()
        task_config = TaskConfig()

        loop = cls.__new__(cls)
        loop.log = get_logger()
        loop.evo_config = evo_config
        loop.sim_config = sim_config
        loop.rl_config = rl_config
        loop.task_config = task_config
        loop.device = device
        loop.seed = data["seed"]
        loop.n_workers = 4
        loop.catalog = dict(DEFAULT_CATALOG)

        loop.rng = np.random.RandomState(data["seed"])
        torch.manual_seed(data["seed"])
        if device == "cuda":
            torch.cuda.manual_seed(data["seed"])

        loop.generator = BodyGenerator(loop.catalog, seed=data["seed"])

        enc_cfg = data.get("morph_encoder_config", {})
        type_registry = _build_type_registry(loop.catalog) if loop.catalog else None
        loop.morph_encoder = MorphologyEncoder(
            hidden_dim=enc_cfg.get("hidden_dim", rl_config.gnn_hidden),
            output_dim=enc_cfg.get("output_dim", rl_config.morph_embed_dim),
            num_layers=enc_cfg.get("num_layers", rl_config.gnn_layers),
            node_feat_dim=enc_cfg.get("node_feat_dim", 32),
            edge_feat_dim=enc_cfg.get("edge_feat_dim", 16),
            part_type_registry=type_registry,
        ).to(device)
        loop.morph_encoder.load_state_dict(data["morph_encoder_weights"])

        loop.evaluator = PopulationEvaluator(
            sim_config=loop.sim_config,
            task_config=loop.task_config,
            rl_config=loop.rl_config,
            catalog=loop.catalog,
            device=device,
            n_workers=loop.n_workers,
        )
        loop.breeder = PopulationBreeder(
            evo_config=loop.evo_config,
            catalog=loop.catalog,
            generator=loop.generator,
            morph_encoder=loop.morph_encoder,
            device=device,
            rng=loop.rng,
        )

        loop.population = pickle.loads(data["population"])
        loop.generation = data["generation"]
        loop.history = data["history"]
        loop.best_body = pickle.loads(data["best_body"]) if data["best_body"] else None
        loop.best_trainer_state = data["best_trainer_state"]
        loop.best_obs_dim = data["best_obs_dim"]
        loop.best_act_dim = data["best_act_dim"]
        loop.best_morph_embed = data["best_morph_embed"]
        loop._stagnation_count = data["_stagnation_count"]
        loop._is_final_gen = False
        loop._curriculum = None
        loop._pareto_objectives = []
        loop._pareto_directions = []
        loop.pareto_front = []

        if data["global_obs_rms"] is not None:
            rms = RunningMeanStd.__new__(RunningMeanStd)
            rms.mean = data["global_obs_rms"]["mean"]
            rms.var = data["global_obs_rms"]["var"]
            rms.count = data["global_obs_rms"]["count"]
            loop.evaluator.global_obs_rms = rms

        if device == "cuda":
            loop.morph_encoder.to("cuda")
            for body in loop.population:
                try:
                    body.policy_state = None
                except Exception:
                    pass

        loop.log = get_logger()
        loop.log.info(f"断点恢复: {path} (从第{loop.generation}代继续)")
        return loop

    # ═══════════════════════════════════════════════════════════════
    # Main Evolution Loop — 5 phases: init → evaluate → breed → export → report
    # ═══════════════════════════════════════════════════════════════
    def run(self, n_generations: Optional[int] = None, safe: bool = True,
            checkpoint_dir: str = "checkpoints",
            on_generation_complete: callable = None):
        if n_generations is None:
            n_generations = self.evo_config.generations

        if safe:
            guard = TrainGuard(
                self,
                checkpoint_dir=checkpoint_dir,
                save_every_n_gens=5,
            )
            guard.run(n_generations=n_generations,
                      on_generation_complete=on_generation_complete)
            return

        # ── 非安全模式 (向后兼容) ──
        if not self.population:
            self.initialize_population()

        if self.device == "cuda":
            device_name = torch.cuda.get_device_name(0)
            self.log.section(f"ForgeCraft 进化开始 [GPU: {device_name}]")
        else:
            self.log.section("ForgeCraft 进化开始 [CPU]")
        self.log.info(f"种群规模: {self.evo_config.population_size} | 并行进程: {self.n_workers}")
        self.log.info(f"计划世代: {n_generations}")
        if self.device == "cuda" and torch.cuda.is_available():
            self.log.info(f"加速: GPU 推理 ({torch.cuda.get_device_name(0)})")
        self.log.info("评估模式: 渐进式 + UCB分配 (早期低预算→后期潜力优先)")
        self.log.info(f"初始种群已生成: {len(self.population)} 个个体")

        ckpt = SafeCheckpointManager(checkpoint_dir=checkpoint_dir, save_every_n_gens=5)

        for gen in range(n_generations):
            gen_start = time.time()

            self._apply_curriculum()

            gens_remaining = n_generations - self.generation
            self._is_final_gen = (gens_remaining <= 2)

            budget = _get_progressive_budget(self.generation, n_generations)
            mode_label = "[决赛]" if self._is_final_gen else "[UCB]"
            self.log.generation_start(self.generation, mode_label, budget)

            self.evolve_one_generation()

            stats = self.history[-1]
            self.log.generation_end(self.generation, stats)

            if (self.generation + 1) % ckpt.save_every_n_gens == 0 or self._is_final_gen:
                ckpt.save(self, self.generation)

        self.log.section("进化完成！")
        if self.best_body:
            self.log.info(f"最佳形态: {self.best_body.name}")
            self.log.info(f"最佳适应度: {self.best_body.fitness:.4f}")
            self.log.info(f"零件数: {self.best_body.num_parts()}")
            self.log.info(f"驱动关节: {len(self.best_body.actuated_joints())}")
            for line in self.best_body.describe().split('\n'):
                self.log.info(line)

    def get_pareto_front(self) -> List[MechanicalBody]:
        from forgecraft.evolution.selection import pareto_elites
        return pareto_elites(self.population, len(self.population),
                              self._pareto_objectives, self._pareto_directions)
