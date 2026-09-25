"""
YAML 配置热加载 — 实验参数持久化

支持从 YAML 文件加载完整实验配置:
  python -m forgecraft.main --config experiments/biped_v2.yaml

YAML 格式:
  experiment:
    name: "biped_v2"
    generations: 100
    population: 64
    catalog: "primitives"
    task: "speed"
    curriculum: true
    export: true

  evolution:
    population_size: 64
    generations: 100
    elite_count: 10
    mutation_rate: 0.35
    crossover_rate: 0.6
    topo_mutation_prob: 0.18
    param_mutation_prob: 0.35
    param_mutation_scale: 0.12

  rl:
    hidden_dim: 128
    gnn_layers: 3
    gnn_hidden: 64
    morph_embed_dim: 64
    actor_lr: 0.0003
    critic_lr: 0.001
    gamma: 0.99
    lam: 0.95
    clip_ratio: 0.2
    entropy_coef: 0.02
    ppo_epochs: 12

  simulation:
    gravity: -9.81
    timestep: 0.005
    friction: 0.6
    max_steps: 500
    substeps: 10
"""

import os
import argparse
from typing import Optional
from dataclasses import asdict

import yaml

from forgecraft.config import EvolutionConfig, RLConfig, SimConfig, TaskConfig, RewardComponent


class ExperimentConfig:
    """从 YAML 加载的完整实验配置"""

    def __init__(self, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)

        exp = self._raw.get("experiment", {})
        evo = self._raw.get("evolution", {})
        rl = self._raw.get("rl", {})
        sim = self._raw.get("simulation", {})
        task = self._raw.get("task_config", {})

        self.name = exp.get("name", os.path.splitext(os.path.basename(yaml_path))[0])
        self.generations = exp.get("generations", 50)
        self.population = exp.get("population", 32)
        self.catalog = exp.get("catalog", "primitives")
        self.task = exp.get("task", "speed")
        self.curriculum = exp.get("curriculum", False)
        self.export = exp.get("export", False)
        self.export_dir = exp.get("export_dir", "design_output")
        self.seed = exp.get("seed", 42)
        self.workers = exp.get("workers", 4)
        self.cuda = exp.get("cuda", False)
        self.quick = exp.get("quick", False)
        self.turbo = exp.get("turbo", False)
        self.pretrain = exp.get("pretrain", False)
        self.checkpoint_every = exp.get("checkpoint_every", 5)
        self.resume = exp.get("resume", None)
        self.dashboard = exp.get("dashboard", False)

        self.evolution = EvolutionConfig(
            population_size=evo.get("population_size", self.population),
            generations=evo.get("generations", self.generations),
            elite_count=evo.get("elite_count", 8),
            mutation_rate=evo.get("mutation_rate", 0.35),
            crossover_rate=evo.get("crossover_rate", 0.6),
            topo_mutation_prob=evo.get("topo_mutation_prob", 0.18),
            param_mutation_prob=evo.get("param_mutation_prob", 0.35),
            param_mutation_scale=evo.get("param_mutation_scale", 0.12),
        )

        self.rl_config = RLConfig(
            hidden_dim=rl.get("hidden_dim", 128),
            gnn_layers=rl.get("gnn_layers", 3),
            gnn_hidden=rl.get("gnn_hidden", 64),
            morph_embed_dim=rl.get("morph_embed_dim", 64),
            actor_lr=rl.get("actor_lr", 3e-4),
            critic_lr=rl.get("critic_lr", 1e-3),
            gamma=rl.get("gamma", 0.99),
            lam=rl.get("lam", 0.95),
            clip_ratio=rl.get("clip_ratio", 0.2),
            entropy_coef=rl.get("entropy_coef", 0.02),
            ppo_epochs=rl.get("ppo_epochs", 12),
        )

        self.sim_config = SimConfig(
            gravity=sim.get("gravity", -9.81),
            timestep=sim.get("timestep", 0.005),
            friction=sim.get("friction", 0.6),
            max_steps=sim.get("max_steps", 500),
            substeps=sim.get("substeps", 10),
        )

        reward_comps = []
        for rc in task.get("reward_components", []):
            if isinstance(rc, dict):
                reward_comps.append(RewardComponent(
                    name=rc.get("name", "unknown"),
                    weight=rc.get("weight", 1.0),
                    expression=rc.get("expression", ""),
                    description=rc.get("description", ""),
                ))
        self.task_config = TaskConfig(
            reward_components=reward_comps if reward_comps else [
                RewardComponent("displacement", 1.0),
                RewardComponent("energy", -0.01),
            ],
            fitness_components=task.get("fitness_components", ["speed", "energy", "displacement"]),
            max_episode_steps=task.get("max_episode_steps", sim.get("max_steps", 500)),
            fall_height=task.get("fall_height", 0.05),
            domain=task.get("domain", "robot"),
        )

    def to_args(self) -> argparse.Namespace:
        """转换为 argparse.Namespace 以兼容 main.py。"""
        return argparse.Namespace(
            generations=self.generations,
            population=self.population,
            elites=self.evolution.elite_count,
            workers=self.workers,
            cuda=self.cuda,
            catalog=self.catalog,
            task=self.task,
            seed=self.seed,
            output=self.export_dir if self.export else None,
            export=self.export,
            export_dir=self.export_dir,
            dashboard=self.dashboard,
            quick=self.quick,
            turbo=self.turbo,
            resume=self.resume,
            checkpoint_every=self.checkpoint_every,
            curriculum=self.curriculum,
            pretrain=self.pretrain,
            pretrain_epochs=200,
            pretrain_bodies=256,
            list_catalogs=False,
            list_tasks=False,
        )

    def save_template(self, output_path: str):
        """保存为模板文件。"""
        template = {
            "experiment": {"name": self.name, "generations": self.generations,
                           "population": self.population, "catalog": self.catalog,
                           "task": self.task},
            "evolution": asdict(self.evolution),
            "rl": asdict(self.rl_config),
            "simulation": asdict(self.sim_config),
            "task_config": {
                "reward_components": [
                    {"name": "displacement", "weight": 1.0},
                    {"name": "energy", "weight": -0.01},
                    {"name": "upright", "weight": 0.1},
                ],
                "fitness_components": ["speed", "energy", "displacement"],
                "max_episode_steps": self.sim_config.max_steps,
            },
        }
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(template, f, default_flow_style=False, allow_unicode=True)
        print(f"配置模板已保存: {output_path}")
