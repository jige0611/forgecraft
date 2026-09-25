# ══════════════════════════════════════════════════════════
# forgecraft.evolution — 进化算法模块
# 种群繁殖、选择、变异、MAP-Elites、分布式、嵌套优化、超参优化
# ══════════════════════════════════════════════════════════

from forgecraft.evolution.loop import EvolutionLoop
from forgecraft.evolution.recovery import (
    TrainGuard, SafeCheckpointManager, NanDetector,
    DegradationConfig, RecoveryStats, Severity,
    safe_evolve,
)
from forgecraft.evolution.breeder import PopulationBreeder
from forgecraft.evolution.selection import pareto_elites
from forgecraft.evolution.adaptive_mutation import AdaptiveMutationController
from forgecraft.evolution.operators import (
    mutate_params, add_part_mutation, delete_part_mutation,
    topological_mutation, crossover, apply_mutations,
)
from forgecraft.evolution.evaluator import PopulationEvaluator
from forgecraft.evolution.bayesian_evaluator import BayesianEvaluator, ThompsonSamplingEvaluator
from forgecraft.evolution.map_elites import (
    MAPElitesConfig, MAPElitesEngine,
    GridArchive, CVTArchive,
    compute_behavior_characteristics, inject_elites,
    OptimizerEmitter,
)
from forgecraft.evolution.distributed import (
    DistributedEvaluator, DistributedEvolutionLoop,
    DistributedConfig, ClusterInfo,
)
from forgecraft.evolution.parallel_engine import (
    ParallelEvaluator, ParallelConfig, BatchResult,
    get_optimal_workers,
)
from forgecraft.evolution.hyperparam_opt import (
    HyperParamOptimizer, HyperParameterSpace,
    ParameterScheduler, EarlyStopper,
    SimpleBayesianOptimizer, ParamSpec,
    OptimizationResult, TrialResult,
    create_hpo_objective, print_optimization_report,
)

# NSGA-III + CMA-ME + 六层嵌套
from forgecraft.evolution.nsga3 import (
    NSGA3Selector, generate_reference_points,
    normalize_objectives, associate_to_reference_points,
    niching_select, nsga3_select, nsga3_pareto_elites,
)
from forgecraft.evolution.cma_me import (
    CMAES, CMAMEmitter, cmame_sample,
)
from forgecraft.evolution.hierarchical import (
    LayerState, NestingLayer, LayerScheduler,
    HierarchicalNestingOptimizer, create_default_layers,
)
from forgecraft.evolution.pareto_archive import (
    ParetoArchive, compute_hypervolume,
    detect_knee_point, reference_point_query,
    spacing_metric, spread_metric,
)

__all__ = [
    "EvolutionLoop", "PopulationBreeder", "PopulationEvaluator",
    "pareto_elites",
    # Error Recovery
    "TrainGuard", "SafeCheckpointManager", "NanDetector",
    "DegradationConfig", "RecoveryStats", "Severity",
    "safe_evolve",
    "AdaptiveMutationController",
    "mutate_params", "add_part_mutation", "delete_part_mutation",
    "topological_mutation", "crossover", "apply_mutations",
    "BayesianEvaluator", "ThompsonSamplingEvaluator",
    # MAP-Elites
    "MAPElitesConfig", "MAPElitesEngine",
    "GridArchive", "CVTArchive", "OptimizerEmitter",
    "compute_behavior_characteristics", "inject_elites",
    # Distributed
    "DistributedEvaluator", "DistributedEvolutionLoop",
    "DistributedConfig", "ClusterInfo",
    # Hyperparameter Optimization
    "HyperParamOptimizer", "HyperParameterSpace",
    "ParameterScheduler", "EarlyStopper",
    "SimpleBayesianOptimizer", "ParamSpec",
    "OptimizationResult", "TrialResult",
    "create_hpo_objective", "print_optimization_report",
    # Parallel evaluation
    "ParallelEvaluator", "ParallelConfig", "BatchResult",
    "get_optimal_workers",
    # NSGA-III + CMA-ME + 六层嵌套
    "NSGA3Selector", "generate_reference_points",
    "normalize_objectives", "associate_to_reference_points",
    "niching_select", "nsga3_select", "nsga3_pareto_elites",
    "CMAES", "CMAMEmitter", "cmame_sample",
    "LayerState", "NestingLayer", "LayerScheduler",
    "HierarchicalNestingOptimizer", "create_default_layers",
    "ParetoArchive", "compute_hypervolume",
    "detect_knee_point", "reference_point_query",
    "spacing_metric", "spread_metric",
]
