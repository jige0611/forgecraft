# ══════════════════════════════════════════════════════════
# forgecraft.evaluation — 评估与指标模块
# 适应度评估、多目标优化、GPU 加速引擎、多精度评估
# ══════════════════════════════════════════════════════════

from forgecraft.evaluation.fitness import (
    MultiObjectiveFitnessEvaluator,
    NSGAIISelector,
    AdaptiveWeightLearner,
)
from forgecraft.evaluation.metrics import compute_manufacturability_detail
from forgecraft.evaluation.gpu_engine import GPUAcceleratedEvaluator
from forgecraft.evaluation.multi_fidelity import (
    FidelityLevel, MultiFidelityEvaluator,
    successive_halving, hyperband, bohb_select,
    FIDELITY_CONFIGS,
)

__all__ = [
    "MultiObjectiveFitnessEvaluator", "NSGAIISelector",
    "AdaptiveWeightLearner",
    "compute_manufacturability_detail",
    "GPUAcceleratedEvaluator",
    # Multi-fidelity evaluation
    "FidelityLevel", "MultiFidelityEvaluator",
    "successive_halving", "hyperband", "bohb_select",
    "FIDELITY_CONFIGS",
]
