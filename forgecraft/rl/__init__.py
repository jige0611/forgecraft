# ══════════════════════════════════════════════════════════
# forgecraft.rl — 强化学习模块
# PPO 训练、形态编码器、迁移学习、预训练、批量推理
# ══════════════════════════════════════════════════════════

from forgecraft.rl.ppo import PPOTrainer
from forgecraft.rl.encoder import MorphologyEncoder, MorphAwareActor, _build_type_registry
from forgecraft.rl.morph_transfer import (
    MorphConditionalActor,
    MorphConditionalCritic,
    TransferLearningManager,
)
from forgecraft.rl.pretrain import pretrain_encoder
from forgecraft.rl.env import ForgeCraftEnv, RunningMeanStd
from forgecraft.rl.batch_encoder import BatchMorphologyEncoder
from forgecraft.rl.batch_ppo import BatchPPORollout
from forgecraft.rl.morph_policy import (
    CoevolutionConfig, MorphPolicyGenome, CoevolutionEngine,
    ESMController, MorphConditionedPolicy,
)

__all__ = [
    "PPOTrainer",
    "MorphologyEncoder", "MorphAwareActor", "_build_type_registry",
    "MorphConditionalActor", "MorphConditionalCritic",
    "TransferLearningManager",
    "pretrain_encoder",
    "ForgeCraftEnv", "RunningMeanStd",
    "BatchMorphologyEncoder", "BatchPPORollout",
    # Co-evolution
    "CoevolutionConfig", "MorphPolicyGenome", "CoevolutionEngine",
    "ESMController", "MorphConditionedPolicy",
]
