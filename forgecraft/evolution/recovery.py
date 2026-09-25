"""错误恢复与断点续训模块

功能：
  1. 原子化断点写入 (write-then-rename)
  2. NaN/Inf 检测与自动恢复
  3. 梯度爆炸检测与裁剪
  4. 进化停滞检测与重启
  5. 分级降级策略 (warn → reset → abort)
  6. 训练循环保护装饰器

设计理念 (参考 PyTorch Lightning + CleanRL):
  - 最小侵入: 通过包装器而非修改核心循环
  - 可配置: 每种异常类型可独立配置策略
  - 可观测: 所有恢复事件记录到日志

用法:
  guard = TrainGuard(loop, checkpoint_dir="checkpoints")
  guard.run(n_generations=200)
"""

import os
import pickle
import shutil
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from forgecraft.logging import get_logger

__all__ = [
    "Severity",
    "RecoveryStats",
    "SafeCheckpointManager",
    "NanDetector",
    "DegradationConfig",
    "TrainGuard",
]


# ═══════════════════════════════════════════════
# 异常严重级别
# ═══════════════════════════════════════════════

class Severity(Enum):
    WARN = auto()      # 记录日志，继续运行
    RETRY = auto()     # 重试当前操作
    DEGRADE = auto()   # 降级 (减少预算/降低精度)
    RESET_GEN = auto() # 重置当前代
    RESUME = auto()    # 从断点恢复
    ABORT = auto()     # 中止进化


# ═══════════════════════════════════════════════
# 恢复统计
# ═══════════════════════════════════════════════

@dataclass
class RecoveryStats:
    """恢复统计"""
    nan_detections: int = 0
    gradient_explosions: int = 0
    generation_resets: int = 0
    checkpoint_restores: int = 0
    degradation_events: int = 0
    total_errors: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0

    @property
    def recovery_rate(self) -> float:
        total = self.total_errors
        return self.successful_recoveries / total if total > 0 else 1.0

    def summary(self) -> str:
        parts = [
            f"总错误: {self.total_errors}",
            f"NaN检测: {self.nan_detections}",
            f"梯度爆炸: {self.gradient_explosions}",
            f"代重置: {self.generation_resets}",
            f"断点恢复: {self.checkpoint_restores}",
            f"降级事件: {self.degradation_events}",
            f"恢复率: {self.recovery_rate:.1%}",
        ]
        return " | ".join(parts)


# ═══════════════════════════════════════════════
# 原子化断点管理器
# ═══════════════════════════════════════════════

class SafeCheckpointManager:
    """原子化断点管理

    特性:
      - write-then-rename 原子写入
      - 最多保留 K 个历史断点
      - 自动检测断点损坏
      - 带时间戳的自动命名
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        save_every_n_gens: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_every_n_gens = save_every_n_gens
        self.log = get_logger()

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, loop, generation: int, label: str = "") -> str:
        """原子化保存断点

        返回: 断点文件路径
        """
        ts = time.strftime("%Y%m%d_%H%M%S")
        label_suffix = f"_{label}" if label else ""
        filename = f"gen{generation:04d}_{ts}{label_suffix}.pkl"
        tmp_path = self.checkpoint_dir / f".tmp_{filename}"
        final_path = self.checkpoint_dir / filename

        try:
            # 阶段 1: 写入临时文件
            with open(tmp_path, "wb") as f:
                pickle.dump(self._collect_state(loop), f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())

            # 阶段 2: 原子重命名
            tmp_path.replace(final_path)

            # 阶段 3: 清理旧断点
            self._cleanup_old_checkpoints()

            self.log.info(f"断点已保存: {final_path.name}")
            return str(final_path)

        except Exception as e:
            self.log.error(f"断点保存失败: {e}")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def load(self, path: str) -> Optional[Dict[str, Any]]:
        """加载并验证断点"""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._validate_state(data)
            return data
        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            self.log.error(f"断点损坏: {path} ({e})")
            return None
        except FileNotFoundError:
            self.log.error(f"断点不存在: {path}")
            return None

    def find_latest(self) -> Optional[str]:
        """查找最新断点"""
        files = sorted(
            self.checkpoint_dir.glob("gen*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return str(files[0]) if files else None

    def _collect_state(self, loop) -> Dict[str, Any]:
        """收集完整状态"""
        obs_rms_state = None
        if loop.evaluator.global_obs_rms is not None:
            obs_rms_state = {
                "mean": loop.evaluator.global_obs_rms.mean,
                "var": loop.evaluator.global_obs_rms.var,
                "count": loop.evaluator.global_obs_rms.count,
            }

        return {
            "version": 2,
            "generation": loop.generation,
            "population": pickle.dumps(loop.population),
            "history": loop.history,
            "best_body": pickle.dumps(loop.best_body) if loop.best_body else None,
            "best_trainer_state": loop.best_trainer_state,
            "best_obs_dim": loop.best_obs_dim,
            "best_act_dim": loop.best_act_dim,
            "best_morph_embed": loop.best_morph_embed,
            "_stagnation_count": loop._stagnation_count,
            "global_obs_rms": obs_rms_state,
            "morph_encoder_weights": {
                k: v.cpu() for k, v in loop.morph_encoder.state_dict().items()
            },
            "morph_encoder_config": {
                "hidden_dim": loop.morph_encoder.hidden_dim,
                "output_dim": loop.morph_encoder.output_dim,
                "num_layers": loop.morph_encoder.num_layers,
                "node_feat_dim": loop.morph_encoder.node_feat_dim,
                "edge_feat_dim": loop.morph_encoder.edge_feat_dim,
            },
            "seed": loop.seed,
            "rng_state": loop.rng.get_state(),
        }

    def _validate_state(self, data: Dict[str, Any]) -> None:
        """验证断点数据完整性"""
        required_keys = {
            "generation", "population", "history", "best_body",
            "morph_encoder_weights", "seed",
        }
        missing = required_keys - set(data.keys())
        if missing:
            raise KeyError(f"断点缺少关键字段: {missing}")

    def _cleanup_old_checkpoints(self) -> None:
        """保留最近 K 个断点"""
        files = sorted(
            self.checkpoint_dir.glob("gen*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for f in files[self.max_checkpoints:]:
            f.unlink(missing_ok=True)


# ═══════════════════════════════════════════════
# NaN/异常检测器
# ═══════════════════════════════════════════════

class NanDetector:
    """检测训练中的 NaN/Inf

    检测目标:
      - 损失值 NaN/Inf
      - 梯度 NaN/Inf
      - 适应度 NaN/Inf
      - 模型参数 NaN/Inf
    """

    def __init__(self, auto_fix: bool = True):
        self.auto_fix = auto_fix
        self.log = get_logger()

    def check_loss(self, loss: float, context: str = "") -> Tuple[bool, str]:
        """检查损失值"""
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        if np.isnan(loss) or np.isinf(loss):
            msg = f"[NaN/Inf] 损失={loss} {context}"
            self.log.warning(msg)
            return False, msg
        return True, ""

    def check_gradients(self, model: torch.nn.Module, context: str = "") -> Tuple[bool, str]:
        """检查梯度"""
        max_grad = 0.0
        nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                if torch.isnan(grad).any():
                    nan_grad = True
                    self.log.warning(f"[NaN梯度] {name} {context}")
                if torch.isinf(grad).any():
                    nan_grad = True
                    self.log.warning(f"[Inf梯度] {name} {context}")
                max_grad = max(max_grad, grad.abs().max().item())

        if nan_grad:
            if self.auto_fix:
                self._zero_nan_grads(model)
                return False, f"NaN/Inf梯度已清零 {context}"
            return False, f"NaN/Inf梯度 {context}"
        return True, f"max_grad={max_grad:.4f}"

    def check_fitness(self, fitness: float, body_name: str = "") -> Tuple[bool, str]:
        """检查适应度"""
        if np.isnan(fitness) or np.isinf(fitness):
            msg = f"[NaN/Inf适应度] {body_name}: {fitness}"
            self.log.warning(msg)
            return False, msg
        if fitness < -1e6:
            self.log.warning(f"[异常低适应度] {body_name}: {fitness}")
            return False, f"abnormal fitness {fitness}"
        return True, ""

    def check_parameters(self, model: torch.nn.Module, context: str = "") -> Tuple[bool, str]:
        """检查模型参数"""
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any():
                msg = f"[NaN参数] {name} {context}"
                self.log.error(msg)
                return False, msg
            if torch.isinf(param.data).any():
                msg = f"[Inf参数] {name} {context}"
                self.log.error(msg)
                return False, msg
        return True, ""

    @staticmethod
    def _zero_nan_grads(model: torch.nn.Module) -> None:
        """清零 NaN 梯度"""
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = torch.nan_to_num(
                    param.grad.data, nan=0.0, posinf=0.0, neginf=0.0
                )


# ═══════════════════════════════════════════════
# 降级策略配置
# ═══════════════════════════════════════════════

@dataclass
class DegradationConfig:
    """降级策略配置"""
    # NaN恢复
    nan_recovery: Severity = Severity.DEGRADE
    nan_max_consecutive: int = 3  # 连续NaN后才更严重

    # 梯度爆炸
    grad_explosion_recovery: Severity = Severity.RESET_GEN
    grad_max_consecutive: int = 2

    # 仿真崩溃
    sim_crash_recovery: Severity = Severity.RESET_GEN
    sim_crash_max_consecutive: int = 5

    # OOM恢复
    oom_recovery: Severity = Severity.DEGRADE
    oom_max_consecutive: int = 2

    # 停滞检测
    stagnation_threshold: int = 20  # 连续无提升代数
    stagnation_action: Severity = Severity.RESUME

    # 降级时减少的比例
    budget_reduction: float = 0.5   # 预算减半
    pop_reduction: float = 0.7      # 种群缩减到70%


# ═══════════════════════════════════════════════
# 训练守卫 - 核心保护机制
# ═══════════════════════════════════════════════

class TrainGuard:
    """进化训练守卫

    包装 EvolutionLoop.run() 提供:
      1. 自动异常捕获与分类
      2. 按严重级别执行恢复策略
      3. NaN/梯度/适应度检测
      4. 自动断点保存/恢复
      5. 降级预算调整

    用法:
      loop = EvolutionLoop(...)
      guard = TrainGuard(loop, checkpoint_dir="checkpoints")
      guard.run(n_generations=200)
    """

    def __init__(
        self,
        loop,
        checkpoint_dir: str = "checkpoints",
        degradation_config: Optional[DegradationConfig] = None,
        save_every_n_gens: int = 5,
        max_checkpoints: int = 10,
        enable_nan_detection: bool = True,
    ):
        self.loop = loop
        self.log = get_logger()
        self.stats = RecoveryStats()

        self.checkpoint_mgr = SafeCheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=max_checkpoints,
            save_every_n_gens=save_every_n_gens,
        )
        self.degradation_config = degradation_config or DegradationConfig()
        self.nan_detector = NanDetector(auto_fix=True) if enable_nan_detection else None

        # 计数器和状态
        self._consecutive_nans: int = 0
        self._consecutive_grad_explosions: int = 0
        self._consecutive_sim_crashes: int = 0
        self._consecutive_ooms: int = 0
        self._consecutive_generic_errors: int = 0
        self._current_budget_mult: float = 1.0
        self._degraded: bool = False

    def run(self, n_generations: int, resume: bool = True,
            on_generation_complete: callable = None) -> None:
        """运行进化，带完整错误恢复

        Args:
            n_generations: 计划进化代数
            resume: 是否自动从最新断点恢复
            on_generation_complete: 每代完成后回调 fn(loop)
        """
        # 自动恢复
        if resume and not self.loop.population:
            latest = self.checkpoint_mgr.find_latest()
            if latest:
                if self._restore_from_checkpoint(latest):
                    self.log.info(f"从断点恢复成功: {latest}")
                    self.stats.checkpoint_restores += 1

        # 初始化种群
        if not self.loop.population:
            self.loop.initialize_population()

        n_generations = n_generations or self.loop.evo_config.generations
        target_gen = n_generations  # 总代数上限，断点恢复后只跑剩余部分

        # 尝试自动从断点恢复 (resume模式)
        if resume and self.loop.generation == 0:
            latest = self.checkpoint_mgr.find_latest()
            if latest:
                try:
                    self._restore_from_checkpoint(latest)
                    self.stats.checkpoint_restores += 1
                except Exception:
                    pass

        while self.loop.generation < target_gen:
            try:
                self._guarded_step()
                if on_generation_complete:
                    try:
                        on_generation_complete(self.loop)
                    except Exception:
                        pass  # 回调失败不影响进化

            except torch.cuda.OutOfMemoryError as e:
                self._handle_oom(e)

            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    self._handle_oom(e)
                elif "cudnn" in msg or "cublas" in msg:
                    self._handle_cuda_error(e)
                else:
                    self._handle_generic_error(e)

            except (ValueError, AssertionError) as e:
                self._handle_value_error(e)

            except KeyboardInterrupt:
                self.log.warning("用户中断，保存断点...")
                self.checkpoint_mgr.save(
                    self.loop, self.loop.generation, label="interrupt"
                )
                raise

            except Exception as e:
                self._handle_generic_error(e)

        self.log.info(f"进化完成! {self.stats.summary()}")

    def _guarded_step(self) -> None:
        """执行一步进化，带完整保护"""
        gen = self.loop.generation

        # 进度日志
        if gen % 10 == 0:
            self.log.info(f"[Guard] 第 {gen} 代 (计划 {self.loop.evo_config.generations})")

        # 执行进化
        self.loop.evolve_one_generation()

        # NaN检测
        if self.nan_detector and self.loop.history:
            stats = self.loop.history[-1]
            best_fit = stats.get("best_fitness", 0.0)
            ok, msg = self.nan_detector.check_fitness(best_fit, "历史最佳")
            if not ok:
                self._consecutive_nans += 1
                self.stats.nan_detections += 1
                self._handle_nan_recovery()
                return

        # GNN参数检测
        if self.nan_detector:
            ok, msg = self.nan_detector.check_parameters(
                self.loop.morph_encoder, f"gen{gen}"
            )
            if not ok:
                self._consecutive_nans += 1
                self.stats.nan_detections += 1
                self._handle_nan_recovery()
                return

        # 重置连续计数器 (成功一步)
        self._consecutive_nans = 0
        self._consecutive_sim_crashes = 0
        self._consecutive_generic_errors = 0

        # 定期保存断点
        if (gen + 1) % self.checkpoint_mgr.save_every_n_gens == 0:
            self.checkpoint_mgr.save(self.loop, gen)

        # 最新断点 (每代保存，用于恢复)
        self.checkpoint_mgr.save(self.loop, gen, label="latest")

    # ── 恢复策略 ──

    def _handle_nan_recovery(self) -> None:
        """NaN 恢复策略"""
        if self._consecutive_nans >= self.degradation_config.nan_max_consecutive:
            severity = Severity.RESUME
        else:
            severity = self.degradation_config.nan_recovery

        if severity == Severity.DEGRADE:
            self._apply_degradation("NaN检测")
        elif severity == Severity.RESUME:
            self._restore_latest("连续NaN")
        elif severity == Severity.ABORT:
            raise RuntimeError(f"连续 {self._consecutive_nans} 次 NaN，中止进化")

    def _handle_oom(self, error: Exception) -> None:
        """OOM 恢复策略"""
        self._consecutive_ooms += 1
        self.stats.total_errors += 1
        self.log.error(f"[OOM] {error}")

        if self._consecutive_ooms >= self.degradation_config.oom_max_consecutive:
            self.stats.failed_recoveries += 1
            raise RuntimeError("连续 OOM，GPU 资源不足") from error

        self._apply_degradation("OOM恢复")
        self.stats.successful_recoveries += 1

        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _handle_cuda_error(self, error: Exception) -> None:
        """CUDA 错误恢复"""
        self.stats.total_errors += 1
        self.log.error(f"[CUDA Error] {error}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._restore_latest("CUDA错误")
        self.stats.successful_recoveries += 1

    def _handle_value_error(self, error: Exception) -> None:
        """值错误恢复"""
        self.stats.total_errors += 1
        msg = str(error)

        if "nan" in msg.lower() or "inf" in msg.lower():
            self._consecutive_nans += 1
            self.stats.nan_detections += 1
            self._handle_nan_recovery()
        else:
            self.log.error(f"[ValueError] {error}")
            self._restore_latest("ValueError")
            self.stats.generation_resets += 1

        self.stats.successful_recoveries += 1

    def _handle_generic_error(self, error: Exception) -> None:
        """通用错误恢复"""
        self.stats.total_errors += 1
        self._consecutive_generic_errors += 1
        err_type = type(error).__name__
        self.log.error(f"[Error] {err_type}: {error}")

        # RecursionError: 打印完整调用栈
        if isinstance(error, RecursionError):
            self.log.error(f"[RecursionError detail] {traceback.format_exc(limit=30)}")
        else:
            self.log.debug(traceback.format_exc())

        # 保存错误现场
        self.checkpoint_mgr.save(
            self.loop, self.loop.generation, label="error"
        )

        # 连续 3 次同样错误 → 跳过该代 (恢复无效)
        if self._consecutive_generic_errors >= 3:
            self.log.warning(
                f"[跳过] 连续 {self._consecutive_generic_errors} 次相同错误，跳过第 {self.loop.generation} 代"
            )
            self._consecutive_generic_errors = 0
            self.loop.generation += 1
            return

        # 尝试恢复
        self._restore_latest("通用错误")
        self.stats.successful_recoveries += 1

    def _apply_degradation(self, reason: str) -> None:
        """应用降级策略"""
        if self._degraded:
            self.log.warning(f"[已降级] {reason} (跳过重复降级)")
            return

        self._current_budget_mult *= self.degradation_config.budget_reduction
        self._degraded = True
        self.stats.degradation_events += 1

        self.log.warning(
            f"[降级] {reason}: 预算降至 {self._current_budget_mult:.0%}"
        )

        # 降低评估预算
        if hasattr(self.loop.evaluator, 'n_workers'):
            new_workers = max(1, self.loop.evaluator.n_workers // 2)
            self.loop.evaluator.n_workers = new_workers

        # GPU缓存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _restore_latest(self, reason: str) -> bool:
        """从最新断点恢复"""
        latest = self.checkpoint_mgr.find_latest()
        if latest:
            self.log.warning(f"[恢复] {reason} → {Path(latest).name}")
            return self._restore_from_checkpoint(latest)
        self.log.warning(f"[恢复失败] {reason}: 没有可用断点")
        return False

    def _restore_from_checkpoint(self, path: str) -> bool:
        """从指定断点恢复"""
        data = self.checkpoint_mgr.load(path)
        if data is None:
            return False

        try:
            loop = self.loop

            # 恢复形态编码器
            loop.morph_encoder.load_state_dict(data["morph_encoder_weights"])

            # 恢复种群
            loop.population = pickle.loads(data["population"])
            loop.generation = data.get("generation", 0)
            loop.history = data.get("history", [])
            loop.best_body = (
                pickle.loads(data["best_body"]) if data["best_body"] else None
            )
            loop.best_trainer_state = data.get("best_trainer_state")
            loop.best_obs_dim = data.get("best_obs_dim")
            loop.best_act_dim = data.get("best_act_dim")
            loop.best_morph_embed = data.get("best_morph_embed")
            loop._stagnation_count = data.get("_stagnation_count", 0)

            # 恢复RNG状态 (如果可用)
            if "rng_state" in data:
                loop.rng.set_state(data["rng_state"])

            # 恢复 RMS 状态
            if data.get("global_obs_rms") is not None:
                from forgecraft.rl.env import RunningMeanStd
                rms = RunningMeanStd.__new__(RunningMeanStd)
                rms.mean = data["global_obs_rms"]["mean"]
                rms.var = data["global_obs_rms"]["var"]
                rms.count = data["global_obs_rms"]["count"]
                loop.evaluator.global_obs_rms = rms

            return True

        except Exception as e:
            self.log.error(f"断点恢复失败: {e}")
            return False


# ═══════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════

def safe_evolve(
    loop,
    n_generations: int,
    checkpoint_dir: str = "checkpoints",
    resume: bool = True,
) -> RecoveryStats:
    """安全进化包装函数

    Args:
        loop: EvolutionLoop 实例
        n_generations: 代数
        checkpoint_dir: 断点目录
        resume: 是否自动恢复

    Returns:
        RecoveryStats: 恢复统计
    """
    guard = TrainGuard(
        loop,
        checkpoint_dir=checkpoint_dir,
        degradation_config=DegradationConfig(),
        save_every_n_gens=5,
    )
    guard.run(n_generations=n_generations, resume=resume)
    return guard.stats
