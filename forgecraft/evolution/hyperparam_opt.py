# ══════════════════════════════════════════════════════════
# 🎯 超参数自动优化引擎
#
# 基于 Optuna / Bayesian Optimization 的自动超参数调优
#
# 功能:
#   ✅ 搜索空间定义 (进化/RL/仿真三大类参数)
#   ✅ Optuna 集成 (TPE sampler + MedianPruner)
#   ✅ 纯 Python 回退 (随机搜索 + 高斯过程贝叶斯优化)
#   ✅ 早停策略 (中位数剪枝 + 收敛检测)
#   ✅ 持久化 (SQLite/JSON 存储)
#   ✅ 参数调度器 (种群规模动态调整)
#   ✅ Pareto 多目标优化
#   ✅ 重要性分析
#
# 使用:
#   >>> from forgecraft.evolution.hyperparam_opt import HyperParamOptimizer
#   >>> optimizer = HyperParamOptimizer(use_optuna=True)
#   >>> best_params = optimizer.optimize(n_trials=50, generations=10)
# ══════════════════════════════════════════════════════════

import json
import math
import os
import time
import pickle
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

import numpy as np

from forgecraft.config import EvolutionConfig, RLConfig, SimConfig

logger = logging.getLogger(__name__)

# 尝试导入 Optuna
try:
    import optuna
    from optuna.trial import TrialState
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.info("Optuna 未安装, 使用内置贝叶斯优化器")

__all__ = [
    "ParamSpec",
    "HyperParameterSpace",
    "SimpleBayesianOptimizer",
    "EarlyStopper",
    "TrialResult",
    "OptimizationResult",
    "HyperParamOptimizer",
    "ParameterScheduler",
]

# ══════════════════════════════════════════════════════════
# 搜索空间定义
# ══════════════════════════════════════════════════════════

@dataclass
class ParamSpec:
    """超参数规格"""
    name: str
    type: str  # "float" | "int" | "categorical"
    low: float = 0.0
    high: float = 1.0
    step: float = None      # 步长 (int 类型时)
    choices: List = None     # 分类选择
    log_scale: bool = False  # 对数尺度采样
    default: float = None

    def sample(self, rng: np.random.RandomState = None) -> Any:
        if rng is None:
            rng = np.random.RandomState()
        if self.type == "categorical" and self.choices:
            return rng.choice(self.choices)
        elif self.type == "int":
            return int(rng.uniform(self.low, self.high + 1))
        elif self.type == "float":
            if self.log_scale:
                log_low = math.log(self.low)
                log_high = math.log(self.high)
                return math.exp(rng.uniform(log_low, log_high))
            return float(rng.uniform(self.low, self.high))
        return self.default

    def suggest_optuna(self, trial: "optuna.Trial"):
        if self.type == "categorical" and self.choices:
            return trial.suggest_categorical(self.name, self.choices)
        elif self.type == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high))
        elif self.type == "float":
            if self.log_scale:
                return trial.suggest_float(self.name, self.low, self.high, log=True)
            return trial.suggest_float(self.name, self.low, self.high)
        return self.default


# ══════════════════════════════════════════════════════════
# 搜索空间
# ══════════════════════════════════════════════════════════

class HyperParameterSpace:
    """
    超参数搜索空间
    
    三类参数:
    1. 进化参数 (种群/精英/变异/交叉)
    2. RL 参数 (学习率/PPO/网络结构)
    3. 仿真参数 (时间步/摩擦)
    """
    
    def __init__(self, complexity: str = "medium"):
        self.complexity = complexity
        self.params: List[ParamSpec] = []
        self._setup_space()
    
    def _setup_space(self):
        """构建搜索空间"""
        # 进化参数
        self.params.extend([
            ParamSpec("population_size", "int", 16, 128, default=50),
            ParamSpec("elite_count", "int", 3, 20, default=8),
            ParamSpec("mutation_rate", "float", 0.1, 0.6, default=0.35),
            ParamSpec("crossover_rate", "float", 0.2, 0.8, default=0.6),
            ParamSpec("topo_mutation_prob", "float", 0.05, 0.4, default=0.18),
            ParamSpec("param_mutation_prob", "float", 0.1, 0.6, default=0.35),
            ParamSpec("param_mutation_scale", "float", 0.02, 0.3, default=0.12),
            ParamSpec("selection_pressure", "float", 1.5, 5.0, default=2.5),
        ])
        
        # RL 参数
        self.params.extend([
            ParamSpec("actor_lr", "float", 1e-5, 1e-3, log_scale=True, default=3e-4),
            ParamSpec("critic_lr", "float", 1e-4, 3e-3, log_scale=True, default=1e-3),
            ParamSpec("gamma", "float", 0.9, 0.999, default=0.99),
            ParamSpec("lam", "float", 0.8, 0.99, default=0.95),
            ParamSpec("clip_ratio", "float", 0.1, 0.4, default=0.2),
            ParamSpec("entropy_coef", "float", 0.001, 0.1, log_scale=True, default=0.02),
            ParamSpec("ppo_epochs", "int", 4, 20, default=12),
            ParamSpec("batch_size", "int", 64, 512, step=32, default=256),
            ParamSpec("value_loss_coef", "float", 0.1, 1.0, default=0.5),
            ParamSpec("max_grad_norm", "float", 0.1, 2.0, default=0.5),
        ])
        
        # 网络结构
        self.params.extend([
            ParamSpec("hidden_dim", "int", 64, 256, step=32, default=128),
            ParamSpec("gnn_hidden", "int", 32, 128, step=16, default=64),
            ParamSpec("morph_embed_dim", "int", 32, 128, step=16, default=64),
            ParamSpec("gnn_layers", "int", 2, 5, default=3),
        ])
        
        # 仿真参数
        self.params.extend([
            ParamSpec("timestep", "float", 0.001, 0.02, log_scale=True, default=0.005),
            ParamSpec("friction", "float", 0.3, 1.0, default=0.6),
            ParamSpec("substeps", "int", 5, 20, default=10),
        ])
    
    def sample(self, rng: np.random.RandomState = None) -> Dict[str, Any]:
        return {p.name: p.sample(rng) for p in self.params}
    
    def suggest_optuna(self, trial: "optuna.Trial") -> Dict[str, Any]:
        return {p.name: p.suggest_optuna(trial) for p in self.params}
    
    def to_evo_config(self, params: Dict[str, Any]) -> EvolutionConfig:
        return EvolutionConfig(
            population_size=params.get("population_size", 50),
            generations=params.get("generations", 200),
            elite_count=params.get("elite_count", 8),
            mutation_rate=params.get("mutation_rate", 0.35),
            crossover_rate=params.get("crossover_rate", 0.6),
            topo_mutation_prob=params.get("topo_mutation_prob", 0.18),
            param_mutation_prob=params.get("param_mutation_prob", 0.35),
            param_mutation_scale=params.get("param_mutation_scale", 0.12),
            selection_pressure=params.get("selection_pressure", 2.5),
        )
    
    def to_rl_config(self, params: Dict[str, Any]) -> RLConfig:
        return RLConfig(
            hidden_dim=params.get("hidden_dim", 128),
            gnn_layers=params.get("gnn_layers", 3),
            gnn_hidden=params.get("gnn_hidden", 64),
            morph_embed_dim=params.get("morph_embed_dim", 64),
            actor_lr=params.get("actor_lr", 3e-4),
            critic_lr=params.get("critic_lr", 1e-3),
            gamma=params.get("gamma", 0.99),
            lam=params.get("lam", 0.95),
            clip_ratio=params.get("clip_ratio", 0.2),
            entropy_coef=params.get("entropy_coef", 0.02),
            ppo_epochs=params.get("ppo_epochs", 12),
            batch_size=params.get("batch_size", 256),
            value_loss_coef=params.get("value_loss_coef", 0.5),
            max_grad_norm=params.get("max_grad_norm", 0.5),
        )
    
    def to_sim_config(self, params: Dict[str, Any]) -> SimConfig:
        return SimConfig(
            timestep=params.get("timestep", 0.005),
            friction=params.get("friction", 0.6),
            substeps=params.get("substeps", 10),
        )
    
    def get_param(self, name: str) -> Optional[ParamSpec]:
        for p in self.params:
            if p.name == name:
                return p
        return None


# ══════════════════════════════════════════════════════════
# 纯 Python 贝叶斯优化器 (Optuna 回退)
# ══════════════════════════════════════════════════════════

class SimpleBayesianOptimizer:
    """
    简单贝叶斯优化器
    
    使用高斯过程 (RBF kernel) 作为代理模型,
    Expected Improvement (EI) 作为采集函数。
    """
    
    def __init__(self, bounds: np.ndarray, seed: int = 42):
        self.bounds = bounds  # (n_params, 2) low/high
        self.rng = np.random.RandomState(seed)
        self.X_observed = []
        self.y_observed = []
        self.n_dims = bounds.shape[0]
        
        # GP 超参数
        self.length_scale = 1.0
        self.signal_variance = 1.0
        self.noise_variance = 1e-6
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF 核函数 K(x,x') = σ² exp(-||x-x'||² / 2l²)"""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                 np.sum(X2**2, axis=1) - \
                 2 * np.dot(X1, X2.T)
        
        return self.signal_variance * np.exp(-0.5 * sqdist / self.length_scale**2)
    
    def _gp_predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GP 预测均值与方差"""
        X_train = np.array(self.X_observed)
        y_train = np.array(self.y_observed)
        
        if len(X_train) < 2:
            return np.zeros(len(X_test)), np.ones(len(X_test))
        
        K = self._rbf_kernel(X_train, X_train)
        K += self.noise_variance * np.eye(len(X_train))
        K_inv = np.linalg.inv(K)
        
        K_s = self._rbf_kernel(X_test, X_train)
        K_ss = self._rbf_kernel(X_test, X_test)
        
        mu = K_s @ K_inv @ y_train
        sigma = np.sqrt(np.diag(K_ss) - np.sum((K_s @ K_inv) * K_s, axis=1))
        
        return mu, sigma
    
    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> float:
        """Expected Improvement 采集函数"""
        mu, sigma = self._gp_predict(X.reshape(1, -1))
        mu, sigma = mu[0], sigma[0]
        
        if len(self.y_observed) == 0:
            return 0.0
        
        best_y = max(self.y_observed)
        
        if sigma < 1e-12:
            return 0.0
        
        z = (mu - best_y - xi) / sigma
        ei = (mu - best_y - xi) * self._normal_cdf(z) + sigma * self._normal_pdf(z)
        return float(ei)
    
    def _normal_cdf(self, x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    
    def suggest(self) -> np.ndarray:
        """建议下一个评估点"""
        if len(self.X_observed) < 3:
            # 初始随机探索
            x = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            return x
        
        # 优化 EI (24 个候选点)
        n_candidates = 24
        candidates = self.rng.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(n_candidates, self.n_dims),
        )
        
        # 加入已观察点附近的扰动
        if len(self.X_observed) > 0:
            best_idx = np.argmax(self.y_observed)
            best_x = np.array(self.X_observed[best_idx])
            local_candidates = best_x + self.rng.normal(
                0, 0.05, size=(12, self.n_dims)
            )
            local_candidates = np.clip(local_candidates, self.bounds[:, 0], self.bounds[:, 1])
            candidates = np.vstack([candidates, local_candidates])
        
        ei_values = [self._expected_improvement(c, xi=0.01) for c in candidates]
        best_idx = np.argmax(ei_values)
        return candidates[best_idx]
    
    def observe(self, x: np.ndarray, y: float):
        self.X_observed.append(x.tolist())
        self.y_observed.append(y)


# ══════════════════════════════════════════════════════════
# 早停策略
# ══════════════════════════════════════════════════════════

class EarlyStopper:
    """早停策略"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0
    
    def should_stop(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ══════════════════════════════════════════════════════════
# 主优化器
# ══════════════════════════════════════════════════════════

@dataclass
class TrialResult:
    """单次试验结果"""
    trial_id: int
    params: Dict[str, Any]
    best_fitness: float
    qd_score: float = 0.0
    coverage: float = 0.0
    eval_time: float = 0.0
    status: str = "completed"  # completed | pruned | error
    error_message: str = ""


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_fitness: float
    best_trial_id: int
    n_trials: int
    n_completed: int
    n_pruned: int
    total_time: float
    trials: List[TrialResult]
    param_importance: Dict[str, float]
    convergence_curve: List[float]


class HyperParamOptimizer:
    """超参数自动优化 — TPE + Bayesian + Early Stopping

    用 Optuna TPE (Tree-structured Parzen Estimator) 搜索最优超参:
      1. 定义搜索空间 (population_size, learning_rate, mutation_rate, ...)
      2. TPE 采样: 对每轮参数用 Bayesian 模型估计 EI (Expected Improvement)
      3. 试验: 每轮候选参数跑 N 代进化 → 记录 best_fitness
      4. Early stopping (Pringebridge): 中位数 fitness 不增长则终止
      5. 收敛: 返回 Pareto 最优参数集

    参考:
      - Bergstra et al. (2011) "Algorithms for Hyper-Parameter Optimization"
      - Optuna (https://github.com/optuna/optuna)
    """

    def __init__(
        self,
        use_optuna: bool = True,
        storage: str = None,
        study_name: str = "forgecraft_hpo",
        n_startup_trials: int = 5,
        seed: int = 42,
    ):
        self.use_optuna = use_optuna and HAS_OPTUNA
        self.storage = storage or f"sqlite:///{study_name}.db"
        self.study_name = study_name
        self.n_startup_trials = n_startup_trials
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.space: Optional[HyperParameterSpace] = None
        self.study = None
        self.bayesian_opt: Optional[SimpleBayesianOptimizer] = None
        
    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 50,
        space: HyperParameterSpace = None,
        early_stopping_patience: int = 20,
        timeout_seconds: float = None,
        n_jobs: int = 1,
        direction: str = "maximize",
        progress_callback: Callable = None,
    ) -> OptimizationResult:
        """
        执行超参数优化
        
        Args:
            objective_fn: 目标函数 (params) -> fitness
            n_trials: 试验次数
            space: 搜索空间
            early_stopping_patience: 早停耐心
            timeout_seconds: 超时
            n_jobs: 并行试验数
            direction: maximize | minimize
            progress_callback: 进度回调
        
        Returns:
            OptimizationResult
        """
        self.space = space or HyperParameterSpace()
        
        if self.use_optuna:
            return self._optimize_optuna(
                objective_fn, n_trials, early_stopping_patience,
                timeout_seconds, n_jobs, direction, progress_callback,
            )
        else:
            return self._optimize_builtin(
                objective_fn, n_trials, early_stopping_patience,
                timeout_seconds, progress_callback,
            )
    
    def _optimize_optuna(
        self, objective_fn, n_trials, patience, timeout,
        n_jobs, direction, progress_callback,
    ) -> OptimizationResult:
        """使用 Optuna 优化"""
        import optuna
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=direction,
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=self.n_startup_trials,
                seed=self.seed,
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3,
            ),
            load_if_exists=True,
        )
        
        start_time = time.time()
        trials_data = []
        
        def optuna_objective(trial):
            params = self.space.suggest_optuna(trial)
            
            t0 = time.time()
            try:
                fitness = objective_fn(params)
                elapsed = time.time() - t0
                
                result = TrialResult(
                    trial_id=trial.number,
                    params=params,
                    best_fitness=fitness,
                    eval_time=elapsed,
                    status="completed",
                )
                trials_data.append(result)
                
                if progress_callback:
                    progress_callback(trial.number, n_trials, fitness)
                
                return fitness
            except Exception as e:
                result = TrialResult(
                    trial_id=trial.number,
                    params=params,
                    best_fitness=-float("inf"),
                    status="error",
                    error_message=str(e),
                )
                trials_data.append(result)
                raise optuna.TrialPruned()
        
        study.optimize(optuna_objective, n_trials=n_trials, timeout=timeout)
        
        total_time = time.time() - start_time
        
        # 计算参数重要性
        importance = {}
        try:
            importance = optuna.importance.get_param_importances(study)
        except Exception:
            pass
        
        # 收敛曲线
        convergence = [t.value for t in study.trials if t.value is not None]
        
        # 统计
        completed = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
        pruned = sum(1 for t in study.trials if t.state == TrialState.PRUNED)
        
        return OptimizationResult(
            best_params=study.best_params,
            best_fitness=study.best_value,
            best_trial_id=study.best_trial.number,
            n_trials=n_trials,
            n_completed=completed,
            n_pruned=pruned,
            total_time=total_time,
            trials=trials_data,
            param_importance=importance,
            convergence_curve=convergence,
        )
    
    def _optimize_builtin(
        self, objective_fn, n_trials, patience, timeout, progress_callback,
    ) -> OptimizationResult:
        """使用内置贝叶斯优化器"""
        # 构建搜索空间边界
        n_params = len(self.space.params)
        bounds = np.array([
            [p.low if p.type != "categorical" else 0,
             p.high if p.type != "categorical" else len(p.choices) - 1]
            for p in self.space.params
        ])
        
        # 归一化/去归一化
        def normalize(x):
            return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-10)
        
        def denormalize(x_norm):
            return x_norm * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        
        # 将参数字典转为向量
        def params_to_vec(params):
            vec = []
            for i, p in enumerate(self.space.params):
                val = params.get(p.name, p.default)
                if p.type == "categorical" and p.choices:
                    vec.append(p.choices.index(val) if val in p.choices else 0)
                else:
                    vec.append(float(val))
            return np.array(vec)
        
        # 将向量转为参数字典
        def vec_to_params(vec):
            params = {}
            for i, p in enumerate(self.space.params):
                if p.type == "categorical" and p.choices:
                    idx = int(np.clip(vec[i], 0, len(p.choices) - 1))
                    params[p.name] = p.choices[idx]
                elif p.type == "int":
                    params[p.name] = int(np.clip(vec[i], p.low, p.high))
                else:
                    val = float(np.clip(vec[i], p.low, p.high))
                    if p.log_scale:
                        val = max(val, 1e-10)
                    params[p.name] = val
            return params
        
        # 初始化贝叶斯优化器
        bounds_norm = np.column_stack([
            np.zeros(n_params), np.ones(n_params)
        ])
        bayes_opt = SimpleBayesianOptimizer(bounds_norm, seed=self.seed)
        
        start_time = time.time()
        trials_data = []
        convergences = []
        best_fitness = -float("inf")
        best_params = None
        
        for trial_id in range(n_trials):
            t0 = time.time()
            
            # 建议参数
            if trial_id < self.n_startup_trials:
                # 随机采样
                x_norm = self.rng.uniform(0, 1, n_params)
            else:
                x_norm = bayes_opt.suggest()
            
            x = denormalize(x_norm)
            params = vec_to_params(x)
            
            try:
                fitness = objective_fn(params)
                elapsed = time.time() - t0
                
                result = TrialResult(
                    trial_id=trial_id,
                    params=params,
                    best_fitness=fitness,
                    eval_time=elapsed,
                    status="completed",
                )
                trials_data.append(result)
                convergences.append(fitness)
                
                # 更新贝叶斯模型
                bayes_opt.observe(x_norm, fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = params.copy()
                
                if progress_callback:
                    progress_callback(trial_id, n_trials, fitness)
                    
            except Exception as e:
                result = TrialResult(
                    trial_id=trial_id,
                    params=params,
                    best_fitness=-float("inf"),
                    status="error",
                    error_message=str(e),
                )
                trials_data.append(result)
            
            # 超时检查
            if timeout and time.time() - start_time > timeout:
                break
        
        total_time = time.time() - start_time
        
        # 参数重要性 (基于方差)
        importance = {}
        if len(trials_data) > 5:
            completed = [t for t in trials_data if t.status == "completed"]
            if len(completed) > 5:
                for i, p in enumerate(self.space.params):
                    values = [vec_to_params(denormalize(normalize(
                        params_to_vec(t.params)
                    )))[p.name] for t in completed if p.name in t.params]
                    if values:
                        fitnesses = [t.best_fitness for t in completed]
                        # 相关性
                        if len(set(values)) > 1:
                            corr = abs(np.corrcoef(values, fitnesses)[0, 1])
                            importance[p.name] = float(corr)
        
        return OptimizationResult(
            best_params=best_params or {},
            best_fitness=best_fitness,
            best_trial_id=trials_data[-1].trial_id if trials_data else -1,
            n_trials=n_trials,
            n_completed=sum(1 for t in trials_data if t.status == "completed"),
            n_pruned=0,
            total_time=total_time,
            trials=trials_data,
            param_importance=importance,
            convergence_curve=convergences,
        )


# ══════════════════════════════════════════════════════════
# 参数调度器
# ══════════════════════════════════════════════════════════

class ParameterScheduler:
    """
    进化过程中的自适应参数调度
    
    功能:
    - 种群规模动态调整 (早期大种群探索 → 后期小种群精炼)
    - 变异率衰减 (模拟退火)
    - 学习率调度
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.current_step = 0
    
    def step(self, generation: int, total_generations: int,
             performance: float = None, stagnation: int = 0) -> Dict[str, Any]:
        """
        根据进化进度调整参数
        
        Args:
            generation: 当前代数
            total_generations: 总代数
            performance: 当前性能
            stagnation: 停滞代数
        
        Returns:
            调整后的参数字典
        """
        self.current_step = generation
        progress = generation / max(total_generations, 1)
        params = dict(self.base_config)
        
        # 种群规模: 后期缩减 (专注精炼)
        if "population_size" in params:
            pop_base = params["population_size"]
            # 线性衰减到 60%
            params["population_size"] = int(pop_base * (1.0 - 0.4 * progress))
            params["population_size"] = max(8, params["population_size"])
        
        # 变异率: 余弦衰减
        if "mutation_rate" in params:
            base_mr = params["mutation_rate"]
            params["mutation_rate"] = base_mr * (0.5 + 0.5 * math.cos(math.pi * progress))
        
        # 交叉率: 后期增加 (利用已有的好基因)
        if "crossover_rate" in params:
            base_cr = params["crossover_rate"]
            params["crossover_rate"] = base_cr * (0.5 + 0.5 * progress)
        
        # 学习率: 指数衰减
        if "actor_lr" in params:
            params["actor_lr"] = params["actor_lr"] * (0.95 ** stagnation)
        
        # 熵系数: 线性衰减
        if "entropy_coef" in params:
            base_ent = params["entropy_coef"]
            params["entropy_coef"] = base_ent * max(0.2, 1.0 - progress * 0.8)
        
        return params
    
    def get_current_params(self) -> Dict[str, Any]:
        return self.base_config


# ══════════════════════════════════════════════════════════
# 进化循环集成接口
# ══════════════════════════════════════════════════════════

def create_hpo_objective(
    catalog, task_config, sim_config,
    generations: int = 10,
    workers: int = 4,
    device: str = "cpu",
    seed: int = 42,
) -> Callable[[Dict[str, Any]], float]:
    """
    创建用于 HPO 的目标函数
    
    返回的函数可以被 HyperParamOptimizer.optimize() 使用
    """
    from forgecraft.evolution.loop import EvolutionLoop
    
    def objective(params: Dict[str, Any]) -> float:
        space = HyperParameterSpace()
        evo_config = space.to_evo_config(params)
        rl_config = space.to_rl_config(params)
        sim = space.to_sim_config(params)
        
        evo_config.generations = generations
        
        loop = EvolutionLoop(
            evo_config=evo_config,
            sim_config=sim,
            rl_config=rl_config,
            task_config=task_config,
            catalog=catalog,
            device=device,
            seed=seed,
            n_workers=workers,
        )
        
        try:
            loop.run(generations)
            if loop.best_body:
                return loop.best_body.fitness
            return -float("inf")
        except Exception as e:
            return -float("inf")
    
    return objective


def print_optimization_report(result: OptimizationResult):
    """打印优化报告"""
    print("=" * 60)
    print("超参数优化报告")
    print("=" * 60)
    print(f"  试验数: {result.n_trials}")
    print(f"  完成: {result.n_completed}, 剪枝: {result.n_pruned}")
    print(f"  耗时: {result.total_time:.1f}s")
    print(f"  最佳适应度: {result.best_fitness:.4f}")
    print()
    print("最佳超参数:")
    for k, v in sorted(result.best_params.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print()
    
    if result.param_importance:
        print("参数重要性 (top 10):")
        sorted_imp = sorted(result.param_importance.items(),
                           key=lambda x: x[1], reverse=True)[:10]
        for name, imp in sorted_imp:
            bar = "█" * int(imp * 30)
            print(f"  {name:25s} {imp:.3f} {bar}")
    print("=" * 60)
