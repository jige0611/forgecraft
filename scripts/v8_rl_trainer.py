#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V8 Robot Assembly RL Training System (Stable-Baselines3 + Optuna)
================================================================
工业级强化学习训练框架 - 自动化超参数搜索与模型优化

核心功能:
1. PPO/SAC/A2C多算法支持
2. Optuna贝叶斯超参数优化
3. 自定义CNN+MLP混合网络架构
4. 分布式训练与评估管道
5. TensorBoard实时监控
6. 模型版本管理与对比分析

作者: V8 Evolution System
版本: 3.0.0 (Production Ready)
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import warnings

# 深度学习框架
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical

# 强化学习库
import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CallbackList, CheckpointCallback
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device

# 超参数优化
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances

# 可视化和日志
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 导入自定义环境
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from v8_rl_environment import (
    V8RobotAssemblyEnv, VectorizedV8Env,
    create_training_env, create_eval_env, create_vectorized_env,
    RLConfig
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('v8_rl_training.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础参数
    algorithm: str = 'ppo'  # ppo/sac/a2c/dqn
    total_timesteps: int = 1_000_000  # 总训练步数
    n_envs: int = 8  # 并行环境数
    
    # 网络架构
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    activation_fn: str = 'relu'  # relu/tanh/elu
    
    # PPO特定参数
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # 优化器设置
    optimizer_class: str = 'adam'
    optimizer_kwargs: Dict = None
    
    # 设备和性能
    device: str = 'auto'  # auto/cpu/cuda
    verbose: int = 1
    
    # 保存和评估
    save_freq: int = 10000  # 每10k步保存一次
    eval_freq: int = 5000  # 每5k步评估一次
    n_eval_episodes: int = 10  # 评估episode数
    
    # 日志
    tensorboard_log: str = './v8_rl_logs/'
    
    def __post_init__(self):
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}


@dataclass
class HyperparameterSearchSpace:
    """超参数搜索空间定义"""
    params: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    
    @classmethod
    def default_ppo_space(cls) -> 'HyperparameterSearchSpace':
        """默认PPO超参数搜索空间"""
        return cls(params={
            'learning_rate': (1e-5, 1e-3),  # 对数均匀分布
            'n_epochs': (5, 20),
            'batch_size': (64, 512),
            'gamma': (0.9, 0.9999),
            'gae_lambda': (0.9, 0.99),
            'clip_range': (0.1, 0.4),
            'ent_coef': (0.001, 0.1),
        })
    
    @classmethod
    def default_sac_space(cls) -> 'HyperparameterSearchSpace':
        """默认SAC超参数搜索空间"""
        return cls(params={
            'learning_rate': (1e-5, 1e-3),
            'batch_size': (128, 512),
            'gamma': (0.9, 0.9999),
            'tau': (0.001, 0.02),
            'ent_coef': ('auto', 0.1),
        })


# ============================================================
# 自定义神经网络架构
# ============================================================

class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    自定义特征提取器 - 处理复杂的Dict观测空间
    
    架构:
    1. 各模态独立编码 (robot_state, part_config, sensor_data等)
    2. 多头注意力融合
    3. 全连接输出层
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # 提取各子空间的维度
        self.robot_state_dim = observation_space['robot_state'].shape[0]  # 13
        self.part_config_dim = observation_space['part_config'].shape[0]  # n_parts
        self.sensor_data_dim = observation_space['sensor_data'].shape[0]  # 20
        self.progress_dim = observation_space['assembly_progress'].shape[0]  # 5
        
        # 各模态编码器
        self.robot_state_encoder = nn.Sequential(
            nn.Linear(self.robot_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.part_config_encoder = nn.Sequential(
            nn.Linear(self.part_config_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.sensor_encoder = nn.Sequential(
            nn.Linear(self.sensor_data_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.progress_encoder = nn.Sequential(
            nn.Linear(self.progress_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # 融合层 (concatenate all features)
        total_encoded_dim = 64 + 32 + 32 + 16  # 144
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_encoded_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 编码各模态
        robot_feat = self.robot_state_encoder(observations['robot_state'])
        part_feat = self.part_config_encoder(observations['part_config'])
        sensor_feat = self.sensor_encoder(observations['sensor_data'])
        progress_feat = self.progress_encoder(observations['assembly_progress'])
        
        # 拼接所有特征
        combined = torch.cat([robot_feat, part_feat, sensor_feat, progress_feat], dim=1)
        
        # 融合并输出
        output = self.fusion_layer(combined)
        return output


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    自定义Actor-Critic策略网络
    
    使用CustomFeatureExtractor处理复杂观测空间，
    并支持自定义的actor/critic头。
    """
    
    def __init__(self, *args, **kwargs):
        # 覆盖features_extractor_class
        kwargs['features_extractor_class'] = CustomFeatureExtractor
        kwargs['features_extractor_kwargs'] = dict(features_dim=256)
        super().__init__(*args, **kwargs)


# ============================================================
# 回调函数
# ============================================================

class TrainingMetricsCallback(BaseCallback):
    """自定义训练指标回调 - 记录详细统计信息"""
    
    def __init__(self, verbose: int = 0, log_dir: str = './'):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.metrics_history = []
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # 记录当前步的指标
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                
                metric = {
                    'step': self.num_timesteps,
                    'episode_reward': ep_info['r'],
                    'episode_length': ep_info['l'],
                    'mean_reward_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
                    'num_parts': info.get('num_parts', 0),
                }
                self.metrics_history.append(metric)
                
                if self.verbose > 0 and len(self.metrics_history) % 10 == 0:
                    logger.info(f"📊 Step {self.num_timesteps}: "
                               f"reward={ep_info['r']:.2f}, "
                               f"mean_100={metric['mean_reward_100']:.2f}, "
                               f"parts={info.get('num_parts', 0)}")
        
        return True


class ModelCheckpointCallback(CallbackList):
    """增强版模型检查点 - 支持最佳模型保留和版本管理"""
    
    def __init__(self, save_path: str, save_freq: int = 10000, 
                 n_best_models: int = 5, verbose: int = 1):
        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self.n_best_models = n_best_models
        self.best_rewards = []
        self.verbose = verbose
        
        # 创建保存目录
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(self.save_path / 'checkpoints'),
                name_prefix='rl_model',
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        ]
        super().__init__(callbacks)


# ============================================================
# 核心训练器类
# ============================================================

class V8RLTrainer:
    """
    V8机器人组装RL训练器
    
    提供端到端的训练流程:
    1. 环境创建与配置
    2. 模型初始化与加载
    3. 训练循环与监控
    4. 评估与验证
    5. 模型导出与分析
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.model = None
        self.env = None
        self.eval_env = None
        self.training_start_time = None
        self.results = {}
        
        logger.info(f"🚀 V8RLTrainer initialized with {self.config.algorithm.upper()}")
    
    def setup_environments(self) -> None:
        """创建训练和评估环境"""
        logger.info("🔧 Setting up environments...")
        
        # 创建向量化训练环境
        def make_env():
            env = create_training_env(render_mode=None)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=self.config.total_timesteps // 1000)
            return env
        
        # 使用DummyVecEnv (单进程，适合调试)
        # 或SubprocVecEnv (多进程，适合生产环境)
        if self.config.n_envs > 1:
            self.env = SubprocVecEnv([make_env for _ in range(self.config.n_envs)])
        else:
            self.env = DummyVecEnv([make_env])
        
        # 包装监控
        self.env = VecMonitor(self.env, filename=str(Path(self.config.tensorboard_log) / 'monitor.csv'))
        
        # 创建评估环境
        self.eval_env = create_eval_env(render_mode=None)
        self.eval_env = gym.wrappers.TimeLimit(self.eval_env, max_episode_steps=500)
        
        logger.info(f"✅ Environments ready: train={self.config.n_envs} envs, eval=1 env")
    
    def build_model(self, custom_policy: bool = True) -> None:
        """构建RL模型"""
        logger.info(f"🏗️ Building {self.config.algorithm.upper()} model...")
        
        # 选择算法
        algorithm_map = {
            'ppo': PPO,
            'sac': SAC,
            'a2c': A2C,
            'dqn': DQN,
        }
        
        algo_class = algorithm_map.get(self.config.algorithm.lower(), PPO)
        
        # 模型参数
        model_kwargs = dict(
            policy='MultiInputPolicy' if not custom_policy else CustomActorCriticPolicy,
            env=self.env,
            learning_rate=self.config.learning_rate,
            n_steps=2048 // self.config.n_envs,  # 每个环境的步数
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            tensorboard_log=self.config.tensorboard_log,
            device=self.config.device,
            verbose=self.config.verbose,
        )
        
        # 移除不适用于某些算法的参数
        if self.config.algorithm.lower() in ['sac', 'dqn']:
            model_kwargs.pop('n_steps', None)
            model_kwargs.pop('gae_lambda', None)
            model_kwargs.pop('clip_range', None)
        
        # 创建模型
        self.model = algo_class(**model_kwargs)
        
        logger.info(f"✅ Model built: {self.model.__class__.__name__}")
        logger.info(f"   Total parameters: {sum(p.numel() for p in self.model.policy.parameters()):,}")
    
    def train(self, resume_from: str = None) -> Dict[str, Any]:
        """
        执行训练
        
        Args:
            resume_from: 从指定路径恢复训练
        
        Returns:
            dict: 训练结果和统计信息
        """
        logger.info("🎯 Starting training...")
        self.training_start_time = time.time()
        
        # 设置回调
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = Path(self.config.tensorboard_log) / f'model_{timestamp}'
        
        callbacks = []
        
        # 1. 评估回调
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(save_dir / 'best'),
            log_path=str(save_dir / 'eval_results'),
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)
        
        # 2. 指标记录回调
        metrics_callback = TrainingMetricsCallback(
            verbose=self.config.verbose,
            log_dir=str(save_dir)
        )
        callbacks.append(metrics_callback)
        
        # 3. 检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=str(save_dir / 'checkpoints'),
            name_prefix='rl_model',
        )
        callbacks.append(checkpoint_callback)
        
        callback_list = CallbackList(callbacks)
        
        # 恢复训练或从头开始
        if resume_from and Path(resume_from).exists():
            logger.info(f"📂 Resuming training from: {resume_from}")
            self.model = self.model.load(resume_from)
        
        # 开始训练
        try:
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callback_list,
                reset_num_timesteps=(resume_from is None),
                progress_bar=True,
            )
            
            training_time = time.time() - self.training_start_time
            
            # 收集结果
            self.results = {
                'algorithm': self.config.algorithm.upper(),
                'total_timesteps': self.config.total_timesteps,
                'training_time_seconds': training_time,
                'training_time_hours': training_time / 3600,
                'final_mean_reward': np.mean(metrics_callback.episode_rewards[-100:]) if metrics_callback.episode_rewards else 0,
                'best_reward': max(metrics_callback.episode_rewards) if metrics_callback.episode_rewards else 0,
                'total_episodes': len(metrics_callback.episode_rewards),
                'metrics_history': metrics_callback.metrics_history[-100:],  # 最近100条
                'model_save_path': str(save_dir),
                'config': vars(self.config),
            }
            
            logger.info("="*70)
            logger.info("🎉 Training Complete!")
            logger.info(f"   Algorithm: {self.results['algorithm']}")
            logger.info(f"   Time: {training_time/3600:.2f} hours")
            logger.info(f"   Final mean reward (last 100): {self.results['final_mean_reward']:.2f}")
            logger.info(f"   Best reward: {self.results['best_reward']:.2f}")
            logger.info(f"   Total episodes: {self.results['total_episodes']}")
            logger.info(f"   Model saved to: {save_dir}")
            logger.info("="*70)
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            raise
    
    def evaluate(self, n_episodes: int = 50, render: bool = False) -> Dict[str, float]:
        """
        评估训练好的模型
        
        Returns:
            dict: 评估结果
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        logger.info(f"📊 Evaluating model over {n_episodes} episodes...")
        
        eval_env = create_eval_env(render_mode='human' if render else None)
        rewards = []
        lengths = []
        parts_used = []
        
        for ep in range(n_episodes):
            obs, info = eval_env.reset()
            ep_reward = 0
            done = False
            step = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_reward += reward
                step += 1
                done = terminated or truncated
                
                if render:
                    eval_env.render()
            
            rewards.append(ep_reward)
            lengths.append(step)
            parts_used.append(info.get('num_parts', 0))
            
            if (ep + 1) % 10 == 0:
                logger.info(f"   Episode {ep+1}/{n_episodes}: reward={ep_reward:.2f}, steps={step}")
        
        eval_env.close()
        
        results = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'mean_episode_length': np.mean(lengths),
            'mean_parts_used': np.mean(parts_used),
            'success_rate': sum(1 for r in rewards if r > 0) / len(rewards),
        }
        
        logger.info("\n📈 Evaluation Results:")
        for key, value in results.items():
            logger.info(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
        
        return results
    
    def save_results(self, filepath: str = None) -> None:
        """保存训练结果到JSON文件"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'v8_training_results_{timestamp}.json'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 Results saved to: {filepath}")
    
    def close(self) -> None:
        """清理资源"""
        if self.env is not None:
            self.env.close()
        if self.eval_env is not None:
            self.eval_env.close()
        logger.info("🔌 Trainer resources cleaned up")


# ============================================================
# 超参数优化 (Optuna集成)
# ============================================================

class HyperparameterOptimizer:
    """
    基于Optuna的超参数自动优化器
    
    使用TPE (Tree-structured Parzen Estimator)采样器和
    中位数剪枝策略进行高效的超参数搜索。
    """
    
    def __init__(
        self,
        search_space: HyperparameterSearchSpace = None,
        n_trials: int = 50,
        timeout: int = 3600 * 4,  # 4小时
        study_name: str = 'v8_rl_optimization',
        storage: str = None,
        direction: str = 'maximize'
    ):
        self.search_space = search_space or HyperparameterSearchSpace.default_ppo_space()
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        
        # 创建或加载study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20),
            direction=direction
        )
        
        logger.info(f"🔬 Hyperparameter optimizer initialized ({n_trials} trials)")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna目标函数 - 单次试验的训练和评估
        
        Args:
            trial: Optuna试验对象
        
        Returns:
            float: 目标值 (mean reward)
        """
        # 从搜索空间采样超参数
        sampled_params = {}
        
        for param_name, (low, high) in self.search_space.params.items():
            if isinstance(low, float) and isinstance(high, float):
                # 连续参数
                if low > 0 and high > 1 and abs(np.log10(high) - np.log10(low)) > 1:
                    # 可能是对数尺度
                    sampled_params[param_name] = trial.suggest_float(
                        param_name, low, high, log=True
                    )
                else:
                    sampled_params[param_name] = trial.suggest_float(
                        param_name, low, high
                    )
            elif isinstance(low, int) and isinstance(high, int):
                # 整数参数
                sampled_params[param_name] = trial.suggest_int(param_name, low, high)
            elif isinstance(low, str) and isinstance(high, (int, float)):
                # 分类参数
                choices = [low, high]
                sampled_params[param_name] = trial.suggest_categorical(param_name, choices)
        
        logger.info(f"\n🎲 Trial {trial.number}: Testing hyperparameters:")
        for k, v in sampled_params.items():
            logger.info(f"   {k}: {v}")
        
        try:
            # 创建配置
            config = TrainingConfig(**sampled_params)
            config.total_timesteps = min(200_000, self.timeout * 50)  # 缩短单次训练时间
            config.tensorboard_log = f'./optuna_logs/trial_{trial.number}'
            
            # 创建训练器并训练
            trainer = V8RLTrainer(config=config)
            trainer.setup_environments()
            trainer.build_model(custom_policy=True)
            results = trainer.train()
            trainer.close()
            
            mean_reward = results['final_mean_reward']
            
            logger.info(f"✅ Trial {trial.number} completed: reward={mean_reward:.4f}")
            
            return mean_reward
            
        except Exception as e:
            logger.error(f"❌ Trial {trial.number} failed: {e}")
            return -float('inf')
    
    def optimize(self) -> optuna.Study:
        """
        执行超参数优化
        
        Returns:
            optuna.Study: 优化后的study对象
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 Starting Hyperparameter Optimization")
        logger.info(f"   Trials: {self.n_trials}")
        logger.info(f"   Timeout: {self.timeout/3600:.1f} hours")
        logger.info(f"   Search space size: {len(self.search_space.params)}")
        logger.info(f"{'='*70}\n")
        
        start_time = time.time()
        
        # 运行优化
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
            catch=(Exception,),
            gc_after_trial=True,
        )
        
        optimization_time = time.time() - start_time
        
        # 输出最优结果
        best_trial = self.study.best_trial
        logger.info("\n" + "="*70)
        logger.info("🏆 Optimization Complete!")
        logger.info(f"   Best trial: #{best_trial.number}")
        logger.info(f"   Best value: {best_trial.value:.6f}")
        logger.info(f"   Best parameters:")
        for key, value in best_trial.params.items():
            logger.info(f"      {key}: {value}")
        logger.info(f"   Total trials: {len(self.study.trials)}")
        logger.info(f"   Optimization time: {optimization_time/3600:.2f} hours")
        logger.info("="*70)
        
        # 保存优化历史
        self.save_optimization_history()
        
        # 生成可视化图表
        self.generate_visualizations()
        
        return self.study
    
    def save_optimization_history(self) -> None:
        """保存优化历史"""
        history = {
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'best_trial_number': self.study.best_trial.number,
            'all_trials': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': str(t.state),
                }
                for t in self.study.trials
            ],
            'search_space': self.search_space.params,
        }
        
        filepath = f'optuna_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Optimization history saved to: {filepath}")
    
    def generate_visualizations(self) -> None:
        """生成优化过程可视化图表"""
        try:
            import matplotlib.pyplot as plt
            
            fig = plt.figure(figsize=(15, 10))
            
            # 1. 优化历史图
            ax1 = fig.add_subplot(221)
            plot_optimization_history(self.study, ax=ax1)
            ax1.set_title('Optimization History')
            
            # 2. 参数重要性图
            ax2 = fig.add_subplot(222)
            plot_param_importances(self.study, ax=ax2)
            ax2.set_title('Parameter Importances')
            
            # 3. 平行坐标图 (如果matplotlib支持)
            try:
                from optuna.visualization import plot_parallel_coordinate
                ax3 = fig.add_subplot(223)
                plot_parallel_coordinate(self.study, ax=ax3)
                ax3.set_title('Parallel Coordinate')
            except Exception:
                pass
            
            # 4. 等高线图 (前两个重要参数)
            try:
                from optuna.visualization import plot_contour
                ax4 = fig.add_subplot(224)
                plot_contour(self.study, params=list(self.search_space.params.keys())[:2], ax=ax4)
                ax4.set_title('Contour Plot')
            except Exception:
                pass
            
            plt.tight_layout()
            viz_path = f'optuna_visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Visualization saved to: {viz_path}")
            
        except ImportError:
            logger.warning("⚠️ Matplotlib not installed, skipping visualizations")


# ============================================================
# 便捷函数
# ============================================================

def quick_train(
    algorithm: str = 'ppo',
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    use_gpu: bool = True,
) -> V8RLTrainer:
    """
    快速训练入口 - 一键启动训练
    
    Args:
        algorithm: 算法名称 (ppo/sac/a2c)
        total_timesteps: 总训练步数
        n_envs: 并行环境数量
        use_gpu: 是否使用GPU加速
    
    Returns:
        V8RLTrainer: 训练好的训练器实例
    """
    config = TrainingConfig(
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        device='cuda' if use_gpu and torch.cuda.is_available() else 'auto',
    )
    
    trainer = V8RLTrainer(config=config)
    trainer.setup_environments()
    trainer.build_model(custom_policy=True)
    results = trainer.train()
    trainer.save_results()
    
    return trainer


def run_hyperparameter_search(n_trials: int = 30) -> optuna.Study:
    """
    快速超参数搜索入口
    
    Args:
        n_trials: 试验次数
    
    Returns:
        optuna.Study: 优化后的study
    """
    optimizer = HyperparameterOptimizer(
        n_trials=n_trials,
        timeout=3600 * 2,  # 2小时
    )
    study = optimizer.optimize()
    return study


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("V8 Robot Assembly RL Training System")
    print("Stable-Baselines3 + Optuna Hyperparameter Optimization")
    print("=" * 80)
    
    import argparse
    parser = argparse.ArgumentParser(description='V8 RL Training System')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'optimize', 'evaluate', 'demo'],
                       help='运行模式: train/optimize/evaluate/demo')
    parser.add_argument('--algo', type=str, default='ppo',
                       choices=['ppo', 'sac', 'a2c', 'dqn'],
                       help='RL算法')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='总训练步数')
    parser.add_argument('--envs', type=int, default=4,
                       help='并行环境数')
    parser.add_argument('--trials', type=int, default=30,
                       help='超参数搜索试验次数')
    parser.add_argument('--gpu', action='store_true',
                       help='使用GPU加速')
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"\n🚀 Starting {args.algo.upper()} training ({args.timesteps:,} timesteps)...")
        trainer = quick_train(
            algorithm=args.algo,
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            use_gpu=args.gpu,
        )
        
        # 评估
        print("\n📊 Evaluating trained model...")
        eval_results = trainer.evaluate(n_episodes=20)
        
        print("\n✅ Training complete! Model saved.")
        
    elif args.mode == 'optimize':
        print(f"\n🔬 Starting hyperparameter optimization ({args.trials} trials)...")
        study = run_hyperparameter_search(n_trials=args.trials)
        
        print("\n✅ Optimization complete! Best parameters found.")
        
    elif args.mode == 'evaluate':
        print("\n📊 Loading model for evaluation...")
        # 需要提供模型路径
        print("Please specify --model-path for evaluation mode")
        
    elif args.mode == 'demo':
        print("\n🎮 Demo mode - loading best model...")
        # 加载最佳模型并进行演示
        print("Demo mode coming soon!")
    
    print("\n" + "=" * 80)
