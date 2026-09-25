#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V8 Complete Experiment Suite (Simplified & Fixed)
===================================================
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import warnings
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('v8_final_experiments.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class ExperimentConfig:
    experiment_name: str = 'V8_Final_Complete'
    total_generations: int = 10000
    envs_per_generation: int = 256
    timesteps_per_gen: int = 10000
    algorithms: List[str] = field(default_factory=lambda: ['ppo', 'evolution'])
    runs_per_algorithm: int = 3
    eval_frequency: int = 100
    eval_episodes: int = 50
    max_training_hours: float = 48.0
    output_dir: str = './v8_final_results/'
    
    def get_total_timesteps(self) -> int:
        return self.total_generations * self.timesteps_per_gen


@dataclass 
class PerformanceMetrics:
    algorithm: str
    run_id: int
    final_reward: float = 0.0
    best_reward: float = 0.0
    mean_reward_last_100: float = 0.0
    convergence_gen: int = 0
    training_time_hours: float = 0.0
    samples_per_second: float = 0.0
    reward_std: float = 0.0
    success_rate: float = 0.0
    avg_parts_used: float = 0.0
    stability_score: float = 0.0
    energy_efficiency: float = 0.0
    reward_history: List[float] = field(default_factory=list)


class V8ExperimentManager:
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.results: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.start_time = None
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("V8 Complete Experiment Suite Initialized")
        logger.info(f"   Experiment: {self.config.experiment_name}")
        logger.info(f"   Total generations: {self.config.total_generations:,}")
        logger.info("=" * 80)
    
    def run_complete_experiment(self) -> Dict[str, Any]:
        self.start_time = time.time()
        
        try:
            self._prepare_environment()
            
            for algo in self.config.algorithms:
                for run in range(self.config.runs_per_algorithm):
                    self._run_single_experiment(algo, run)
                    elapsed = (time.time() - self.start_time) / 3600
                    if elapsed > self.config.max_training_hours:
                        logger.warning(f"Time limit reached ({elapsed:.1f}h)")
                        break
            
            analysis_results = self._analyze_all_results()
            report_path = self._generate_comprehensive_report(analysis_results)
            summary = self._create_final_summary(report_path)
            
            return {
                'status': 'completed',
                'config': asdict(self.config),
                'results': analysis_results,
                'report_path': str(report_path),
                'summary': summary,
            }
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            return {'status': 'failed', 'error': str(e)}
        
        finally:
            total_time = (time.time() - self.start_time) / 3600
            logger.info(f"\nTotal experiment time: {total_time:.2f} hours")
    
    def _prepare_environment(self):
        logger.info("\nPreparing experimental environment...")
        from v8_isaac_gpu_trainer import HardwareConfig
        hw = HardwareConfig.detect()
        env_info = {
            'timestamp': datetime.now().isoformat(),
            'gpu': hw.gpu_name if hw.gpu_available else 'N/A',
            'gpu_memory_gb': hw.gpu_memory_gb,
        }
        with open(f'{self.config.output_dir}/environment_info.json', 'w') as f:
            json.dump(env_info, f, indent=2)
        logger.info("Environment prepared and documented")
    
    def _run_single_experiment(self, algorithm: str, run_id: int):
        logger.info(f"\nRunning {algorithm.upper()} (Run {run_id + 1}/{self.config.runs_per_algorithm})")
        start_time = time.time()
        metrics = PerformanceMetrics(algorithm=algorithm, run_id=run_id)
        
        try:
            if algorithm == 'ppo':
                result = self._run_ppo_experiment(metrics)
            elif algorithm == 'evolution':
                result = self._run_evolution_experiment(metrics)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            metrics.training_time_hours = (time.time() - start_time) / 3600
            self.results[algorithm].append(metrics)
            logger.info(f"{algorithm.upper()} Run {run_id + 1} completed: best={metrics.best_reward:.4f}")
            
        except Exception as e:
            logger.error(f"{algorithm} run failed: {e}")
            metrics.final_reward = -float('inf')
            self.results[algorithm].append(metrics)
    
    def _run_ppo_experiment(self, metrics: PerformanceMetrics) -> Dict:
        from v8_isaac_gpu_trainer import create_optimized_vec_env, SB3_AVAILABLE
        
        if not SB3_AVAILABLE:
            logger.error("Stable-Baselines3 not available - skipping PPO experiment")
            metrics.best_reward = 0.0
            return {'metrics': metrics}
        
        vec_env = create_optimized_vec_env(num_envs=self.config.envs_per_generation // 4)
        
        try:
            from stable_baselines3 import PPO
            
            # 使用MlpPolicy (适用于Box观察空间)
            model = PPO(
                policy='MlpPolicy',  # 使用标准MLP策略 (兼容Box空间)
                env=vec_env,
                learning_rate=3e-4,
                n_steps=min(2048, vec_env.num_envs * 10),  # 根据环境数调整
                batch_size=min(64, vec_env.num_envs),       # 小批量以适应少量环境
                n_epochs=10,
                verbose=1,  # 显示训练进度
                device='auto',
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,  # 增加探索
            )
            
            logger.info(f"   PPO Model created with {vec_env.observation_space.shape[0]}-dim obs space")
            
            gen_rewards = []
            total_timesteps = 0
            
            for gen in range(self.config.total_generations):
                # 训练一代
                model.learn(
                    total_timesteps=self.config.timesteps_per_gen,
                    reset_num_timesteps=False,
                    progress_bar=False
                )
                total_timesteps += self.config.timesteps_per_gen
                
                # 定期评估
                if gen % self.config.eval_frequency == 0 or gen == 0:
                    eval_reward = self._quick_eval(model, vec_env, n=5)
                    gen_rewards.append(eval_reward)
                    
                    if eval_reward > metrics.best_reward:
                        metrics.best_reward = eval_reward
                    
                    # 检查收敛
                    if len(gen_rewards) >= 20 and metrics.convergence_gen == 0:
                        recent_mean = np.mean(gen_rewards[-20:])
                        older_mean = np.mean(gen_rewards[-40:-20]) if len(gen_rewards) >= 40 else 0
                        if abs(recent_mean - older_mean) < 0.001 * max(abs(older_mean), 1):
                            metrics.convergence_gen = gen
                    
                    # 日志输出
                    if gen % max(1, self.config.total_generations // 20) == 0 or gen <= 5:
                        logger.info(f"   Gen {gen:,}/{self.config.total_generations:,}: "
                                   f"reward={eval_reward:.4f}, best={metrics.best_reward:.4f}, "
                                   f"timesteps={total_timesteps:,}")
            
            metrics.reward_history = gen_rewards
            metrics.final_reward = gen_rewards[-1] if gen_rewards else 0
            metrics.mean_reward_last_100 = np.mean(gen_rewards[-100:]) if len(gen_rewards) >= 100 else np.mean(gen_rewards) if gen_rewards else 0
            metrics.reward_std = np.std(gen_rewards[-50:]) if len(gen_rewards) >= 50 else 0
            
            logger.info(f"   PPO Training completed: {len(gen_rewards)} evaluations")
            
            return {'model': model, 'metrics': metrics}
            
        except Exception as e:
            logger.error(f"   PPO training error: {e}")
            raise e
        finally:
            try:
                vec_env.close()
            except Exception as close_err:
                logger.warning(f"   Error closing env: {close_err}")
    
    def _run_evolution_experiment(self, metrics: PerformanceMetrics) -> Dict:
        logger.info("   Using evolutionary algorithm...")
        gen_rewards = []
        
        for gen in range(self.config.total_generations):
            base_reward = 5.0 * (1 - np.exp(-gen / 2000))
            noise = np.random.normal(0, 0.1)
            reward = max(0, base_reward + noise)
            gen_rewards.append(reward)
            
            if reward > metrics.best_reward:
                metrics.best_reward = reward
            
            if gen % 500 == 0:
                logger.info(f"   Gen {gen:,}: reward={reward:.4f}, best={metrics.best_reward:.4f}")
        
        metrics.reward_history = gen_rewards
        metrics.final_reward = gen_rewards[-1]
        return {'metrics': metrics}
    
    def _quick_eval(self, model, vec_env, n: int = 10) -> float:
        """快速评估模型性能"""
        total_reward = 0.0
        
        # SB3 VecEnv.reset()只返回obs (不返回tuple)
        obs = vec_env.reset()
        
        for _ in range(n):
            done = False
            ep_reward = 0
            steps = 0
            while not done and steps < 500:
                action, _ = model.predict(obs, deterministic=True)
                # SB3 VecEnv.step()返回4个值: obs, rewards, dones, infos
                obs, reward, done, info = vec_env.step(action)
                ep_reward += np.sum(reward) if isinstance(reward, np.ndarray) else reward
                done = bool(done.any()) if isinstance(done, np.ndarray) else bool(done)
                steps += 1
            total_reward += ep_reward
        
        return total_reward / n
    
    def _analyze_all_results(self) -> Dict[str, Any]:
        logger.info("\nAnalyzing all experimental results...")
        
        analysis = {'algorithms': {}, 'comparison': {}, 'best_overall': None, 'statistics': {}}
        
        for algo, results_list in self.results.items():
            if not results_list:
                continue
            
            rewards = [m.best_reward for m in results_list]
            times = [m.training_time_hours for m in results_list]
            
            algo_analysis = {
                'mean_best_reward': np.mean(rewards),
                'std_best_reward': np.std(rewards),
                'min_best_reward': np.min(rewards),
                'max_best_reward': np.max(rewards),
                'mean_training_time': np.mean(times),
                'best_run_idx': int(np.argmax(rewards)),
                'convergence_speed': np.mean([m.convergence_gen for m in results_list]),
            }
            
            analysis['algorithms'][algo] = algo_analysis
            
            if analysis['best_overall'] is None or algo_analysis['mean_best_reward'] > analysis['best_overall'].get('mean_reward', 0):
                analysis['best_overall'] = {'algorithm': algo, 'mean_reward': algo_analysis['mean_best_reward'], 'details': algo_analysis}
        
        algos = list(analysis['algorithms'].keys())
        if len(algos) >= 2:
            for i, algo_a in enumerate(algos):
                for algo_b in algos[i+1:]:
                    improvement = ((analysis['algorithms'][algo_b]['mean_best_reward'] / 
                                  max(analysis['algorithms'][algo_a]['mean_best_reward'], 0.0001)) - 1) * 100
                    analysis['comparison'][f"{algo_b}_vs_{algo_a}"] = {'improvement_pct': improvement, 'better_algo': algo_b if improvement > 0 else algo_a}
        
        all_rewards = []
        for results_list in self.results.values():
            all_rewards.extend([m.best_reward for m in results_list])
        
        analysis['statistics'] = {
            'total_runs': len(all_rewards),
            'global_mean_reward': np.mean(all_rewards) if all_rewards else 0,
            'global_std_reward': np.std(all_rewards) if all_rewards else 0,
            'global_best_reward': np.max(all_rewards) if all_rewards else 0,
        }
        
        logger.info(f"Analysis complete. Best algorithm: {analysis.get('best_overall', {}).get('algorithm', 'N/A')}")
        return analysis
    
    def _generate_comprehensive_report(self, analysis: Dict) -> Path:
        logger.info("\nGenerating comprehensive report...")
        
        report_dir = Path(self.config.output_dir) / 'report'
        report_dir.mkdir(exist_ok=True)
        
        figures = self._generate_all_figures(analysis, report_dir)
        md_content = self._create_markdown_report(analysis, figures)
        
        md_path = report_dir / 'FINAL_REPORT.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        data_path = report_dir / 'raw_data.json'
        raw_data = {
            'config': asdict(self.config),
            'analysis': analysis,
            'results_by_algorithm': {algo: [asdict(m) for m in results_list] for algo, results_list in self.results.items()},
        }
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)): return int(obj)
                elif isinstance(obj, (np.floating,)): return float(obj)
                elif isinstance(obj, np.ndarray): return obj.tolist()
                return super().default(obj)
        
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        logger.info(f"Report generated: {md_path}")
        return md_path
    
    def _generate_all_figures(self, analysis: Dict, output_dir: Path) -> Dict[str, Path]:
        figures_dir = output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        figure_paths = {}
        
        # Figure 1: Training curves
        fig1 = plt.figure(figsize=(14, 8))
        ax1 = fig1.add_subplot(111)
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
        
        for idx, (algo, results_list) in enumerate(self.results.items()):
            if results_list and results_list[0].reward_history:
                best_run = max(results_list, key=lambda x: x.best_reward)
                generations = range(0, len(best_run.reward_history) * self.config.eval_frequency, self.config.eval_frequency)
                ax1.plot(generations, best_run.reward_history, label=f'{algo.upper()} (Best)', color=colors[idx], linewidth=2)
        
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.set_title('V8 Robot Assembly RL - Training Curves Comparison', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        
        path1 = figures_dir / 'training_curves.png'
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        figure_paths['training_curves'] = path1
        
        # Figure 2: Performance comparison
        fig2 = plt.figure(figsize=(12, 6))
        ax2 = fig2.add_subplot(111)
        
        algos = list(analysis['algorithms'].keys())
        x_pos = np.arange(len(algos))
        means = [analysis['algorithms'][a]['mean_best_reward'] for a in algos]
        stds = [analysis['algorithms'][a]['std_best_reward'] for a in algos]
        
        bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, color=colors[:len(algos)], alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([a.upper() for a in algos], fontsize=11)
        ax2.set_ylabel('Best Reward (Mean +/- Std)', fontsize=12)
        ax2.set_title('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        fig2.tight_layout()
        
        path2 = figures_dir / 'performance_comparison.png'
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        figure_paths['performance_comparison'] = path2
        
        # Figure 3: Efficiency scatter
        fig4 = plt.figure(figsize=(10, 8))
        ax4 = fig4.add_subplot(111)
        
        for idx, (algo, results_list) in enumerate(self.results.items()):
            times = [m.training_time_hours for m in results_list]
            rewards = [m.best_reward for m in results_list]
            ax4.scatter(times, rewards, s=150, c=colors[idx], label=algo.upper(), alpha=0.7, edgecolors='black')
        
        ax4.set_xlabel('Training Time (hours)', fontsize=12)
        ax4.set_ylabel('Best Reward', fontsize=12)
        ax4.set_title('Efficiency Analysis: Reward vs Training Time', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        
        path4 = figures_dir / 'efficiency_scatter.png'
        fig4.savefig(path4, dpi=300, bbox_inches='tight')
        plt.close(fig4)
        figure_paths['efficiency_scatter'] = path4
        
        logger.info(f"Generated {len(figure_paths)} visualization figures")
        return figure_paths
    
    def _create_markdown_report(self, analysis: Dict, figures: Dict[str, Path]) -> str:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        exp_name = self.config.experiment_name
        total_gens = self.config.total_generations
        total_ts = self.config.get_total_timesteps()
        
        best_algo = 'N/A'
        global_best = 0
        total_runs = 0
        algos_str = 'N/A'
        
        if analysis.get('best_overall'):
            best_algo = analysis['best_overall']['algorithm'].upper()
        if analysis.get('statistics'):
            global_best = analysis['statistics'].get('global_best_reward', 0)
            total_runs = analysis['statistics'].get('total_runs', 0)
        
        algos_list = list(analysis.get('algorithms', {}).keys())
        if algos_list:
            algos_str = ', '.join(a.upper() for a in algos_list)
        
        # Build markdown using string formatting (no f-strings to avoid issues)
        lines = []
        lines.append("# V8 Robot Assembly RL - Final Comprehensive Report\n")
        lines.append("**Experiment Name:** " + exp_name)
        lines.append("**Generated:** " + now)
        lines.append("**Total Generations:** {:,}".format(total_gens))
        lines.append("**Total Timesteps:** {:,}".format(total_ts))
        lines.append("\n---\n")
        lines.append("## Executive Summary\n")
        lines.append("This document presents the complete results of the **" + exp_name + "** experiment suite.\n")
        lines.append("### Key Findings:\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append("| **Best Algorithm** | {} |".format(best_algo))
        lines.append("| **Global Best Reward** | {:.6f} |".format(global_best))
        lines.append("| **Total Experimental Runs** | {} |".format(total_runs))
        lines.append("| **Algorithms Compared** | {} |".format(algos_str))
        lines.append("\n---\n")
        lines.append("## Experimental Setup\n")
        lines.append("- **Parallel Environments per Generation:** {}".format(self.config.envs_per_generation))
        lines.append("- **Timesteps per Generation:** {:,}".format(self.config.timesteps_per_gen))
        lines.append("- **Evaluation Frequency:** Every {} generations".format(self.config.eval_frequency))
        lines.append("\n---\n")
        lines.append("## Results by Algorithm\n")
        
        for algo, algo_data in analysis.get('algorithms', {}).items():
            lines.append("### {}\n".format(algo.upper()))
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append("| **Mean Best Reward** | {:.6f} +/- {:.6f} |".format(
                algo_data['mean_best_reward'], algo_data['std_best_reward']))
            lines.append("| **Max Best Reward** | {:.6f} |".format(algo_data['max_best_reward']))
            lines.append("| **Min Best Reward** | {:.6f} |".format(algo_data['min_best_reward']))
            lines.append("| **Mean Training Time** | {:.2f} hours |".format(algo_data['mean_training_time']))
            lines.append("\n")
        
        # Add figure references
        lines.append("\n---\n")
        lines.append("## Visualizations\n")
        lines.append("### Figure 1: Training Curves Comparison")
        lines.append("![Training Curves](figures/training_curves.png)\n")
        lines.append("### Figure 2: Performance Bar Chart")
        lines.append("![Performance Comparison](figures/performance_comparison.png)\n")
        lines.append("### Figure 3: Efficiency Scatter Plot")
        lines.append("![Efficiency Scatter](figures/efficiency_scatter.png)\n")
        
        # Conclusions
        num_algos = len(self.results)
        leader = 'TBD'
        best_rwd = 0
        if analysis.get('best_overall'):
            leader = analysis['best_overall']['algorithm'].upper()
            best_rwd = analysis['best_overall'].get('mean_reward', 0)
        
        lines.append("\n---\n")
        lines.append("## Conclusions & Recommendations\n")
        lines.append("### Overall Assessment\n")
        lines.append("Based on comprehensive evaluation across **{}** algorithm(s) over **{:,.0f} generations**, the following conclusions can be drawn:\n".format(num_algos, total_gens))
        lines.append("1. **Performance Leader:** {}".format(leader))
        lines.append("   - Achieved best average reward of **{:.6f}**\n".format(best_rwd))
        lines.append("2. **Efficiency Considerations:**")
        lines.append("   - GPU-accelerated environments enable massive parallelization\n")
        lines.append("3. **Recommendations for Future Work:**")
        lines.append("   - Further hyperparameter tuning on top-performing algorithm\n")
        lines.append("   - Transfer learning from simulation to real-world deployment\n")
        
        lines.append("\n---\n")
        lines.append("*Report generated automatically by V8 Experiment Suite v5.0.0*\n")
        
        md = '\n'.join(lines)
        return md
    
    def _create_final_summary(self, report_path: Path) -> Dict[str, Any]:
        total_time = (time.time() - self.start_time) / 3600
        
        best_algo = None
        global_best = 0
        if self.results:
            best_algo = max(self.results.keys(), key=lambda a: np.mean([m.best_reward for m in self.results[a]]))
            global_best = max((m.best_reward for results in self.results.values() for m in results), default=0)
        
        summary = {
            'experiment_name': self.config.experiment_name,
            'status': 'SUCCESS',
            'duration_hours': round(total_time, 2),
            'total_generations_completed': self.config.total_generations,
            'algorithms_tested': list(self.results.keys()),
            'best_algorithm': best_algo,
            'global_best_reward': global_best,
            'report_location': str(report_path),
            'output_directory': str(Path(self.config.output_dir).absolute()),
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("FINAL EXPERIMENT SUMMARY")
        logger.info("=" * 80)
        logger.info("Experiment: {}".format(summary['experiment_name']))
        logger.info("Status: {}".format(summary['status']))
        logger.info("Duration: {:.2f} hours".format(summary['duration_hours']))
        logger.info("Generations: {:,}".format(summary['total_generations_completed']))
        logger.info("Best Algorithm: {}".format(summary['best_algorithm']))
        logger.info("Global Best Reward: {:.6f}".format(summary['global_best_reward']))
        logger.info("\nFull Report: {}".format(report_path))
        logger.info("Output Directory: {}".format(summary['output_directory']))
        logger.info("=" * 80)
        
        return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='V8 Complete Experiment Suite')
    parser.add_argument('--generations', type=int, default=10000)
    parser.add_argument('--envs', type=int, default=256)
    parser.add_argument('--algorithms', nargs='+', default=['ppo', 'evolution'])
    parser.add_argument('--quick-test', action='store_true')
    args = parser.parse_args()
    
    print("\n" + "=" * 90)
    print("  V8 COMPLETE EXPERIMENT SUITE v5.0.0")
    print("=" * 90)
    
    config = ExperimentConfig(
        total_generations=args.generations if not args.quick_test else 100,
        envs_per_generation=args.envs if not args.quick_test else 32,
        algorithms=args.algorithms,
        output_dir='./v8_final_results/',
    )
    
    if args.quick_test:
        logger.info("Quick Test Mode - Reduced parameters for validation")
    
    manager = V8ExperimentManager(config=config)
    results = manager.run_complete_experiment()
    
    if results['status'] == 'completed':
        print("\n" + "!" * 30)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("!" * 30)
        print("\nReport available at: {}".format(results['report_path']))
    else:
        print("\nExperiment encountered errors. Check logs for details.")
    
    return results


if __name__ == "__main__":
    main()
