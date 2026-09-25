#!/usr/bin/env python3
"""
两电机竞速 PPO 训练脚本
=========================
- 仅 2× M3508 电机
- 目标：不倒的前提下跑最快
- 训练无时间上限
- 最优参数配置，追求最终策略最佳表现
"""

import os, sys, time, json
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from racer_env import TwoWheelRacerEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# ============================================================
# 最优配置
# ============================================================
OUTPUT_DIR = Path(__file__).parent / "training_output"
OUTPUT_DIR.mkdir(exist_ok=True)
STATUS_FILE = Path(__file__).parent / "status.json"

TOTAL_TIMESTEPS = 20_000_000
N_ENVS = 4                     # 并行环境：4 个可提供更丰富的探索多样性
EVAL_FREQ = 30_000
SAVE_FREQ = 30_000
N_EVAL_EPISODES = 10


def make_env(rank: int = 0, seed: int = 42):
    def _init():
        env = TwoWheelRacerEnv(render_mode=None, max_steps=3000)
        env.reset(seed=seed + rank)
        return env
    return _init


class TrainingMonitor(BaseCallback):
    """训练监控 + 写 status.json 给仪表盘"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = time.time()
        self.episode_rewards = []
        self.last_status_write = 0

    def _on_step(self):
        if self.n_calls % 1000 == 0:
            elapsed = time.time() - self.start_time
            fps = int(self.n_calls / max(elapsed, 1))
            hours = self.num_timesteps / max(fps, 1) / 3600 if fps > 0 else 0

            if len(self.model.ep_info_buffer) > 0:
                buf = list(self.model.ep_info_buffer)
                recent = buf[-min(50, len(buf)):] if len(buf) > 0 else []
                avg_reward = np.mean([e.get('r', 0) for e in recent]) if recent else 0
                avg_len = np.mean([e.get('l', 0) for e in recent]) if recent else 0
            else:
                avg_reward, avg_len = 0, 0

            self.episode_rewards.append(avg_reward)
            if len(self.episode_rewards) > 1000:
                self.episode_rewards = self.episode_rewards[-1000:]

            print(
                f"📊 Step {self.num_timesteps:>8d}/{TOTAL_TIMESTEPS} | "
                f"FPS: {fps:>5d} | "
                f"R_avg: {avg_reward:>7.1f} | "
                f"Len_avg: {avg_len:>5.0f} | "
                f"⏱ {elapsed/3600:.1f}h | "
                f"ETA: {hours:.1f}h"
            )

            # 每 5000 步写一次状态文件
            if self.n_calls - self.last_status_write >= 5000:
                self._write_status()
                self.last_status_write = self.n_calls

        return True

    def _write_status(self):
        try:
            status = {
                "timestamp": time.strftime("%H:%M:%S"),
                "step": self.num_timesteps,
                "total": TOTAL_TIMESTEPS,
                "progress_pct": round(self.num_timesteps / TOTAL_TIMESTEPS * 100, 2),
                "status": "running",
                "n_envs": N_ENVS,
            }

            # 评估奖励
            eval_log = OUTPUT_DIR / "eval_logs" / "evaluations.npz"
            if eval_log.exists():
                try:
                    d = np.load(str(eval_log))
                    status["eval_reward"] = round(float(d['results'].mean()), 1)
                except:
                    status["eval_reward"] = 0

            # 检查点 / 最佳模型
            for dname, key in [("checkpoints", "checkpoint_count"), ("best_model", "best_models")]:
                d = OUTPUT_DIR / dname
                if d.exists():
                    status[key] = len([f for f in os.listdir(d) if f.endswith('.zip')])

            STATUS_FILE.write_text(
                json.dumps(status, ensure_ascii=False), encoding='utf-8'
            )
        except:
            pass


def create_eval_env():
    return TwoWheelRacerEnv(render_mode=None, max_steps=3000)


def main():
    print("=" * 60)
    print("🏎️  两电机竞速 PPO 训练系统 (最优配置)")
    print("=" * 60)
    print(f"   机器人:   2× M3508 + 被动底盘")
    print(f"   目标:     不倒的前提下跑最快")
    print(f"   总步数:   {TOTAL_TIMESTEPS:,}")
    print(f"   并行环境: {N_ENVS}")
    print(f"   设备:     {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # ── 环境 ──
    print(f"\n🔄 创建环境 ({N_ENVS} 并行)...")
    vec_env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)
    eval_env = DummyVecEnv([lambda: create_eval_env()])
    eval_env = VecMonitor(eval_env)

    # ── PPO 最优参数 ──
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=torch.nn.ReLU,
    )

    print("🧠 初始化 PPO 模型 (最优参数)...")
    resume_path = OUTPUT_DIR / "racer_final.zip"
    if resume_path.exists():
        print(f"📂 续训: {resume_path}")
        model = PPO.load(str(resume_path), vec_env, device='auto')
    else:
        print("   从头训练")
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=3e-4,
            n_steps=2048,           # 更长的 rollout → 更好的优势估计
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,          # 适度探索
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device='auto',
            tensorboard_log=str(OUTPUT_DIR / "tensorboard"),
        )

    # ── 回调 ──
    train_monitor = TrainingMonitor()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(OUTPUT_DIR / "best_model"),
        log_path=str(OUTPUT_DIR / "eval_logs"),
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=False, render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=str(OUTPUT_DIR / "checkpoints"),
        name_prefix="racer_ppo",
    )

    callbacks = CallbackList([train_monitor, eval_callback, checkpoint_callback])

    # ── 训练 ──
    print(f"\n🚀 开始训练 (最优配置: envs={N_ENVS}, n_steps=2048, pi=[256,128])")
    print(f"   目标: {TOTAL_TIMESTEPS:,} 步")
    print(f"   Ctrl+C 停止并保存\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name="two_motor_racer",
        )
    except (KeyboardInterrupt, Exception) as e:
        print(f"\n⚠️ 训练中断: {e}，保存模型...")

    # ── 保存 ──
    final_path = OUTPUT_DIR / "racer_final.zip"
    try:
        model.save(str(final_path))
        print(f"\n✅ 最终模型已保存: {final_path}")
    except:
        print("\n⚠️ 保存失败")

    # ── 最终评估 ──
    print("\n📊 最终评估...")
    final_env = create_eval_env()
    obs, _ = final_env.reset()

    total_reward, max_speed, survived = 0, 0, 0
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = final_env.step(action)
        total_reward += reward
        max_speed = max(max_speed, info.get('forward_speed', 0))
        survived = i + 1
        if terminated:
            print(f"   ⚠️ 翻倒！步数: {survived}")
            break
        if truncated:
            print(f"   ✅ 完成！存活全部 {survived} 步")
            break

    print(f"\n   {'='*40}")
    print(f"   总奖励:     {total_reward:.1f}")
    print(f"   最高速度:   {max_speed:.3f} m/s ({max_speed*3.6:.1f} km/h)")
    print(f"   存活步数:   {survived}/1000")
    print(f"   {'='*40}")

    final_env.close()
    vec_env.close()
    eval_env.close()
    print("\n🏁 训练完成！")


if __name__ == "__main__":
    main()
