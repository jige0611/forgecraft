"""ONNX 部署验证模块

增强 ONNX 导出流程:
  1. torch.onnx.export → policy.onnx
  2. onnx.checker.check_model → 验证格式正确性
  3. onnxruntime.InferenceSession → 加载 + 推理
  4. 对比输出 (PyTorch vs ONNX) → MSE ≤ 1e-6
  5. 可选: FP16/INT8 量化

Usage:
    result = export_onnx_verified(
        policy_state, obs_dim, act_dim, morph_dim, output_dir,
        verify=True, quantize="fp16"
    )
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

_logger = logging.getLogger(__name__)

__all__ = ["export_onnx_verified", "verify_onnx_inference_speed"]


def export_onnx_verified(
    policy_state: dict,
    obs_dim: int,
    act_dim: int,
    morph_dim: int,
    output_dir: str,
    verify: bool = True,
    quantize: Optional[str] = None,
) -> Dict[str, Any]:
    """导出 + 验证 + 量化一站式

    三步流水线:
      1. torch.onnx.export  → 导出 ONNX graph
      2. onnxruntime 推理   → 验证输出一致性
      3. 可选量化 (int8)   → 减小模型体积 + 加速

    Parameters
    ----------
    policy_state : dict
        策略网络权重状态。
    obs_dim : int
        观测维度。
    act_dim : int
        动作维度。
    morph_dim : int
        形态嵌入维度。
    output_dir : str
        输出目录。
    verify : bool
        是否运行时验证 (需要 onnxruntime)。
    quantize : Optional[str]
        "fp16" 进行 FP16 量化 (需要 onnxruntime + onnx)。

    Returns
    -------
    result : dict
        {onnx_path, verification_passed, mse, quantized_path, error}
    """
    import onnx
    import onnxruntime

    result = {
        "onnx_path": "",
        "verification_passed": False,
        "mse": None,
        "quantized_path": None,
        "error": None,
    }

    os.makedirs(output_dir, exist_ok=True)

    # 1. 导出 ONNX
    onnx_path = os.path.join(output_dir, "forgecraft_policy.onnx")

    try:
        from forgecraft.rl.encoder import MorphAwareActor

        # 提取 actor 权重
        actor_weights = _extract_actor_weights(policy_state)
        if actor_weights is None:
            result["error"] = "无法从 policy_state 提取 actor 权重"
            return result

        hidden_dim = actor_weights["shared.0.weight"].shape[0]
        actor = MorphAwareActor(obs_dim, act_dim, morph_dim, hidden_dim)
        actor.load_state_dict(actor_weights)
        actor.eval()

        dummy_obs = torch.randn(1, obs_dim)
        dummy_morph = torch.randn(1, morph_dim)

        torch.onnx.export(
            actor,
            (dummy_obs, dummy_morph),
            onnx_path,
            input_names=["observation", "morphology_embedding"],
            output_names=["action_mean", "action_log_std"],
            dynamic_axes={
                "observation": {0: "batch"},
                "morphology_embedding": {0: "batch"},
                "action_mean": {0: "batch"},
                "action_log_std": {0: "batch"},
            },
            opset_version=17,
        )

        result["onnx_path"] = onnx_path
        _logger.info("ONNX 导出完成: %s", onnx_path)

    except Exception as e:
        result["error"] = f"ONNX 导出失败: {e}"
        _logger.error(result["error"])
        return result

    # 2. 验证格式
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        _logger.info("ONNX 格式验证通过")
    except Exception as e:
        result["error"] = f"ONNX 格式验证失败: {e}"
        _logger.error(result["error"])
        return result

    # 3. 推理验证
    if verify:
        try:
            # PyTorch 推理
            with torch.no_grad():
                torch_mean, torch_std = actor(dummy_obs, dummy_morph)

            # ONNX Runtime 推理
            session = onnxruntime.InferenceSession(
                onnx_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

            onnx_outputs = session.run(
                None,
                {
                    "observation": dummy_obs.numpy(),
                    "morphology_embedding": dummy_morph.numpy(),
                },
            )

            onnx_mean = onnx_outputs[0]
            onnx_std = onnx_outputs[1]

            # 对比
            mse_mean = float(np.mean((torch_mean.numpy() - onnx_mean) ** 2))
            mse_std = float(np.mean((torch_std.numpy() - onnx_std) ** 2))
            mse = max(mse_mean, mse_std)
            result["mse"] = mse
            result["verification_passed"] = mse < 1e-6

            if result["verification_passed"]:
                _logger.info("ONNX 推理验证通过 (MSE=%.2e)", mse)
            else:
                _logger.warning("ONNX 推理验证失败 (MSE=%.2e > 1e-6)", mse)

        except Exception as e:
            result["error"] = f"ONNX 推理验证失败: {e}"
            _logger.error(result["error"])
            return result

    # 4. 量化 (可选)
    if quantize == "fp16":
        try:
            from onnxruntime.transformers import float16

            fp16_path = onnx_path.replace(".onnx", "_fp16.onnx")
            float16.convert_float_to_float16(model, fp16_path)

            result["quantized_path"] = fp16_path
            _logger.info("FP16 量化完成: %s", fp16_path)
        except ImportError:
            result["error"] = "FP16 量化失败: onnxruntime.transformers 不可用"
            _logger.error(result["error"])
        except Exception as e:
            result["error"] = f"FP16 量化失败: {e}"
            _logger.error(result["error"])

    return result


def _extract_actor_weights(policy_state: dict) -> Optional[dict]:
    """从 policy_state 提取 actor 权重"""
    if "actor" in policy_state and isinstance(policy_state["actor"], dict):
        return policy_state["actor"]

    extracted = {}
    for k, v in policy_state.items():
        if k.startswith("actor."):
            extracted[k[6:]] = v

    return extracted if extracted else None


def verify_onnx_inference_speed(
    onnx_path: str,
    obs_dim: int,
    morph_dim: int,
    n_iterations: int = 1000,
) -> Dict[str, float]:
    """ONNX 推理速度测试

    Parameters
    ----------
    onnx_path : str
        ONNX 模型路径。
    obs_dim, morph_dim : int
        输入维度。
    n_iterations : int
        测试迭代数。

    Returns
    -------
    stats : dict
        {mean_ms, p50_ms, p99_ms, fps}
    """
    import onnxruntime
    import time

    session = onnxruntime.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    obs = np.random.randn(1, obs_dim).astype(np.float32)
    morph = np.random.randn(1, morph_dim).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {"observation": obs, "morphology_embedding": morph})

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        session.run(None, {"observation": obs, "morphology_embedding": morph})
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p99_ms": float(np.percentile(times, 99)),
        "fps": float(1000.0 / np.mean(times)),
    }
