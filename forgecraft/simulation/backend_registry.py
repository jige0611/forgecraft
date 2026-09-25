"""仿真后端注册表

支持通过环境变量或代码动态切换仿真后端。

使用方式:
  # 环境变量
  export FORGECRAFT_BACKEND=genesis
  export FORGECRAFT_BACKEND=mujoco
  export FORGECRAFT_BACKEND=mjx

  # 代码
  from forgecraft.simulation.backend_registry import BackendRegistry
  evaluator = BackendRegistry.create_evaluator("genesis", **kwargs)
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

_logger = logging.getLogger(__name__)

__all__ = ["BackendRegistry", "get_backend_status", "create_simulation_backend"]


class BackendRegistry:
    """后端注册表 — 工厂模式切换 MuJoCo/Genesis/MJX

    一行切换:
        FORGECRAFT_BACKEND=genesis python -m forgecraft.main

    代码切换:
        backend = BackendRegistry.get_backend("genesis")
        evaluator = BackendRegistry.create_evaluator("genesis", config, catalog)

    注册新后端只需实现:
        BackendRegistry.register(name, sim_cls, evaluator_cls)
    """
    """仿真后端注册表

    管理多个仿真后端 (MuJoCo/Genesis/MJX) 的注册、发现和工厂创建。
    所有后端遵循统一接口:
      - open(bodies, sim_config, task_config, catalog)
      - reset_all() → observations
      - rollout(policy_params, n_steps) → reward
      - close()
    """

    # 后端注册: backend_name → (module_path, class_name)
    _backends: Dict[str, Tuple[str, str]] = {
        "mujoco": ("forgecraft.rl.env", "ForgeCraftEnv"),
        "genesis": ("forgecraft.simulation.genesis_backend", "GenesisSimBackend"),
        # "mjx": ("forgecraft.simulation.mjx_backend", "MJXSimBackend"),
    }

    # 当前活动的后端
    _active: Optional[str] = None

    @classmethod
    def register(cls, name: str, module_path: str, class_name: str) -> None:
        """注册新的仿真后端

        Parameters
        ----------
        name : str
            后端名称 (用于 FORGECRAFT_BACKEND 环境变量)。
        module_path : str
            Python 模块路径 (e.g. "forgecraft.rl.env")。
        class_name : str
            后端类名 (e.g. "ForgeCraftEnv")。
        """
        if name in cls._backends:
            _logger.warning("后端 %s 已注册，将被覆盖", name)
        cls._backends[name] = (module_path, class_name)
        _logger.info("注册后端: %s → %s.%s", name, module_path, class_name)

    @classmethod
    def list_backends(cls) -> Dict[str, bool]:
        """列出所有已注册后端及其可用性

        Returns
        -------
        Dict[str, bool]
            {backend_name: is_available}
        """
        result = {}
        for name, (module_path, class_name) in cls._backends.items():
            try:
                import importlib
                mod = importlib.import_module(module_path)
                getattr(mod, class_name)
                result[name] = True
            except (ImportError, AttributeError, Exception):
                result[name] = False
        return result

    @classmethod
    def get_backend(cls, name: Optional[str] = None) -> str:
        """获取当前应使用的后端名称

        优先级: 参数 > 环境变量 > 默认(mujoco)

        Parameters
        ----------
        name : Optional[str]
            显式指定的后端名称。

        Returns
        -------
        str
            后端名称。
        """
        if name:
            return name
        env_backend = os.environ.get("FORGECRAFT_BACKEND", "")
        if env_backend:
            return env_backend
        return "mujoco"

    @classmethod
    def create_backend(
        cls,
        name: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """工厂方法: 创建仿真后端实例

        根据后端名称创建对应的仿真环境/后端实例。

        Parameters
        ----------
        name : Optional[str]
            后端名称。None 使用环境变量或默认值。
        **kwargs
            传递给后端构造函数的参数。

        Returns
        -------
        Any
            仿真后端实例 (ForgeCraftEnv / GenesisSimBackend / ...)

        Raises
        ------
        ValueError
            如果指定的后端未注册。
        ImportError
            如果后端依赖未安装。
        """
        backend_name = cls.get_backend(name)

        if backend_name not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(
                f"未知后端: {backend_name}。可用: {available}"
            )

        module_path, class_name = cls._backends[backend_name]

        try:
            import importlib
            mod = importlib.import_module(module_path)
            backend_cls = getattr(mod, class_name)
        except ImportError as e:
            raise ImportError(
                f"后端 {backend_name} 模块加载失败 ({module_path}): {e}\n"
                f"请安装: pip install genesis-world"
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f"后端类 {class_name} 未在 {module_path} 中找到"
            ) from e

        cls._active = backend_name
        _logger.info("创建后端: %s (%s.%s)", backend_name, module_path, class_name)

        instance = backend_cls(**kwargs)
        return instance

    @classmethod
    def create_evaluator(
        cls,
        name: Optional[str] = None,
        sim_config: Any = None,
        task_config: Any = None,
        rl_config: Any = None,
        catalog: Optional[Dict] = None,
        device: str = "cpu",
        n_workers: int = 4,
    ):
        """工厂方法: 创建带后端选择的评估器

        这是一个方便的工厂方法，返回 PopulationEvaluator，
        但内部会根据 backend 选择不同的评估路径。

        Parameters
        ----------
        name : Optional[str]
            后端名称。
        sim_config, task_config, rl_config, catalog, device, n_workers
            标准 PopulationEvaluator 参数。

        Returns
        -------
        PopulationEvaluator
            配置好后端选择的评估器。
        """
        from forgecraft.evolution.evaluator import PopulationEvaluator

        backend_name = cls.get_backend(name)
        evaluator = PopulationEvaluator(
            sim_config=sim_config,
            task_config=task_config,
            rl_config=rl_config,
            catalog=catalog or {},
            device=device,
            n_workers=n_workers,
        )
        # 注入后端信息
        evaluator._active_backend = backend_name
        return evaluator

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """获取后端状态摘要

        Returns
        -------
        dict
            {active, available_backends, env_var}
        """
        return {
            "active": cls._active or cls.get_backend(),
            "available_backends": cls.list_backends(),
            "env_var": os.environ.get("FORGECRAFT_BACKEND", "(未设置)"),
            "registered": list(cls._backends.keys()),
        }


# 便捷函数
def get_backend_status() -> Dict[str, Any]:
    """获取当前后端状态"""
    return BackendRegistry.get_status()


def create_simulation_backend(
    backend: Optional[str] = None,
    **kwargs,
) -> Any:
    """快速创建仿真后端 (便捷函数)

    Usage:
        env = create_simulation_backend("genesis", body=body, ...)
    """
    return BackendRegistry.create_backend(name=backend, **kwargs)
