"""
ForgeCraft 统一日志配置

  级别控制: FORGECRAFT_LOG_LEVEL env (默认 INFO)
  格式:      [时间] [级别] [模块名] [函数名:行号] 消息
  输出:      控制台 + 可选文件 (FORGECRAFT_LOG_FILE env)
"""

import logging
import sys
import time
from typing import Optional

__all__ = ["ForgeLogger", "get_logger", "set_logger"]

_LOGGER: Optional["ForgeLogger"] = None


class ForgeLogger:
    LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

    def __init__(self, name: str = "forgecraft", level: str = "INFO"):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self._logger.addHandler(handler)
        self._start_time = time.time()
        self._gen_times: list = []
        self._metrics: dict = {}

    def debug(self, msg: str):
        self._logger.debug(msg)

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def section(self, title: str):
        self._logger.info("=" * 60)
        self._logger.info(f"  {title}")
        self._logger.info("=" * 60)

    def generation_start(self, gen: int, mode_label: str, budget: dict):
        self._gen_times.append(time.time())
        self._logger.info(
            f"--- 第 {gen} 代 {mode_label} "
            f"[上限: {budget['episodes']}ep×{budget['steps_per_ep']}步] ---"
        )

    def generation_end(self, gen: int, stats: dict):
        elapsed = self._gen_times[-1] - (self._gen_times[-2] if len(self._gen_times) > 1 else self._start_time)
        self._gen_times.pop()
        self._logger.info(f"  最佳适应度: {stats['best_fitness']:.4f}")
        self._logger.info(f"  平均适应度: {stats['mean_fitness']:.4f}")
        self._logger.info(f"  中位数适应度: {stats['median_fitness']:.4f}")
        self._logger.info(f"  耗时: {elapsed:.1f}s")

    def new_best(self, fitness: float, mfg: float, n_parts: int, n_motors: int, pareto_info: str = ""):
        self._logger.info(
            f"  >>> 新一代最佳形态！适应度: {fitness:.4f}  "
            f"可制造性:{mfg:.2f} 零件:{n_parts} 马达:{n_motors}"
        )
        if pareto_info:
            self._logger.info(f"  {pareto_info}")

    def adaptive(self, msg: str):
        self._logger.info(f"  [自适应变异] {msg}")

    def diversity(self, sim: float, n_inject: int):
        self._logger.info(f"  [多样性低] 相似度{sim:.3f} → Novelty注入{n_inject}个个体")

    def eval_progress(self, msg: str):
        self._logger.info(f"  {msg}")

    def metric(self, key: str, value):
        self._metrics[key] = value

    @property
    def metrics(self) -> dict:
        return dict(self._metrics)

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time


def get_logger(name: str = "forgecraft", level: str = "INFO") -> ForgeLogger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = ForgeLogger(name, level)
    return _LOGGER


def set_logger(logger: ForgeLogger):
    global _LOGGER
    _LOGGER = logger
