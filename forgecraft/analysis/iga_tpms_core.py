# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_tpms_core — TPMS 水平集数学核心
#
#   实现 8 种 TPMS 隐式方程及其梯度。
#   支持空间变化的频率 ω(x) 和壁厚 t(x)。
#
#   实体判定: |f(ωx·x, ωy·y, ωz·z)| < t/2
#   等效模量: Gibson-Ashby E_eff = C · E_solid · ρ_rel^n
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from forgecraft.analysis.iga_tpms_types import (
    TPMSType, TPMSParameters, TPMSGradingSettings, TPMS_DEFAULTS,
)

__all__ = [
    "tpms_level_set",
    "tpms_level_set_gradient",
    "tpms_is_solid",
    "compute_relative_density",
    "effective_youngs_modulus",
    "effective_shear_modulus",
    "compute_tpms_volume_fraction",
    "cell_size_to_frequency",
    "frequency_to_cell_size",
]


# ══════════════════════════════════════════════════════════
#  TPMS 隐式方程 (标量)
# ══════════════════════════════════════════════════════════

def tpms_level_set(
    x: float, y: float, z: float,
    tpms_type: TPMSType,
    wx: float = 2.0 * np.pi,
    wy: float = 2.0 * np.pi,
    wz: float = 2.0 * np.pi,
) -> float:
    """计算 TPMS 水平集值 f(ωx·x, ωy·y, ωz·z)

    所有方程都归一化到大致 [-1, 1] 范围。

    Args:
        x, y, z: 空间坐标
        tpms_type: TPMS 类型
        wx, wy, wz: 各向频率 ω = 2π / cell_size

    Returns:
        水平集值 (正=实体方向, 负=空方向, 零=曲面)
    """
    u = wx * x
    v = wy * y
    w = wz * z

    if tpms_type == TPMSType.GYROID:
        return float(np.sin(u) * np.cos(v) + np.sin(v) * np.cos(w) + np.sin(w) * np.cos(u))
    elif tpms_type == TPMSType.SCHWARZ_P:
        return float(np.cos(u) + np.cos(v) + np.cos(w))
    elif tpms_type == TPMSType.SCHWARZ_D:
        return float(np.cos(u) * np.cos(v) * np.cos(w) - np.sin(u) * np.sin(v) * np.sin(w))
    elif tpms_type == TPMSType.DIAMOND:
        su, cu = np.sin(u), np.cos(u)
        sv, cv = np.sin(v), np.cos(v)
        sw, cw = np.sin(w), np.cos(w)
        return float(su * sv * sw + su * cv * cw + cu * sv * cw + cu * cv * sw)
    elif tpms_type == TPMSType.SPLIT_P:
        s2u, c2u = np.sin(2 * u), np.cos(2 * u)
        s2v, c2v = np.sin(2 * v), np.cos(2 * v)
        s2w, c2w = np.sin(2 * w), np.cos(2 * w)
        return float(1.1 * (s2u * np.sin(w) * np.cos(v) +
                            s2v * np.sin(u) * np.cos(w) +
                            s2w * np.sin(v) * np.cos(u)) -
                     0.2 * (c2u * c2v + c2v * c2w + c2w * c2u))
    elif tpms_type == TPMSType.LIDINOID:
        s2u, c2u = np.sin(2 * u), np.cos(2 * u)
        s2v, c2v = np.sin(2 * v), np.cos(2 * v)
        s2w, c2w = np.sin(2 * w), np.cos(2 * w)
        return float(0.5 * (s2u * np.sin(w) * np.cos(v) +
                            s2v * np.sin(u) * np.cos(w) +
                            s2w * np.sin(v) * np.cos(u)) -
                     0.5 * (c2u * c2v + c2v * c2w + c2w * c2u))
    elif tpms_type == TPMSType.NEOVIUS:
        return float(3 * (np.cos(u) + np.cos(v) + np.cos(w)) +
                     4 * np.cos(u) * np.cos(v) * np.cos(w))
    elif tpms_type == TPMSType.IWP:
        return float(np.cos(u) * np.cos(v) + np.cos(v) * np.cos(w) + np.cos(w) * np.cos(u))
    else:
        raise ValueError(f"未知 TPMS 类型: {tpms_type}")


# ══════════════════════════════════════════════════════════
#  TPMS 梯度 (解析)
# ══════════════════════════════════════════════════════════

def tpms_level_set_gradient(
    x: float, y: float, z: float,
    tpms_type: TPMSType,
    wx: float = 2.0 * np.pi,
    wy: float = 2.0 * np.pi,
    wz: float = 2.0 * np.pi,
) -> Tuple[float, float, float]:
    """计算 TPMS 水平集的解析梯度 ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)

    Args:
        x, y, z: 空间坐标
        tpms_type: TPMS 类型
        wx, wy, wz: 各向频率

    Returns:
        (dfdx, dfdy, dfdz)
    """
    u = wx * x
    v = wy * y
    w = wz * z

    if tpms_type == TPMSType.GYROID:
        dfdu = float(np.cos(u) * np.cos(v) - np.sin(w) * np.sin(u))
        dfdv = float(-np.sin(u) * np.sin(v) + np.cos(v) * np.cos(w))
        dfdw = float(-np.sin(v) * np.sin(w) + np.cos(w) * np.cos(u))
        return (dfdu * wx, dfdv * wy, dfdw * wz)

    elif tpms_type == TPMSType.SCHWARZ_P:
        dfdu = float(-np.sin(u))
        dfdv = float(-np.sin(v))
        dfdw = float(-np.sin(w))
        return (dfdu * wx, dfdv * wy, dfdw * wz)

    elif tpms_type == TPMSType.SCHWARZ_D:
        su, cu = np.sin(u), np.cos(u)
        sv, cv = np.sin(v), np.cos(v)
        sw, cw = np.sin(w), np.cos(w)
        dfdu = float(-su * cv * cw - cu * sv * sw)
        dfdv = float(-cu * sv * cw - su * cv * sw)
        dfdw = float(-cu * cv * sw - su * sv * cw)
        return (dfdu * wx, dfdv * wy, dfdw * wz)

    elif tpms_type == TPMSType.SCHWARZ_P:
        return (float(-np.sin(u) * wx),
                float(-np.sin(v) * wy),
                float(-np.sin(w) * wz))

    elif tpms_type == TPMSType.DIAMOND:
        su, cu = np.sin(u), np.cos(u)
        sv, cv = np.sin(v), np.cos(v)
        sw, cw = np.sin(w), np.cos(w)
        dfdu = float(cu * sv * sw - su * cv * cw + cu * cv * cw - su * sv * sw)
        dfdv = float(su * cv * sw - cu * sv * cw - su * sv * cw + cu * cv * sw)
        dfdw = float(su * sv * cw + su * cv * cw - cu * sv * sw + cu * cv * sw)
        return (dfdu * wx, dfdv * wy, dfdw * wz)

    elif tpms_type == TPMSType.NEOVIUS:
        su, cu = np.sin(u), np.cos(u)
        sv, cv = np.sin(v), np.cos(v)
        sw, cw = np.sin(w), np.cos(w)
        dfdu = float(-3 * su - 4 * su * cv * cw)
        dfdv = float(-3 * sv - 4 * cu * sv * cw)
        dfdw = float(-3 * sw - 4 * cu * cv * sw)
        return (dfdu * wx, dfdv * wy, dfdw * wz)

    elif tpms_type == TPMSType.IWP:
        su, cu = np.sin(u), np.cos(u)
        sv, cv = np.sin(v), np.cos(v)
        sw, cw = np.sin(w), np.cos(w)
        dfdu = float(-su * cv - su * cw)
        dfdv = float(-cu * sv - sv * cw)
        dfdw = float(-cu * sw - cv * sw)
        return (dfdu * wx, dfdv * wy, dfdw * wz)

    else:
        # SPLIT_P, LIDINOID: 数值梯度
        eps = 1e-6
        f0 = tpms_level_set(x, y, z, tpms_type, wx, wy, wz)
        fx = tpms_level_set(x + eps, y, z, tpms_type, wx, wy, wz)
        fy = tpms_level_set(x, y + eps, z, tpms_type, wx, wy, wz)
        fz = tpms_level_set(x, y, z + eps, tpms_type, wx, wy, wz)
        return ((fx - f0) / eps, (fy - f0) / eps, (fz - f0) / eps)


# ══════════════════════════════════════════════════════════
#  TPMS 实体判定
# ══════════════════════════════════════════════════════════

def _tpms_level_set_vectorized(
    points: np.ndarray,
    tpms_type: TPMSType,
    wx: np.ndarray,
    wy: np.ndarray,
    wz: np.ndarray,
) -> np.ndarray:
    """批量计算 TPMS 水平集值 (快速路径)

    Args:
        points: (N, 3) 空间坐标
        tpms_type: TPMS 类型
        wx, wy, wz: (N,) 各向频率

    Returns:
        (N,) 水平集值
    """
    u = wx * points[:, 0]
    v = wy * points[:, 1]
    w = wz * points[:, 2]

    if tpms_type == TPMSType.GYROID:
        return np.sin(u) * np.cos(v) + np.sin(v) * np.cos(w) + np.sin(w) * np.cos(u)
    elif tpms_type == TPMSType.SCHWARZ_P:
        return np.cos(u) + np.cos(v) + np.cos(w)
    elif tpms_type == TPMSType.SCHWARZ_D:
        return np.cos(u) * np.cos(v) * np.cos(w) - np.sin(u) * np.sin(v) * np.sin(w)
    elif tpms_type == TPMSType.DIAMOND:
        su, cu = np.sin(u), np.cos(u)
        sv, cv = np.sin(v), np.cos(v)
        sw, cw = np.sin(w), np.cos(w)
        return su * sv * sw + su * cv * cw + cu * sv * cw + cu * cv * sw
    elif tpms_type == TPMSType.NEOVIUS:
        return 3 * (np.cos(u) + np.cos(v) + np.cos(w)) + 4 * np.cos(u) * np.cos(v) * np.cos(w)
    elif tpms_type == TPMSType.IWP:
        return np.cos(u) * np.cos(v) + np.cos(v) * np.cos(w) + np.cos(w) * np.cos(u)
    else:
        # SPLIT_P, LIDINOID: 逐点计算
        n = len(points)
        result = np.empty(n)
        for i in range(n):
            result[i] = tpms_level_set(
                points[i, 0], points[i, 1], points[i, 2],
                tpms_type, float(wx[i]), float(wy[i]), float(wz[i]),
            )
        return result


def tpms_is_solid(
    points: np.ndarray,
    tpms_type: TPMSType,
    params: TPMSParameters,
) -> np.ndarray:
    """判断点是否在 TPMS 实体内部

    实体判定: |f(ωx, ωy, ωz) - offset| < thickness / 2

    Args:
        points: (N, 3) 空间坐标
        tpms_type: TPMS 类型
        params: TPMS 参数 (均匀)

    Returns:
        (N,) bool 数组
    """
    n = len(points)
    wx = np.full(n, params.wx)
    wy = np.full(n, params.wy)
    wz = np.full(n, params.wz)

    f_vals = _tpms_level_set_vectorized(points, tpms_type, wx, wy, wz)
    return np.abs(f_vals - params.offset) < params.thickness / 2.0


def tpms_is_solid_graded(
    points: np.ndarray,
    tpms_type: TPMSType,
    cell_size: np.ndarray,
    thickness: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """判断点是否在梯度 TPMS 实体内部

    每个点有独立的参数 (cell_size, thickness, offset)。

    Args:
        points: (N, 3) 空间坐标
        tpms_type: TPMS 类型
        cell_size: (N,) 各点晶胞尺寸
        thickness: (N,) 各点壁厚
        offset: (N,) 各点偏移

    Returns:
        (N,) bool 数组
    """
    n = len(points)
    wx = 2.0 * np.pi / np.clip(cell_size, 1e-10, None)
    wy = wx.copy()
    wz = wx.copy()

    f_vals = _tpms_level_set_vectorized(points, tpms_type, wx, wy, wz)
    return np.abs(f_vals - offset) < thickness / 2.0


# ══════════════════════════════════════════════════════════
#  体积分数 & 相对密度
# ══════════════════════════════════════════════════════════

def compute_relative_density(
    params: TPMSParameters,
    tpms_type: TPMSType = TPMSType.GYROID,
    n_samples: int = 10000,
) -> float:
    """通过采样估算均匀 TPMS 的相对密度

    在一个晶胞内随机采样, 统计实体点比例。

    Args:
        params: TPMS 参数
        tpms_type: TPMS 类型
        n_samples: 采样点数

    Returns:
        相对密度 ρ_rel = V_solid / V_total
    """
    L = params.cell_size
    # 在 [0, L]³ 立方体内均匀采样
    points = np.random.uniform(0, L, size=(n_samples, 3))
    solid_mask = tpms_is_solid(points, tpms_type, params)
    return float(np.mean(solid_mask))


def compute_tpms_volume_fraction(
    params: TPMSParameters,
    tpms_type: TPMSType = TPMSType.GYROID,
    n_samples: int = 10000,
) -> float:
    """同 compute_relative_density, 但返回体积分数"""
    return compute_relative_density(params, tpms_type, n_samples)


# ══════════════════════════════════════════════════════════
#  等效材料属性 (Gibson-Ashby)
# ══════════════════════════════════════════════════════════

def effective_youngs_modulus(
    relative_density: float,
    E_solid: float,
    tpms_type: TPMSType = TPMSType.GYROID,
) -> float:
    """Gibson-Ashby 等效杨氏模量

    E_eff = C · E_solid · (ρ_rel)^n

    参数来自文献综述 (TPMS_DEFAULTS)。

    Args:
        relative_density: 相对密度 ρ/ρ_solid ∈ [0, 1]
        E_solid: 固体材料杨氏模量 (Pa)
        tpms_type: TPMS 类型

    Returns:
        等效杨氏模量 (Pa)
    """
    defaults = TPMS_DEFAULTS.get(tpms_type, {"C": 0.5, "n": 2.0})
    C = defaults["C"]
    n_exp = defaults["n"]
    return C * E_solid * (relative_density ** n_exp)


def effective_shear_modulus(
    relative_density: float,
    E_solid: float,
    nu_solid: float = 0.3,
    tpms_type: TPMSType = TPMSType.GYROID,
) -> float:
    """有效剪切模量

    G_eff = E_eff / (2 * (1 + ν_eff))

    假设 TPMS 晶格有效泊松比 ν_eff ≈ ν_solid (对于 Gyroid 等各向同性 TPMS)。

    Args:
        relative_density: 相对密度
        E_solid: 固体杨氏模量
        nu_solid: 固体泊松比
        tpms_type: TPMS 类型

    Returns:
        等效剪切模量 (Pa)
    """
    E_eff = effective_youngs_modulus(relative_density, E_solid, tpms_type)
    # 有效泊松比近似为固体泊松比
    return E_eff / (2.0 * (1.0 + nu_solid))


# ══════════════════════════════════════════════════════════
#  频率与晶胞尺寸互转
# ══════════════════════════════════════════════════════════

def cell_size_to_frequency(cell_size: float) -> float:
    """晶胞尺寸 → 频率 ω = 2π / L"""
    return 2.0 * np.pi / max(cell_size, 1e-10)


def frequency_to_cell_size(omega: float) -> float:
    """频率 → 晶胞尺寸 L = 2π / ω"""
    return 2.0 * np.pi / max(omega, 1e-10)
