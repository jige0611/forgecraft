"""
PBR 物理材质库

为每种零件类型定义物理准确的 PBR 材质 (metallic-roughness 模型),
用于 glTF 导出和高质量渲染。

材质参考:
- 碳纤维管: 高刚度复合材料, 哑光深灰, 编织纹理
- 轴承钢: 高抛光铬钢, 镜面反射
- 铝合金: 拉丝铝, 中等金属度
- 无刷电机: 深灰钢壳 + 铜色绕组
- LiPo 电池: 热缩套管塑料
- 弹簧: 弹簧钢, 有光泽
- 竞速脚: 橡胶/TPU, 粗糙无金属
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PBRMaterial:
    """PBR Metallic-Roughness 材质"""
    name: str
    base_color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    metallic: float = 0.0
    roughness: float = 0.5
    # 可选增强
    emissive: Tuple[float, float, float] = (0, 0, 0)
    emissive_strength: float = 0.0
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.0
    normal_scale: float = 1.0
    # 描述
    description: str = ""


# ================================================================
# 材质预设
# ================================================================

MATERIAL_PRESETS: Dict[str, PBRMaterial] = {
    # ── 结构件 ──
    "carbon_tube": PBRMaterial(
        name="Carbon Fiber Tube",
        base_color=(0.22, 0.22, 0.24, 1.0),
        metallic=0.15,
        roughness=0.55,
        description="Matte dark gray, woven texture visible, slight anisotropic reflection",
    ),
    "alloy_chassis": PBRMaterial(
        name="7075 Aluminum Chassis",
        base_color=(0.65, 0.66, 0.68, 1.0),
        metallic=0.85,
        roughness=0.35,
        description="Brushed aluminum, medium metallic, clear reflections",
    ),

    # ── 传动件 ──
    "micro_bearing": PBRMaterial(
        name="Chrome Steel Bearing",
        base_color=(0.78, 0.78, 0.79, 1.0),
        metallic=1.0,
        roughness=0.12,
        description="High polish chrome steel, mirror finish, very low roughness",
    ),
    "launch_spring": PBRMaterial(
        name="Spring Steel",
        base_color=(0.45, 0.48, 0.52, 1.0),
        metallic=1.0,
        roughness=0.28,
        description="Spring steel, glossy, tempered blue/gray tone",
    ),

    # ── 执行器 ──
    "brushless_motor": PBRMaterial(
        name="Brushless Motor (Steel)",
        base_color=(0.25, 0.27, 0.30, 1.0),
        metallic=0.9,
        roughness=0.40,
        description="Dark gray steel shell, cooling ribs slightly rougher",
    ),
    "brushless_motor_compact": PBRMaterial(
        name="Compact BLDC Motor",
        base_color=(0.25, 0.27, 0.30, 1.0),
        metallic=0.9,
        roughness=0.40,
        description="Same as standard brushless motor",
    ),

    # ── 能源 ──
    "compact_lipo": PBRMaterial(
        name="LiPo Battery (Compact)",
        base_color=(0.85, 0.35, 0.08, 1.0),
        metallic=0.0,
        roughness=0.65,
        description="Heat shrink sleeve, orange, non-metallic, high roughness plastic",
    ),
    "high_power_lipo": PBRMaterial(
        name="LiPo Battery (High Power)",
        base_color=(0.82, 0.15, 0.05, 1.0),
        metallic=0.0,
        roughness=0.65,
        description="Red heat shrink sleeve, high power marking",
    ),

    # ── 接地/终端 ──
    "sprint_foot": PBRMaterial(
        name="TPU Sprint Foot",
        base_color=(0.12, 0.13, 0.14, 1.0),
        metallic=0.0,
        roughness=0.85,
        description="TPU/rubber, extremely rough, non-metallic, high friction",
    ),

    # ── 回退 ──
    "generic": PBRMaterial(
        name="Generic PLA/ABS",
        base_color=(0.55, 0.55, 0.58, 1.0),
        metallic=0.0,
        roughness=0.50,
        description="Default generic plastic",
    ),
}


# ================================================================
# 子零件材质 (用于多材质网格)
# ================================================================

SUB_MATERIALS: Dict[str, Dict[str, PBRMaterial]] = {
    "brushless_motor": {
        "body": PBRMaterial("Steel Shell", (0.25, 0.27, 0.30, 1.0), 0.9, 0.4,
                            description="Motor body"),
        "shaft": PBRMaterial("Shaft Steel", (0.70, 0.71, 0.72, 1.0), 1.0, 0.15,
                             description="Output shaft"),
        "flange": PBRMaterial("Flange Alu", (0.60, 0.61, 0.63, 1.0), 0.8, 0.30,
                              description="Mounting flange"),
    },
    "micro_bearing": {
        "outer_ring": PBRMaterial("Outer Ring", (0.78, 0.78, 0.79, 1.0), 1.0, 0.12,
                                  description="Outer bearing ring"),
        "inner_ring": PBRMaterial("Inner Ring", (0.78, 0.78, 0.79, 1.0), 1.0, 0.10,
                                  description="Inner bearing ring"),
        "balls": PBRMaterial("Balls", (0.85, 0.85, 0.86, 1.0), 1.0, 0.05,
                             description="Bearing balls"),
        "shield": PBRMaterial("Shield", (0.55, 0.56, 0.58, 1.0), 0.7, 0.25,
                              description="Dust shield"),
    },
    "alloy_chassis": {
        "frame": PBRMaterial("Frame Aluminum", (0.65, 0.66, 0.68, 1.0), 0.85, 0.35,
                             description="Edge beams + cross ribs"),
        "plate": PBRMaterial("CF Plate", (0.22, 0.22, 0.24, 1.0), 0.15, 0.55,
                             description="Thin base plate"),
        "bosses": PBRMaterial("Boss Aluminum", (0.60, 0.61, 0.63, 1.0), 0.80, 0.30,
                              description="Mounting bosses"),
    },
}


def get_material(part_type: str) -> PBRMaterial:
    """获取零件类型的 PBR 材质, 无匹配时返回 generic"""
    return MATERIAL_PRESETS.get(part_type, MATERIAL_PRESETS["generic"])


def get_sub_material(part_type: str, sub_name: str) -> PBRMaterial:
    """获取子零件 PBR 材质"""
    subs = SUB_MATERIALS.get(part_type, {})
    return subs.get(sub_name, get_material(part_type))


def to_trimesh_material(mat: PBRMaterial):
    """将 PBRMaterial 转为 trimesh PBR material 对象"""
    try:
        from trimesh.visual.material import PBRMaterial as TriPBR
    except ImportError:
        from trimesh.visual.material import SimpleMaterial as TriPBR

    return TriPBR(
        name=mat.name,
        baseColorFactor=list(mat.base_color),
        metallicFactor=mat.metallic,
        roughnessFactor=mat.roughness,
        emissiveFactor=list(mat.emissive),
    )


def list_presets() -> List[str]:
    return sorted(MATERIAL_PRESETS.keys())
