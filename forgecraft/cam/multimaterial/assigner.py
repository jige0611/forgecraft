"""
材料分配器 — Part → Filament 映射

根据零件类型和功能分配最佳打印材料。
"""

from typing import Dict, Optional

from forgecraft.core.morphology import Part
from forgecraft.cam.multimaterial.compatibility import MaterialCompatibilityMatrix

__all__ = ["MaterialAssigner"]


# 零件类型 → 材料映射规则
_DEFAULT_PART_MATERIAL_MAP = {
    # 结构件 → 刚性材料
    "base": "PLA",
    "chassis": "PLA",
    "link": "PLA",
    "limb": "PETG",
    "foot": "TPU",           # 足部用柔性材料
    "wheel": "TPU",          # 轮子用 TPU 增加抓地力
    "gripper_pad": "TPU",    # 手爪垫用柔性

    # 关节件 → 耐磨材料
    "hinge": "PETG",
    "ball_joint": "Nylon",
    "axle": "PLA",
    "bearing": "Nylon",

    # 电机座 → 耐热
    "motor_mount": "PETG",

    # 支撑 → 可溶性
    "support": "PVA",

    # 外壳 → ABS (耐用)
    "housing": "ABS",
    "cover": "PLA",
}

# 结构件高耐磨用
_WEAR_RESISTANT_TYPES = {"hinge", "ball_joint", "bearing", "axle", "slide"}

# 高柔性用
_FLEXIBLE_TYPES = {"foot", "wheel", "gripper_pad", "spring", "damper"}

# 需要支撑
_SUPPORT_TYPES = {"support", "overhang_support"}


class MaterialAssigner:
    """零件→丝材分配器

    Usage:
        assigner = MaterialAssigner()
        material = assigner.assign(part)
    """

    def __init__(self, default_material: str = "PLA"):
        self.default_material = default_material
        self._compat = MaterialCompatibilityMatrix()
        self._custom_map: Dict[str, str] = {}

    def assign(self, part: Part) -> str:
        """为单个零件分配材料"""
        part_type = part.part_type.lower()

        # 1. 自定义映射优先
        if part_type in self._custom_map:
            return self._custom_map[part_type]

        # 2. 默认映射表
        if part_type in _DEFAULT_PART_MATERIAL_MAP:
            return _DEFAULT_PART_MATERIAL_MAP[part_type]

        # 3. 规则推断
        if any(t in part_type for t in _FLEXIBLE_TYPES):
            return "TPU"
        if any(t in part_type for t in _WEAR_RESISTANT_TYPES):
            return "Nylon"
        if any(t in part_type for t in _SUPPORT_TYPES):
            return "PVA"
        if "motor" in part_type or "mount" in part_type:
            return "PETG"

        return self.default_material

    def assign_all(self, parts: Dict[str, Part]) -> Dict[str, str]:
        """批量分配"""
        return {pid: self.assign(part) for pid, part in parts.items()}

    def set_custom(self, part_type: str, material: str):
        """自定义零件类型材料"""
        self._custom_map[part_type.lower()] = material.upper()

    @staticmethod
    def get_filament_density(material: str) -> float:
        """丝材密度 g/cm³"""
        data = MaterialCompatibilityMatrix().get_material_data(material)
        return data.get("density", 1.24)

    @staticmethod
    def get_nozzle_temp(material: str) -> float:
        """推荐喷嘴温度"""
        data = MaterialCompatibilityMatrix().get_material_data(material)
        return data.get("nozzle_temp", 210)

    @staticmethod
    def get_bed_temp(material: str) -> float:
        """推荐热床温度"""
        data = MaterialCompatibilityMatrix().get_material_data(material)
        return data.get("bed_temp", 60)
