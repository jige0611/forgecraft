"""
材料相容性矩阵

定义不同材料之间的结合强度、清洗系数、推荐界面类型。
"""

from typing import Dict, Optional, Tuple

__all__ = ["MaterialCompatibilityMatrix", "COMPATIBILITY_MATRIX"]


# 默认相容性数据
COMPATIBILITY_MATRIX: Dict[Tuple[str, str], dict] = {
    ("PLA", "PETG"):   {"bond": "weak",     "purge_factor": 1.2, "interface": "interlock"},
    ("PETG", "PLA"):   {"bond": "weak",     "purge_factor": 1.2, "interface": "interlock"},
    ("PLA", "TPU"):    {"bond": "moderate", "purge_factor": 1.8, "interface": "interlock"},
    ("TPU", "PLA"):    {"bond": "moderate", "purge_factor": 1.8, "interface": "interlock"},
    ("PETG", "TPU"):   {"bond": "moderate", "purge_factor": 1.5, "interface": "interlock"},
    ("TPU", "PETG"):   {"bond": "moderate", "purge_factor": 1.5, "interface": "interlock"},
    ("PLA", "PVA"):    {"bond": "good",     "purge_factor": 1.0, "interface": "graded"},
    ("PVA", "PLA"):    {"bond": "good",     "purge_factor": 1.0, "interface": "graded"},
    ("PLA", "ABS"):    {"bond": "weak",     "purge_factor": 1.5, "interface": "interlock"},
    ("ABS", "PLA"):    {"bond": "weak",     "purge_factor": 1.5, "interface": "interlock"},
    ("ABS", "HIPS"):   {"bond": "good",     "purge_factor": 1.0, "interface": "graded"},
    ("HIPS", "ABS"):   {"bond": "good",     "purge_factor": 1.0, "interface": "graded"},
    ("PLA", "PLA"):    {"bond": "excellent","purge_factor": 0.0, "interface": "none"},
    ("PETG", "PETG"):  {"bond": "excellent","purge_factor": 0.0, "interface": "none"},
    ("TPU", "TPU"):    {"bond": "excellent","purge_factor": 0.0, "interface": "none"},
    ("ABS", "ABS"):    {"bond": "excellent","purge_factor": 0.0, "interface": "none"},
    ("Nylon", "Nylon"): {"bond": "excellent","purge_factor": 0.0, "interface": "none"},
}

# 材料密度和价格数据
MATERIAL_DATA = {
    "PLA":  {"density": 1.24, "price_per_kg": 25.0,  "nozzle_temp": 210, "bed_temp": 60, "viscosity": 1.0},
    "PETG": {"density": 1.27, "price_per_kg": 30.0,  "nozzle_temp": 240, "bed_temp": 80, "viscosity": 1.4},
    "TPU":  {"density": 1.20, "price_per_kg": 40.0,  "nozzle_temp": 230, "bed_temp": 50, "viscosity": 2.5},
    "ABS":  {"density": 1.04, "price_per_kg": 22.0,  "nozzle_temp": 250, "bed_temp": 100,"viscosity": 1.1},
    "PVA":  {"density": 1.19, "price_per_kg": 60.0,  "nozzle_temp": 200, "bed_temp": 60, "viscosity": 1.3},
    "HIPS": {"density": 1.04, "price_per_kg": 35.0,  "nozzle_temp": 240, "bed_temp": 100,"viscosity": 1.2},
    "Nylon":{"density": 1.14, "price_per_kg": 50.0,  "nozzle_temp": 260, "bed_temp": 80, "viscosity": 1.6},
    "PC":   {"density": 1.20, "price_per_kg": 45.0,  "nozzle_temp": 280, "bed_temp": 110,"viscosity": 1.3},
    "ASA":  {"density": 1.07, "price_per_kg": 38.0,  "nozzle_temp": 250, "bed_temp": 100,"viscosity": 1.2},
}


class MaterialCompatibilityMatrix:
    """材料相容性查询"""

    def __init__(self):
        self._matrix = dict(COMPATIBILITY_MATRIX)
        self._materials = dict(MATERIAL_DATA)

    def check(self, mat_a: str, mat_b: str) -> Optional[dict]:
        """查询两材料的相容性"""
        key = (mat_a.upper(), mat_b.upper())
        # 尝试精确匹配
        if key in self._matrix:
            return self._matrix[key]
        # 尝试通用化 (取前缀)
        for (a, b), info in self._matrix.items():
            if mat_a.upper().startswith(a) and mat_b.upper().startswith(b):
                return info
        # Same material
        if mat_a.upper() == mat_b.upper():
            return {"bond": "excellent", "purge_factor": 0.0, "interface": "none"}
        # 默认不兼容
        return {"bond": "unknown", "purge_factor": 2.0, "interface": "interlock"}

    def purge_factor(self, mat_a: str, mat_b: str) -> float:
        """清洗体积系数"""
        compat = self.check(mat_a, mat_b)
        return compat.get("purge_factor", 1.5) if compat else 1.5

    def recommended_interface(self, mat_a: str, mat_b: str) -> str:
        """推荐界面类型"""
        compat = self.check(mat_a, mat_b)
        return compat.get("interface", "interlock") if compat else "interlock"

    def bond_strength(self, mat_a: str, mat_b: str) -> str:
        """结合强度等级"""
        compat = self.check(mat_a, mat_b)
        return compat.get("bond", "unknown") if compat else "unknown"

    def get_material_data(self, material: str) -> dict:
        """获取材料物理数据"""
        mat_upper = material.upper()
        if mat_upper in self._materials:
            return self._materials[mat_upper]
        # 默认 PLA
        return MATERIAL_DATA["PLA"]
