"""
挤出机调度器 — 多材料打印时的切换时序管理

按层调度挤出机切换，最小化切换次数。
"""

from typing import Dict, List, Optional, Tuple

from forgecraft.cam.types import LayerToolpath, SliceLayer

__all__ = ["ExtruderScheduler", "ToolChangeEvent"]


class ToolChangeEvent:
    """换料事件"""
    def __init__(
        self,
        layer_index: int,
        from_material: str,
        to_material: str,
        from_extruder: int,
        to_extruder: int,
        purge_volume: float,
    ):
        self.layer_index = layer_index
        self.from_material = from_material
        self.to_material = to_material
        self.from_extruder = from_extruder
        self.to_extruder = to_extruder
        self.purge_volume = purge_volume

    def __repr__(self):
        return (
            f"ToolChange(layer={self.layer_index}, "
            f"{self.from_material}(T{self.from_extruder}) → "
            f"{self.to_material}(T{self.to_extruder}), "
            f"purge={self.purge_volume:.1f}mm³)"
        )


class ExtruderScheduler:
    """挤出机调度器

    Usage:
        sched = ExtruderScheduler(extruder_assignments={"PLA": 0, "PETG": 1})
        timeline = sched.schedule(layer_materials, purge_calculator)
    """

    def __init__(self, extruder_assignments: Dict[str, int] = None):
        """
        Args:
            extruder_assignments: {material_name: extruder_id}
        """
        self.extruder_assignments = extruder_assignments or {}
        self._next_extruder = 0

    def assign_extruders(self, materials: set) -> Dict[str, int]:
        """为材料集分配挤出机 ID"""
        assignments = {}
        for mat in sorted(materials):
            if mat in self.extruder_assignments:
                assignments[mat] = self.extruder_assignments[mat]
            else:
                assignments[mat] = self._next_extruder
                self._next_extruder += 1
        self.extruder_assignments.update(assignments)
        return assignments

    def schedule(
        self,
        layer_materials: List[str],
        purge_volume_func,
    ) -> List[ToolChangeEvent]:
        """生成换料事件序列

        Args:
            layer_materials: 每层使用的材料列表
            purge_volume_func: (from_mat, to_mat) → purge_volume 的回调

        Returns:
            按层号排序的换料事件列表
        """
        events = []
        current_material = None

        for layer_idx, material in enumerate(layer_materials):
            if current_material is None:
                current_material = material
                continue

            if material != current_material:
                from_ext = self.extruder_assignments.get(current_material, 0)
                to_ext = self.extruder_assignments.get(material, 1)
                purge_vol = purge_volume_func(current_material, material)

                events.append(ToolChangeEvent(
                    layer_index=layer_idx,
                    from_material=current_material,
                    to_material=material,
                    from_extruder=from_ext,
                    to_extruder=to_ext,
                    purge_volume=purge_vol,
                ))
                current_material = material

        return events

    def schedule_by_layer(
        self,
        toolpaths: List[LayerToolpath],
        part_materials: Dict[str, str],
    ) -> Tuple[List[ToolChangeEvent], List[List[str]]]:
        """根据 toolpath 和零件→材料映射调度

        Returns:
            (events, layer_material_list)
        """
        # 简化：每层只有一个材料 (假设不需要同层多材料)
        layer_mats = []
        for tp in toolpaths:
            # 取第一个非空段所在材料
            mat = "PLA"  # default
            if tp.perimeters:
                mat = part_materials.get("default", "PLA")
            layer_mats.append(mat)

        events = self.schedule(layer_mats, lambda a, b: 100.0)  # placeholder
        return events, layer_mats
