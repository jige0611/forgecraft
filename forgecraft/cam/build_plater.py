"""
排版优化器 — 2D 装箱 + GA 优化

功能:
  - Skyline 算法 (天际线 2D 装箱)
  - GA 优化零件排列顺序
  - 按材料分组排版
"""

import math
import random
from typing import Callable, List, Optional, Tuple

import numpy as np

from forgecraft.cam.types import BuildPlateLayout, PlacedPart

__all__ = ["BuildPlater", "SkylinePacker"]


# ══════════════════════════════════════════════════════════
# Skyline 2D 装箱
# ══════════════════════════════════════════════════════════

class SkylineSegment:
    """天际线段"""
    __slots__ = ("x", "y", "width")

    def __init__(self, x: float, y: float, width: float):
        self.x = x
        self.y = y
        self.width = width


class SkylinePacker:
    """天际线 2D 装箱算法

    算法描述：
      1. 维护一条"天际线"—— 一条阶梯状的水平线段列表
      2. 每个零件放置在天际线最低的位置
      3. 放置后更新天际线
    """

    def __init__(
        self,
        plate_width: float = 220.0,
        plate_depth: float = 220.0,
        gap: float = 5.0,
    ):
        self.width = plate_width
        self.depth = plate_depth
        self.gap = gap
        self.skyline = [SkylineSegment(0, 0, plate_width)]

    def reset(self):
        self.skyline = [SkylineSegment(0, 0, self.width)]

    def pack(
        self, parts: List[PlacedPart]
    ) -> Tuple[List[PlacedPart], List[PlacedPart]]:
        """执行装箱

        Returns:
            (placed_parts, unplaced_parts)
        """
        # 按面积降序排序 (大件优先)
        sorted_parts = sorted(
            parts,
            key=lambda p: p.bbox[0] * p.bbox[1],
            reverse=True,
        )

        placed = []
        unplaced = []

        for part in sorted_parts:
            w = part.bbox[0] + self.gap
            d = part.bbox[1] + self.gap

            x, y = self._find_best_position(w, d)
            if y + d <= self.depth:
                part.pos_x = x
                part.pos_y = y
                self._update_skyline(x, y, w, d)
                placed.append(part)
            else:
                unplaced.append(part)

        return placed, unplaced

    def _find_best_position(self, part_w: float, part_d: float) -> Tuple[float, float]:
        """找到天际线最低可容纳位置"""
        best_x, best_y, best_score = 0.0, float('inf'), float('inf')

        for seg in self.skyline:
            if seg.width >= part_w:
                y = seg.y + part_d
                if y <= self.depth:
                    score = y * 10000 + seg.x  # 优先低位置 + 靠左
                    if score < best_score:
                        best_x = seg.x
                        best_y = seg.y
                        best_score = score

        return best_x, best_y

    def _update_skyline(self, x: float, y: float, w: float, d: float):
        """放置零件后更新天际线"""
        new_skyline = []
        new_y = y + d

        for seg in self.skyline:
            seg_end = seg.x + seg.width
            if seg_end <= x or seg.x >= x + w:
                # 不重叠，保留
                new_skyline.append(seg)
            else:
                # 重叠区域处理
                if seg.x < x:
                    new_skyline.append(SkylineSegment(seg.x, seg.y, x - seg.x))
                if seg_end > x + w:
                    new_skyline.append(SkylineSegment(x + w, seg.y, seg_end - x - w))

        # 添加新占据区域
        new_skyline.append(SkylineSegment(x, new_y, w))

        # 按 x 排序
        new_skyline.sort(key=lambda s: s.x)

        # 合并同高度的相邻段
        self.skyline = self._merge(new_skyline)

    def _merge(self, segments: List[SkylineSegment]) -> List[SkylineSegment]:
        """合并同 Y 的相邻段"""
        if not segments:
            return []

        result = [segments[0]]
        for seg in segments[1:]:
            last = result[-1]
            if abs(last.y - seg.y) < 0.01 and abs(last.x + last.width - seg.x) < 0.01:
                last.width += seg.width
            else:
                result.append(seg)
        return result

    def max_height(self) -> float:
        """当前最高天际线高度"""
        return max(s.y for s in self.skyline) if self.skyline else 0.0

    def utilization(self) -> float:
        """面积利用率"""
        total_area = self.width * self.depth
        used = 0.0
        for seg in self.skyline:
            used += seg.width * (self.depth - seg.y)
        return 1.0 - (used / total_area)


# ══════════════════════════════════════════════════════════
# GA 优化器
# ══════════════════════════════════════════════════════════

class BatchOptimizer:
    """遗传算法优化零件排列顺序"""

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 30,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def optimize(self, parts: List[PlacedPart]) -> List[PlacedPart]:
        """GA 优化 → 返回最优排列后的 placed parts"""
        n = len(parts)
        if n <= 1:
            return parts

        # 初始化种群
        population = []
        for _ in range(self.population_size):
            order = list(range(n))
            random.shuffle(order)
            population.append(order)

        # 适应度缓存
        fitness_cache = {}

        def evaluate(order: Tuple[int, ...]) -> float:
            if order in fitness_cache:
                return fitness_cache[order]
            packer = SkylinePacker()
            ordered_parts = [copy_part(parts[i]) for i in order]
            placed, unplaced = packer.pack(ordered_parts)
            score = len(placed) * 10000 - packer.max_height() * 100 - len(unplaced) * 10000
            fitness_cache[order] = score
            return score

        for gen in range(self.generations):
            # 评估
            fitness = [evaluate(tuple(ind)) for ind in population]

            # 选择 (锦标赛)
            new_pop = []
            # 精英保留
            elite_idx = np.argmax(fitness)
            new_pop.append(population[elite_idx][:])

            while len(new_pop) < self.population_size:
                # 锦标赛选择
                p1 = self._tournament(population, fitness, 3)
                p2 = self._tournament(population, fitness, 3)

                # 交叉
                if random.random() < self.crossover_rate:
                    c1, c2 = self._pmx_crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                # 变异
                if random.random() < self.mutation_rate:
                    self._swap_mutation(c1)
                if random.random() < self.mutation_rate:
                    self._swap_mutation(c2)

                new_pop.append(c1)
                if len(new_pop) < self.population_size:
                    new_pop.append(c2)

            population = new_pop
            fitness_cache.clear()

        # 返回最优
        final_fitness = [evaluate(tuple(ind)) for ind in population]
        best_order = population[np.argmax(final_fitness)]
        return [parts[i] for i in best_order]

    def _tournament(self, population, fitness, k):
        """锦标赛选择"""
        candidates = random.sample(range(len(population)), k)
        best = max(candidates, key=lambda i: fitness[i])
        return population[best][:]

    def _pmx_crossover(self, p1, p2):
        """部分匹配交叉 (PMX)"""
        n = len(p1)
        a, b = sorted(random.sample(range(n), 2))
        c1, c2 = [-1] * n, [-1] * n

        # 复制区间
        mapping_1to2 = {}
        mapping_2to1 = {}
        for i in range(a, b + 1):
            c1[i] = p2[i]
            c2[i] = p1[i]
            mapping_1to2[p2[i]] = p1[i]
            mapping_2to1[p1[i]] = p2[i]

        # 填充其余位置
        for i in list(range(a)) + list(range(b + 1, n)):
            val = p1[i]
            while val in mapping_2to1:
                val = mapping_2to1[val]
            c1[i] = val

            val = p2[i]
            while val in mapping_1to2:
                val = mapping_1to2[val]
            c2[i] = val

        return c1, c2

    def _swap_mutation(self, individual):
        """交换两个位置"""
        a, b = random.sample(range(len(individual)), 2)
        individual[a], individual[b] = individual[b], individual[a]


def copy_part(part: PlacedPart) -> PlacedPart:
    """浅拷贝 PlacedPart"""
    return PlacedPart(
        part_id=part.part_id,
        mesh=part.mesh,
        orientation=part.orientation.copy(),
        bbox=part.bbox,
        material=part.material,
    )


# ══════════════════════════════════════════════════════════
# BuildPlater 主类
# ══════════════════════════════════════════════════════════

class BuildPlater:
    """排版总控

    Usage:
        plater = BuildPlater(plate_width=220, plate_depth=220)
        layout = plater.layout(parts)
    """

    def __init__(
        self,
        plate_width: float = 220.0,
        plate_depth: float = 220.0,
        gap: float = 5.0,
        use_ga: bool = True,
    ):
        self.plate_width = plate_width
        self.plate_depth = plate_depth
        self.gap = gap
        self.use_ga = use_ga
        self._packer = SkylinePacker(plate_width, plate_depth, gap)

    def layout(self, parts: List[PlacedPart]) -> BuildPlateLayout:
        """排版所有零件

        1. 按材料分组
        2. 每组内用 GA 优化排序 + Skyline 装箱
        3. 返回 BuildPlateLayout
        """
        import copy as cp

        # 按材料分组
        groups: dict = {}
        for part in parts:
            mat = part.material
            if mat not in groups:
                groups[mat] = []
            groups[mat].append(part)

        all_placed = []
        for material, group_parts in groups.items():
            if self.use_ga and len(group_parts) > 2:
                optimizer = BatchOptimizer()
                ordered = optimizer.optimize(group_parts)
            else:
                ordered = group_parts

            self._packer.reset()
            placed, unplaced = self._packer.pack(ordered)
            all_placed.extend(placed)

            if unplaced:
                # 未放下的零件尝试新一轮
                self._packer.reset()
                placed2, _ = self._packer.pack(unplaced)
                all_placed.extend(placed2)

        utilization = self._packer.utilization() if all_placed else 0.0

        return BuildPlateLayout(
            plate_width=self.plate_width,
            plate_depth=self.plate_depth,
            parts=all_placed,
            utilization=utilization,
        )
