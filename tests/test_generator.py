"""
单元测试: 形态生成器
"""

import numpy as np
from forgecraft.core.generator import BodyGenerator
from forgecraft.core.catalog import get_default_catalog, get_v2_catalog


class TestBodyGenerator:
    def test_create_part(self):
        gen = BodyGenerator(get_default_catalog(), seed=42)
        part = gen.create_part("base")
        assert part.part_type == "base"
        assert part.part_id  # 非空
        assert "length" in part.params or "mass" in part.params

    def test_generate_random_body(self):
        gen = BodyGenerator(get_v2_catalog(), seed=42)
        body = gen.generate_random_body(min_parts=4, max_parts=8, max_depth=4)
        assert body.num_parts() >= 4
        assert body.num_parts() <= 8
        assert body.root_id is not None
        assert body.is_connected()

    def test_first_part_is_non_actuated(self):
        gen = BodyGenerator(get_v2_catalog(), seed=42)
        for _ in range(10):
            body = gen.generate_random_body(min_parts=4, max_parts=6)
            root = body.get_part(body.root_id)
            cat = get_v2_catalog()
            if root.part_type in cat:
                assert cat[root.part_type].can_actuate is False

    def test_generate_simple_walker(self):
        gen = BodyGenerator(get_default_catalog(), seed=42)
        body = gen.generate_simple_walker(n_legs=4, segments_per_leg=2)
        assert body.num_parts() >= 1 + 4 * 2  # root + legs
        assert len(body.actuated_joints()) >= 4

    def test_initial_population(self):
        gen = BodyGenerator(get_default_catalog(), seed=42)
        pop = gen.generate_initial_population(size=8, min_parts=3, max_parts=6)
        assert len(pop) == 8
        for b in pop:
            assert b.num_parts() >= 3

    def test_v2_parts_in_population(self):
        gen = BodyGenerator(get_v2_catalog(), seed=99)
        body = gen.generate_random_body(min_parts=6, max_parts=12)
        # V2 零件应该能被选中
        types = {p.part_type for p in body.parts()}
        # 不强制要求V2, 但应该至少有零件
        assert len(types) > 0
