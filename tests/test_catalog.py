"""
单元测试: 零件目录 (PartSpec / Catalog)
"""

import numpy as np
from forgecraft.config import PartSpec
from forgecraft.core.catalog import (
    DEFAULT_CATALOG, V2_INNOVATIVE_PARTS,
    get_default_catalog, get_v2_catalog
)


class TestPartSpec:
    def test_default_spec(self):
        s = PartSpec(part_type="test", mass=0.5, shape="box", size_range=[0.01, 0.3])
        assert s.can_actuate is False
        assert s.max_torque == 0.0
        assert len(s.color) == 4

    def test_get_range(self):
        s = PartSpec("test", 0.5, "box", [0.05, 0.15],
                      param_ranges={"length": [0.1, 0.5]})
        lo, hi = s.get_range("length")
        assert lo == 0.1
        assert hi == 0.5

        lo, hi = s.get_range("nonexistent")
        assert lo == 0.05
        assert hi == 0.15

    def test_clamp_param(self):
        s = PartSpec("test", 0.5, "box", [0.05, 0.15],
                      param_ranges={"length": [0.1, 0.5]})
        assert s.clamp_param("length", 0.05) == 0.1
        assert s.clamp_param("length", 0.3) == 0.3
        assert s.clamp_param("length", 1.0) == 0.5

    def test_actuated_spec(self):
        s = PartSpec("motor", 0.12, "box", [0.02, 0.06],
                      can_actuate=True, max_torque=5.0,
                      joint_type="hinge")
        assert s.can_actuate is True
        assert s.joint_type == "hinge"

    def test_friction(self):
        s = PartSpec("test", 0.5, "box", [0.05, 0.1],
                      friction=[1.2, 0.01, 0.01])
        assert s.friction[0] == 1.2


class TestDefaultCatalog:
    def test_default_has_base(self):
        cat = get_default_catalog()
        assert "base" in cat
        assert "motor" in cat
        assert "foot" in cat
        assert "wheel" in cat
        assert cat["motor"].can_actuate is True
        assert cat["base"].can_actuate is False

    def test_v2_catalog_has_innovations(self):
        cat = get_v2_catalog()
        assert "spring_element" in cat
        assert "hollow_tube" in cat
        assert "hemisphere_foot" in cat
        spring = cat["spring_element"]
        assert "stiffness" in spring.param_ranges
        assert spring.shape == "helical_spring"

    def test_v2_parts_have_ranges(self):
        cat = get_v2_catalog()
        for name, spec in cat.items():
            assert spec.mass > 0, f"{name} mass 必须 > 0"
            assert len(spec.color) == 4
            assert spec.friction[0] > 0
