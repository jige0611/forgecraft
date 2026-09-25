"""
单元测试: MuJoCo 模型构建器
"""

import numpy as np
import pytest
from forgecraft.core.morphology import MechanicalBody, Part, Joint
from forgecraft.core.catalog import get_default_catalog
from forgecraft.simulation.builder import build_mjcf_model, _part_geom, _joint_xml


class TestPartGeom:
    def test_box(self):
        from forgecraft.config import PartSpec
        spec = PartSpec("box_test", 0.5, "box", [0.05, 0.15])
        p = Part("box_test", {"length": 0.1, "height": 0.02, "width": 0.02})
        xml = _part_geom(p, spec)
        assert 'type="box"' in xml
        assert 'mass="0.5000"' in xml

    def test_cylinder(self):
        from forgecraft.config import PartSpec
        spec = PartSpec("cyl", 0.3, "cylinder", [0.01, 0.1])
        p = Part("cyl", {"radius": 0.03, "length": 0.2})
        xml = _part_geom(p, spec)
        assert 'type="cylinder"' in xml

    def test_sphere(self):
        from forgecraft.config import PartSpec
        spec = PartSpec("sph", 0.1, "sphere", [0.01, 0.05])
        p = Part("sph", {"radius": 0.04})
        xml = _part_geom(p, spec)
        assert 'type="sphere"' in xml

    def test_hollow_cylinder(self):
        from forgecraft.config import PartSpec
        spec = PartSpec("hollow", 0.07, "hollow_cylinder", [0.05, 0.5])
        p = Part("hollow_tube", {"outer_diameter": 0.04, "length": 0.2})
        xml = _part_geom(p, spec)
        assert 'type="cylinder"' in xml

    def test_helical_spring(self):
        from forgecraft.config import PartSpec
        spec = PartSpec("spring", 0.03, "helical_spring", [0.02, 0.15])
        p = Part("spring_element", {"coil_diameter": 0.03, "free_length": 0.1})
        xml = _part_geom(p, spec)
        assert 'type="cylinder"' in xml

    def test_hemisphere_shell(self):
        from forgecraft.config import PartSpec
        spec = PartSpec("hemi", 0.015, "hemisphere_shell", [0.005, 0.03])
        p = Part("hemisphere_foot", {"radius": 0.02})
        xml = _part_geom(p, spec)
        assert 'type="sphere"' in xml

    def test_unknown_shape_fallback(self):
        from forgecraft.config import PartSpec
        spec = PartSpec("unknown", 0.1, "torus", [0.01, 0.1])
        p = Part("u", {"length": 0.1, "thickness": 0.02})
        xml = _part_geom(p, spec)
        assert 'type="box"' in xml  # fallback


class TestJointXml:
    def test_hinge(self):
        xml = _joint_xml("j1", "hinge", np.array([0, 1, 0]),
                          {"range_min": -1.5, "range_max": 1.5, "damping": 0.5},
                          np.zeros(3), np.array([0, 0, -0.1]))
        assert xml is not None
        assert 'type="hinge"' in xml

    def test_fixed(self):
        xml = _joint_xml("j2", "fixed", np.zeros(3), {}, np.zeros(3), np.zeros(3))
        assert xml is None

    def test_ball(self):
        xml = _joint_xml("j3", "ball", np.zeros(3), {}, np.zeros(3), np.array([0, 0, 0]))
        assert xml is not None
        assert 'type="ball"' in xml

    def test_free(self):
        xml = _joint_xml("j4", "free", np.zeros(3), {}, np.zeros(3), np.array([0, 0, 0]))
        assert xml is not None
        assert 'type="free"' in xml


class TestBuildMjcf:
    def test_simple_body(self):
        cat = get_default_catalog()
        body = MechanicalBody("test")
        root = Part("base", {"length": 0.1, "mass": 0.5}, np.array([0, 0, 0.5]))
        body.add_part(root)
        xml, jm, mm, si = build_mjcf_model(body, cat)
        assert '<mujoco model="forgecraft_body">' in xml
        assert 'name="floor"' in xml

    def test_body_with_leg(self):
        cat = get_default_catalog()
        body = MechanicalBody("walker")
        root = Part("base", {"length": 0.1, "mass": 0.5}, np.array([0, 0, 0.5]))
        body.add_part(root)
        motor = Part("motor", {"actuated": 1.0, "max_torque": 5.0, "max_velocity": 10.0, "length": 0.04},
                     np.array([0.02, 0, 0.45]))
        body.add_part(motor)
        body.add_joint(Joint("hinge", root.part_id, motor.part_id,
                             anchor=np.array([0.02, 0, -0.05]),
                             axis=np.array([0, 1, 0]),
                             params={"range_min": -1.5, "range_max": 1.5, "damping": 0.5}))
        xml, jm, mm, si = build_mjcf_model(body, cat)
        assert len(jm) >= 1
        assert len(mm) >= 1

    def test_touch_sensor_for_foot(self):
        cat = get_default_catalog()
        body = MechanicalBody("sensor_test")
        root = Part("base", {"length": 0.1}, np.array([0, 0, 0.5]))
        body.add_part(root)
        foot = Part("foot", {"radius": 0.02}, np.array([0, 0, 0.42]))
        body.add_part(foot)
        body.add_joint(Joint("fixed", root.part_id, foot.part_id,
                             anchor=np.array([0, 0, -0.08])))
        xml, jm, mm, si = build_mjcf_model(body, cat)
        assert len(si["touch_nodes"]) >= 1
