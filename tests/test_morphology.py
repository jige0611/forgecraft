"""
单元测试: 形态学核心 (Part / Joint / MechanicalBody)
"""

import numpy as np
import pytest
import networkx as nx
from forgecraft.core.morphology import Part, Joint, MechanicalBody


class TestPart:
    def test_create_part(self):
        p = Part(part_type="base", params={"length": 0.1})
        assert p.part_type == "base"
        assert p.params["length"] == 0.1
        assert len(p.part_id) == 8
        assert p.position.shape == (3,)

    def test_clone_part(self):
        p = Part("motor", {"max_torque": 5.0}, position=np.array([1, 2, 3]))
        c = p.clone()
        assert c.part_id == p.part_id
        assert c.part_type == p.part_type
        assert c.params["max_torque"] == 5.0
        assert np.allclose(c.position, [1, 2, 3])
        # 确保深拷贝
        c.position[0] = 999
        assert p.position[0] == 1.0

    def test_to_dict(self):
        p = Part("foot", {"radius": 0.03}, np.array([0, -1, 0.2]))
        d = p.to_dict()
        assert d["part_type"] == "foot"
        assert d["params"]["radius"] == 0.03
        assert d["position"] == pytest.approx([0.0, -1.0, 0.2])


class TestJoint:
    def test_create_joint(self):
        j = Joint("hinge", "a", "b", anchor=np.array([0, 0.1, 0]))
        assert j.joint_type == "hinge"
        assert j.parent_id == "a"
        assert j.child_id == "b"
        assert j.axis.shape == (3,)

    def test_clone_joint(self):
        j = Joint("ball", "p1", "c1", params={"damping": 0.5})
        c = j.clone()
        assert c.params["damping"] == 0.5
        c.params["damping"] = 999
        assert j.params["damping"] == 0.5

    def test_to_dict(self):
        j = Joint("slide", "root", "leg1")
        d = j.to_dict()
        assert d["joint_type"] == "slide"
        assert d["parent_id"] == "root"


class TestMechanicalBody:
    def make_body(self):
        body = MechanicalBody("test")
        root = Part("base", {"length": 0.1}, np.array([0, 0, 0.5]))
        body.add_part(root)
        return body, root.part_id

    def test_add_part(self):
        body, rid = self.make_body()
        assert body.root_id == rid
        assert body.num_parts() == 1
        leg = Part("segment", {"length": 0.2})
        body.add_part(leg)
        assert body.num_parts() == 2

    def test_add_joint(self):
        body, rid = self.make_body()
        leg = Part("segment", {"length": 0.2}, np.array([0, 0, 0.4]))
        body.add_part(leg)
        j = Joint("hinge", rid, leg.part_id)
        body.add_joint(j)
        assert body.num_joints() == 1
        assert body.get_joint(rid, leg.part_id).joint_type == "hinge"

    def test_is_connected(self):
        body, _ = self.make_body()
        assert body.is_connected()
        isolated = Part("motor", {})
        body.add_part(isolated)
        # 孤立节点还是一条有向边 — 这里有向图弱连通仍需检查
        # 通过 add_joint 添加边后应连通
        assert body.num_parts() >= 1

    def test_depth_first_order(self):
        body, rid = self.make_body()
        leg1 = Part("segment", {}, np.array([0, 0, 0.3]))
        body.add_part(leg1)
        body.add_joint(Joint("hinge", rid, leg1.part_id))
        foot = Part("foot", {}, np.array([0, 0, 0.15]))
        body.add_part(foot)
        body.add_joint(Joint("hinge", leg1.part_id, foot.part_id))
        order = body.depth_first_order()
        assert order[0] == rid
        assert len(order) == 3

    def test_actuated_joints(self):
        body, rid = self.make_body()
        motor = Part("motor", {"actuated": 1.0}, np.array([0, 0, 0.4]))
        body.add_part(motor)
        body.add_joint(Joint("hinge", rid, motor.part_id))
        assert len(body.actuated_joints()) == 1

    def test_clone_body(self):
        body, rid = self.make_body()
        leg = Part("segment", {"length": 0.2}, np.array([0, 0, 0.4]))
        body.add_part(leg)
        body.add_joint(Joint("hinge", rid, leg.part_id))
        body.fitness = 0.5
        c = body.clone()
        assert c.num_parts() == 2
        assert c.fitness == 0.5
        c.fitness = 999
        assert body.fitness == 0.5

    def test_from_dict_roundtrip(self):
        body, rid = self.make_body()
        leg = Part("segment", {"length": 0.2}, np.array([0, 0, 0.4]))
        body.add_part(leg)
        body.add_joint(Joint("hinge", rid, leg.part_id))
        d = body.to_dict()
        restored = MechanicalBody.from_dict(d)
        assert restored.num_parts() == 2
        assert restored.num_joints() == 1

    def test_cache_invalidation(self):
        body, rid = self.make_body()
        body._cached_xml = "<test/>"
        body.invalidate_feature_cache()
        assert body._cached_xml is None


class TestBodyNullSafety:
    def test_empty_body(self):
        body = MechanicalBody("empty")
        assert body.num_parts() == 0
        assert body.depth_first_order() == []
        assert body.max_depth() == 0

    def test_missing_node(self):
        body, _ = TestMechanicalBody().make_body()
        assert body.get_level("nonexistent") == -1
