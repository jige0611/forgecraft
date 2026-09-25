import copy
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

__all__ = ["Part", "Joint", "MechanicalBody"]

_logger = logging.getLogger(__name__)


class Part:
    """机械零件 — 形态的基本组成单元

    Attributes:
        part_id:      8 位 hex UUID (自动生成, clone 保留)
        part_type:    零件类型名, 对应 PartSpec.part_type
        params:       {param_name: float} 物理参数 (长度/半径/质量/扭矩/...)
        position:     (3,) float32 相对父节点的局部位置
        orientation:  (4,) float32 四元数 [w,x,y,z]

    设计约束:
        - __slots__ 禁用 __dict__, 每个零件固定 112B 内存
        - clone() 保留 part_id (同一基因型标识)
        - params 存储所有 PartSpec 定义的参数值
    """

    __slots__ = ("part_id", "part_type", "params", "position", "orientation")

    def __init__(
        self,
        part_type: str,
        params: Optional[Dict[str, float]] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ):
        self.part_id = uuid.uuid4().hex[:8]
        self.part_type = part_type
        self.params = params or {}
        self.position = np.array(position, dtype=np.float32) if position is not None else np.zeros(3, dtype=np.float32)
        self.orientation = np.array(orientation, dtype=np.float32) if orientation is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def clone(self) -> "Part":
        p = Part(
            part_type=self.part_type,
            params=dict(self.params),
            position=self.position.copy(),
            orientation=self.orientation.copy(),
        )
        p.part_id = self.part_id
        return p

    def to_dict(self) -> dict:
        return {
            "part_id": self.part_id,
            "part_type": self.part_type,
            "params": self.params,
            "position": self.position.tolist(),
            "orientation": self.orientation.tolist(),
        }


class Joint:
    """机械关节 — 连接两个零件

    Attributes:
        joint_type:  fixed/hinge/ball/slide/free
        parent_id:   父零件 part_id (图中有向边起点)
        child_id:    子零件 part_id (图中有向边终点)
        anchor:      (3,) float32 关节锚点位置
        axis:        (3,) float32 关节旋转轴 (hinge/ball)
        params:      {param_name: float} 关节参数 (range_min/range_max/damping/...)
    """

    __slots__ = ("joint_type", "parent_id", "child_id", "anchor", "axis", "params")

    def __init__(
        self,
        joint_type: str,
        parent_id: str,
        child_id: str,
        anchor: Optional[np.ndarray] = None,
        axis: Optional[np.ndarray] = None,
        params: Optional[Dict[str, float]] = None,
    ):
        self.joint_type = joint_type
        self.parent_id = parent_id
        self.child_id = child_id
        self.anchor = np.array(anchor, dtype=np.float32) if anchor is not None else np.zeros(3, dtype=np.float32)
        self.axis = np.array(axis, dtype=np.float32) if axis is not None else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.params = params or {}

    def clone(self) -> "Joint":
        return Joint(
            joint_type=self.joint_type,
            parent_id=self.parent_id,
            child_id=self.child_id,
            anchor=self.anchor.copy(),
            axis=self.axis.copy(),
            params=dict(self.params),
        )

    def to_dict(self) -> dict:
        return {
            "joint_type": self.joint_type,
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "anchor": self.anchor.tolist(),
            "axis": self.axis.tolist(),
            "params": self.params,
        }


class MechanicalBody:
    """机械形态体 — 零件树 + GNN 编码

    核心数据结构:
        - nx.DiGraph 有向图 (root→leaves 的零件树)
        - root_id: 根节点 (必为 can_actuate=False 的零件)
        - fitness + fitness_components: 适应度 (由评估器计算)

    缓存机制:
        - _cached_features: GNN 特征 (node_feats, edge_feats, senders, receivers)
        - _cached_xml:      MJCF XML 字符串
        - _cached_mjcf:     MuJoCo model + data
        - invalidate_feature_cache(): 变异后调用, 清除所有缓存

    设计约束:
        - 无 __hash__ (可变对象, 缓存用 id(body) 做 key)
        - clone() 只深拷贝 graph (保留 fitness 和 part_id)
    """

    def __init__(self, name: str = "body"):
        self.name = name
        self.graph = nx.DiGraph()
        self.root_id: Optional[str] = None
        self.fitness: float = 0.0
        self.fitness_components: Dict[str, float] = {}
        self.policy_state: Optional[Dict[str, Any]] = None
        self._cached_features: Optional[tuple] = None
        self._cached_xml: Optional[str] = None
        self._cached_mjcf: Optional[tuple] = None

    def invalidate_feature_cache(self) -> None:
        self._cached_features = None
        self._cached_xml = None
        self._cached_mjcf = None

    # ── Graph mutation: add part as child with optional joint
    def add_part(self, part: Part) -> str:
        self.graph.add_node(part.part_id, part=part)
        if self.root_id is None:
            self.root_id = part.part_id
        return part.part_id

    def add_joint(self, joint: Joint) -> None:
        self.graph.add_edge(joint.parent_id, joint.child_id, joint=joint)

    # ── Graph lookup: get Part by node_id
    def get_part(self, part_id: str) -> Part:
        return self.graph.nodes[part_id]["part"]

    # ── Graph lookup: get Joint by (parent_id, child_id)
    def get_joint(self, parent_id: str, child_id: str) -> Joint:
        return self.graph.edges[parent_id, child_id]["joint"]

    def parts(self) -> List[Part]:
        return [self.graph.nodes[n]["part"] for n in self.graph.nodes]

    def joints(self) -> List[Joint]:
        return [self.graph.edges[u, v]["joint"] for u, v in self.graph.edges]

    def parts_dict(self) -> Dict[str, Part]:
        return {n: self.graph.nodes[n]["part"] for n in self.graph.nodes}

    def num_parts(self) -> int:
        return self.graph.number_of_nodes()

    def num_joints(self) -> int:
        return self.graph.number_of_edges()

    def leaf_parts(self) -> List[str]:
        return [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]

    def non_root_parts(self) -> List[str]:
        return [n for n in self.graph.nodes if n != self.root_id]

    # ── Graph traversal: get direct children of a node
    def children_of(self, part_id: str) -> List[str]:
        return list(self.graph.successors(part_id))

    def parent_of(self, part_id: str) -> Optional[str]:
        preds = list(self.graph.predecessors(part_id))
        return preds[0] if preds else None

    def depth_first_order(self) -> List[str]:
        if self.root_id is None:
            return []
        return list(nx.dfs_preorder_nodes(self.graph, self.root_id))

    def is_connected(self) -> bool:
        if self.graph.number_of_nodes() <= 1:
            return True
        return nx.is_weakly_connected(self.graph)

    def copy_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for n in self.graph.nodes:
            g.add_node(n, part=self.graph.nodes[n]["part"].clone())
        for u, v in self.graph.edges:
            g.add_edge(u, v, joint=self.graph.edges[u, v]["joint"].clone())
        return g

    def sync_from_graph(self, g: nx.DiGraph) -> None:
        self.graph = g
        if self.root_id not in self.graph.nodes:
            self.root_id = next(iter(self.graph.nodes)) if self.graph.nodes else None

    def clone(self) -> "MechanicalBody":
        new_body = MechanicalBody(name=self.name)
        new_body.root_id = self.root_id
        new_body.graph = self.copy_graph()
        new_body.fitness = self.fitness
        new_body.fitness_components = dict(self.fitness_components)
        new_body.policy_state = copy.deepcopy(self.policy_state) if self.policy_state else None
        return new_body

    def bfs_order_from_root(self) -> List[str]:
        if self.root_id is None:
            return []
        return list(nx.bfs_tree(self.graph, self.root_id).nodes())

    def get_level(self, part_id: str) -> int:
        if self.root_id is None or part_id not in self.graph:
            return -1
        try:
            return nx.shortest_path_length(self.graph, self.root_id, part_id)
        except nx.NetworkXNoPath:
            return -1

    def max_depth(self) -> int:
        return max(self.get_level(n) for n in self.graph.nodes) if self.graph.nodes else 0

    def actuated_joints(self) -> List[Tuple[str, str]]:
        result = []
        for u, v in self.graph.edges:
            child_part = self.graph.nodes[v]["part"]
            if child_part.params.get("actuated", 0.0) > 0.5:
                result.append((u, v))
        return result

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "root_id": self.root_id,
            "fitness": self.fitness,
            "fitness_components": self.fitness_components,
            "parts": [self.graph.nodes[n]["part"].to_dict() for n in self.graph.nodes],
            "edges": [
                {
                    "parent": u,
                    "child": v,
                    "joint": self.graph.edges[u, v]["joint"].to_dict(),
                }
                for u, v in self.graph.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MechanicalBody":
        body = cls(name=data["name"])
        body.root_id = data["root_id"]
        body.fitness = data.get("fitness", 0.0)
        body.fitness_components = data.get("fitness_components", {})
        for part_dict in data["parts"]:
            part = Part(
                part_type=part_dict["part_type"],
                params=part_dict["params"],
                position=np.array(part_dict["position"], dtype=np.float32),
                orientation=np.array(part_dict["orientation"], dtype=np.float32),
            )
            part.part_id = part_dict["part_id"]
            body.graph.add_node(part.part_id, part=part)
        for edge_dict in data.get("edges", []):
            joint_dict = edge_dict["joint"]
            joint = Joint(
                joint_type=joint_dict["joint_type"],
                parent_id=joint_dict["parent_id"],
                child_id=joint_dict["child_id"],
                anchor=np.array(joint_dict["anchor"], dtype=np.float32),
                axis=np.array(joint_dict["axis"], dtype=np.float32),
                params=joint_dict["params"],
            )
            body.graph.add_edge(edge_dict["parent"], edge_dict["child"], joint=joint)
        return body

    def describe(self) -> str:
        lines = [f"MechanicalBody: {self.name}"]
        lines.append(f"  Parts: {self.num_parts()}, Joints: {self.num_joints()}")
        lines.append(f"  Root: {self.root_id}")
        lines.append(f"  Fitness: {self.fitness:.4f}")
        for n in self.depth_first_order():
            part = self.get_part(n)
            parent = self.parent_of(n)
            joint_str = ""
            if parent:
                j = self.get_joint(parent, n)
                joint_str = f" [{j.joint_type}]"
            lines.append(f"    {n}: {part.part_type}{joint_str} pos={part.position}")
        return "\n".join(lines)
