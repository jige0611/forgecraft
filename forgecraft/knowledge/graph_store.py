"""图知识库 — 零件本体图 & 形态图索引

用途:
  - 零件组合推理: "rotary_joint 通常和哪些零件搭配?"
  - 约束传播: "如果选了 base_plate, 必须至少有一个 attachment 面"
  - 形态子图匹配: "有没有成功的形态包含 wheel-motor-wheel 三联体?"
  - 功能替换: "如果不能用 motor, 可以用什么代替?"

实现:
  - NetworkX DiGraph 存储本体图
  - 节点: 零件类型 + 功能标签
  - 边: 组合关系 (is_a, connects_to, replaces, co_occurs_with)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from forgecraft.logging import get_logger

_logger = get_logger(__name__)

# ── 边类型 ───────────────────────────────────────────────────


class EdgeType:
    IS_A = "is_a"               # 特化关系: motor → rotary_joint (motor 是 rotary_joint 的特化)
    CONNECTS_TO = "connects_to" # 组合关系: motor 连接到 wheel
    REPLACES = "replaces"       # 功能等效: wheel 替代 foot
    CO_OCCURS = "co_occurs"     # 高频共现: base_plate 通常和 motor 一起出现
    CONFLICTS = "conflicts"     # 冲突: wheel 和 finger 材质不兼容
    REQUIRES = "requires"       # 依赖: motor 需要 power_source
    SYMMETRIC_TO = "symmetric_to" # 对称镜像: left_finger ↔ right_finger


def part_ontology_to_graph(ontology: Dict) -> nx.DiGraph:
    """将 PART_ONTOLOGY 字典转换为 NetworkX 图"""
    G = nx.DiGraph()

    groups = ontology.get("functional_groups", {})

    # 添加节点: 每个零件 + 功能组
    for func_name, func_data in groups.items():
        # 功能组节点
        G.add_node(
            func_name,
            node_type="functional_group",
            description=func_data.get("description", ""),
        )

        for part_name in func_data.get("parts", []):
            G.add_node(
                part_name,
                node_type="part",
                function=func_name,
            )
            # 零件 → 功能组
            G.add_edge(part_name, func_name, type=EdgeType.IS_A)

    # 添加组合规则中的关系
    rules = ontology.get("composition_rules", [])
    for rule in rules:
        rule_name = rule.get("rule", "")
        G.add_node(
            rule_name,
            node_type="composition_rule",
            description=rule.get("description", ""),
        )

    return G


class GraphKB:
    """图知识库 — 零件本体 + 成功形态子图索引

    支持:
      - 功能查询: get_parts_by_function("locomotion")
      - 组合查询: get_compatible_parts("motor")  → 哪些零件可以连到 motor?
      - 功能替换: find_replacements("wheel")     → 哪些零件在功能上等效于 wheel?
      - 子图匹配: find_morphology_patterns(pattern) → 是否存在类似子图?
    """

    def __init__(self, ontology: Optional[Dict] = None):
        if ontology is None:
            from forgecraft.knowledge.ontology import PART_ONTOLOGY
            ontology = PART_ONTOLOGY

        self.ontology_graph = part_ontology_to_graph(ontology)
        self._morphology_graphs: Dict[str, nx.DiGraph] = {}  # hash → 形态子图
        self._pattern_index: Dict[str, List[str]] = {}        # 模式标签 → 案例列表

    # ── 功能查询 ────────────────────────────────────────────

    def get_parts_by_function(self, function: str) -> List[str]:
        """获取指定功能的所有零件"""
        parts = []
        for node, attrs in self.ontology_graph.nodes(data=True):
            if attrs.get("function") == function or (
                attrs.get("node_type") == "part" and function == "any"
            ):
                parts.append(node)
        return parts

    def get_function_of_part(self, part_name: str) -> Optional[str]:
        """查询零件的功能归类"""
        if part_name in self.ontology_graph:
            attrs = self.ontology_graph.nodes[part_name]
            if attrs.get("node_type") == "part":
                return attrs.get("function")
            elif attrs.get("node_type") == "functional_group":
                return part_name
        return None

    # ── 组合查询 ────────────────────────────────────────────

    def get_compatible_parts(self, part_name: str) -> List[Tuple[str, str]]:
        """查询可与指定零件组合的其他零件

        Returns:
            [(part_name, relation_type), ...]
        """
        if part_name not in self.ontology_graph:
            return []

        results = []
        for neighbor in self.ontology_graph.successors(part_name):
            edge_data = self.ontology_graph.get_edge_data(part_name, neighbor)
            if edge_data:
                results.append((neighbor, edge_data.get("type", "unknown")))

        for predecessor in self.ontology_graph.predecessors(part_name):
            edge_data = self.ontology_graph.get_edge_data(predecessor, part_name)
            if edge_data:
                results.append((predecessor, edge_data.get("type", "unknown")))

        return results

    def get_frequently_co_occurring(self, part_name: str) -> List[str]:
        """查询与指定零件高频共现的其他零件"""
        from forgecraft.knowledge.ontology import PART_ONTOLOGY
        groups = PART_ONTOLOGY.get("functional_groups", {})
        part_func = self.get_function_of_part(part_name)

        result = set()
        # 同功能组内其他零件
        if part_func and part_func in groups:
            for p in groups[part_func].get("parts", []):
                if p != part_name:
                    result.add(p)

        # 任务优先组合
        for task_name, config in groups.get(
            "task_specific_configs", {}
        ).items():
            priority_parts = config.get("priority_parts", [])
            for pp in priority_parts:
                func = pp.get("function")
                if func:
                    for p in groups.get(func, {}).get("parts", []):
                        if p != part_name:
                            result.add(p)

        return list(result)[:20]

    # ── 功能替换 ────────────────────────────────────────────

    def find_replacements(self, part_name: str, function: Optional[str] = None) -> List[str]:
        """查找功能等效的替代零件

        Args:
            part_name: 原始零件名
            function: 限制在指定功能组内 (可选)

        Returns:
            替代零件列表 (同功能组其他零件)
        """
        func = function or self.get_function_of_part(part_name)
        if not func:
            return []

        from forgecraft.knowledge.ontology import PART_ONTOLOGY
        groups = PART_ONTOLOGY.get("functional_groups", {})
        if func in groups:
            return [p for p in groups[func].get("parts", []) if p != part_name]
        return []

    # ── 形态图索引 ───────────────────────────────────────────

    def index_morphology(self, case_id: str, body_graph: nx.DiGraph,
                         performance: Dict, tags: Optional[List[str]] = None):
        """将成功形态图索引到知识库

        Args:
            case_id: 设计案例 ID
            body_graph: MechanicalBody.graph (NetworkX DiGraph)
            performance: 性能指标字典
            tags: 标签列表 (如 ["fast", "stable", "6-legged"])
        """
        graph_hash = _hash_graph(body_graph)
        self._morphology_graphs[graph_hash] = body_graph.copy()

        # 按标签索引
        for tag in (tags or []):
            if tag not in self._pattern_index:
                self._pattern_index[tag] = []
            self._pattern_index[tag].append(case_id)

        # 按拓扑特征索引
        n_nodes = len(body_graph.nodes)
        depth = _max_depth(body_graph)
        leaf_count = _count_leaves(body_graph)

        topo_key = f"n{n_nodes}_d{depth}_l{leaf_count}"
        if topo_key not in self._pattern_index:
            self._pattern_index[topo_key] = []
        self._pattern_index[topo_key].append(case_id)

    def find_similar_morphologies(
        self,
        body_graph: nx.DiGraph,
        top_k: int = 5,
    ) -> List[str]:
        """查找与给定形态图最相似的成功案例

        当前实现: 基于拓扑特征 (节点数, 深度, 叶子数) 匹配
        未来可扩展为 GNN 图嵌入匹配
        """
        n_nodes = len(body_graph.nodes)
        depth = _max_depth(body_graph)
        leaf_count = _count_leaves(body_graph)

        # 精确匹配拓扑
        topo_key = f"n{n_nodes}_d{depth}_l{leaf_count}"
        exact = self._pattern_index.get(topo_key, [])
        if exact:
            return exact[:top_k]

        # 模糊匹配: 相似拓扑
        candidates = []
        for key, cases in self._pattern_index.items():
            if key.startswith("n"):
                try:
                    parts = key.split("_")
                    kn = int(parts[0][1:])
                    kd = int(parts[1][1:])
                    kl = int(parts[2][1:])
                    # 节点数相差 50% 以内
                    if abs(kn - n_nodes) / max(n_nodes, 1) <= 0.5:
                        candidates.extend(cases)
                except (ValueError, IndexError):
                    pass

        # 去重
        seen = set()
        result = []
        for c in candidates + exact:
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result[:top_k]

    # ── 规则检索 ────────────────────────────────────────────

    def get_composition_rules(self):
        """获取所有组合规则"""
        from forgecraft.knowledge.ontology import PART_ONTOLOGY
        return PART_ONTOLOGY.get("composition_rules", [])

    def check_constraint(self, part_a: str, part_b: str) -> List[str]:
        """检查两个零件之间的约束是否满足

        Returns:
            违反的规则列表 (空列表 = 通过)
        """
        violations = []
        rules = self.get_composition_rules()

        func_a = self.get_function_of_part(part_a)
        func_b = self.get_function_of_part(part_b)

        for rule in rules:
            ante = rule.get("antecedent", {})
            conseq = rule.get("consequent", [])

            # 简化检查: 如果两零件功能与规则不匹配则跳过
            if ante.get("function") in (func_a, func_b):
                if rule["rule"] == "actuator_connects_structural":
                    if func_a not in ("structural",) and func_b not in ("structural",):
                        violations.append(rule["rule"])

        return violations

    def get_statistics(self) -> Dict:
        return {
            "ontology_nodes": len(self.ontology_graph.nodes),
            "ontology_edges": len(self.ontology_graph.edges),
            "indexed_morphologies": len(self._morphology_graphs),
            "pattern_index_size": len(self._pattern_index),
        }


# ── 图工具函数 ───────────────────────────────────────────────


def _hash_graph(g: nx.DiGraph) -> str:
    """生成图的确定性哈希"""
    import hashlib

    data = f"{len(g.nodes)}:{len(g.edges)}:"
    for u, v in sorted(g.edges):
        data += f"{u}-{v};"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _max_depth(g: nx.DiGraph) -> int:
    """从根节点计算最大深度"""
    roots = [n for n, d in g.in_degree() if d == 0]
    if not roots:
        roots = [list(g.nodes)[0]] if g.nodes else []

    max_d = 0
    for root in roots:
        try:
            lengths = nx.single_source_shortest_path_length(g, root)
            max_d = max(max_d, max(lengths.values()))
        except Exception:
            pass
    return max_d


def _count_leaves(g: nx.DiGraph) -> int:
    """计算叶节点数 (出度为 0)"""
    return sum(1 for n, d in g.out_degree() if d == 0)
