"""
爆炸图生成器

基于装配体关节层级自动计算爆炸视图位置偏移。
支持:
- 多级爆炸深度 (沿运动链逐级推开)
- 关节轴对齐展开 (铰链沿旋转轴, 固定沿法向)
- 动画步进 (assembled → exploded 插值)
- 分组爆炸 (子装配体整体位移)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def get_joint_ends(joint: dict) -> Tuple[str, str, str]:
    """读取关节的 (父零件, 子零件, 类型)

    兼容两种序列化键名:
      - MechanicalBody.Joint.to_dict(): parent_id / child_id / joint_type
      - 早期草稿格式: part1 / part2 / type
    """
    a = joint.get("parent_id") or joint.get("part1") or ""
    b = joint.get("child_id") or joint.get("part2") or ""
    jtype = joint.get("joint_type") or joint.get("type") or "fixed"
    return a, b, jtype


def _build_adjacency(parts: List[dict], joints: List[dict]
                     ) -> Dict[str, List[Tuple[str, dict]]]:
    """构建零件邻接图: part_id → [(other_part_id, joint_info), ...]"""
    adj = defaultdict(list)
    for j in joints:
        p1, p2, _ = get_joint_ends(j)
        if p1 and p2:
            adj[p1].append((p2, j))
            adj[p2].append((p1, j))
    return adj


def _find_root(parts: List[dict], adj: Dict) -> str:
    """寻找根零件 (优先 chassis > tube > 连接度最高的零件)"""
    # 策略1: 按 part_type 优先级
    PRIORITY = ["alloy_chassis", "carbon_tube", "brushless_motor"]
    for pt in PRIORITY:
        for p in parts:
            if p.get("part_type") == pt and p.get("part_id", "") in adj:
                return p["part_id"]

    # 策略2: 连接度最高
    best, best_deg = None, -1
    for p in parts:
        pid = p.get("part_id", "")
        deg = len(adj.get(pid, []))
        if deg > best_deg:
            best, best_deg = pid, deg
    return best or (parts[0].get("part_id", "") if parts else "")


def _bfs_order(root: str, adj: Dict) -> List[Tuple[str, int, Optional[str]]]:
    """BFS 遍历, 返回 (part_id, depth, parent_id) 列表"""
    visited = {root}
    queue = [(root, 0, None)]
    order = [(root, 0, None)]
    while queue:
        cur, depth, parent = queue.pop(0)
        for nxt, joint in adj.get(cur, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, depth + 1, cur))
                order.append((nxt, depth + 1, cur))
    return order


def _explode_direction(joint: dict, from_part_pos: np.ndarray,
                       to_part_pos: np.ndarray) -> np.ndarray:
    """从关节类型和位置计算爆炸方向"""
    jtype = get_joint_ends(joint)[2]

    # 从 from 指向 to 的向量
    to_pt = np.asarray(to_part_pos, dtype=np.float64)
    from_pt = np.asarray(from_part_pos, dtype=np.float64)
    diff = to_pt - from_pt
    dist = np.linalg.norm(diff)

    if jtype == "hinge":
        # 铰链: 沿旋转轴展开
        axis = np.array(joint.get("axis", [0, 0, 1]), dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        return axis
    elif jtype == "prismatic":
        axis = np.array(joint.get("axis", [0, 0, 1]), dtype=np.float64)
        return axis / (np.linalg.norm(axis) + 1e-10)
    else:
        # fixed: 沿零件间连线方向, 或 Y 轴
        if dist > 1e-6:
            return diff / dist
        return np.array([0, 1, 0], dtype=np.float64)


def compute_exploded_positions(
    body_data: dict,
    explode_distance: float = 0.15,
    depth_factor: float = 1.3,
    num_steps: int = 5,
    direction_bias: Optional[np.ndarray] = None,
) -> Dict:
    """计算爆炸视图的零件位移

    Args:
        body_data: {parts: [...], joints: [...]}
        explode_distance: 每级爆炸的基础距离 (米)
        depth_factor: 每深一级的距离倍率
        num_steps: 动画步数
        direction_bias: 主展开方向 (默认自动检测)

    Returns:
        {
            'assembled': {part_id: (4x4 matrix)},     # 原始位置
            'exploded':  {part_id: (4x4 matrix)},      # 爆炸位置
            'steps': [{part_id: (4x4 matrix)}, ...],    # 动画步进
            'root': str,                                # 根零件ID
            'depth_map': {part_id: depth},              # 各零件深度
            'directions': {part_id: (dx,dy,dz)},        # 各零件展开方向
        }
    """
    parts = body_data.get("parts", [])
    joints = body_data.get("joints", [])

    if not parts:
        return {}

    # 建立位置映射
    pos_map = {}
    for p in parts:
        pid = p.get("part_id", "")
        pos = np.array(p.get("position", [0, 0, 0]), dtype=np.float64)
        pos_map[pid] = pos

    adj = _build_adjacency(parts, joints)
    root = _find_root(parts, adj)
    order = _bfs_order(root, adj)

    # 计算每个零件的爆炸方向
    explode_map: Dict[str, np.ndarray] = {}
    depth_map: Dict[str, int] = {}
    parent_map: Dict[str, Optional[str]] = {}

    for pid, depth, parent in order:
        depth_map[pid] = depth
        parent_map[pid] = parent
        if parent is None:
            # 根零件不移动
            explode_map[pid] = np.zeros(3)
            continue

        # 找到连接此父子对的关节
        p_pos = pos_map.get(parent, np.zeros(3))
        c_pos = pos_map.get(pid, np.zeros(3))
        direction = np.zeros(3)
        found = False
        for j in joints:
            ja, jb, _ = get_joint_ends(j)
            if (ja == pid and jb == parent) or (ja == parent and jb == pid):
                direction = _explode_direction(j, p_pos, c_pos)
                found = True
                break
        if not found:
            # 从父指向子的方向
            diff = c_pos - p_pos
            dist = np.linalg.norm(diff)
            direction = diff / dist if dist > 1e-6 else np.array([0, 1, 0], dtype=np.float64)

        # 确保方向朝外 (远离根)
        parent_offset = explode_map.get(parent, np.zeros(3))
        # 计算父零件爆炸后的位置
        parent_exploded = p_pos + parent_offset
        child_from_root = c_pos - parent_exploded
        if np.dot(direction, child_from_root) < 0:
            direction = -direction

        explode_map[pid] = direction

    # 计算累积位移
    displacement: Dict[str, np.ndarray] = {}
    for pid, depth, parent in order:
        base_dist = explode_distance * (depth_factor ** depth)
        local_disp = explode_map[pid] * base_dist
        if parent is not None and parent in displacement:
            displacement[pid] = displacement[parent] + local_disp
        else:
            displacement[pid] = local_disp if parent is not None else np.zeros(3)

    # 构建 4x4 变换矩阵
    assembled = {}
    exploded = {}
    steps = [{} for _ in range(num_steps)]

    for pid in pos_map:
        orig = np.eye(4)
        orig[:3, 3] = pos_map[pid]
        assembled[pid] = orig.copy()

        exp_t = orig.copy()
        disp = displacement.get(pid, np.zeros(3))
        exp_t[:3, 3] += disp
        exploded[pid] = exp_t

        for si in range(num_steps):
            alpha = (si + 1) / num_steps
            step_t = orig.copy()
            step_t[:3, 3] += disp * alpha
            steps[si][pid] = step_t

    return {
        "assembled": assembled,
        "exploded": exploded,
        "steps": steps,
        "root": root,
        "depth_map": depth_map,
        "directions": {pid: tuple(map(float, v)) for pid, v in explode_map.items()},
        "displacement": {pid: tuple(map(float, v)) for pid, v in displacement.items()},
    }


def create_exploded_body_data(
    body_data: dict,
    explode_distance: float = 0.15,
    depth_factor: float = 1.3,
) -> dict:
    """生成爆炸视图的 body_data (零件位置已偏移)"""
    result = compute_exploded_positions(body_data, explode_distance, depth_factor)
    if not result:
        return body_data

    exploded_body = {"parts": [], "joints": body_data.get("joints", [])}
    for p in body_data.get("parts", []):
        pid = p.get("part_id", "")
        new_p = dict(p)
        if pid in result["exploded"]:
            new_p["position"] = tuple(result["exploded"][pid][:3, 3].tolist())
        exploded_body["parts"].append(new_p)

    return exploded_body


def tight_assemble(body_data: dict, gen) -> dict:
    """紧装配: 调整零件位置使连接表面贴合, 消除浮动间隙

    对每个关节连接的零件对, 沿关节方向平移使两者 bounding box
    表面接触。结果是视觉上"装在一起的"装配体。
    """
    parts = body_data.get("parts", [])
    joints = body_data.get("joints", [])

    # 生成 bounding boxes
    bboxes = {}
    pos_map = {}
    for p in parts:
        pid = p.get("part_id", "")
        pos = np.array(p.get("position", [0, 0, 0]), dtype=np.float64)
        pos_map[pid] = pos
        mesh = gen.make(p.get("part_type", "unknown"), p.get("params", {}))
        if mesh is not None:
            bboxes[pid] = (mesh.bounds[0].copy(), mesh.bounds[1].copy())

    # 构建邻接
    adj = _build_adjacency(parts, joints)
    root = _find_root(parts, adj)

    # BFS, 记录偏移
    offsets: Dict[str, np.ndarray] = {root: np.zeros(3)}
    visited = {root}
    queue = [root]

    while queue:
        cur = queue.pop(0)
        cur_pos = pos_map[cur] + offsets[cur]
        cur_bbox = bboxes.get(cur)

        for nxt, joint in adj.get(cur, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)

            nxt_pos = pos_map[nxt]
            nxt_bbox = bboxes.get(nxt)

            if cur_bbox is None or nxt_bbox is None:
                offsets[nxt] = np.zeros(3)
                continue

            # 计算两个 bbox 中心之间的方向
            cur_center = (cur_bbox[0] + cur_bbox[1]) / 2
            nxt_center = (nxt_bbox[0] + nxt_bbox[1]) / 2
            direction = nxt_center - cur_center
            dist = np.linalg.norm(direction)
            if dist < 1e-8:
                direction = np.array([0, 1, 0], dtype=np.float64)
            else:
                direction = direction / dist

            # 计算 cur 在方向上的投影半长 + nxt 在反方向上的投影半长
            def projected_half_extent(bbox, d):
                """bbox 在方向 d 上的投影半长"""
                ext = (bbox[1] - bbox[0]) / 2
                return abs(ext[0] * d[0]) + abs(ext[1] * d[1]) + abs(ext[2] * d[2])

            half_cur = projected_half_extent(cur_bbox, direction)
            half_nxt = projected_half_extent(nxt_bbox, direction)

            # nxt 应该移到 cur 表面 + nxt 自身半长 的位置
            target = cur_center + direction * (half_cur + half_nxt)
            offset = target - nxt_center
            offsets[nxt] = offset

    # 应用偏移
    new_body = {"parts": [], "joints": list(joints)}
    for p in parts:
        pid = p.get("part_id", "")
        new_p = dict(p)
        orig = np.array(p.get("position", [0, 0, 0]), dtype=np.float64)
        off = offsets.get(pid, np.zeros(3))
        new_p["position"] = tuple((orig + off).tolist())
        new_body["parts"].append(new_p)

    return new_body
