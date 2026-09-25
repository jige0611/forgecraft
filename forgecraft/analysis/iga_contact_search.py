# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_contact_search — 接触碰撞检测
#
#   AABB 树 (Axis-Aligned Bounding Box)
#   Newton 最近点投影 (Closest Point Projection)
#   间隙函数 / 穿透检测
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.geometry.evaluator import LimitEvaluator, SurfacePoint
from forgecraft.analysis._iga_base import IrregularFaceCache
from forgecraft.analysis.iga_contact_types import (
    ContactSettings, ContactPair, GapResult, FrictionModel,
)

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "AABBTree",
    "AABBNode",
    "find_closest_point",
    "closest_point_newton",
    "compute_gap",
    "find_contact_pairs",
    "build_contact_pairs",
]


# ══════════════════════════════════════════════════════════
#  AABB 包围盒树
# ══════════════════════════════════════════════════════════

@dataclass
class AABBNode:
    """AABB 树节点"""
    face_id: int = -1            # 叶节点: 面 ID, 内部节点: -1
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    left: Optional[AABBNode] = None
    right: Optional[AABBNode] = None
    
    @property
    def is_leaf(self) -> bool:
        return self.face_id >= 0
    
    def bbox_area(self) -> float:
        """包围盒表面积 (用于 SAH 建树)"""
        extents = self.bbox_max - self.bbox_min
        return 2.0 * (extents[0]*extents[1] + extents[1]*extents[2] + extents[0]*extents[2])


class AABBTree:
    """AABB 包围盒树 (Binary BVH)
    
    为 CC 控制网格的每个面构建包围盒，支持快速候选面对筛选。
    
    用法:
        tree = AABBTree(mesh)
        candidates = tree.query_face_envelope(face_id, expansion=0.01)
    """
    
    def __init__(self, mesh: PolyMesh, expansion: float = 0.0):
        self.mesh = mesh
        self.expansion = expansion
        self.face_bboxes: List[Tuple[np.ndarray, np.ndarray]] = []
        self._build_bboxes()
        self.root = self._build_tree(list(range(mesh.n_faces)))
    
    def _build_bboxes(self):
        """为每个面计算包围盒"""
        for f in range(self.mesh.n_faces):
            verts = self.mesh.face_vertices(f)
            if len(verts) < 3:
                self.face_bboxes.append((np.zeros(3), np.zeros(3)))
                continue
            
            # 用顶点 + 控制点 (1 环) 估计包围盒
            pts = []
            seen = set(verts)
            for v in verts:
                pts.append(self.mesh.vertices[v])
                ring = self.mesh.vertex_ring(v)
                for r in ring[:8]:  # 最多 8 个邻域点
                    if r not in seen:
                        pts.append(self.mesh.vertices[r])
                        seen.add(r)
            
            pts = np.array(pts)
            pmin = pts.min(axis=0) - self.expansion
            pmax = pts.max(axis=0) + self.expansion
            self.face_bboxes.append((pmin, pmax))
    
    def _build_tree(self, face_ids: List[int], depth: int = 0) -> Optional[AABBNode]:
        """递归构建 AABB 树 (中位数分割, O(n log n))"""
        if not face_ids:
            return None
        
        if len(face_ids) == 1:
            fid = face_ids[0]
            return AABBNode(
                face_id=fid,
                bbox_min=self.face_bboxes[fid][0].copy(),
                bbox_max=self.face_bboxes[fid][1].copy(),
            )
        
        # 合并包围盒
        merged_min = np.full(3, np.inf)
        merged_max = np.full(3, -np.inf)
        for fid in face_ids:
            merged_min = np.minimum(merged_min, self.face_bboxes[fid][0])
            merged_max = np.maximum(merged_max, self.face_bboxes[fid][1])
        
        # 沿最长轴分割
        extents = merged_max - merged_min
        split_axis = np.argmax(extents)
        
        # 按包围盒中心排序
        centers = [
            (self.face_bboxes[fid][0][split_axis] + self.face_bboxes[fid][1][split_axis]) / 2.0
            for fid in face_ids
        ]
        median = np.median(centers)
        
        left_faces = [fid for fid, c in zip(face_ids, centers) if c < median]
        right_faces = [fid for fid, c in zip(face_ids, centers) if c >= median]
        
        if not left_faces or not right_faces:
            # 退化: 平均分割
            mid = len(face_ids) // 2
            left_faces = face_ids[:mid]
            right_faces = face_ids[mid:]
        
        node = AABBNode(
            bbox_min=merged_min.copy(),
            bbox_max=merged_max.copy(),
            left=self._build_tree(left_faces, depth + 1),
            right=self._build_tree(right_faces, depth + 1),
        )
        return node
    
    def query(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> List[int]:
        """查询与给定包围盒相交的所有面
        
        Args:
            bbox_min: (3,) 查询包围盒最小角
            bbox_max: (3,) 查询包围盒最大角
        
        Returns:
            候选面 ID 列表
        """
        candidates = []
        self._query_node(self.root, bbox_min, bbox_max, candidates)
        return candidates
    
    def _query_node(
        self, node: Optional[AABBNode],
        bbox_min: np.ndarray, bbox_max: np.ndarray,
        candidates: List[int],
    ):
        """递归查询节点"""
        if node is None:
            return
        
        # 包围盒相交检测
        if not self._bbox_overlap(node.bbox_min, node.bbox_max, bbox_min, bbox_max):
            return
        
        if node.is_leaf:
            candidates.append(node.face_id)
        else:
            self._query_node(node.left, bbox_min, bbox_max, candidates)
            self._query_node(node.right, bbox_min, bbox_max, candidates)
    
    def query_face(self, face_id: int, expansion: float = 0.0) -> List[int]:
        """查询与指定面可能相交的所有面
        
        Args:
            face_id: 面 ID
            expansion: 包围盒扩展量
        
        Returns:
            候选面 ID 列表 (不含自身)
        """
        fmin, fmax = self.face_bboxes[face_id]
        fmin = fmin - expansion
        fmax = fmax + expansion
        candidates = self.query(fmin, fmax)
        return [c for c in candidates if c != face_id]
    
    @staticmethod
    def _bbox_overlap(
        a_min: np.ndarray, a_max: np.ndarray,
        b_min: np.ndarray, b_max: np.ndarray,
    ) -> bool:
        """两个轴对齐包围盒是否相交"""
        return (
            a_min[0] <= b_max[0] and a_max[0] >= b_min[0] and
            a_min[1] <= b_max[1] and a_max[1] >= b_min[1] and
            a_min[2] <= b_max[2] and a_max[2] >= b_min[2]
        )


# ══════════════════════════════════════════════════════════
#  最近点投影 (Closest Point Projection)
# ══════════════════════════════════════════════════════════

def closest_point_newton(
    mesh: PolyMesh,
    face_id: int,
    query_point: np.ndarray,          # (3,)
    init_u: float = 0.5,
    init_v: float = 0.5,
    max_iter: int = 20,
    tolerance: float = 1e-8,
    evaluator: Optional[LimitEvaluator] = None,
    cache: Optional[IrregularFaceCache] = None,
) -> GapResult:
    """Newton-Raphson 最近点投影
    
    求解: min_{u,v ∈ [0,1]²} ||query_point - P(u,v)||²
    
    使用完整 Hessian (含二阶导) 保证二次收敛。
    二阶导通过数值差分近似。
    
    Args:
        mesh: CC 控制网格
        face_id: 面 ID
        query_point: (3,) 查询点物理坐标
        init_u, init_v: 初始参数猜测
        max_iter: 最大迭代数
        tolerance: 收敛容差
        evaluator: LimitEvaluator 实例 (可选, 默认创建)
        cache: 非常面缓存
    
    Returns:
        GapResult
    """
    if evaluator is None:
        evaluator = LimitEvaluator(max_subdivisions=4)
    
    u_k = np.clip(init_u, 0.0, 1.0)
    v_k = np.clip(init_v, 0.0, 1.0)
    
    converged = False
    iterations = 0
    
    # 有限差分步长
    eps = 1e-6
    
    for it in range(max_iter):
        iterations = it + 1
        
        # 求值: P(u,v), dP/du, dP/dv
        pt = evaluator.evaluate(mesh, face_id, u_k, v_k)
        P = pt.position
        dPdu = pt.du
        dPdv = pt.dv
        
        # 残差向量
        diff = P - query_point
        
        # 梯度: g = [diff·dPdu, diff·dPdv]
        g = np.array([np.dot(diff, dPdu), np.dot(diff, dPdv)])
        
        # 收敛检查
        if np.linalg.norm(g) < tolerance:
            converged = True
            break
        
        # Hessian 近似 (二阶导数值差分)
        d2P_duu, d2P_duv, d2P_dvv = _compute_second_derivatives(
            mesh, face_id, u_k, v_k, eps, evaluator,
        )
        
        # H = [dPdu·dPdu + diff·d2P_duu,   dPdu·dPdv + diff·d2P_duv;
        #      dPdv·dPdu + diff·d2P_duv,   dPdv·dPdv + diff·d2P_dvv]
        H = np.zeros((2, 2))
        H[0, 0] = np.dot(dPdu, dPdu) + np.dot(diff, d2P_duu)
        H[0, 1] = np.dot(dPdu, dPdv) + np.dot(diff, d2P_duv)
        H[1, 0] = H[0, 1]
        H[1, 1] = np.dot(dPdv, dPdv) + np.dot(diff, d2P_dvv)
        
        # 求解 Δ = -H^{-1} g
        try:
            delta = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            # Hessian 奇异 → 梯度下降
            delta = -0.1 * g
        
        # 线搜索 (backtracking)
        alpha = 1.0
        for _ in range(8):
            u_try = np.clip(u_k + alpha * delta[0], 0.0, 1.0)
            v_try = np.clip(v_k + alpha * delta[1], 0.0, 1.0)
            pt_try = evaluator.evaluate(mesh, face_id, u_try, v_try)
            f_try = 0.5 * np.sum((pt_try.position - query_point)**2)
            f_current = 0.5 * np.sum(diff**2)
            
            if f_try < f_current:
                break
            alpha *= 0.5
        else:
            alpha = 0.0  # 接受当前值
        
        u_k = np.clip(u_k + alpha * delta[0], 0.0, 1.0)
        v_k = np.clip(v_k + alpha * delta[1], 0.0, 1.0)
    
    # 最终求值
    pt_final = evaluator.evaluate(mesh, face_id, u_k, v_k)
    n = pt_final.normal
    if n is None or np.linalg.norm(n) < 1e-15:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / np.linalg.norm(n)
    
    gap_vec = query_point - pt_final.position
    gap_val = np.dot(gap_vec, n)
    dist = np.linalg.norm(gap_vec)
    
    return GapResult(
        position=pt_final.position.copy(),
        normal=n.copy(),
        parametric_coords=np.array([u_k, v_k]),
        gap=float(gap_val),
        distance=float(dist),
        converged=converged,
        iterations=iterations,
    )


def _compute_second_derivatives(
    mesh: PolyMesh,
    face_id: int,
    u: float, v: float,
    eps: float,
    evaluator: LimitEvaluator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """中心差分计算二阶导 ∂²P/∂u², ∂²P/∂u∂v, ∂²P/∂v²"""
    # duu
    pt_up = evaluator.evaluate(mesh, face_id, min(u + eps, 1.0), v)
    pt_um = evaluator.evaluate(mesh, face_id, max(u - eps, 0.0), v)
    d2P_duu = (pt_up.du - pt_um.du) / (2.0 * eps)
    
    # dvv
    pt_vp = evaluator.evaluate(mesh, face_id, u, min(v + eps, 1.0))
    pt_vm = evaluator.evaluate(mesh, face_id, u, max(v - eps, 0.0))
    d2P_dvv = (pt_vp.dv - pt_vm.dv) / (2.0 * eps)
    
    # duv (混合差分)
    d2P_duv = (pt_up.dv - pt_um.dv) / (2.0 * eps)
    
    return d2P_duu, d2P_duv, d2P_dvv


def find_closest_point(
    mesh: PolyMesh,
    face_id: int,
    query_point: np.ndarray,
    init_uv: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> GapResult:
    """便捷接口: 求最近点"""
    if init_uv is None:
        init_u, init_v = 0.5, 0.5
    else:
        init_u, init_v = init_uv
    
    return closest_point_newton(
        mesh, face_id, query_point, init_u, init_v, **kwargs,
    )


def compute_gap(
    mesh: PolyMesh,
    face_id: int,
    query_point: np.ndarray,
    **kwargs,
) -> float:
    """计算带符号间隙值"""
    result = find_closest_point(mesh, face_id, query_point, **kwargs)
    return result.gap


# ══════════════════════════════════════════════════════════
#  接触对检测
# ══════════════════════════════════════════════════════════

def find_contact_pairs(
    slave_mesh: PolyMesh,
    master_mesh: PolyMesh,
    master_tree: AABBTree,
    settings: ContactSettings,
    evaluator: Optional[LimitEvaluator] = None,
    slave_cache: Optional[IrregularFaceCache] = None,
    master_cache: Optional[IrregularFaceCache] = None,
) -> List[ContactPair]:
    """检测两体之间的所有活跃接触对
    
    流程:
      1. 对 slave 每个面的 Gauss 积分点
      2. 用 AABB 树查询候选 master 面
      3. 对每个候选面做 Newton 最近点投影
      4. 筛选 gap < 0 (穿透) 的接触对
    
    Args:
        slave_mesh: 从面网格
        master_mesh: 主面网格
        master_tree: 主面 AABB 树
        settings: 接触设置
        evaluator: LimitEvaluator
        slave_cache: 从面非常面缓存
        master_cache: 主面非常面缓存
    
    Returns:
        活跃接触对列表
    """
    if evaluator is None:
        evaluator = LimitEvaluator(max_subdivisions=4)
    
    from forgecraft.analysis.iga_quadrature import gauss_legendre_2d
    
    # 接触面积分点
    q_order = settings.quadrature_order_contact
    gauss_pts, gauss_wts = gauss_legendre_2d(q_order)
    
    contact_pairs = []
    expansion = 0.01  # AABB 搜索扩展
    
    for s_face in range(slave_mesh.n_faces):
        s_verts = slave_mesh.face_vertices(s_face)
        if len(s_verts) != 4:
            continue
        
        # 从面 Gauss 积分点
        for k in range(len(gauss_wts)):
            u_s, v_s = gauss_pts[k, 0], gauss_pts[k, 1]
            
            # 从面积分点物理坐标
            try:
                s_pt = evaluator.evaluate(slave_mesh, s_face, u_s, v_s)
            except Exception:
                continue
            
            s_pos = s_pt.position
            
            # AABB 查询候选主面
            bbox_min = s_pos - expansion
            bbox_max = s_pos + expansion
            candidates = master_tree.query(bbox_min, bbox_max)
            
            # 窄相: Newton 投影到每个候选面
            best_gap = np.inf
            best_result = None
            best_mface = -1
            
            for m_face in candidates:
                result = closest_point_newton(
                    master_mesh, m_face, s_pos,
                    evaluator=evaluator,
                    cache=master_cache,
                )
                
                if result.gap < best_gap:
                    best_gap = result.gap
                    best_result = result
                    best_mface = m_face
            
            # 穿透判断
            if best_result is not None and best_gap < settings.gap_tolerance:
                pair = ContactPair(
                    slave_face=s_face,
                    master_face=best_mface,
                    slave_uv=np.array([u_s, v_s]),
                    master_uv=best_result.parametric_coords,
                    slave_position=s_pos.copy(),
                    master_position=best_result.position.copy(),
                    normal=best_result.normal.copy(),
                    gap=best_gap,
                    active=True,
                )
                contact_pairs.append(pair)
    
    return contact_pairs


def build_contact_pairs(
    body_a: PolyMesh,
    body_b: PolyMesh,
    settings: ContactSettings,
    evaluator: Optional[LimitEvaluator] = None,
) -> List[ContactPair]:
    """构建两组网格间的所有接触对 (完整流程)
    
    Args:
        body_a: 第一个 CC 网格 (slave)
        body_b: 第二个 CC 网格 (master)
        settings: 接触设置
        evaluator: LimitEvaluator
    
    Returns:
        接触对列表
    """
    # 构建 master AABB 树
    tree_b = AABBTree(body_b, expansion=0.01)
    
    # 检测接触对
    pairs = find_contact_pairs(
        body_a, body_b, tree_b, settings, evaluator,
    )
    
    if settings.method != "penalty":
        _logger.info(f"Found {len(pairs)} active contact pairs")
    
    return pairs
