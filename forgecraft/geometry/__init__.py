# ══════════════════════════════════════════════════════════
# forgecraft.geometry — 统一几何表示引擎
#
#   基于 Catmull-Clark 细分面 + T-Splines 等效自适应细分的
#   统一几何框架。提供:
#
#   - 半边数据结构 (PolyMesh)
#   - Catmull-Clark 细分 (极限 = 精确球面/环面)
#   - Stam 1998 极限曲面求值 (点/法向量/曲率)
#   - 自适应局部细分 (T-junction 裂缝处理 = T-Splines 等效)
#   - B-Rep ↔ Catmull-Clark 桥接 (cadquery 集成)
#   - 曲率驱动自适应镶嵌 (出 STL)
#   - 布尔运算 (CGAL C++ 加速 / trimesh 回退)
#
#   解决了 4 个弱点:
#     1. C-C ≠ T-Splines → 自适应细分 + 裂缝处理等效
#     2. 纯 Python 慢 → Numba JIT + CGAL C++ 可选
#     3. 非常点 C¹ 非 C² → 改进权重加速收敛
#     4. 无 GPU → Numba CUDA 可选 (CUDA 设备可用时)
# ══════════════════════════════════════════════════════════

from __future__ import annotations

# ── 核心数据结构 ──
from forgecraft.geometry.mesh import (
    HalfEdge,
    PolyMesh,
    TetMesh,
    mesh_from_trimesh,
    mesh_to_trimesh,
    build_tet_mesh,
    validate_manifold,
)

# ── Catmull-Clark 细分 ──
from forgecraft.geometry.catmull_clark import (
    CatmullClark,
    CCResult,
    subdivide,
    subdivide_n,
    verify_sphere,
    verify_torus,
)

# ── 极限曲面求值 ──
from forgecraft.geometry.evaluator import (
    SurfacePoint,
    LimitEvaluator,
    evaluate_bicubic_bspline,
    compute_curvature,
    compute_principal_curvatures,
    bspectrum,
)

# ── 自适应细分 ──
from forgecraft.geometry.adaptive import (
    AdaptiveSubdivider,
    AdaptiveConfig,
    AdaptiveResult,
    planarity_metric,
    curvature_refine_criterion,
    feature_edge_criterion,
)

# ── B-Rep 桥接 ──
from forgecraft.geometry.brep_bridge import (
    BRepBridge,
    BRepToCCResult,
    SurfaceKind,
    brep_to_cc_mesh,
    cc_mesh_to_brep,
)

# ── 自适应镶嵌 ──
from forgecraft.geometry.tessellator import (
    AdaptiveTessellator,
    TessellationConfig,
    tessellate_to_stl,
    tessellate_to_trimesh,
)

# ── 运算子 ──
from forgecraft.geometry.operators import (
    MeshOperator,
    BooleanOp,
    CGALBridge,
    OpResult,
    union_meshes,
    intersect_meshes,
    difference_meshes,
    offset_surface,
    bevel_edges,
    fillet_corners,
)

# ── 参数化零件 ──
from forgecraft.geometry.parametric import (
    ParametricGenerator,
    QUALITY_PRESETS,
    GENERATOR_REGISTRY,
)

# ── Phase 3: PBR 渲染管线 ──
from forgecraft.geometry.materials import (
    PBRMaterial,
    MATERIAL_PRESETS,
    SUB_MATERIALS,
    get_material,
    get_sub_material,
    to_trimesh_material,
    list_presets,
)
from forgecraft.geometry.exploded import (
    compute_exploded_positions,
    create_exploded_body_data,
)
from forgecraft.geometry.gltf_scene import (
    build_gltf_scene,
    export_glb,
    export_gltf,
)
from forgecraft.geometry.pbr_renderer import (
    CameraAngle,
    PRESET_ANGLES,
    PAPER_ANGLES,
    render_assembly,
    render_paper_suite,
)

# ── 管线工具 ──


def unified_pipeline(solid, quality: str = "high"):
    """全流程: cadquery Solid 或 PolyMesh → 统一几何 → STL
    
    Args:
        solid: cadquery Solid 或 PolyMesh (直接使用 CC 控制网格)
        quality: 'low'|'medium'|'high'|'ultra'
    
    Returns:
        dict with 'cc_mesh', 'trimesh', 'edges', 'normals', 'curvatures'
    """
    from forgecraft.geometry.catmull_clark import CatmullClark
    
    quality_map = {
        'low':    {'subdivisions': 1, 'chord_error': 0.1},
        'medium': {'subdivisions': 2, 'chord_error': 0.05},
        'high':   {'subdivisions': 3, 'chord_error': 0.01},
        'ultra':  {'subdivisions': 4, 'chord_error': 0.001},
    }
    
    q = quality_map.get(quality, quality_map['high'])
    
    # 1. 输入 → CC 控制网格
    if isinstance(solid, PolyMesh):
        cc_mesh = solid
    else:
        bridge = BRepBridge()
        cc_mesh = bridge.convert_solid(solid).control_mesh
    
    # 2. Catmull-Clark 细分
    cc = CatmullClark()
    result = cc.subdivide_n(cc_mesh, q['subdivisions'])
    refined = result.mesh
    
    # 3. 自适应镶嵌 → STL
    from forgecraft.geometry.tessellator import AdaptiveTessellator, TessellationConfig
    tess = AdaptiveTessellator(TessellationConfig(chord_error=q['chord_error']))
    tri_mesh = tess.tessellate(refined)
    
    # 4. 曲率分析
    from forgecraft.geometry.evaluator import LimitEvaluator, compute_curvature
    evaluator = LimitEvaluator(max_subdivisions=2)
    curvatures = []
    for f in range(min(refined.n_faces, 10)):
        pt = evaluator.evaluate(refined, f, 0.5, 0.5)
        pt = compute_curvature(pt)
        curvatures.append({
            'face': f,
            'gaussian': pt.gaussian_curvature,
            'mean': pt.mean_curvature,
        })
    
    return {
        'cc_mesh': refined,
        'trimesh': tri_mesh,
        'edges': refined.n_edges,
        'normals': refined.vertex_normals,
        'curvatures': curvatures,
    }


def geometry_summary(poly_mesh: PolyMesh) -> str:
    """生成几何体的详细摘要"""
    lines = [
        f"Geometry Summary",
        f"  Vertices: {poly_mesh.n_vertices}",
        f"  Edges:    {poly_mesh.n_edges}",
        f"  Faces:    {poly_mesh.n_faces}",
        f"  Halfedges: {poly_mesh.n_halfedges}",
    ]
    
    # Manifold 检查
    from forgecraft.geometry.mesh import validate_manifold
    valid, info = validate_manifold(poly_mesh)
    lines.append(f"  Manifold: {'YES' if valid else 'NO'} ({info['non_manifold_edges']} non-manifold edges)")
    
    # 边界
    boundary_count = sum(1 for e_id, he_idx in poly_mesh._edge_to_he.items()
                         if poly_mesh.is_boundary(he_idx))
    lines.append(f"  Boundary edges: {boundary_count}")
    
    # 四边形化程度
    quad_count = sum(1 for f in range(poly_mesh.n_faces)
                     if poly_mesh.face_vertex_count(f) == 4)
    lines.append(f"  Quad faces: {quad_count}/{poly_mesh.n_faces} ({100*quad_count/max(1,poly_mesh.n_faces):.0f}%)")
    
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
#  公开 API
# ══════════════════════════════════════════════════════════

__all__ = [
    # mesh
    "HalfEdge", "PolyMesh", "TetMesh",
    "mesh_from_trimesh", "mesh_to_trimesh",
    "build_tet_mesh", "validate_manifold",
    
    # catmull_clark
    "CatmullClark", "CCResult",
    "subdivide", "subdivide_n",
    "verify_sphere", "verify_torus",
    
    # evaluator
    "SurfacePoint", "LimitEvaluator",
    "evaluate_bicubic_bspline",
    "compute_curvature", "compute_principal_curvatures",
    "bspectrum",
    
    # adaptive
    "AdaptiveSubdivider", "AdaptiveConfig", "AdaptiveResult",
    "planarity_metric", "curvature_refine_criterion", "feature_edge_criterion",
    
    # brep_bridge
    "BRepBridge", "BRepToCCResult", "SurfaceKind",
    "brep_to_cc_mesh", "cc_mesh_to_brep",
    
    # tessellator
    "AdaptiveTessellator", "TessellationConfig",
    "tessellate_to_stl", "tessellate_to_trimesh",
    
    # operators
    "MeshOperator", "BooleanOp", "CGALBridge", "OpResult",
    "union_meshes", "intersect_meshes", "difference_meshes",
    "offset_surface", "bevel_edges", "fillet_corners",
    
    # pipeline
    "unified_pipeline", "geometry_summary",

    # parametric
    "ParametricGenerator", "QUALITY_PRESETS", "GENERATOR_REGISTRY",

    # Phase 3: PBR materials
    "PBRMaterial", "MATERIAL_PRESETS", "SUB_MATERIALS",
    "get_material", "get_sub_material", "to_trimesh_material", "list_presets",

    # Phase 3: exploded
    "compute_exploded_positions", "create_exploded_body_data",

    # Phase 3: glTF
    "build_gltf_scene", "export_glb", "export_gltf",

    # Phase 3: PBR renderer
    "CameraAngle", "PRESET_ANGLES", "PAPER_ANGLES",
    "render_assembly", "render_paper_suite",
]
