# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_post — 后处理
#
#   compute_stress:     应力恢复 (积分点 → 控制点外推)
#   export_vtk:         导出 VTK 非结构网格 (.vtu)
#   compute_von_mises:  von Mises 应力
#   compute_principal:  主应力
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    DOFMap, StressField, StaticResult, IGASettings, IrregularFaceCache,
)

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "compute_stress",
    "compute_von_mises",
    "compute_principal_stresses",
    "export_vtk",
    "export_vtk_displacement",
]


# ══════════════════════════════════════════════════════════
#  应力恢复
# ══════════════════════════════════════════════════════════

def compute_stress(
    mesh: PolyMesh,
    integrator,
    material,
    dof_map: DOFMap,
    u: np.ndarray,
    settings: IGASettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> StressField:
    """应力恢复: 积分点 → 控制点外推
    
    流程:
      1. 在每个积分点计算应变 ε = B u_e
      2. 用本构矩阵计算应力 σ = D ε
      3. 逐面求积分点应力
      4. 简单外推到控制顶点 (面积加权平均)
    
    Args:
        mesh: CC 控制网格
        integrator: ElementIntegrator 实例
        material: 材料模型
        dof_map: DOF 映射
        u: (n_dof,) 位移解
        settings: IGA 设置
        irregular_cache: 非常面缓存
    
    Returns:
        StressField (von Mises, principal, components, positions)
    """
    from forgecraft.analysis.iga_shape import (
        cc_shape_functions, cc_element_control_points, is_face_regular,
    )
    from forgecraft.analysis.iga_quadrature import cc_quadrature
    
    n_faces = mesh.n_faces
    n_vertices = mesh.n_vertices
    
    # 应力收集
    all_stress = []          # (n_quad_total, 6) 各积分点应力
    all_positions = []       # (n_quad_total, 3) 各积分点物理位置
    
    # 顶点应力累加 (面积加权)
    vertex_stress_sum = np.zeros((n_vertices, 6))
    vertex_stress_weight = np.zeros(n_vertices)
    
    D_membrane = _plane_stress_D_numba(material.E, material.nu)
    
    for face_id in range(n_faces):
        verts = mesh.face_vertices(face_id)
        if len(verts) != 4:
            continue
        
        # 构建单元 DOF 列表
        all_verts = list(verts)
        seen = set(verts)
        for v in verts:
            ring = mesh.vertex_ring(v)
            for r in ring:
                if r not in seen and len(all_verts) < 16:
                    all_verts.append(r)
                    seen.add(r)
        while len(all_verts) < 16:
            all_verts.append(all_verts[-1])
        all_verts = all_verts[:16]
        
        # 提取单元位移
        u_e = np.zeros(48)
        for i, vi in enumerate(all_verts):
            dx, dy, dz = dof_map.vertex_dofs(vi)
            u_e[3 * i] = u[dx]
            u_e[3 * i + 1] = u[dy]
            u_e[3 * i + 2] = u[dz]
        
        # 积分点遍历
        qps = cc_quadrature(mesh, face_id, settings.quadrature_order, irregular_cache)
        
        for qp in qps:
            u_param, v_param = qp.u, qp.v
            
            if qp.sub_face_id >= 0 and irregular_cache is not None:
                shape = cc_shape_functions(mesh, face_id, u_param, v_param, irregular_cache)
            else:
                shape = cc_shape_functions(mesh, face_id, u_param, v_param, irregular_cache)
            
            if shape.detJ < 1e-15:
                continue
            
            # 局部坐标系
            J = shape.J
            t1 = J[:, 0].copy()
            t1 /= np.linalg.norm(t1)
            t2 = J[:, 1].copy()
            t2 -= np.dot(t2, t1) * t1
            tn = np.linalg.norm(t2)
            if tn < 1e-15:
                continue
            t2 /= tn
            
            # B 矩阵 (全局应变-位移)
            B_glob = np.zeros((3, 48))
            for node_i in range(16):
                dNdx = shape.dN_dx[node_i]
                dN_dt1 = np.dot(dNdx, t1)
                dN_dt2 = np.dot(dNdx, t2)
                
                i3 = node_i * 3
                B_glob[0, i3 + 0] = dN_dt1 * t1[0]
                B_glob[0, i3 + 1] = dN_dt1 * t1[1]
                B_glob[0, i3 + 2] = dN_dt1 * t1[2]
                
                B_glob[1, i3 + 0] = dN_dt2 * t2[0]
                B_glob[1, i3 + 1] = dN_dt2 * t2[1]
                B_glob[1, i3 + 2] = dN_dt2 * t2[2]
                
                B_glob[2, i3 + 0] = dN_dt1 * t2[0] + dN_dt2 * t1[0]
                B_glob[2, i3 + 1] = dN_dt1 * t2[1] + dN_dt2 * t1[1]
                B_glob[2, i3 + 2] = dN_dt1 * t2[2] + dN_dt2 * t1[2]
            
            # 膜应变
            eps_membrane = B_glob @ u_e  # (3,)
            
            # 膜应力 (3 分量)
            sigma_membrane = D_membrane @ eps_membrane
            
            # 扩展到 6 分量 (仅面内非零)
            # σ = [σ_tt1, σ_tt2, 0, τ_t1t2, 0, 0]
            sigma_6 = np.zeros(6)
            sigma_6[0] = sigma_membrane[0]  # σ_xx (in t1 direction)
            sigma_6[1] = sigma_membrane[1]  # σ_yy (in t2 direction)
            sigma_6[3] = sigma_membrane[2]  # τ_xy (shear)
            
            all_stress.append(sigma_6)
            
            # 物理位置
            from forgecraft.analysis.iga_shape import _uniform_bspline_basis
            control, _ = cc_element_control_points(mesh, face_id)
            if control.shape == (4, 4, 3):
                Nu = _uniform_bspline_basis(u_param, 3)
                Nv = _uniform_bspline_basis(v_param, 3)
                pos = np.zeros(3)
                for i in range(4):
                    for j in range(4):
                        pos += Nu[i] * Nv[j] * control[i, j]
                all_positions.append(pos)
            
            # 面积加权累加到顶点
            area = qp.weight * shape.detJ
            for node_i, vi in enumerate(all_verts):
                vertex_stress_sum[vi] += sigma_6 * area
                vertex_stress_weight[vi] += area
    
    # 顶点应力 (面积加权平均)
    vertex_stress = np.zeros((n_vertices, 6))
    for vi in range(n_vertices):
        if vertex_stress_weight[vi] > 1e-15:
            vertex_stress[vi] = vertex_stress_sum[vi] / vertex_stress_weight[vi]
    
    # von Mises
    all_stress_arr = np.array(all_stress) if all_stress else np.zeros((0, 6))
    von_mises = compute_von_mises(vertex_stress)
    principal = compute_principal_stresses(vertex_stress)
    
    return StressField(
        von_mises=von_mises,
        principal_stresses=principal,
        stress_components=all_stress_arr,
        quad_points_positions=np.array(all_positions) if all_positions else np.zeros((0, 3)),
    )


# ══════════════════════════════════════════════════════════
#  von Mises 应力
# ══════════════════════════════════════════════════════════

def compute_von_mises(stress_6: np.ndarray) -> np.ndarray:
    """计算 von Mises 应力
    
    对于平面应力:
      σ_vm = sqrt(σₓ² + σᵧ² - σₓσᵧ + 3τₓᵧ²)
    
    Args:
        stress_6: (n, 6) or (6,) [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_zx]
    
    Returns:
        (n,) or scalar von Mises stress
    """
    if stress_6.ndim == 1:
        sx, sy, sz, txy, tyz, tzx = stress_6
        return np.sqrt(sx**2 + sy**2 + sz**2 - sx*sy - sy*sz - sz*sx
                       + 3*(txy**2 + tyz**2 + tzx**2))
    
    sx = stress_6[:, 0]
    sy = stress_6[:, 1]
    sz = stress_6[:, 2]
    txy = stress_6[:, 3]
    tyz = stress_6[:, 4]
    tzx = stress_6[:, 5]
    
    return np.sqrt(sx**2 + sy**2 + sz**2 - sx*sy - sy*sz - sz*sx
                   + 3*(txy**2 + tyz**2 + tzx**2))


def compute_principal_stresses(stress_6: np.ndarray) -> np.ndarray:
    """计算主应力
    
    Args:
        stress_6: (n, 6) or (6,) 应力分量
    
    Returns:
        (n, 3) or (3,) 主应力 [σ₁, σ₂, σ₃] 降序
    """
    if stress_6.ndim == 1:
        S = np.array([
            [stress_6[0], stress_6[3], stress_6[5]],
            [stress_6[3], stress_6[1], stress_6[4]],
            [stress_6[5], stress_6[4], stress_6[2]],
        ])
        vals = np.linalg.eigvalsh(S)
        return np.sort(vals)[::-1]
    
    n = stress_6.shape[0]
    result = np.zeros((n, 3))
    
    for i in range(n):
        S = np.array([
            [stress_6[i, 0], stress_6[i, 3], stress_6[i, 5]],
            [stress_6[i, 3], stress_6[i, 1], stress_6[i, 4]],
            [stress_6[i, 5], stress_6[i, 4], stress_6[i, 2]],
        ])
        vals = np.linalg.eigvalsh(S)
        result[i] = np.sort(vals)[::-1]
    
    return result


# ══════════════════════════════════════════════════════════
#  VTK 导出
# ══════════════════════════════════════════════════════════

def export_vtk(
    filename: str,
    mesh: PolyMesh,
    result: StaticResult,
    dof_map: DOFMap,
    stress_field: Optional[StressField] = None,
    displ_scale: float = 1.0,
):
    """导出 VTK 非结构网格 (.vtu) 用于 ParaView 可视化
    
    导出:
      - 原始网格几何
      - 位移场 (DISPLACEMENT)
      - von Mises 应力 (VON_MISES, 可选)
      - 主应力 (PRINCIPAL_1/2/3, 可选)
    
    Args:
        filename: 输出 .vtu 文件路径
        mesh: CC 控制网格
        result: 静力求解结果
        dof_map: DOF 映射
        stress_field: 应力场 (可选)
        displ_scale: 位移缩放因子 (用于变形显示)
    """
    try:
        from evtk.hl import pointsToVTK, gridToVTK
        _use_evtk = True
    except ImportError:
        _use_evtk = False
        _logger.warning("evtk 不可用，使用原始 XML 写入")
    
    n_vertices = mesh.n_vertices
    n_faces = mesh.n_faces
    
    # 提取顶点坐标
    points = mesh.vertices.copy()
    points_flat = points.ravel()
    
    # 位移
    u = result.displacements
    displ = np.zeros((n_vertices, 3))
    for vi in range(n_vertices):
        dx, dy, dz = dof_map.vertex_dofs(vi)
        displ[vi, 0] = u[dx]
        displ[vi, 1] = u[dy]
        displ[vi, 2] = u[dz]
    
    # 变形后的坐标
    points_deformed = points + displ_scale * displ
    
    # 单元 (四边形面 → VTK 四边形单元)
    cells = []
    for f in range(n_faces):
        verts = mesh.face_vertices(f)
        if len(verts) == 4:
            cells.append(list(verts))
    
    if _use_evtk:
        _export_evtk(
            filename, points, displ, stress_field, cells,
        )
    else:
        _export_raw_vtk(
            filename, points, displ, stress_field, cells,
        )


def _export_evtk(filename, points, displ, stress_field, cells):
    """使用 evtk 导出"""
    from evtk.hl import pointsToVTK
    
    n_vertices = points.shape[0]
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    z = points[:, 2].copy()
    
    data = {}
    data["displacement"] = displ.copy()
    
    if stress_field is not None:
        data["von_mises"] = stress_field.von_mises.copy()
        if stress_field.principal_stresses.shape[0] == n_vertices:
            ps = stress_field.principal_stresses
            data["principal_1"] = ps[:, 0].copy()
            data["principal_2"] = ps[:, 1].copy()
            data["principal_3"] = ps[:, 2].copy()
    
    pointsToVTK(filename, x, y, z, data=data)


def _export_raw_vtk(filename, points, displ, stress_field, cells):
    """原始 XML VTU 写入"""
    n_vertices = points.shape[0]
    n_cells = len(cells)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{n_vertices}" NumberOfCells="{n_cells}">\n')
        
        # 点坐标
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for i in range(n_vertices):
            f.write(f'          {points[i, 0]} {points[i, 1]} {points[i, 2]}\n')
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')
        
        # 单元
        f.write('      <Cells>\n')
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        for cell in cells:
            f.write(f'          {" ".join(str(c) for c in cell)}\n')
        f.write('        </DataArray>\n')
        
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        offset = 0
        for cell in cells:
            offset += len(cell)
            f.write(f'          {offset}\n')
        f.write('        </DataArray>\n')
        
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        for _ in cells:
            f.write('          9\n')  # VTK_QUAD = 9
        f.write('        </DataArray>\n')
        f.write('      </Cells>\n')
        
        # 点数据
        f.write('      <PointData>\n')
        
        f.write('        <DataArray type="Float64" Name="displacement" NumberOfComponents="3" format="ascii">\n')
        for i in range(n_vertices):
            f.write(f'          {displ[i, 0]} {displ[i, 1]} {displ[i, 2]}\n')
        f.write('        </DataArray>\n')
        
        if stress_field is not None:
            f.write('        <DataArray type="Float64" Name="von_mises" format="ascii">\n')
            for i in range(n_vertices):
                f.write(f'          {stress_field.von_mises[i]}\n')
            f.write('        </DataArray>\n')
            
            if stress_field.principal_stresses.shape[0] == n_vertices:
                ps = stress_field.principal_stresses
                f.write('        <DataArray type="Float64" Name="principal_1" format="ascii">\n')
                for i in range(n_vertices):
                    f.write(f'          {ps[i, 0]}\n')
                f.write('        </DataArray>\n')
                
                f.write('        <DataArray type="Float64" Name="principal_2" format="ascii">\n')
                for i in range(n_vertices):
                    f.write(f'          {ps[i, 1]}\n')
                f.write('        </DataArray>\n')
                
                f.write('        <DataArray type="Float64" Name="principal_3" format="ascii">\n')
                for i in range(n_vertices):
                    f.write(f'          {ps[i, 2]}\n')
                f.write('        </DataArray>\n')
        
        f.write('      </PointData>\n')
        f.write('    </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')


def export_vtk_displacement(
    filename: str,
    mesh: PolyMesh,
    u: np.ndarray,
    dof_map: DOFMap,
):
    """仅导出行位移场 (轻量级)
    
    Args:
        filename: 输出 .vtu 文件路径
        mesh: CC 控制网格
        u: (n_dof,) 位移向量
        dof_map: DOF 映射
    """
    n_vertices = mesh.n_vertices
    displ = np.zeros((n_vertices, 3))
    
    for vi in range(n_vertices):
        dx, dy, dz = dof_map.vertex_dofs(vi)
        displ[vi, 0] = u[dx]
        displ[vi, 1] = u[dy]
        displ[vi, 2] = u[dz]
    
    try:
        from evtk.hl import pointsToVTK
        
        x = mesh.vertices[:, 0].copy()
        y = mesh.vertices[:, 1].copy()
        z = mesh.vertices[:, 2].copy()
        
        data = {"displacement": displ}
        pointsToVTK(filename, x, y, z, data=data)
    except ImportError:
        _export_raw_vtk(filename, points, displ, None, [])


# ══════════════════════════════════════════════════════════
#  内联平面应力 D 矩阵 (避免循环导入)
# ══════════════════════════════════════════════════════════

def _plane_stress_D_numba(E: float, nu: float) -> np.ndarray:
    """3×3 平面应力本构矩阵 (无 numba 依赖)"""
    D = np.zeros((3, 3))
    factor = E / (1.0 - nu * nu)
    D[0, 0] = 1.0
    D[1, 1] = 1.0
    D[0, 1] = nu
    D[1, 0] = nu
    D[2, 2] = (1.0 - nu) / 2.0
    D *= factor
    return D
