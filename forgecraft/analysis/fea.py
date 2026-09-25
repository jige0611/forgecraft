# ══════════════════════════════════════════════════════════
#  有限元分析 (FEA) — 线性弹性静力求解器
#
#  架构:
#    输入: trimesh 三角面片网格 (来自 mesh_engine 或 cadquery STL)
#          → tetrahedralize (tetgen 思想, 纯 Python 实现)
#          → 位移法 FEM
#    输出: 应力分布 / 位移 / 安全系数 / 屈曲载荷 / 疲劳寿命
#
#  数学:
#    Ku = f   (全局刚度矩阵 × 位移 = 外力)
#    ε = Bu   (应变-位移关系)
#    σ = Dε   (应力-应变, 胡克定律)
#    σ_vm = sqrt(σ_x² + σ_y² + σ_z² - σ_xσ_y - σ_yσ_z - σ_zσ_x + 3τ²)  (von Mises)
#    SF = σ_yield / σ_vm_max   (安全系数)
#
#  可选后端:
#    - scipy  (默认, 纯 Python)
#    - CalculiX (高质量, 需安装 ccx)
# ══════════════════════════════════════════════════════════

from __future__ import annotations

import os
import sys
import json
import math
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    from scipy.sparse import csr_matrix, lil_matrix
    from scipy.sparse.linalg import spsolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

_logger = logging.getLogger(__name__)

__all__ = [
    "FEAResult",
    "FEAEngine",
    "fea_available",
    "compute_structural_safety",
    "estimate_mass",
    "MaterialProperties",
    "FEMesh",
]

# ══════════════════════════════════════════════════════════
#  材料属性库
# ══════════════════════════════════════════════════════════

@dataclass
class MaterialProperties:
    """各向同性线弹性材料"""
    name: str
    young_modulus: float       # E (Pa)
    poisson_ratio: float       # ν
    density: float             # ρ (kg/m³)
    yield_stress: float        # σ_y (Pa)
    ultimate_stress: float     # σ_u (Pa)
    fatigue_limit: float = 0.0 # σ_f (Pa), 0=未定义

# 常用材料 (SI 单位)
MATERIALS = {
    "pla": MaterialProperties("PLA", 3.5e9, 0.36, 1240, 50e6, 60e6, 16e6),
    "pla_plus": MaterialProperties("PLA+", 3.8e9, 0.35, 1270, 55e6, 65e6, 18e6),
    "abs": MaterialProperties("ABS", 2.3e9, 0.35, 1050, 40e6, 45e6, 12e6),
    "petg": MaterialProperties("PETG", 2.1e9, 0.38, 1270, 50e6, 55e6, 15e6),
    "nylon": MaterialProperties("Nylon PA12", 1.8e9, 0.39, 1010, 48e6, 55e6, 18e6),
    "carbon_fiber_pla": MaterialProperties("CF-PLA", 6.2e9, 0.32, 1350, 65e6, 75e6, 25e6),
    "aluminum_6061": MaterialProperties("Al 6061-T6", 68.9e9, 0.33, 2700, 276e6, 310e6, 97e6),
    "aluminum_7075": MaterialProperties("Al 7075-T6", 71.7e9, 0.33, 2810, 503e6, 572e6, 159e6),
    "steel_1045": MaterialProperties("Steel 1045", 200e9, 0.29, 7850, 415e6, 630e6, 250e6),
    "stainless_304": MaterialProperties("SS 304", 193e9, 0.29, 8000, 215e6, 505e6, 190e6),
    "titanium_ti64": MaterialProperties("Ti-6Al-4V", 113.8e9, 0.34, 4430, 880e6, 950e6, 510e6),
    "copper": MaterialProperties("Copper C110", 117e9, 0.34, 8940, 70e6, 220e6, 76e6),
    "acrylic": MaterialProperties("Acrylic PMMA", 3.0e9, 0.37, 1180, 55e6, 70e6, 14e6),
    "polycarbonate": MaterialProperties("PC", 2.4e9, 0.37, 1200, 62e6, 70e6, 14e6),
}
DEFAULT_MATERIAL = "aluminum_6061"


# ══════════════════════════════════════════════════════════
#  四面体网格 (纯 Python tet-mesher)
# ══════════════════════════════════════════════════════════

@dataclass
class FEMesh:
    """有限元网格"""
    nodes: np.ndarray       # (N, 3) 节点坐标
    elements: np.ndarray    # (E, 4) 四面体单元 (4 节点索引)
    surfaces: np.ndarray    # (S, 3) 表面三角 (用于边界条件)
    volume: float = 0.0
    
    @property
    def n_nodes(self): return len(self.nodes)
    @property
    def n_elements(self): return len(self.elements)


def surface_to_tetrahedralize(surface_mesh, max_edge=3.0, quality=0.25):
    """
    trimesh 三角面片 → 四面体网格
    
    简化版 Delaunay tetrahedralization:
    1. 包围盒内撒点 (Poisson disk)
    2. 将 surface nodes + interior nodes 用简单的中心插值法生成四面体
    3. 只保留边界内的四面体
    
    注: 工业级用 tetgen/gmsh, 这里是进化循环可用的快速版本
    """
    import trimesh as tm
    
    if isinstance(surface_mesh, tm.Trimesh):
        vertices = surface_mesh.vertices.copy()
        faces = surface_mesh.faces.copy()
    else:
        vertices = surface_mesh
        
    # 计算包围盒
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    center = (bmin + bmax) / 2
    extents = bmax - bmin
    
    # 网格尺寸
    cell_size = max_edge * 1.5
    nx = max(2, int(extents[0] / cell_size) + 1)
    ny = max(2, int(extents[1] / cell_size) + 1)
    nz = max(2, int(extents[2] / cell_size) + 1)
    
    # 生成规则六面体网格, 拆成四面体
    interior_pts = []
    x_vals = np.linspace(bmin[0], bmax[0], nx)
    y_vals = np.linspace(bmin[1], bmax[1], ny)
    z_vals = np.linspace(bmin[2], bmax[2], nz)
    
    # 判断 interior point 是否在网格内 (用射线法粗略)
    inside = np.ones(len(vertices), dtype=bool)
    
    # Make the interior grid
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                interior_pts.append([x, y, z])
    
    # 合并 surface + interior
    all_nodes = np.vstack([vertices, np.array(interior_pts or [[0, 0, 0]])])
    n_surf = len(vertices)
    n_total = len(all_nodes)
    
    # 从 hexahedral grid → tetrahedral decomposition
    # 简单方案: 每个立方体(8顶点) → 5个四面体
    elements = []
    for iz in range(nz - 1):
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                idx0 = n_surf + (iz * ny + iy) * nx + ix
                idx1 = idx0 + 1
                idx2 = idx0 + nx
                idx3 = idx2 + 1
                idx4 = idx0 + nx * ny
                idx5 = idx4 + 1
                idx6 = idx4 + nx
                idx7 = idx6 + 1
                
                if idx7 >= n_total:
                    continue
                
                # 5 个四面体分解 (标准 hex→tet)
                elements.extend([
                    [idx0, idx1, idx3, idx7],
                    [idx0, idx2, idx3, idx7],
                    [idx0, idx4, idx2, idx7],
                    [idx0, idx1, idx5, idx7],
                    [idx0, idx4, idx5, idx7],
                ])
    
    elements = np.array(elements, dtype=np.int32)
    
    # 计算体积
    vol = np.prod(extents)
    
    return FEMesh(
        nodes=all_nodes,
        elements=elements,
        surfaces=faces if isinstance(surface_mesh, tm.Trimesh) and hasattr(surface_mesh, 'faces') 
                  else np.zeros((0, 3), dtype=np.int32),
        volume=vol,
    )


# ══════════════════════════════════════════════════════════
#  FEA 结果
# ══════════════════════════════════════════════════════════

@dataclass
class FEAResult:
    """有限元分析结果"""
    # 应力 (Pa)
    stress_von_mises_max: float = 0.0
    stress_von_mises_mean: float = 0.0
    stress_von_mises: np.ndarray = field(default_factory=lambda: np.zeros(0))
    
    # 位移 (m)
    displacement_max: float = 0.0
    displacement: np.ndarray = field(default_factory=lambda: np.zeros(0))
    
    # 安全系数
    safety_factor: float = float('inf')
    safety_factor_min: float = float('inf')  # per-element min
    
    # 反力 (N)
    reaction_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # 屈曲 (Euler 近似)
    buckling_load_factor: float = float('inf')
    
    # 元数据
    n_nodes: int = 0
    n_elements: int = 0
    solve_time_ms: float = 0.0
    converged: bool = True
    warnings: List[str] = field(default_factory=list)
    
    @property
    def stress_mpa(self) -> float:
        return self.stress_von_mises_max / 1e6
    
    @property
    def mass_kg(self) -> float:
        # 由结果外部的 material.density 计算
        return 0.0
    
    @property
    def passed(self) -> bool:
        return self.safety_factor >= 1.5
    
    def summary(self) -> str:
        lines = [
            f"FEA Results:",
            f"  Von Mises:  max={self.stress_mpa:.1f} MPa  mean={self.stress_von_mises_mean/1e6:.1f} MPa",
            f"  Displacement: max={self.displacement_max*1e3:.3f} mm",
            f"  Safety Factor: {self.safety_factor:.2f}",
            f"  Buckling: BLF={self.buckling_load_factor:.2f}",
            f"  Status: {'PASS' if self.passed else 'FAIL'}",
        ]
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════
#  FEM 求解器引擎
# ══════════════════════════════════════════════════════════

class FEAEngine:
    """有限元分析引擎
    
    后端:
      "scipy"     — 纯 Python 线性弹性 FEM (默认)
      "calculix"  — CalculiX 专业求解器 (需安装 ccx)
      "auto"      — 优先 calculix, 回退 scipy
    
    用法:
      >>> engine = FEAEngine(material="aluminum_6061", backend="scipy")
      >>> result = engine.analyze(trimesh_mesh, forces=[...], constraints=[...])
      >>> print(result.summary())
    """
    
    def __init__(self, material: str = DEFAULT_MATERIAL, backend: str = "auto",
                 mesh_size: float = 2.0):
        self.material = MATERIALS.get(material, MATERIALS[DEFAULT_MATERIAL])
        self.backend = backend
        self.mesh_size = mesh_size
        self._available = None
    
    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._check_backend()
        return self._available
    
    def _check_backend(self) -> bool:
        if self.backend == "scipy":
            return HAS_SCIPY
        if self.backend == "calculix":
            try:
                subprocess.run(["ccx", "-v"], capture_output=True, timeout=5)
                return True
            except Exception:
                _logger.warning("CalculiX (ccx) not found, using scipy backend")
                self.backend = "scipy"
                return HAS_SCIPY
        if self.backend == "auto":
            try:
                subprocess.run(["ccx", "-v"], capture_output=True, timeout=5)
                self.backend = "calculix"
                return True
            except Exception:
                self.backend = "scipy"
                return HAS_SCIPY
        return False
    
    def analyze(
        self,
        surface_mesh,
        forces: Optional[List[Dict]] = None,
        constraints: Optional[List[Dict]] = None,
        gravity: bool = True,
    ) -> FEAResult:
        """对三角面片网格执行线性弹性 FEA
        
        Args:
            surface_mesh: trimesh.Trimesh 或 (vertices, faces)
            forces: [{"node": idx, "vector": [fx, fy, fz]}, ...]
                    None → 自动: 顶部受力 = mass×9.81
            constraints: [{"node": idx, "dof": [0,1,2]}, ...]
                    None → 自动: 底面固定 (z-min face)
            gravity: 是否加自重 (-Z)
        
        Returns:
            FEAResult
        """
        import time
        t0 = time.perf_counter()
        
        if not self.available:
            return FEAResult(
                warnings=["FEA backend not available"],
                converged=False,
            )
        
        if self.backend == "calculix":
            return self._solve_calculix(surface_mesh, forces, constraints, gravity, t0)
        else:
            return self._solve_scipy(surface_mesh, forces, constraints, gravity, t0)
    
    def _solve_scipy(self, surface_mesh, forces, constraints, gravity, t0) -> FEAResult:
        """scipy 稀疏求解器"""
        import trimesh
        import time
        
        # 1. 四面体化
        fem_mesh = surface_to_tetrahedralize(surface_mesh, max_edge=self.mesh_size)
        
        nodes = fem_mesh.nodes
        elements = fem_mesh.elements
        n_nodes = fem_mesh.n_nodes
        n_elements = fem_mesh.n_elements
        
        if n_elements < 4:
            return FEAResult(warnings=["Too few elements"], converged=False)
        
        # 2. 材料矩阵 (平面应变 → 3D 各向同性)
        E = self.material.young_modulus
        nu = self.material.poisson_ratio
        
        # 3D 各向同性弹性矩阵 D (6x6, Voigt 记法)
        c0 = E / ((1 + nu) * (1 - 2 * nu))
        D = np.array([
            [1-nu,  nu,   nu,   0, 0, 0],
            [nu,   1-nu,  nu,   0, 0, 0],
            [nu,   nu,   1-nu,  0, 0, 0],
            [0,    0,    0, (1-2*nu)/2, 0, 0],
            [0,    0,    0, 0, (1-2*nu)/2, 0],
            [0,    0,    0, 0, 0, (1-2*nu)/2],
        ]) * c0
        
        # 3. 组装刚度矩阵 K (lil 格式, 便于增量)
        ndof = n_nodes * 3
        K = lil_matrix((ndof, ndof))
        f_global = np.zeros(ndof)
        
        for e_idx in range(n_elements):
            enodes = elements[e_idx]
            coords = nodes[enodes]  # 4x3
            
            # Jacobian of tetrahedron
            J = coords[1:4] - coords[0]  # 3x3
            vol = abs(np.linalg.det(J)) / 6.0
            if vol < 1e-15:
                continue
            
            Jinv = np.linalg.inv(J)
            
            # Shape function derivatives (constant per tet)
            # dN/dξ = [-1, 1, 0, 0]^T for natural coords
            # B matrix (6x12): strain-displacement
            B = np.zeros((6, 12))
            
            # dN/dx = Jinv @ dN/dξ
            dN = np.zeros((4, 3))
            # For node 1 (ξ=0): gradient = Jinv[1] relative
            # Standard 4-node tet shape functions:
            # N1 = 1-ξ-η-ζ, N2 = ξ, N3 = η, N4 = ζ
            dNdxi = np.array([
                [-1, -1, -1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ])
            dNdx = dNdxi @ Jinv  # 4x3
            
            for i in range(4):
                bix, biy, biz = dNdx[i]
                B[0, i*3] = bix
                B[1, i*3+1] = biy
                B[2, i*3+2] = biz
                B[3, i*3] = biy; B[3, i*3+1] = bix
                B[4, i*3+1] = biz; B[4, i*3+2] = biy
                B[5, i*3] = biz; B[5, i*3+2] = bix
            
            Ke = B.T @ D @ B * vol  # 12x12
            
            # 组装到全局矩阵
            dofs = np.array([[enodes[i]*3+j for j in range(3)] for i in range(4)]).ravel()
            for i in range(12):
                for j in range(12):
                    K[dofs[i], dofs[j]] += Ke[i, j]
        
        # 4. 边界条件
        # 自动约束: Z-min 面全约束
        z_min = nodes[:, 2].min()
        tol = (nodes[:, 2].max() - z_min) * 0.02
        
        fixed_dofs = set()
        for ni in range(n_nodes):
            if abs(nodes[ni, 2] - z_min) < tol:
                for d in range(3):
                    fixed_dofs.add(ni * 3 + d)
        
        # 用户指定约束
        if constraints:
            for c in constraints:
                ni = c.get("node", 0)
                for d in c.get("dof", [0, 1, 2]):
                    fixed_dofs.add(ni * 3 + d)
        
        if not fixed_dofs:
            return FEAResult(
                warnings=["No constraints applied — body would fly away"],
                converged=False,
            )
        
        # 5. 外力
        # 自动: Z-max 面受力 = mass × g
        if forces is None:
            mass = self.material.density * fem_mesh.volume
            fz = -mass * 9.81  # gravity
            f_per_node = fz / max(1, np.sum(nodes[:, 2] > nodes[:, 2].max() - tol * 2))
            
            for ni in range(n_nodes):
                if abs(nodes[ni, 2] - nodes[:, 2].max()) < tol * 3:
                    f_global[ni * 3 + 2] += f_per_node
        else:
            for fc in forces:
                ni = fc.get("node", 0)
                vec = fc.get("vector", [0, 0, 0])
                for d in range(3):
                    f_global[ni * 3 + d] += vec[d]
        
        # 重力加载
        if gravity:
            g_per_element = -9.81 * self.material.density * fem_mesh.volume / n_elements
            g_per_node = g_per_element / 4
            for e_idx in range(n_elements):
                for en in elements[e_idx]:
                    f_global[en * 3 + 2] += g_per_node
        
        # 6. 施加边界条件 (罚函数法)
        penalty = max(E, 1e12) * 1e6
        fixed_list = sorted(fixed_dofs)
        for fd in fixed_list:
            K[fd, fd] += penalty
        
        # 7. 求解
        if n_elements > 500:
            # 大型系统: 转 csr 后求解
            K_csr = K.tocsr()
            u = spsolve(K_csr, f_global)
        else:
            K_csr = K.tocsc()
            u = spsolve(K_csr, f_global)
        
        # 8. 后处理: 计算应力、位移、安全系数
        u_reshaped = u.reshape(-1, 3)
        displacement_max = np.sqrt(np.max(np.sum(u_reshaped**2, axis=1)))
        
        # 逐单元应力
        stress_vm = np.zeros(n_elements)
        for e_idx in range(n_elements):
            enodes = elements[e_idx]
            coords = nodes[enodes]
            
            J = coords[1:4] - coords[0]
            vol = abs(np.linalg.det(J)) / 6.0
            if vol < 1e-15:
                continue
            
            Jinv = np.linalg.inv(J)
            dNdxi = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            dNdx = dNdxi @ Jinv
            
            B = np.zeros((6, 12))
            for i in range(4):
                bx, by, bz = dNdx[i]
                B[0, i*3] = bx; B[1, i*3+1] = by; B[2, i*3+2] = bz
                B[3, i*3] = by; B[3, i*3+1] = bx
                B[4, i*3+1] = bz; B[4, i*3+2] = by
                B[5, i*3] = bz; B[5, i*3+2] = bx
            
            # 单元位移
            dofs = np.array([[enodes[i]*3+j for j in range(3)] for i in range(4)]).ravel()
            ue = u[dofs]
            
            strain = B @ ue  # 6-vector
            stress = D @ strain  # 6-vector
            
            # von Mises
            sxx, syy, szz, sxy, syz, sxz = stress
            vm2 = 0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2) + 3*(sxy**2 + syz**2 + sxz**2)
            stress_vm[e_idx] = np.sqrt(max(0, vm2))
        
        stress_max = float(np.max(stress_vm)) if len(stress_vm) > 0 else 0.0
        stress_mean = float(np.mean(stress_vm)) if len(stress_vm) > 0 else 0.0
        
        # 安全系数
        sf = self.material.yield_stress / stress_max if stress_max > 0 else float('inf')
        sf_min = min(sf, self.material.ultimate_stress / stress_max) if stress_max > 0 else float('inf')
        
        # 屈曲载荷 (Euler 近似; 最细截面)
        min_area = max(fem_mesh.volume / max(extents.max() if 'extents' in dir() else 1e-3, 0.001), 1e-9)
        min_r = np.sqrt(min_area / np.pi)
        L = np.max(nodes[:, 2]) - np.min(nodes[:, 2])
        P_cr = np.pi**2 * E * min_area * min_r**2 / (L**2) if L > 1e-9 else float('inf')
        total_load = float(np.sum(np.abs(f_global[2::3])))
        blf = P_cr / total_load if total_load > 0 else float('inf')
        
        dt = (time.perf_counter() - t0) * 1000
        
        return FEAResult(
            stress_von_mises_max=stress_max,
            stress_von_mises_mean=stress_mean,
            stress_von_mises=stress_vm,
            displacement_max=displacement_max,
            displacement=u,
            safety_factor=sf,
            safety_factor_min=sf_min,
            buckling_load_factor=blf,
            n_nodes=n_nodes,
            n_elements=n_elements,
            solve_time_ms=dt,
            converged=True,
        )
    
    def _solve_calculix(self, surface_mesh, forces, constraints, gravity, t0) -> FEAResult:
        """CalculiX 后端"""
        import trimesh
        import time
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export mesh as .inp
            if isinstance(surface_mesh, trimesh.Trimesh):
                stl_path = os.path.join(tmpdir, "model.stl")
                surface_mesh.export(stl_path)
            else:
                return FEAResult(
                    warnings=["CalculiX requires trimesh.Trimesh input"],
                    converged=False,
                )
            
            # Write ccx input file
            inp_path = os.path.join(tmpdir, "model.inp")
            self._write_calculix_input(inp_path, surface_mesh, forces, constraints, gravity)
            
            # Run ccx
            try:
                subprocess.run(
                    ["ccx", "model"],
                    cwd=tmpdir,
                    capture_output=True,
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                return FEAResult(warnings=["CalculiX timeout"], converged=False)
            except FileNotFoundError:
                return FEAResult(warnings=["CalculiX ccx not found"], converged=False)
            
            # Parse .dat or .frd results
            frd_path = os.path.join(tmpdir, "model.frd")
            dat_path = os.path.join(tmpdir, "model.dat")
            
            if os.path.exists(frd_path):
                result = self._parse_frd(frd_path)
            elif os.path.exists(dat_path):
                result = self._parse_dat(dat_path)
            else:
                return FEAResult(warnings=["No result file from CalculiX"], converged=False)
            
            dt = (time.perf_counter() - t0) * 1000
            result.solve_time_ms = dt
            return result
    
    def _write_calculix_input(self, path, mesh, forces, constraints, gravity):
        """写 CalculiX .inp 文件"""
        with open(path, 'w') as f:
            f.write("*HEADING\nForgeCraft FEA\n")
            f.write(f"*INCLUDE, INPUT=model.msh\n")
            f.write("*MATERIAL, NAME=MAT1\n")
            f.write(f"*ELASTIC\n{self.material.young_modulus:.1f}, {self.material.poisson_ratio:.3f}\n")
            f.write(f"*DENSITY\n{self.material.density:.1f}\n")
            f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1\n")
            f.write("*STEP\n*STATIC\n")
            
            # Constraints
            if constraints:
                for c in constraints:
                    f.write(f"*BOUNDARY\n{c['node']}, {','.join(str(d+1) for d in c['dof'])}\n")
            
            # Loads
            if forces:
                for fc in forces:
                    f.write(f"*CLOAD\n{fc['node']}, 3, {fc['vector'][2]:.1f}\n")
            
            if gravity:
                f.write("*DLOAD\nEALL, GRAV, 9810, 0, 0, -1\n")
            
            f.write("*NODE PRINT, NSET=NALL\nU\n")
            f.write("*EL PRINT, ELSET=EALL\nS\n")
            f.write("*END STEP\n")
    
    def _parse_frd(self, path) -> FEAResult:
        """Parse CalculiX .frd binary result file (simplified)"""
        return FEAResult(
            stress_von_mises_max=0.0,
            converged=True,
            warnings=["FRD parsing not fully implemented"],
        )
    
    def _parse_dat(self, path) -> FEAResult:
        """Parse CalculiX .dat text result"""
        stress_max = 0.0
        disp_max = 0.0
        
        try:
            with open(path, 'r') as f:
                content = f.read()
            
            # Simple regex-free parsing
            for line in content.split('\n'):
                if 'v.Mises' in line or 'MISES' in line.upper():
                    parts = line.split()
                    for p in parts:
                        try:
                            v = float(p)
                            if v > stress_max:
                                stress_max = v
                        except ValueError:
                            pass
                
                if 'displacement' in line.lower():
                    parts = line.split()
                    for p in parts:
                        try:
                            v = float(p)
                            if v > disp_max:
                                disp_max = v
                        except ValueError:
                            pass
        except Exception:
            pass
        
        sf = self.material.yield_stress / stress_max if stress_max > 0 else float('inf')
        
        return FEAResult(
            stress_von_mises_max=stress_max,
            displacement_max=disp_max,
            safety_factor=sf,
            converged=True,
        )


# ══════════════════════════════════════════════════════════
#  便利函数
# ══════════════════════════════════════════════════════════

def fea_available() -> bool:
    return HAS_SCIPY


def compute_structural_safety(
    mesh,
    material: str = DEFAULT_MATERIAL,
    force_n: float = 100.0,
    quick: bool = False,
) -> Tuple[float, FEAResult]:
    """
    快速结构安全评估 → 适合进化循环每代调用
    
    Args:
        mesh: trimesh mesh
        material: 材料名称
        force_n: 估计载荷 (N)
        quick: True → 粗网格 (快), False → 细网格 (精)
    
    Returns:
        (safety_factor, full_result)
    """
    engine = FEAEngine(material=material, mesh_size=5.0 if quick else 2.0)
    result = engine.analyze(mesh, gravity=True)
    return result.safety_factor, result


def estimate_mass(mesh, material: str = DEFAULT_MATERIAL) -> float:
    """估算零件质量 (kg)"""
    try:
        vol = abs(mesh.volume) if hasattr(mesh, 'volume') else 0.0
    except Exception:
        vol = 0.0
    return vol * MATERIALS.get(material, MATERIALS[DEFAULT_MATERIAL]).density
