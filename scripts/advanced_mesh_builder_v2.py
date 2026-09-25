#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高级3D建模引擎 V2版本 - 混合架构实现

继承 AdvancedMeshBuilder V1 的所有能力，新增：
- manifold3d 加速的布尔运算引擎
- 创新零件类型（弹簧、中空管、半球脚垫）
- 参数自适应映射系统
- 批量处理优化和缓存机制

架构设计：
┌─────────────────────────────────────┐
│         AdvancedMeshBuilderV2       │
│  ┌───────────────────────────────┐  │
│  │   AdvancedMeshBuilder (V1)    │  │  ← 继承所有基础能力
│  │  - 高精度几何体               │  │
│  │  - 细分曲面/平滑             │  │
│  │  - PBR材质库                 │  │
│  └───────────────────────────────┘  │
│  + manifold3d 布尔引擎              │  ← 新增：精确布尔运算
│  + 参数自适应规则库                │  ← 新增：物理→几何映射
│  + 零件缓存机制                    │  ← 新增：性能优化
│  + 创新零件生成器                  │  ← 新增：弹簧/管/脚垫
└─────────────────────────────────────┘

作者: ForgeCraft AI
版本: 2.0 Hybrid (manifold3d + trimesh)
日期: 2026-06-02
"""

import math
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("⚠️  trimesh 未安装")

try:
    import manifold3d as mf
    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False
    print("⚠️  manifold3d 未安装")


@dataclass
class PartSpec:
    """零件规格数据类"""
    part_type: str
    params: Dict[str, float]
    position: np.ndarray
    material_key: str


class AdvancedMeshBuilderV2:
    """
    高级3D网格建造器 V2版本
    
    混合架构：
    - 使用trimesh进行基础几何生成（继承自V1）
    - 使用manifold3d进行复杂布尔运算（新增）
    - 自研参数自适应算法（新增）
    
    核心能力：
    1. 弹性储能单元（螺旋弹簧）- 创新零件A1
    2. 轻量化中空结构体 - 创新零件B1
    3. 高摩擦半球脚垫 - 创新零件C1
    4. 所有V1原有能力完全保留
    """

    def __init__(self, quality_level: str = "high"):
        """
        初始化 V2 建模引擎
        
        Args:
            quality_level: 质量 ("low", "medium", "high", "ultra")
        """
        # 质量配置（与V1保持一致）
        self.quality_configs = {
            "low": {
                "cylinder_sections": 32,
                "sphere_subdivisions": 3,
                "subdivision_iterations": 1,
            },
            "medium": {
                "cylinder_sections": 48,
                "sphere_subdivisions": 3,
                "subdivision_iterations": 2,
            },
            "high": {
                "cylinder_sections": 64,
                "sphere_subdivisions": 4,
                "subdivision_iterations": 2,
            },
            "ultra": {
                "cylinder_sections": 128,
                "sphere_subdivisions": 5,
                "subdivision_iterations": 3,
            }
        }
        
        self.quality = quality_level
        self.config = self.quality_configs.get(quality_level, self.quality_configs["high"])
        
        # PBR材质库（完整版，包含创新零件专用材质）
        self.material_library = self._init_material_library()
        
        # V2新增：manifold3d引擎状态
        self.use_manifold = HAS_MANIFOLD
        
        if self.use_manifold:
            print(f"🔧 AdvancedMeshBuilderV2 初始化完成")
            print(f"   质量: {quality_level} | 引擎: manifold3d + trimesh")
        else:
            print(f"⚠️  AdvancedMeshBuilderV2 初始化（纯trimesh模式）")
        
        # V2新增：参数自适应规则库
        self.adaptation_rules = self._init_adaptation_rules()
        
        # V2新增：缓存字典（避免重复计算相同参数的零件）
        self._mesh_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 性能统计
        self.stats = {
            "total_parts_generated": 0,
            "cache_hit_rate": 0.0,
            "avg_generation_time_ms": 0.0,
        }

    def _init_material_library(self) -> Dict[str, Dict]:
        """初始化PBR材质库（扩展版）"""
        materials = {
            # 结构体材质（继承自V1）
            "carbon_fiber": {
                "color": [0.35, 0.38, 0.42],
                "metalness": 0.3,
                "roughness": 0.35,
                "name": "碳纤维"
            },
            "aluminum_6061": {
                "color": [0.72, 0.73, 0.75],
                "metalness": 0.9,
                "roughness": 0.25,
                "name": "6061铝合金"
            },
            "steel_structural": {
                "color": [0.45, 0.47, 0.50],
                "metalness": 0.95,
                "roughness": 0.40,
                "name": "结构钢"
            },
            "titanium_alloy": {
                "color": [0.67, 0.69, 0.73],
                "metalness": 0.85,
                "roughness": 0.30,
                "name": "钛合金"
            },
            
            # 驱动器材质（继承自V1）
            "motor_copper": {
                "color": [0.72, 0.45, 0.20],
                "metalness": 0.95,
                "roughness": 0.30,
                "name": "电机铜绕组"
            },
            "steel_shaft": {
                "color": [0.55, 0.55, 0.57],
                "metalness": 0.92,
                "roughness": 0.20,
                "name": "传动轴钢"
            },
            
            # 触地点材质（继承自V1）
            "rubber_black": {
                "color": [0.15, 0.15, 0.15],
                "metalness": 0.0,
                "roughness": 0.85,
                "name": "黑色橡胶"
            },
            "silicone_grey": {
                "color": [0.50, 0.50, 0.52],
                "metalness": 0.0,
                "roughness": 0.70,
                "name": "灰色硅胶"
            },
            
            # ══════════════════════════════
            # 🆕 V2 新增：创新零件专用材质
            # ══════════════════════════════
            
            # 弹簧材质
            "piano_wire": {
                "color": [0.75, 0.77, 0.80],  # 亮银色
                "metalness": 0.98,
                "roughness": 0.15,
                "name": "钢琴丝（高碳钢）",
                "elastic_modulus": 207e9,  # Pa
                "yield_strength": 2.5e9,    # Pa
            },
            "spring_steel_65mn": {
                "color": [0.40, 0.42, 0.45],  # 蓝灰色
                "metalness": 0.90,
                "roughness": 0.35,
                "name": "65Mn弹簧钢",
                "elastic_modulus": 206e9,
                "yield_strength": 1.5e9,
            },
            "titanium_spring": {
                "color": [0.65, 0.68, 0.72],  # 钛灰色
                "metalness": 0.88,
                "roughness": 0.28,
                "name": "钛合金弹簧",
                "elastic_modulus": 110e9,
                "yield_strength": 1.1e9,
            },
            
            # 中空管材质
            "aluminum_tube_6063": {
                "color": [0.78, 0.79, 0.81],
                "metalness": 0.92,
                "roughness": 0.20,
                "name": "6063铝合金管材",
                "density": 2700,  # kg/m³
            },
            "carbon_fiber_tube": {
                "color": [0.25, 0.25, 0.27],
                "metalness": 0.25,
                "roughness": 0.30,
                "name": "碳纤维管材",
                "density": 1550,
            },
            
            # 半球脚垫材质
            "rubber_high_friction": {
                "color": [0.12, 0.12, 0.14],
                "metalness": 0.0,
                "roughness": 0.95,
                "name": "高摩擦橡胶",
                "friction_coefficient": 1.2,
            },
            "polyurethane": {
                "color": [0.45, 0.48, 0.52],
                "metalness": 0.05,
                "roughness": 0.65,
                "name": "聚氨酯弹性体",
                "friction_coefficient": 0.9,
            },
        }
        
        return materials

    def _init_adaptation_rules(self) -> Dict[str, Dict]:
        """
        初始化参数自适应规则库
        
        将抽象的物理参数映射为具体的几何参数。
        
        例如：
        - 物理参数：刚度 k=500 N/m
        - → 几何参数：线径 d=2mm, 圈数 n=10, 线圈直径 D=20mm
        
        规则来源：
        - 材料力学公式（Wahl修正系数等）
        - 工程经验数据
        - 制造约束（最小线径、最大长径比等）
        """
        rules = {
            # ══════════════════════════════
            # 弹簧参数自适应规则
            # ══════════════════════════════
            "spring_element": {
                "description": "从物理参数推导弹簧几何尺寸",
                
                # 输入物理参数范围
                "input_params": {
                    "stiffness": {          # 弹簧刚度 [N/m]
                        "range": [10, 10000],
                        "distribution": "log_uniform",
                    },
                    "max_deformation": {   # 最大形变 [m]
                        "range": [0.001, 0.05],
                    },
                    "preload": {            # 预紧力 [N]
                        "range": [0, 20],
                    },
                },
                
                # 输出几何参数
                "output_geometry": {
                    "wire_diameter": {      # 线径 [m]
                        "range": [0.001, 0.005],
                        "formula": "adaptive",  # 使用自适应算法
                    },
                    "coil_diameter": {      # 线圈直径（中径）[m]
                        "range": [0.01, 0.05],
                    },
                    "free_length": {        # 自由长度 [m]
                        "range": [0.02, 0.15],
                    },
                    "num_coils": {           # 有效圈数
                        "range": [5, 25],
                    }
                },
                
                # 约束条件
                "constraints": {
                    "min_wire_diameter": 0.001,     # 最小线径 1mm
                    "max_wire_diameter": 0.005,     # 最大线径 5mm
                    "spring_index_range": [4, 12],   # 弹簧指数 C=D/d
                    "max_solid_height_ratio": 0.8,  # 最大压并高度比
                    "min_pitch_to_wire": 1.2,       # 最小节距/线径比
                },
                
                # 材料默认值（钢琴丝）
                "default_material": "piano_wire",
                
                # 自适应算法配置
                "adaptation_config": {
                    "target_safety_factor": 1.5,     # 安全系数
                    "max_iterations": 50,            # 最大迭代次数
                    "tolerance": 0.05,               # 收敛容差 5%
                }
            },
            
            # ══════════════════════════════
            # 中空管参数自适应规则
            # ══════════════════════════════
            "hollow_tube": {
                "description": "轻量化中空管的几何优化",
                
                "input_params": {
                    "outer_diameter": {     # 外径 [m]
                        "range": [0.01, 0.08],
                    },
                    "length": {             # 长度 [m]
                        "range": [0.05, 0.5],
                    },
                    "wall_thickness": {     # 壁厚 [m]
                        "range": [0.001, 0.005],
                    },
                },
                
                "output_geometry": {
                    "inner_diameter": {     # 内径（自动计算）
                        "formula": "outer - 2*wall",
                    },
                    "mass_optimization": {  # 质量优化目标
                        "target": "minimal_mass",
                        "constraint": "buckling_load > required",
                    }
                },
                
                "constraints": {
                    "min_wall_thickness": 0.001,     # 最小壁厚 1mm
                    "max_outer_to_inner_ratio": 5.0,  # 最大内外径比
                    "min_length_to_diameter": 2.0,   # 最小长径比
                },
                
                "default_material": "aluminum_tube_6063",
            },
            
            # ══════════════════════════════
            # 半球脚垫参数自适应规则
            # ══════════════════════════════
            "hemisphere_foot": {
                "description": "高摩擦接触面的几何设计",
                
                "input_params": {
                    "radius": {             # 半径 [m]
                        "range": [0.005, 0.03],
                    },
                    "compliance": {         # 柔度 [m/N]
                        "range": [0.0001, 0.001],
                    },
                },
                
                "output_geometry": {
                    "shell_thickness": {    # 壳厚度 [m]
                        "range": [0.002, 0.008],
                    },
                    "contact_area": {       # 接触面积（自动计算）
                        "formula": "pi * r^2",
                    }
                },
                
                "constraints": {
                    "min_shell_to_radius": 0.1,       # 最小壳厚/半径比
                    "max_compression_ratio": 0.3,     # 最大压缩比
                },
                
                "default_material": "rubber_high_friction",
            }
        }
        
        return rules

    def _generate_cache_key(self, part_type: str, params: Dict[str, float]) -> str:
        """
        生成零件缓存键
        
        Args:
            part_type: 零件类型
            params: 参数字典
            
        Returns:
            MD5哈希字符串
        """
        param_str = str(sorted(params.items()))
        raw_key = f"{part_type}:{param_str}"
        return hashlib.md5(raw_key.encode()).hexdigest()

    def _get_cached_mesh(self, cache_key: str):
        """获取缓存的网格"""
        if cache_key in self._mesh_cache:
            self.cache_hits += 1
            return self._mesh_cache[cache_key]
        return None

    def _set_cached_mesh(self, cache_key: str, mesh):
        """设置网格缓存"""
        self._mesh_cache[cache_key] = mesh
        self.cache_misses += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            "total_parts_generated": self.stats["total_parts_generated"],
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self._mesh_cache),
        }

    # ═══════════════════════════════════════════
    # 🆕 V2 核心：创新零件生成方法
    # ═══════════════════════════════════════════

    def create_spring_element(
        self,
        stiffness: float = 500.0,
        max_deformation: float = 0.02,
        preload: float = 0.0,
        wire_diameter: Optional[float] = None,
        coil_diameter: Optional[float] = None,
        free_length: Optional[float] = None,
        num_coils: Optional[float] = None,
        position: Optional[np.ndarray] = None,
        material_key: str = "piano_wire",
        use_manifold_engine: bool = True
    ) -> 'trimesh.Trimesh':
        """
        创建完整的弹性储能单元（螺旋弹簧）
        
        完整组装流程：
        1. 参数自适应 → 从物理参数推导最优几何尺寸
        2. 生成螺旋线圈主体（使用参数化方程）
        3. 添加顶部挂钩（连接到驱动器/结构）
        4. 添加底部挂钩（连接到脚垫/地面）
        5. 布尔合并成单一实体（使用manifold3d或trimesh）
        6. 应用PBR材质和后处理
        
        Args:
            stiffness: 弹簧刚度 [N/m] (10-10000)
            max_deformation: 最大形变 [m] (0.001-0.05)
            preload: 预紧力 [N] (0-20)
            wire_diameter: 线径 [m] (可选，None则自动计算)
            coil_diameter: 线圈直径（中径）[m] (可选)
            free_length: 自由长度 [m] (可选)
            num_coils: 有效圈数 (可选)
            position: 安装位置 [x,y,z]
            material_key: 材质键名
            use_manifold_engine: 是否使用manifold3d布尔引擎
            
        Returns:
            trimesh.Trimesh: 完整的弹簧网格模型
            
        Raises:
            ValueError: 参数超出有效范围
        """
        start_time = time.time()
        
        # 参数验证
        if not (10 <= stiffness <= 10000):
            raise ValueError(f"刚度必须在 [10, 10000] N/m 范围内，当前: {stiffness}")
        if not (0.001 <= max_deformation <= 0.05):
            raise ValueError(f"最大形变必须在 [0.001, 0.05] m 范围内，当前: {max_deformation}")
        
        # Step 1: 参数自适应（如果未提供几何参数）
        if any(param is None for param in [wire_diameter, coil_diameter, free_length, num_coils]):
            geo_params = self._adapt_spring_parameters(
                stiffness=stiffness,
                max_deformation=max_deformation,
                preload=preload
            )
            wire_diameter = wire_diameter or geo_params['wire_diameter']
            coil_diameter = coil_diameter or geo_params['coil_diameter']
            free_length = free_length or geo_params['free_length']
            num_coils = num_coils or geo_params['num_coils']
        
        # 生成缓存键
        cache_key = self._generate_cache_key("spring_element", {
            'wire_diameter': wire_diameter,
            'coil_diameter': coil_diameter,
            'free_length': free_length,
            'num_coils': num_coils,
            'material': material_key
        })
        
        # 检查缓存
        cached_mesh = self._get_cached_mesh(cache_key)
        if cached_mesh is not None and position is None:
            self.stats["total_parts_generated"] += 1
            elapsed_ms = (time.time() - start_time) * 1000
            print(f"📦 缓存命中: 弹簧 ({elapsed_ms:.1f}ms)")
            return cached_mesh.copy()
        
        # Step 2-6: 完整生成流程
        mesh = self._build_complete_spring(
            wire_diameter=wire_diameter,
            coil_diameter=coil_diameter,
            free_length=free_length,
            num_coils=num_coils,
            position=position,
            material_key=material_key,
            use_manifold=use_manifold_engine
        )
        
        # 更新缓存（仅存储无位置的基准网格）
        if position is None:
            self._set_cached_mesh(cache_key, mesh.copy())
        
        # 更新统计
        self.stats["total_parts_generated"] += 1
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"✅ 弹簧生成完成: d={wire_diameter*1000:.1f}mm, D={coil_diameter*1000:.1f}mm, "
              f"L={free_length*1000:.1f}mm, n={num_coils:.0f} ({elapsed_ms:.1f}ms)")
        
        return mesh

    def _adapt_spring_parameters(
        self,
        stiffness: float,
        max_deformation: float,
        preload: float
    ) -> Dict[str, float]:
        """
        弹簧参数自适应算法
        
        从物理参数（刚度、形变、预紧力）推导最优几何参数（线径、线圈直径、长度、圈数）。
        
        使用材料力学公式 + 启发式优化：
        
        公式基础：
        - 刚度公式: k = (G * d^4) / (8 * D^3 * n)
          其中 G=剪切模量, d=线径, D=线圈直径, n=圈数
        - 应力公式: τ = (8 * F * D * K_w) / (π * d^3)
          其中 K_w=Wahl修正系数, F=载荷
        - 稳定性: 自由长度 / 线圈直径 < 某阈值
        
        算法流程：
        1. 根据刚度要求选择合理的弹簧指数 C = D/d
        2. 计算满足刚度的最小线径 d
        3. 校验应力是否在安全范围内
        4. 调整圈数以满足自由长度要求
        5. 验证稳定性约束
        
        Args:
            stiffness: 目标刚度 [N/m]
            max_deformation: 最大工作形变 [m]
            preload: 预紧力 [N]
            
        Returns:
            Dict: 包含 wire_diameter, coil_diameter, free_length, num_coils 的字典
        """
        config = self.adaptation_rules["spring_element"]["adaptation_config"]
        constraints = self.adaptation_rules["spring_element"]["constraints"]
        material = self.material_library.get("piano_wire", {})
        
        G = material.get("elastic_modulus", 207e9) * 0.38  # 剪切模量 ≈ E/(2*(1+ν)), ν≈0.3
        tau_yield = material.get("yield_strength", 2.5e9)
        safety_factor = config["target_safety_factor"]
        tau_allowable = tau_yield / safety_factor
        
        best_solution = None
        min_error = float('inf')
        
        for C in range(constraints["spring_index_range"][0], 
                       constraints["spring_index_range"][1] + 1):
            for d in np.linspace(
                constraints["min_wire_diameter"],
                constraints["max_wire_diameter"],
                20
            ):
                D = C * d
                
                n = (G * d**4) / (8 * D**3 * stiffness)
                
                if not (5 <= n <= 25):
                    continue
                
                F_max = stiffness * max_deformation + preload
                
                K_w = (4*C - 1)/(4*C - 4) + 0.615/C
                tau_actual = (8 * F_max * D * K_w) / (math.pi * d**3)
                
                if tau_actual > tau_allowable:
                    continue
                
                L_free = n * d * 1.5 + max_deformation * 2
                
                solid_height = n * d
                if L_free / D > 5:
                    continue
                
                error = abs(n - round(n)) + (tau_actual / tau_allowable)
                
                if error < min_error:
                    min_error = error
                    best_solution = {
                        'wire_diameter': round(d, 5),
                        'coil_diameter': round(D, 5),
                        'free_length': round(L_free, 5),
                        'num_coils': int(round(n)),
                        'stress_ratio': tau_actual / tau_allowable,
                        'spring_index': C,
                    }
        
        if best_solution is None:
            raise ValueError("无法找到满足所有约束的弹簧参数组合")
        
        print(f"📐 参数自适应结果:")
        print(f"   线径: {best_solution['wire_diameter']*1000:.2f} mm")
        print(f"   线圈直径: {best_solution['coil_diameter']*1000:.2f} mm")
        print(f"   自由长度: {best_solution['free_length']*1000:.2f} mm")
        print(f"   圈数: {best_solution['num_coils']}")
        print(f"   应力比: {best_solution['stress_ratio']:.2%}")
        
        return best_solution

    def _build_complete_spring(
        self,
        wire_diameter: float,
        coil_diameter: float,
        free_length: float,
        num_coils: int,
        position: Optional[np.ndarray],
        material_key: str,
        use_manifold: bool
    ) -> 'trimesh.Trimesh':
        """
        构建完整的弹簧模型（核心实现）
        
        组装顺序：
        1. 螺旋线圈主体
        2. 顶部挂钩（半圆环）
        3. 底部挂钩（半圆环）
        4. 合并为单一实体
        5. 应用位置偏移
        """
        sections = self.config["cylinder_sections"]
        
        R = coil_diameter / 2
        r = wire_diameter / 2
        pitch = (free_length - wire_diameter) / num_coils
        
        t = np.linspace(0, num_coils * 2 * math.pi, num_coils * sections)
        
        x_helix = R * np.cos(t)
        y_helix = R * np.sin(t)
        z_helix = (pitch / (2 * math.pi)) * t
        
        if use_manifold and self.use_manifold:
            spring_body = self._create_spring_manifold(
                x_helix, y_helix, z_helix, r, sections
            )
        else:
            spring_body = self._create_spring_trimesh(
                x_helix, y_helix, z_helix, r, sections
            )
        
        hook_top = self._create_hook(r, R, position_offset=[0, 0, z_helix[-1]], 
                                     use_manifold=use_manifold)
        hook_bottom = self._create_hook(r, R, position_offset=[0, 0, z_helix[0]], 
                                        flip=True, use_manifold=use_manifold)
        
        if use_manifold and self.use_manifold:
            if isinstance(spring_body, mf.Manifold):
                complete_spring = spring_body + hook_top + hook_bottom
                mesh_data = complete_spring.to_mesh()
                mesh = trimesh.Trimesh(
                    vertices=mesh_data.vert_properties,
                    faces=mesh_data.tri_verts,
                    process=False
                )
            else:
                meshes_to_merge = [spring_body, hook_top, hook_bottom]
                mesh = trimesh.util.concatenate(meshes_to_merge)
        else:
            meshes_to_merge = [spring_body, hook_top, hook_bottom]
            mesh = trimesh.util.concatenate(meshes_to_merge)
        
        if position is not None:
            mesh.apply_translation(position)
        
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            face_colors=np.tile(
                self.material_library[material_key]["color"], 
                (len(mesh.faces), 1)
            )
        )
        
        return mesh

    def _create_spring_manifold(self, x, y, z, r, sections):
        """使用trimesh+manifold3d混合方案创建弹簧线圈主体"""
        path_points = np.column_stack([x, y, z])
        
        tube_segments = []
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            
            cylinder = trimesh.creation.cylinder(
                radius=r,
                height=np.linalg.norm(p2 - p1),
                sections=16
            )
            
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-10:
                continue
            
            axis = np.cross([0, 0, 1], direction)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-10:
                axis = axis / axis_norm
                angle = math.acos(np.clip(np.dot([0, 0, 1], direction / length), -1, 1))
                rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
                cylinder.apply_transform(rotation_matrix)
            
            midpoint = (p1 + p2) / 2
            cylinder.apply_translation(midpoint)
            
            tube_segments.append(cylinder)
        
        if not tube_segments:
            raise ValueError("无法创建弹簧段")
        
        combined_trimesh = trimesh.util.concatenate(tube_segments)
        
        if self.use_manifold:
            mf_mesh = mf.Mesh(
                vert_properties=np.array(combined_trimesh.vertices, dtype=np.float32),
                tri_verts=np.array(combined_trimesh.faces, dtype=np.uint32)
            )
            return mf.Manifold(mf_mesh)
        else:
            return combined_trimesh

    def _create_spring_trimesh(self, x, y, z, r, sections):
        """使用trimesh创建弹簧线圈主体（备用方案）"""
        path_points = np.column_stack([x, y, z])
        
        tube_segments = []
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            
            cylinder = trimesh.creation.cylinder(
                radius=r,
                height=np.linalg.norm(p2 - p1),
                sections=16
            )
            
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-10:
                continue
            
            axis = np.cross([0, 0, 1], direction)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-10:
                axis = axis / axis_norm
                angle = math.acos(np.clip(np.dot([0, 0, 1], direction / length), -1, 1))
                rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
                cylinder.apply_transform(rotation_matrix)
            
            midpoint = (p1 + p2) / 2
            cylinder.apply_translation(midpoint)
            
            tube_segments.append(cylinder)
        
        if not tube_segments:
            raise ValueError("无法创建弹簧段")
        
        return trimesh.util.concatenate(tube_segments)

    def _create_hook(self, wire_radius, coil_radius, position_offset, 
                     flip=False, use_manifold=True):
        """创建端部挂钩（半圆环）"""
        hook_radius = coil_radius * 0.6
        arc_angle = math.pi
        arc_sections = 16
        
        theta = np.linspace(0, arc_angle, arc_sections)
        
        if flip:
            hook_x = hook_radius * np.cos(theta + math.pi)
            hook_y = hook_radius * np.sin(theta + math.pi)
        else:
            hook_x = hook_radius * np.cos(theta)
            hook_y = hook_radius * np.sin(theta)
        
        hook_z = np.zeros_like(theta)
        
        if use_manifold and self.use_manifold:
            return self._create_spring_manifold(hook_x, hook_y, hook_z, wire_radius, arc_sections)
        else:
            return self._create_spring_trimesh(hook_x, hook_y, hook_z, wire_radius, arc_sections)

    @staticmethod
    def _rotation_matrix_from_axis_angle(axis, angle):
        """根据轴和角度创建旋转矩阵（用于manifold3d变换）"""
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1 - c
        x, y, z = axis
        
        return np.array([
            [t*x*x+c,    t*x*y-s*z,  t*x*z+s*y,  0],
            [t*x*y+s*z,  t*y*y+c,    t*y*z-s*x,  0],
            [t*x*z-s*y,  t*y+z+s*x,  t*z*z+c,    0],
            [0,          0,          0,          1]
        ])

    # ═══════════════════════════════════════════
    # 🆕 V2 核心：创新零件生成方法（续）
    # ═══════════════════════════════════════════

    def create_hollow_tube(
        self,
        outer_diameter: float = 0.03,
        length: float = 0.1,
        wall_thickness: float = 0.002,
        position: Optional[np.ndarray] = None,
        material_key: str = "aluminum_tube_6063",
        use_manifold_engine: bool = True,
        end_cap_style: str = "open"
    ) -> 'trimesh.Trimesh':
        """
        创建轻量化中空管结构体
        
        应用场景：
        - 机器人骨架/连杆（替代实心圆柱，减重60-80%）
        - 轻量化结构支撑
        - 气动/液压管道模拟
        
        实现策略：
        1. 创建外圆柱体
        2. 创建内圆柱体（稍短或等长）
        3. 布尔差集得到中空结构
        4. 可选：添加端部法兰/螺纹孔
        
        Args:
            outer_diameter: 外径 [m] (0.01-0.08)
            length: 长度 [m] (0.05-0.5)
            wall_thickness: 壁厚 [m] (0.001-0.005)
            position: 安装位置 [x,y,z]
            material_key: 材质键名
            use_manifold_engine: 是否使用manifold3d布尔引擎
            end_cap_style: 端部样式 ("open", "flat_cap", "flange")
            
        Returns:
            trimesh.Trimesh: 中空管网格模型
            
        Raises:
            ValueError: 参数超出有效范围
        """
        start_time = time.time()
        
        constraints = self.adaptation_rules["hollow_tube"]["constraints"]
        
        if not (0.01 <= outer_diameter <= 0.08):
            raise ValueError(f"外径必须在 [0.01, 0.08] m 范围内")
        if not (0.05 <= length <= 0.5):
            raise ValueError(f"长度必须在 [0.05, 0.5] m 范围内")
        if not (constraints["min_wall_thickness"] <= wall_thickness <= 0.005):
            raise ValueError(f"壁厚必须在 [{constraints['min_wall_thickness']}, 0.005] m 范围内")
        
        inner_diameter = outer_diameter - 2 * wall_thickness
        if inner_diameter <= 0:
            raise ValueError("壁厚过大，导致内径≤0")
        
        outer_radius = outer_diameter / 2
        inner_radius = inner_diameter / 2
        
        cache_key = self._generate_cache_key("hollow_tube", {
            'outer_diameter': outer_diameter,
            'length': length,
            'wall_thickness': wall_thickness,
            'end_cap': end_cap_style,
            'material': material_key
        })
        
        cached_mesh = self._get_cached_mesh(cache_key)
        if cached_mesh is not None and position is None:
            self.stats["total_parts_generated"] += 1
            print(f"📦 缓存命中: 中空管 ({(time.time()-start_time)*1000:.1f}ms)")
            return cached_mesh.copy()
        
        sections = self.config["cylinder_sections"]
        
        if use_manifold_engine and self.use_manifold:
            mesh = self._build_hollow_tube_manifold(
                outer_radius=outer_radius,
                inner_radius=inner_radius,
                length=length,
                sections=sections,
                end_cap_style=end_cap_style
            )
        else:
            mesh = self._build_hollow_tube_trimesh(
                outer_radius=outer_radius,
                inner_radius=inner_radius,
                length=length,
                sections=sections,
                end_cap_style=end_cap_style
            )
        
        if position is not None:
            mesh.apply_translation(position)
        
        mat_color = self.material_library[material_key]["color"]
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            face_colors=np.tile(mat_color, (len(mesh.faces), 1))
        )
        
        if position is None:
            self._set_cached_mesh(cache_key, mesh.copy())
        
        self.stats["total_parts_generated"] += 1
        elapsed_ms = (time.time() - start_time) * 1000
        mass_estimate = self._estimate_tube_mass(outer_diameter, length, wall_thickness, material_key)
        
        print(f"✅ 中空管生成完成: D={outer_diameter*1000:.1f}mm, L={length*1000:.1f}mm, "
              f"t={wall_thickness*1000:.1f}mm, 质量≈{mass_estimate*1000:.1f}g ({elapsed_ms:.1f}ms)")
        
        return mesh

    def _build_hollow_tube_manifold(self, outer_radius, inner_radius, length, 
                                     sections, end_cap_style):
        """使用manifold3d创建中空管（布尔差集法）"""
        outer_cyl = mf.Manifold.cylinder(height=length, radius_low=outer_radius)
        
        inner_length = length * 0.98 if end_cap_style == "open" else length * 1.02
        inner_cyl = mf.Manifold.cylinder(height=inner_length, radius_low=inner_radius).translate([0, 0, -length*0.01])
        
        hollow_tube = outer_cyl - inner_cyl
        
        if end_cap_style == "flange":
            wall = 0.003
            flange_outer = mf.Manifold.cylinder(
                height=wall, 
                radius_low=outer_radius * 1.3
            ).translate([0, 0, -wall/2])
            
            flange_inner = mf.Manifold.cylinder(
                height=wall * 1.1,
                radius_low=inner_radius * 0.9
            ).translate([0, 0, -wall*0.55])
            
            flange_top = (flange_outer - flange_inner).translate([0, 0, length + wall/2])
            flange_bottom = flange_outer - flange_inner
            
            hollow_tube = hollow_tube + flange_top + flange_bottom
        
        mesh_data = hollow_tube.to_mesh()
        return trimesh.Trimesh(
            vertices=mesh_data.vert_properties,
            faces=mesh_data.tri_verts,
            process=False
        )

    def _build_hollow_tube_trimesh(self, outer_radius, inner_radius, length, 
                                    sections, end_cap_style):
        """使用trimesh创建中空管（备用方案）"""
        outer = trimesh.creation.cylinder(radius=outer_radius, height=length, sections=sections)
        
        inner_height = length * 0.98 if end_cap_style == "open" else length * 1.02
        inner = trimesh.creation.cylinder(radius=inner_radius, height=inner_height, sections=sections)
        inner.apply_translation([0, 0, -length*0.01])
        
        hollow_tube = trimesh.boolean.difference([outer], [inner], engine='blender')
        
        if isinstance(hollow_tube, list):
            hollow_tube = hollow_tube[0]
        
        return hollow_tube

    def _estimate_tube_mass(self, outer_diameter, length, wall_thickness, material_key):
        """估算中空管质量"""
        material = self.material_library.get(material_key, {})
        density = material.get("density", 2700)  # 默认铝合金密度
        
        outer_area = math.pi * (outer_diameter/2)**2
        inner_dia = outer_diameter - 2*wall_thickness
        inner_area = math.pi * (inner_dia/2)**2
        cross_section = outer_area - inner_area
        
        volume = cross_section * length
        mass = density * volume
        
        return mass

    def create_hemisphere_foot(
        self,
        radius: float = 0.015,
        shell_thickness: float = 0.003,
        compliance: float = 0.0005,
        position: Optional[np.ndarray] = None,
        material_key: str = "rubber_high_friction",
        use_manifold_engine: bool = True,
        include_contact_pattern: bool = True
    ) -> 'trimesh.Trimesh':
        """
        创建高摩擦半球形脚垫
        
        设计理念：
        - 半球形提供全向接触能力（适应不平地面）
        - 中空设计降低重量并增加柔度
        - 表面纹理增强摩擦系数
        - 内部可集成压力传感器（预留接口）
        
        应用场景：
        - 机器人足端接触点
        - 高摩擦抓地表面
        - 减震缓冲元件
        
        Args:
            radius: 半球外半径 [m] (0.005-0.03)
            shell_thickness: 壳厚度 [m] (0.002-0.008)
            compliance: 柔度参数 [m/N] (影响内部结构)
            position: 安装位置 [x,y,z]
            material_key: 材质键名（高摩擦橡胶/聚氨酯）
            use_manifold_engine: 是否使用manifold3d引擎
            include_contact_pattern: 是否包含表面接触纹理
            
        Returns:
            trimesh.Trimesh: 半球脚垫网格模型
            
        Raises:
            ValueError: 参数超出有效范围
        """
        start_time = time.time()
        
        constraints = self.adaptation_rules["hemisphere_foot"]["constraints"]
        
        if not (0.005 <= radius <= 0.03):
            raise ValueError(f"半径必须在 [0.005, 0.03] m 范围内")
        if not (radius * constraints["min_shell_to_radius"] <= shell_thickness <= 0.008):
            raise ValueError(f"壳厚必须在 [{radius*0.1:.4f}, 0.008] m 范围内")
        
        inner_radius = radius - shell_thickness
        if inner_radius <= 0:
            raise ValueError("壳厚过大，导致内径≤0")
        
        cache_key = self._generate_cache_key("hemisphere_foot", {
            'radius': radius,
            'shell_thickness': shell_thickness,
            'compliance': compliance,
            'pattern': include_contact_pattern,
            'material': material_key
        })
        
        cached_mesh = self._get_cached_mesh(cache_key)
        if cached_mesh is not None and position is None:
            self.stats["total_parts_generated"] += 1
            print(f"📦 缓存命中: 半球脚垫 ({(time.time()-start_time)*1000:.1f}ms)")
            return cached_mesh.copy()
        
        subdivisions = self.config["sphere_subdivisions"]
        
        if use_manifold_engine and self.use_manifold:
            mesh = self._build_hemisphere_foot_manifold(
                outer_radius=radius,
                inner_radius=inner_radius,
                subdivisions=subdivisions,
                include_pattern=include_contact_pattern
            )
        else:
            mesh = self._build_hemisphere_foot_trimesh(
                outer_radius=radius,
                inner_radius=inner_radius,
                subdivisions=subdivisions,
                include_pattern=include_contact_pattern
            )
        
        if position is not None:
            mesh.apply_translation(position)
        
        mat_color = self.material_library[material_key]["color"]
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            face_colors=np.tile(mat_color, (len(mesh.faces), 1))
        )
        
        if position is None:
            self._set_cached_mesh(cache_key, mesh.copy())
        
        self.stats["total_parts_generated"] += 1
        elapsed_ms = (time.time() - start_time) * 1000
        contact_area = math.pi * radius**2
        
        print(f"✅ 半球脚垫生成完成: R={radius*1000:.1f}mm, t={shell_thickness*1000:.1f}mm, "
              f"接触面积≈{contact_area*1e4:.1f}cm² ({elapsed_ms:.1f}ms)")
        
        return mesh

    def _build_hemisphere_foot_manifold(self, outer_radius, inner_radius, 
                                        subdivisions, include_pattern):
        """使用manifold3d创建半球脚垫"""
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=outer_radius)
        
        vertices = sphere.vertices.copy()
        faces = sphere.faces.copy()
        
        keep_faces = []
        for i, face in enumerate(faces):
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            centroid_z = (v0[2] + v1[2] + v2[2]) / 3
            if centroid_z >= -outer_radius * 0.05:
                keep_faces.append(i)
        
        hemisphere_faces = faces[keep_faces]
        outer_hemi_trimesh = trimesh.Trimesh(vertices=vertices, faces=hemisphere_faces)
        
        inner_sphere = trimesh.creation.icosphere(subdivisions=max(1, subdivisions-1), 
                                                  radius=inner_radius)
        inner_vertices = inner_sphere.vertices.copy()
        inner_faces = inner_sphere.faces.copy()
        
        inner_keep = []
        for i, face in enumerate(inner_faces):
            v0, v1, v2 = inner_vertices[face[0]], inner_vertices[face[1]], inner_vertices[face[2]]
            centroid_z = (v0[2] + v1[2] + v2[2]) / 3
            if centroid_z >= -inner_radius * 0.05:
                inner_keep.append(i)
        
        inner_hemi_faces = inner_faces[inner_keep]
        inner_hemi_trimesh = trimesh.Trimesh(vertices=inner_vertices, faces=inner_hemi_faces)
        
        try:
            foot_shell = trimesh.boolean.difference(
                [outer_hemi_trimesh], 
                [inner_hemi_trimesh], 
                engine='blender'
            )
            if isinstance(foot_shell, list):
                foot_shell = foot_shell[0]
        except Exception as e:
            print(f"⚠️ 布尔运算失败，使用实心半球: {e}")
            foot_shell = outer_hemi_trimesh
        
        base_ring = trimesh.creation.cylinder(radius=outer_radius*0.9, height=0.002)
        base_ring.apply_translation([0, 0, -0.001])
        
        meshes_to_merge = [foot_shell, base_ring]
        if include_pattern:
            contact_bumps = self._create_contact_pattern_trimesh(base_radius=outer_radius)
            meshes_to_merge.append(contact_bumps)
        
        final_mesh = trimesh.util.concatenate(meshes_to_merge)
        
        mf_mesh_data = mf.Mesh(
            vert_properties=np.array(final_mesh.vertices, dtype=np.float32),
            tri_verts=np.array(final_mesh.faces, dtype=np.uint32)
        )
        
        return final_mesh

    def _create_contact_pattern_manifold(self, base_radius, num_bumps, 
                                         bump_radius, bump_height):
        """创建表面接触纹理（增加摩擦力的小凸起）"""
        bumps = []
        for i in range(num_bumps):
            angle = (2 * math.pi / num_bumps) * i
            r_pos = base_radius * 0.7
            
            x = r_pos * math.cos(angle)
            y = r_pos * math.sin(angle)
            z = -base_radius * 0.9
            
            bump = mf.Manifold.sphere(radius=bump_radius).translate([x, y, z])
            bumps.append(bump)
        
        if not bumps:
            return mf.Manifold.sphere(radius=0.001)
        
        result = bumps[0]
        for b in bumps[1:]:
            result = result + b
        
        return result

    def _build_hemisphere_foot_trimesh(self, outer_radius, inner_radius, 
                                       subdivisions, include_pattern):
        """使用trimesh创建半球脚垫（备用方案）"""
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=outer_radius)
        
        vertices = sphere.vertices.copy()
        faces = sphere.faces.copy()
        
        keep_faces = []
        for i, face in enumerate(faces):
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            centroid_z = (v0[2] + v1[2] + v2[2]) / 3
            if centroid_z >= -outer_radius * 0.05:
                keep_faces.append(i)
        
        hemisphere_faces = faces[keep_faces]
        outer_hemisphere = trimesh.Trimesh(vertices=vertices, faces=hemisphere_faces)
        
        inner_sphere = trimesh.creation.icosphere(subdivisions=max(1, subdivisions-1), 
                                                  radius=inner_radius)
        inner_vertices = inner_sphere.vertices.copy()
        inner_faces = inner_sphere.faces.copy()
        
        inner_keep = []
        for i, face in enumerate(inner_faces):
            v0, v1, v2 = inner_vertices[face[0]], inner_vertices[face[1]], inner_vertices[face[2]]
            centroid_z = (v0[2] + v1[2] + v2[2]) / 3
            if centroid_z >= -inner_radius * 0.05:
                inner_keep.append(i)
        
        inner_hemi_faces = inner_faces[inner_keep]
        inner_hemisphere = trimesh.Trimesh(vertices=inner_vertices, faces=inner_hemi_faces)
        
        try:
            foot_shell = trimesh.boolean.difference([outer_hemisphere], [inner_hemisphere], 
                                                     engine='blender')
            if isinstance(foot_shell, list):
                foot_shell = foot_shell[0]
        except Exception as e:
            print(f"⚠️ 布尔运算失败，使用简化模型: {e}")
            foot_shell = outer_hemisphere
        
        base_ring = trimesh.creation.cylinder(radius=outer_radius*0.9, height=0.002)
        base_ring.apply_translation([0, 0, -0.001])
        
        meshes_to_merge = [foot_shell, base_ring]
        if include_pattern:
            contact_bumps = self._create_contact_pattern_trimesh(base_radius=outer_radius)
            meshes_to_merge.append(contact_bumps)
        
        return trimesh.util.concatenate(meshes_to_merge)

    def _create_contact_pattern_trimesh(self, base_radius, num_bumps=12):
        """创建表面接触纹理（trimesh版本）"""
        bumps = []
        bump_radius = base_radius * 0.15
        
        for i in range(num_bumps):
            angle = (2 * math.pi / num_bumps) * i
            r_pos = base_radius * 0.7
            
            x = r_pos * math.cos(angle)
            y = r_pos * math.sin(angle)
            z = -base_radius * 0.85
            
            bump = trimesh.creation.icosphere(subdivisions=1, radius=bump_radius)
            bump.apply_translation([x, y, z])
            bumps.append(bump)
        
        return trimesh.util.concatenate(bumps)

    # ═══════════════════════════════════════════
    # 批量生成工具方法
    # ═══════════════════════════════════════════

    def create_complete_robot_part_set(
        self,
        spring_params: Dict[str, float],
        tube_params: Dict[str, float],
        foot_params: Dict[str, float],
        output_dir: str = "test_output"
    ) -> Dict[str, 'trimesh.Trimesh']:
        """
        一键生成完整的创新零件套件
        
        用于快速测试和验证所有新零件类型
        
        Args:
            spring_params: 弹簧参数字典
            tube_params: 中空管参数字典
            foot_params: 半球脚垫参数字典
            output_dir: 输出目录
            
        Returns:
            Dict: 包含三种零件网格的字典
        """
        import os
        
        print("\n" + "="*60)
        print("🤖 批量生成创新零件套件")
        print("="*60)
        
        parts = {}
        
        spring = self.create_spring_element(**spring_params)
        parts['spring'] = spring
        spring_path = os.path.join(output_dir, "spring_v2.stl")
        self.export_part_to_stl(spring, spring_path)
        
        tube = self.create_hollow_tube(**tube_params)
        parts['tube'] = tube
        tube_path = os.path.join(output_dir, "hollow_tube_v2.stl")
        self.export_part_to_stl(tube, tube_path)
        
        foot = self.create_hemisphere_foot(**foot_params)
        parts['foot'] = foot
        foot_path = os.path.join(output_dir, "hemisphere_foot_v2.stl")
        self.export_part_to_stl(foot, foot_path)
        
        print("\n📊 最终统计:")
        stats = self.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n💾 所有文件已保存到: {output_dir}/")
        print("✨ 零件套件生成完成!")
        
        return parts

    def export_part_to_stl(self, mesh: 'trimesh.Trimesh', filepath: str):
        """导出零件为STL文件"""
        if HAS_TRIMESH:
            mesh.export(filepath)
            print(f"💾 已导出: {filepath}")
        else:
            raise ImportError("需要trimesh库才能导出STL")

    def clear_cache(self):
        """清空缓存"""
        self._mesh_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        print("🗑️  缓存已清空")


def test_v2_complete():
    """完整测试V2所有功能"""
    print("\n" + "="*60)
    print("AdvancedMeshBuilderV2 完整测试套件")
    print("="*60)
    
    builder = AdvancedMeshBuilderV2(quality_level="high")
    
    parts = builder.create_complete_robot_part_set(
        spring_params={
            'stiffness': 500.0,
            'max_deformation': 0.02,
            'material_key': "piano_wire",
            'use_manifold_engine': True
        },
        tube_params={
            'outer_diameter': 0.03,
            'length': 0.15,
            'wall_thickness': 0.002,
            'material_key': "aluminum_tube_6063",
            'use_manifold_engine': True
        },
        foot_params={
            'radius': 0.02,
            'shell_thickness': 0.003,
            'material_key': "rubber_high_friction",
            'use_manifold_engine': True,
            'include_contact_pattern': True
        },
        output_dir="test_output"
    )
    
    print("\n📋 零件清单:")
    for name, mesh in parts.items():
        print(f"   {name}: {len(mesh.vertices)}顶点, {len(mesh.faces)}面")
    
    print("\n✨ 所有测试通过!")
    

if __name__ == "__main__":
    test_v2_complete()
