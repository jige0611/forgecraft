"""零件本体 & 物理启发式规则 & 策略模板

预置知识, 在知识库冷启动时提供最小可行推理能力。

本体内容:
  1. PART_ONTOLOGY — 零件功能分类 & 组合约束
  2. PHYSICS_HEURISTICS — 工程设计启发式规则 (~30条)
  3. STRATEGY_TEMPLATES — 进化策略模板 (按任务类型)
"""

from typing import Dict, List, Optional

# ══════════════════════════════════════════════════════════════
#  1. 零件本体 (Part Ontology)
# ══════════════════════════════════════════════════════════════

PART_ONTOLOGY: Dict = {
    "functional_groups": {
        "structural": {
            "description": "提供结构支撑的零件",
            "parts": [
                "base", "base_plate", "segment", "arm_segment",
                "chassis", "limb_segment", "hollow_tube",
                "body", "frame", "beam", "pillar", "plate",
            ],
            "typical_properties": {
                "high_mass": True,
                "high_stiffness": True,
                "actuated": False,
            },
        },
        "actuation": {
            "description": "产生运动和力的驱动零件",
            "parts": [
                "motor", "rotary_joint", "servo", "wrist",
                "hinge", "actuator", "dc_motor", "stepper",
            ],
            "typical_properties": {
                "high_torque": True,
                "actuated": True,
                "joint_type": "hinge",
            },
        },
        "locomotion": {
            "description": "与环境接触产生位移的零件",
            "parts": [
                "wheel", "foot", "hemisphere_foot", "leg",
                "crawler", "track", "propeller", "fin",
            ],
            "typical_properties": {
                "high_friction": True,
                "ground_contact": True,
                "actuated": False,
            },
        },
        "manipulation": {
            "description": "抓取和操作物体的零件",
            "parts": [
                "finger", "suction_cup", "gripper_tip",
                "claw", "hook", "magnet", "vacuum_pad",
            ],
            "typical_properties": {
                "high_friction": True,
                "small_size": True,
                "often_symmetric": True,
            },
        },
        "energy": {
            "description": "能源存储与供应",
            "parts": [
                "battery", "solar_panel", "fuel_tank",
                "capacitor", "power_supply",
            ],
            "typical_properties": {
                "high_mass": True,
                "central_placement": True,
            },
        },
        "sensing": {
            "description": "感知环境的传感器",
            "parts": [
                "camera", "imu", "touch_sensor",
                "lidar", "microphone", "proximity_sensor",
            ],
            "typical_properties": {
                "lightweight": True,
                "requires_mounting": True,
            },
        },
        "connecting": {
            "description": "连接其他零件的无动力元件",
            "parts": [
                "joint_connector", "adapter", "bracket",
                "mount", "coupling", "flange",
            ],
            "typical_properties": {
                "small_size": True,
                "actuated": False,
                "multiple_faces": True,
            },
        },
        "spring_damper": {
            "description": "弹性/阻尼元件",
            "parts": [
                "spring_element", "damper", "tendon",
                "shock_absorber", "elastic_band",
            ],
            "typical_properties": {
                "compliant": True,
                "energy_storage": True,
            },
        },
    },
    "composition_rules": [
        {
            "rule": "actuator_connects_structural",
            "description": "驱动器必须连接在两个结构件之间",
            "antecedent": {"function": "actuation"},
            "consequent": [
                {"function": "structural", "relation": "parent"},
                {"function": "structural", "relation": "child"},
            ],
        },
        {
            "rule": "locomotion_at_leaves",
            "description": "运动零件 (轮/脚) 应位于图末端 (叶节点)",
            "antecedent": {"function": "locomotion"},
            "consequent": [{"is_leaf": True}],
        },
        {
            "rule": "symmetric_manipulation",
            "description": "操作类零件应成对对称布局",
            "antecedent": {"function": "manipulation"},
            "consequent": [{"sibling": {"function": "manipulation", "symmetric": True}}],
        },
        {
            "rule": "energy_near_root",
            "description": "能源零件应靠近根部 (重心低, 稳定性好)",
            "antecedent": {"function": "energy"},
            "consequent": [{"max_depth_from_root": 2}],
        },
        {
            "rule": "sensing_on_surface",
            "description": "传感器应位于外表面 (叶节点或接近叶节点)",
            "antecedent": {"function": "sensing"},
            "consequent": [{"near_leaf": True}],
        },
        {
            "rule": "no_chain_of_actuators",
            "description": "避免串联过多驱动器 (>3), 增加控制复杂度",
            "max_consecutive": 3,
        },
    ],
    "task_specific_configs": {
        "speed": {
            "priority_parts": [
                {"function": "locomotion", "reason": "需要高效位移"},
                {"function": "actuation", "reason": "需要强大驱动"},
                {"function": "structural", "reason": "需要轻量刚体"},
            ],
            "avoid": [
                {"function": "manipulation", "reason": "增加不必要质量和复杂度"},
                {"function": "sensing", "reason": "速度任务通常不需要精密感知"},
            ],
        },
        "climbing": {
            "priority_parts": [
                {"function": "manipulation", "reason": "需要抓握力"},
                {"function": "locomotion", "reason": "需要稳定附着"},
            ],
            "avoid": [
                {"function": "energy", "reason": "攀爬需要轻量化"},
            ],
            "prefer": {
                "high_friction": True,
                "actuated_count_range": [4, 12],
            },
        },
        "manipulation": {
            "priority_parts": [
                {"function": "manipulation", "reason": "抓取核心功能"},
                {"function": "actuation", "reason": "需要精确控制"},
                {"function": "sensing", "reason": "需要感知反馈"},
            ],
            "prefer": {
                "symmetric": True,
                "actuated_count_range": [2, 8],
            },
        },
        "efficiency": {
            "priority_parts": [
                {"function": "structural", "reason": "需要优化结构质量分布"},
            ],
            "avoid": [
                {"function": "energy", "reason": "电池通常增加死重"},
            ],
            "prefer": {
                "low_mass": True,
                "actuated_count_range": [1, 4],
            },
        },
        "structure": {
            "priority_parts": [
                {"function": "structural", "reason": "结构完整性"},
                {"function": "spring_damper", "reason": "能量吸收"},
            ],
            "prefer": {
                "high_stiffness": True,
            },
        },
    },
}


# ══════════════════════════════════════════════════════════════
#  2. 物理启发式规则 (Physics Heuristics)
# ══════════════════════════════════════════════════════════════

PHYSICS_HEURISTICS: List[Dict] = [
    # ── 运动学 ──
    {
        "id": "PH-01",
        "name": "legged_stability",
        "description": "至少需要 3 条腿才能实现静态稳定 (tripod gait)",
        "condition": {"task_type": ["speed", "climbing", "structure"]},
        "rule": "n_legs >= 3",
        "confidence": "high",
        "source": "engineering_textbook",
    },
    {
        "id": "PH-02",
        "name": "wheeled_mobility",
        "description": "轮式驱动的机器人需要偶数个对称轮 + 至少 2 个驱动电机 (差速转向)",
        "condition": {"has_wheels": True},
        "rule": "n_wheels % 2 == 0 AND n_motors >= 2",
        "confidence": "high",
        "source": "engineering_textbook",
    },
    {
        "id": "PH-03",
        "name": "gripper_closure",
        "description": "夹持器需要对称排列的手指以实现力闭合",
        "condition": {"has_fingers": True},
        "rule": "fingers are pairwise symmetric",
        "confidence": "high",
        "source": "robotics_handbook",
    },
    {
        "id": "PH-04",
        "name": "flying_lift",
        "description": "飞行器总质量必须小于推进器提供的最大升力",
        "condition": {"task_type": ["flying"]},
        "rule": "total_mass < n_propellers * max_thrust_per_propeller",
        "confidence": "high",
        "source": "aerodynamics_textbook",
    },
    # ── 稳定性 ──
    {
        "id": "PH-05",
        "name": "low_center_of_mass",
        "description": "重心越低越稳定 (降低翻倒风险)",
        "condition": {"task_type": ["speed", "climbing", "structure"]},
        "rule": "CoM_z < 0.5 * body_height",
        "confidence": "medium",
        "source": "physics_intuition",
    },
    {
        "id": "PH-06",
        "name": "wide_support_base",
        "description": "支撑多边形面积越大越稳定",
        "condition": {"task_type": ["structure", "climbing"]},
        "rule": "support_polygon_area > 0.25 * body_bounding_box_area",
        "confidence": "medium",
        "source": "physics_intuition",
    },
    # ── 效率 ──
    {
        "id": "PH-07",
        "name": "serial_actuator_inefficiency",
        "description": "串联驱动器越多, 控制精度越低, 能耗越高",
        "condition": {"any": True},
        "rule": "max_consecutive_actuated_parts <= 4",
        "confidence": "medium",
        "source": "control_theory",
    },
    {
        "id": "PH-08",
        "name": "mass_distribution",
        "description": "非驱动质量应尽可能远离驱动源, 减少惯性负载",
        "condition": {"has_actuators": True},
        "rule": "actuators near root, passive mass at leaves",
        "confidence": "medium",
        "source": "mechanics",
    },
    {
        "id": "PH-09",
        "name": "energy_pareto",
        "description": "在速度和能耗之间存在 Pareto 最优权衡: 更快通常更耗能",
        "condition": {"task_type": ["efficiency", "speed_density"]},
        "rule": "speed and energy_consumption are conflicting objectives",
        "confidence": "high",
        "source": "observed_pattern",
    },
    # ── 形态拓扑 ──
    {
        "id": "PH-10",
        "name": "depth_fanout_tradeoff",
        "description": "深图 (长链) vs 宽图 (扇出) 的权衡: 深图利于平衡, 宽图利于稳定",
        "condition": {"any": True},
        "rule": "deep_graph(good for dynamic balance) vs wide_graph(good for static stability)",
        "confidence": "medium",
        "source": "morphology_study",
    },
    {
        "id": "PH-11",
        "name": "symmetry_efficiency",
        "description": "双侧对称的形态通常更高效 (控制策略可复用)",
        "condition": {"task_type": ["speed", "climbing"]},
        "rule": "bilateral_symmetry preferred",
        "confidence": "medium",
        "source": "evolutionary_observation",
    },
    # ── 材料与制造 ──
    {
        "id": "PH-12",
        "name": "stress_concentration",
        "description": "尖锐几何过渡导致应力集中, 应避免半径突变",
        "condition": {"has_joints": True},
        "rule": "fillet radius > 0.05 * part_dimension",
        "confidence": "medium",
        "source": "mechanics_of_materials",
    },
    {
        "id": "PH-13",
        "name": "manufacturability_constraint",
        "description": "过度复杂的零件可能无法制造 (悬垂, 薄壁, 内腔)",
        "condition": {"manufacturing": "3d_printing"},
        "rule": "wall_thickness > 1mm AND overhang_angle < 45deg",
        "confidence": "medium",
        "source": "additive_manufacturing",
    },
    # ── 任务特化 ──
    {
        "id": "PH-14",
        "name": "speed_needs_lightweight",
        "description": "速度任务中质量是敌人: 每减少 10% 质量通常提升 ~5-15% 速度",
        "condition": {"task_type": ["speed"]},
        "rule": "minimize non-essential mass",
        "confidence": "high",
        "source": "empirical_data",
    },
    {
        "id": "PH-15",
        "name": "climbing_needs_grip",
        "description": "攀爬任务中末端摩擦系数是关键: μ > 0.6 才能可靠附着",
        "condition": {"task_type": ["climbing"]},
        "rule": "end_effector_friction > 0.6",
        "confidence": "medium",
        "source": "robotics_experiment",
    },
]


# ══════════════════════════════════════════════════════════════
#  3. 策略模板 (Strategy Templates)
# ══════════════════════════════════════════════════════════════

STRATEGY_TEMPLATES: Dict = {
    "speed": {
        "recommended_strategy": "map_elites",
        "reason": "速度任务需要多样性探索 (不同步态对应不同行为特征区间)",
        "evolution_config": {
            "population_size": 80,
            "generations": 300,
            "elite_count": 10,
            "mutation_rate": 0.30,
            "crossover_rate": 0.65,
            "topo_mutation_prob": 0.12,
            "param_mutation_prob": 0.30,
            "param_mutation_scale": 0.08,
            "selection_pressure": 2.0,
        },
        "emitter_config": {
            "num_emitters": 4,
            "emitter_batch_size": 10,
            "emitter_types": ["improvement", "improvement", "random", "optimizer"],
        },
        "map_elites_config": {
            "bc_names": ["speed", "stability", "energy_efficiency"],
            "bc_ranges": [[0, 8], [0, 1], [0, 5]],
            "archive_size": 100,
            "use_cvt": True,
            "inject_every_n_gens": 8,
            "inject_count": 4,
        },
    },
    "climbing": {
        "recommended_strategy": "nsga3",
        "reason": "攀爬有多个冲突目标 (抓力, 稳定性, 能耗), NSGA-III 适合多目标权衡",
        "evolution_config": {
            "population_size": 60,
            "generations": 250,
            "elite_count": 8,
            "mutation_rate": 0.35,
            "crossover_rate": 0.55,
            "topo_mutation_prob": 0.20,
            "param_mutation_prob": 0.35,
            "param_mutation_scale": 0.15,
            "selection_pressure": 2.5,
        },
    },
    "manipulation": {
        "recommended_strategy": "hierarchical",
        "reason": "操作任务受益于分层优化: 先找合理形态, 再精细调参",
        "evolution_config": {
            "population_size": 50,
            "generations": 200,
            "mutation_rate": 0.25,
            "crossover_rate": 0.70,
        },
    },
    "efficiency": {
        "recommended_strategy": "cma_me",
        "reason": "效率优化是连续参数空间问题, CMA-ES 擅长在连续空间搜索",
        "evolution_config": {
            "population_size": 40,
            "generations": 200,
            "mutation_rate": 0.20,
            "param_mutation_prob": 0.40,
            "param_mutation_scale": 0.05,
        },
    },
    "structure": {
        "recommended_strategy": "map_elites",
        "reason": "结构任务需要发现多样的拓扑形态",
        "evolution_config": {
            "population_size": 100,
            "generations": 400,
            "mutation_rate": 0.40,
            "topo_mutation_prob": 0.25,
            "crossover_rate": 0.50,
        },
    },
    "default": {
        "recommended_strategy": "map_elites",
        "reason": "通用任务默认使用 MAP-Elites 以保证多样性",
        "evolution_config": {
            "population_size": 50,
            "generations": 200,
            "elite_count": 8,
            "mutation_rate": 0.35,
            "crossover_rate": 0.60,
        },
    },
}

# ── 停滞检测规则 ─────────────────────────────────────────────

STAGNATION_RULES: List[Dict] = [
    {
        "id": "ST-01",
        "name": "coverage_stall",
        "description": "QD-score 连续 20 代无显著增长",
        "detection": "max(qd_score[-20:]) - qd_score[-1] < 0.01 * qd_score[-1]",
        "suggested_action": "increase_mutation_rate",
        "params": {"mutation_rate_multiplier": 1.5},
    },
    {
        "id": "ST-02",
        "name": "diversity_collapse",
        "description": "种群拓扑多样性低于阈值",
        "detection": "mean_pairwise_topo_distance < 0.1",
        "suggested_action": "switch_to_random_emitter",
        "params": {"random_emitter_ratio": 0.5},
    },
    {
        "id": "ST-03",
        "name": "early_convergence",
        "description": "前 50 代已接近最大 QD-score 且不再增长",
        "detection": "generation > 50 AND qd_score_slope < 0.001",
        "suggested_action": "switch_strategy",
        "params": {"strategy": "cma_me", "reason": "MAP-Elites converged early, switch to CMA-ME for local refinement"},
    },
    {
        "id": "ST-04",
        "name": "slow_start",
        "description": "前 30 代覆盖率为 0",
        "detection": "generation == 30 AND coverage == 0",
        "suggested_action": "switch_catalog_or_relax_constraints",
        "params": {},
    },
]

# ── 便捷查询函数 ─────────────────────────────────────────────


def get_parts_by_function(function: str) -> List[str]:
    """获取指定功能组的所有零件名称"""
    groups = PART_ONTOLOGY.get("functional_groups", {})
    if function in groups:
        return groups[function].get("parts", [])
    return []


def get_heuristics_for_task(task_type: str) -> List[Dict]:
    """获取与任务类型相关的启发式规则"""
    result = []
    for rule in PHYSICS_HEURISTICS:
        cond = rule.get("condition", {})
        if cond.get("any", False):
            result.append(rule)
            continue
        task_types = cond.get("task_type", [])
        if task_type in task_types or "any" in task_types:
            result.append(rule)
    return result


def get_strategy_template(task_type: str) -> Dict:
    """获取任务类型对应的策略模板"""
    return STRATEGY_TEMPLATES.get(
        task_type, STRATEGY_TEMPLATES["default"]
    )


def get_all_task_types() -> List[str]:
    """获取所有已知任务类型"""
    return list(STRATEGY_TEMPLATES.keys())
