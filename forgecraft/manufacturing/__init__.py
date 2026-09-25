"""
ForgeCraft 制造导出模块

功能：
1. STL 导出 — 零件 → 3D 网格文件（含连接结构/支撑/公差）
2. STEP 导出 — 纯 Python AP242 格式，SolidWorks/Fusion360 可导入
3. URDF 导出 — MuJoCo XML → ROS 标准格式
4. 制造可行性评分 — 壁厚、悬垂、重心
5. ONNX 导出 — PyTorch 策略 → 跨平台推理
6. 一键打包 — 进化结果自动导出 design_output/

用法：
  python -m forgecraft.manufacturing --body-file best_body.json
  python -m forgecraft.manufacturing --body-file best_body.json --catalog primitives
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Phase 4 模块导入 ──
try:
    from forgecraft.manufacturing.step_exporter import export_step, export_step_parts
except ImportError:
    export_step = None
    export_step_parts = None

try:
    from forgecraft.manufacturing.drawing_generator import (
        generate_engineering_drawing, export_dxf,
    )
except ImportError:
    generate_engineering_drawing = None
    export_dxf = None

try:
    from forgecraft.manufacturing.interference_checker import (
        check_interferences, generate_interference_report,
    )
except ImportError:
    check_interferences = None
    generate_interference_report = None

try:
    from forgecraft.manufacturing.production_packager import build_production_package
except ImportError:
    build_production_package = None

# ── 懒初始化参数化生成器 ──
_parametric_gen = None


def _get_parametric_generator():
    global _parametric_gen
    if _parametric_gen is None:
        try:
            from forgecraft.geometry.parametric import ParametricGenerator
            _parametric_gen = ParametricGenerator(quality="high")
        except Exception:
            _parametric_gen = False  # 标记为不可用
    return _parametric_gen if _parametric_gen is not False else None


# ================================================================
# 1. STL 导出：零件 → 3D 网格
# ================================================================

def _make_box_mesh(length: float, width: float, height: float):
    import trimesh
    return trimesh.creation.box(extents=(length, width, height))


def _make_cylinder_mesh(radius: float, length: float, sections: int = 24):
    import trimesh
    return trimesh.creation.cylinder(radius=radius, height=length, sections=sections)


def _make_sphere_mesh(radius: float, subdivisions: int = 2):
    import trimesh
    return trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)


def _part_to_mesh(part_dict: dict, spec: dict, generator=None):
    import trimesh

    params = part_dict.get("params", {})
    part_type = part_dict.get("part_type", "unknown")
    shape = spec.get("geometry", "box")

    length = params.get("length", 0.1)
    radius = params.get("radius", 0.03)
    thickness = params.get("thickness", 0.02)
    width = params.get("width", thickness)

    # ── 优先使用参数化生成器 ──
    if generator is not None and generator.has_generator(part_type):
        mesh = generator.make(part_type, params)
        if mesh is not None:
            position = np.array(part_dict.get("position", [0, 0, 0]), dtype=np.float64)
            mesh.apply_translation(position)
            return mesh

    # ── 回退到基本几何 ──
    if shape == "box":
        mesh = _make_box_mesh(length, width if width > 0 else length * 0.5, thickness)
    elif shape == "cylinder":
        mesh = _make_cylinder_mesh(radius, length)
    elif shape == "sphere":
        mesh = _make_sphere_mesh(radius)
    else:
        mesh = _make_box_mesh(length, length * 0.5, thickness)

    position = np.array(part_dict.get("position", [0, 0, 0]), dtype=np.float64)
    mesh.apply_translation(position)

    return mesh


def export_stl(body_data: dict, catalog_specs: dict, output_dir: str, generator=None) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    parts = body_data.get("parts", [])
    joints = body_data.get("joints", [])
    exported: Dict[str, str] = {}

    for i, part in enumerate(parts):
        part_type = part.get("part_type", "unknown")
        spec_dict = catalog_specs.get(part_type, {})
        mesh = _part_to_mesh(part, spec_dict, generator=generator)

        part_id = part.get("part_id", f"part_{i}")
        part_name = part_type.replace("/", "_")
        filename = f"{part_name}_{part_id}.stl"
        filepath = os.path.join(output_dir, filename)

        mesh.export(filepath)
        exported[part_id] = filepath

    assembly = _create_assembly_stl(body_data, catalog_specs, generator=generator)
    assembly_path = os.path.join(output_dir, "assembly.stl")
    assembly.export(assembly_path)
    exported["assembly"] = assembly_path

    return exported


def _create_assembly_stl(body_data: dict, catalog_specs: dict, generator=None):
    import trimesh

    parts = body_data.get("parts", [])
    meshes = []

    for part in parts:
        part_type = part.get("part_type", "unknown")
        spec_dict = catalog_specs.get(part_type, {})
        mesh = _part_to_mesh(part, spec_dict, generator=generator)
        if part_type in catalog_specs:
            spec = catalog_specs[part_type]
            color = spec.get("color", [0.6, 0.6, 0.6, 1.0])[:3]
            mesh.visual.face_colors = [int(c * 255) for c in color] + [255]
        meshes.append(mesh)

    if meshes:
        return trimesh.util.concatenate(meshes)
    return _make_box_mesh(0.1, 0.1, 0.1)


# ================================================================
# 2. URDF 导出：MuJoCo XML → ROS 格式
# ================================================================

def export_urdf(body_data: dict, catalog_specs: dict, output_dir: str) -> str:
    parts = body_data.get("parts", [])
    joints = body_data.get("joints", [])
    name = body_data.get("name", "forgecraft_robot")

    lines = ['<?xml version="1.0"?>']
    lines.append(f'<robot name="{name}">')
    lines.append("")

    for i, part in enumerate(parts):
        part_id = part.get("part_id", f"part_{i}")
        part_type = part.get("part_type", "unknown")
        spec = catalog_specs.get(part_type, {})
        params = part.get("params", {})
        position = part.get("position", [0, 0, 0])

        shape = spec.get("geometry", "box")
        color = spec.get("color", [0.6, 0.6, 0.6, 1.0])
        mass = params.get("mass", 0.1)
        length = params.get("length", 0.1)
        radius = params.get("radius", 0.03)
        thickness = params.get("thickness", 0.02)

        lines.append(f'  <link name="{part_id}">')
        lines.append("    <inertial>")
        lines.append(f'      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append(f'      <mass value="{mass:.4f}"/>')
        inertia = mass * 0.01
        lines.append(f'      <inertia ixx="{inertia:.6f}" ixy="0" ixz="0" iyy="{inertia:.6f}" iyz="0" izz="{inertia:.6f}"/>')
        lines.append("    </inertial>")
        lines.append("    <visual>")

        if shape == "box":
            lines.append(f'      <geometry><box size="{length:.4f} {thickness:.4f} {length:.4f}"/></geometry>')
        elif shape == "cylinder":
            lines.append(f'      <geometry><cylinder radius="{radius:.4f}" length="{length:.4f}"/></geometry>')
        elif shape == "sphere":
            lines.append(f'      <geometry><sphere radius="{radius:.4f}"/></geometry>')
        else:
            lines.append(f'      <geometry><box size="{length:.4f} {thickness:.4f} {length:.4f}"/></geometry>')

        r, g, b, a = color
        lines.append(f'      <material name="{part_type}_mat"><color rgba="{r:.2f} {g:.2f} {b:.2f} {a:.2f}"/></material>')
        lines.append("    </visual>")
        lines.append("    <collision>")

        if shape == "box":
            lines.append(f'      <geometry><box size="{length:.4f} {thickness:.4f} {length:.4f}"/></geometry>')
        elif shape == "cylinder":
            lines.append(f'      <geometry><cylinder radius="{radius:.4f}" length="{length:.4f}"/></geometry>')
        elif shape == "sphere":
            lines.append(f'      <geometry><sphere radius="{radius:.4f}"/></geometry>')

        lines.append("    </collision>")
        lines.append(f"  </link>")
        lines.append("")

    for joint in joints:
        jtype = joint.get("joint_type", "fixed")
        parent_id = joint.get("parent_id", "")
        child_id = joint.get("child_id", "")
        anchor = joint.get("anchor", [0, 0, 0])
        axis = joint.get("axis", [0, 1, 0])
        params_j = joint.get("params", {})

        if jtype == "fixed":
            urdf_type = "fixed"
        elif jtype == "hinge":
            urdf_type = "revolute"
        elif jtype == "ball":
            urdf_type = "floating"
        elif jtype == "slide":
            urdf_type = "prismatic"
        else:
            urdf_type = "fixed"

        lines.append(f'  <joint name="j_{parent_id}_{child_id}" type="{urdf_type}">')
        lines.append(f'    <parent link="{parent_id}"/>')
        lines.append(f'    <child link="{child_id}"/>')
        lines.append(f'    <origin xyz="{anchor[0]:.4f} {anchor[1]:.4f} {anchor[2]:.4f}" rpy="0 0 0"/>')

        if urdf_type in ("revolute", "continuous", "prismatic"):
            ax = axis if len(axis) >= 3 else [0, 1, 0]
            lines.append(f'    <axis xyz="{ax[0]:.4f} {ax[1]:.4f} {ax[2]:.4f}"/>')

        if urdf_type in ("revolute", "prismatic"):
            lo = params_j.get("range_min", -1.5)
            hi = params_j.get("range_max", 1.5)
            torque = params_j.get("max_torque", 5.0)
            velocity = params_j.get("max_velocity", 10.0)
            lines.append(f'    <limit lower="{lo:.4f}" upper="{hi:.4f}" effort="{torque:.2f}" velocity="{velocity:.2f}"/>')

        lines.append("  </joint>")
        lines.append("")

    lines.append(f'  <gazebo>')
    lines.append(f'    <plugin name="forgecraft_control" filename="libforgecraft_controller.so">')
    for i, part in enumerate(parts):
        part_id = part.get("part_id", f"part_{i}")
        params_p = part.get("params", {})
        if params_p.get("actuated", False) or params_p.get("can_actuate", False):
            lines.append(f'      <joint name="j_{part_id}"/>')
            lines.append(f'      <torque>{params_p.get("max_torque", 5.0):.2f}</torque>')
    lines.append("    </plugin>")
    lines.append("  </gazebo>")

    lines.append("</robot>")

    urdf_path = os.path.join(output_dir, f"{name}.urdf")
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return urdf_path


# ================================================================
# 3. 制造可行性评分
# ================================================================

def compute_manufacturability_report(body_data: dict, catalog_specs: dict) -> dict:
    parts = body_data.get("parts", [])
    joints = body_data.get("joints", [])

    scores = {}
    details = []

    for part in parts:
        part_id = part.get("part_id", "?")
        part_type = part.get("part_type", "?")
        spec = catalog_specs.get(part_type, {})
        params = part.get("params", {})
        length = params.get("length", 0.1)
        radius = params.get("radius", 0.03)
        thickness = params.get("thickness", 0.02)
        mass_val = params.get("mass", 0.1)

        score = 1.0
        issues = []

        min_wall = min(length, thickness, radius * 2)
        if min_wall < 0.001:
            score -= 0.4
            issues.append("壁厚过薄 (<1mm)，FDM打印不可行")

        if thickness > 0 and length / max(thickness, 0.001) > 10:
            score -= 0.2
            issues.append("长宽比过大，结构不稳定")

        if len(parts) > 15:
            score -= 0.1
            issues.append("零件过多，装配复杂度高")

        if len(parts) > 30:
            score -= 0.2
            issues.append("零件超30，制造成本极高")

        scores[part_id] = {
            "part_type": part_type,
            "score": max(0.0, score),
            "issues": issues,
        }
        details.append({"part_id": part_id, "part_type": part_type, "score": max(0.0, score), "issues": issues})

    if scores:
        overall = sum(s["score"] for s in scores.values()) / len(scores)
    else:
        overall = 1.0

    unique_types = len(set(p.get("part_type") for p in parts))
    if unique_types <= 1:
        overall -= 0.3

    com_height = _compute_com_height(parts)
    if com_height < 0.02:
        overall -= 0.2

    overall = max(0.0, overall)

    return {
        "overall_score": round(overall, 4),
        "grade": _score_to_grade(overall),
        "com_height": com_height,
        "unique_part_types": unique_types,
        "detail": details,
    }


def _compute_com_height(parts: list) -> float:
    total_mass = 0.0
    weighted_z = 0.0
    for part in parts:
        pos = part.get("position", [0, 0, 0])
        mass = part.get("params", {}).get("mass", 0.1)
        weighted_z += mass * (pos[2] if len(pos) > 2 else 0)
        total_mass += mass
    return weighted_z / max(total_mass, 0.001)


def _score_to_grade(score: float) -> str:
    if score >= 0.9:
        return "A (可直接制造)"
    elif score >= 0.7:
        return "B (建议微调)"
    elif score >= 0.5:
        return "C (需要修改)"
    elif score >= 0.3:
        return "D (重大设计缺陷)"
    else:
        return "F (不可制造)"


# ================================================================
# 4. ONNX 导出 + C 推理模板
# ================================================================

def export_onnx(policy_state: dict, obs_dim: int, act_dim: int, morph_dim: int, output_dir: str) -> str:
    import torch
    from forgecraft.rl.encoder import MorphAwareActor

    actor_weights = None
    if "actor" in policy_state and isinstance(policy_state["actor"], dict):
        actor_weights = policy_state["actor"]
    else:
        extracted = {}
        for k, v in policy_state.items():
            if k.startswith("actor."):
                extracted[k[6:]] = v
        if extracted:
            actor_weights = extracted

    if actor_weights is None:
        raise ValueError("无法从策略权重中提取 actor 参数")

    hidden_dim = actor_weights["shared.0.weight"].shape[0]
    actor = MorphAwareActor(obs_dim, act_dim, morph_dim, hidden_dim)
    actor.load_state_dict(actor_weights)
    actor.eval()

    dummy_obs = torch.randn(1, obs_dim)
    dummy_morph = torch.randn(1, morph_dim)

    traced = torch.jit.trace(actor, (dummy_obs, dummy_morph))

    onnx_path = os.path.join(output_dir, "forgecraft_policy.onnx")
    torch.onnx.export(
        traced,
        (dummy_obs, dummy_morph),
        onnx_path,
        input_names=["observation", "morphology_embedding"],
        output_names=["action_mean", "action_log_std"],
        dynamic_axes={
            "observation": {0: "batch"},
            "morphology_embedding": {0: "batch"},
            "action_mean": {0: "batch"},
            "action_log_std": {0: "batch"},
        },
        opset_version=14,
        dynamo=False,
    )

    return onnx_path


def export_c_template(output_dir: str) -> str:
    cpp_code = r'''/**
 * ForgeCraft 控制策略推理模板
 *
 * 用法:
 *   1. 用 ONNX Runtime 加载 forgecraft_policy.onnx
 *   2. 调用 inference() 每步获取 action
 *
 * 编译 (Linux arm):
 *   g++ -O2 forgecraft_inference.cpp -lonnxruntime -o forgecraft_controller
 *
 * 依赖: ONNX Runtime C API
 *   sudo apt install libonnxruntime-dev
 */

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cmath>

class ForgeCraftController {
public:
    ForgeCraftController(int obs_dim, int morph_dim, int act_dim, const char* model_path)
        : obs_dim_(obs_dim), morph_dim_(morph_dim), act_dim_(act_dim)
    {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ForgeCraft");
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session(env_, model_path, session_options_);

        Ort::AllocatorWithDefaultOptions allocator;
        input_names_.push_back(session_.GetInputName(0, allocator));
        input_names_.push_back(session_.GetInputName(1, allocator));
        output_names_.push_back(session_.GetOutputName(0, allocator));
        output_names_.push_back(session_.GetOutputName(1, allocator));
    }

    void inference(const float* obs, const float* morph_embed, float* action) {
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        int64_t obs_shape[] = {1, obs_dim_};
        int64_t morph_shape[] = {1, morph_dim_};

        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, (float*)obs, obs_dim_, obs_shape, 2));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, (float*)morph_embed, morph_dim_, morph_shape, 2));

        auto outputs = session_.Run(Ort::RunOptions{nullptr},
            input_names_.data(), inputs.data(), inputs.size(),
            output_names_.data(), output_names_.size());

        float* mean_data = outputs[0].GetTensorMutableData<float>();
        for (int i = 0; i < act_dim_; i++) {
            action[i] = mean_data[i];
        }
    }

    void sample_action(const float* obs, const float* morph_embed, float* action) {
        inference(obs, morph_embed, action);
        // 添加高斯噪声用于探索
        for (int i = 0; i < act_dim_; i++) {
            float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            action[i] += noise;
            if (action[i] > 1.0f) action[i] = 1.0f;
            if (action[i] < -1.0f) action[i] = -1.0f;
        }
    }

private:
    int obs_dim_, morph_dim_, act_dim_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};

// ===== 使用示例 =====
int main() {
    const int OBS_DIM = 10;
    const int MORPH_DIM = 32;
    const int ACT_DIM = 2;

    ForgeCraftController controller(OBS_DIM, MORPH_DIM, ACT_DIM,
        "forgecraft_policy.onnx");

    float obs[OBS_DIM] = {0};
    float morph[MORPH_DIM] = {0};
    float action[ACT_DIM] = {0};

    // 每步调用（替代 Python PPO 推理）
    for (int step = 0; step < 500; step++) {
        controller.sample_action(obs, morph, action);
        // 将 action 发送给底层控制 (PID/CAN)
        // send_motor_command(action);
    }

    return 0;
}
'''
    cpp_path = os.path.join(output_dir, "forgecraft_inference.cpp")
    with open(cpp_path, "w", encoding="utf-8") as f:
        f.write(cpp_code)
    return cpp_path


# ================================================================
# 5. 材料清单 (BOM)
# ================================================================

def export_bom(body_data: dict, catalog_specs: dict, output_dir: str) -> str:
    parts = body_data.get("parts", [])

    bom = {}
    for part in parts:
        part_type = part.get("part_type", "unknown")
        if part_type not in bom:
            bom[part_type] = {"count": 0, "total_mass": 0.0, "spec": catalog_specs.get(part_type, {})}
        bom[part_type]["count"] += 1
        bom[part_type]["total_mass"] += part.get("params", {}).get("mass", 0.1)

    lines = ["# ForgeCraft 材料清单 (BOM)", ""]
    lines.append(f"## 机器人: {body_data.get('name', 'unnamed')}")
    lines.append(f"## 总零件数: {len(parts)}")
    lines.append("")
    lines.append("| 零件名 | 数量 | 单件质量(g) | 总质量(g) | 材质建议 |")
    lines.append("|--------|------|------------|----------|---------|")
    for ptype, data in sorted(bom.items()):
        count = data["count"]
        avg_mass = data["total_mass"] / count * 1000
        total_mass = data["total_mass"] * 1000
        material = "PLA" if avg_mass < 80 else "ABS" if avg_mass < 150 else "PETG/碳纤维"
        lines.append(f"| {ptype} | {count} | {avg_mass:.1f} | {total_mass:.1f} | {material} |")

    lines.append("")
    total = sum(d["total_mass"] * 1000 for d in bom.values())
    lines.append(f"**总质量: {total:.1f}g**")

    bom_path = os.path.join(output_dir, "BOM.md")
    with open(bom_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return bom_path


# ================================================================
# 6. 一键打包
# ================================================================

def export_all(
    body_data: dict,
    catalog_specs: dict,
    output_dir: str,
    policy_state: Optional[dict] = None,
    obs_dim: int = 0,
    act_dim: int = 0,
    morph_dim: int = 32,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    gen = _get_parametric_generator()

    stl_files = export_stl(body_data, catalog_specs, os.path.join(output_dir, "stl"), generator=gen)
    results["stl"] = stl_files

    urdf_path = export_urdf(body_data, catalog_specs, output_dir)
    results["urdf"] = urdf_path

    report = compute_manufacturability_report(body_data, catalog_specs)
    report_path = os.path.join(output_dir, "manufacturability.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    results["manufacturability"] = report_path
    results["manufacturability_report"] = report

    if policy_state is not None and obs_dim > 0 and act_dim > 0:
        try:
            onnx_path = export_onnx(policy_state, obs_dim, act_dim, morph_dim, output_dir)
            results["onnx"] = onnx_path
            cpp_path = export_c_template(output_dir)
            results["c_template"] = cpp_path
        except Exception as e:
            print(f"  [警告] ONNX 导出失败: {e}")
            results["onnx"] = None
            results["c_template"] = None
    else:
        results["onnx"] = None
        results["c_template"] = None

    bom_path = export_bom(body_data, catalog_specs, output_dir)
    results["bom"] = bom_path

    body_json_path = os.path.join(output_dir, "body.json")
    with open(body_json_path, "w", encoding="utf-8") as f:
        json.dump(body_data, f, indent=2, ensure_ascii=False)
    results["body_json"] = body_json_path

    results["output_dir"] = output_dir
    results["grade"] = report.get("grade", "?")

    return results


def export_all_enhanced(
    body_data: dict,
    catalog_specs: dict,
    output_dir: str,
    policy_state: Optional[dict] = None,
    obs_dim: int = 0,
    act_dim: int = 0,
    morph_dim: int = 32,
    fdm_profile: str = "PLA_04mm",
    add_connectors: bool = True,
    add_supports: bool = True,
    add_tolerances: bool = True,
) -> dict:
    from forgecraft.manufacturing.step_writer import export_step_from_body_parts
    from forgecraft.manufacturing.connectors import (
        CONNECTION_PRESETS, add_bolt_holes_to_part,
        add_hinge_bearing, add_snap_fit_to_parts,
    )
    from forgecraft.manufacturing.supports import (
        generate_breakaway_supports, generate_raft,
        compute_support_volume_ratio, optimize_orientation,
    )
    from forgecraft.manufacturing.tolerances import (
        FDM_PROFILES, compute_joint_clearance, apply_hinge_clearance,
        compute_print_report, auto_compensate_mesh,
    )
    import trimesh

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    fdm = FDM_PROFILES.get(fdm_profile, FDM_PROFILES["PLA_04mm"])

    stl_basic_dir = os.path.join(output_dir, "stl")
    stl_enhanced_dir = os.path.join(output_dir, "stl_enhanced")
    step_dir = os.path.join(output_dir, "step")
    support_dir = os.path.join(output_dir, "supports")

    for d in [stl_basic_dir, stl_enhanced_dir, step_dir, support_dir]:
        os.makedirs(d, exist_ok=True)

    stl_files = export_stl(body_data, catalog_specs, stl_basic_dir)
    results["stl"] = stl_files

    parts = body_data.get("parts", [])
    joints = body_data.get("joints", [])

    enhanced_meshes = {}
    joint_graph = {}
    for joint in joints:
        parent_id = joint.get("parent_id", "")
        child_id = joint.get("child_id", "")
        joint_graph[child_id] = {
            "parent": parent_id,
            "type": joint.get("joint_type", "fixed"),
            "anchor": joint.get("anchor", [0, 0, 0]),
            "axis": joint.get("axis", [0, 1, 0]),
        }

    print(f"\n  🏭 增强制造导出 (连接={add_connectors}, 支撑={add_supports}, 公差={add_tolerances})")

    gen = _get_parametric_generator()
    if gen is not None:
        print(f"    参数化零件生成器已启用 ({len(gen.available_types())} 种零件类型)")

    for i, part in enumerate(parts):
        part_type = part.get("part_type", "structure")
        part_id = part.get("part_id", f"part_{i}")
        params = part.get("params", {})
        position = np.array(part.get("position", [0, 0, 0]), dtype=np.float64)
        spec = catalog_specs.get(part_type, {})

        # 优先参数化生成
        if gen is not None and gen.has_generator(part_type):
            mesh = gen.make(part_type, params, position=position)
            if mesh is not None:
                enhanced_meshes[part_id] = mesh
                continue

        # 回退: 基本几何
        shape = spec.get("geometry", "cylinder")
        length = float(params.get("length", 0.1))
        radius = float(params.get("radius", 0.03))
        width = float(params.get("width", 0.02))
        height = float(params.get("height", 0.02))
        thickness = float(params.get("thickness", 0.02))

        if shape == "box":
            mesh = trimesh.creation.box(extents=(length, width, height))
        elif shape == "sphere":
            mesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        else:
            mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=32)

        mesh.apply_translation(position)
        enhanced_meshes[part_id] = mesh

    if add_tolerances:
        print("    施加 FDM 热收缩补偿...")
        for part_id, mesh in enhanced_meshes.items():
            enhanced_meshes[part_id] = auto_compensate_mesh(mesh, fdm)

    if add_connectors:
        print("    生成连接结构 (螺栓孔/铰链轴承座)...")
        for joint in joints:
            child_id = joint.get("child_id", "")
            parent_id = joint.get("parent_id", "")
            jtype = joint.get("joint_type", "fixed")
            anchor = np.array(joint.get("anchor", [0, 0, 0]), dtype=np.float64)
            axis = np.array(joint.get("axis", [0, 1, 0]), dtype=np.float64)

            parent_params = {}
            child_params = {}
            for part in parts:
                if part.get("part_id") == parent_id:
                    parent_params = part.get("params", {})
                if part.get("part_id") == child_id:
                    child_params = part.get("params", {})

            if jtype == "hinge":
                shaft_r = child_params.get("radius", 0.003)
                clearance, _ = compute_joint_clearance(fdm, "hinge", shaft_r, "H7_f6")
                if child_id in enhanced_meshes:
                    enhanced_meshes[child_id] = add_hinge_bearing(
                        enhanced_meshes[child_id], anchor, axis,
                        shaft_radius=shaft_r * 0.6, clearance=clearance,
                    )
            elif jtype == "fixed":
                if child_id in enhanced_meshes and parent_id in enhanced_meshes:
                    connection_points = [(anchor, axis)]
                    enhanced_meshes[child_id] = add_bolt_holes_to_part(
                        enhanced_meshes[child_id], connection_points,
                        bolt_radius=0.0015, clearance=fdm.standard_clearance,
                    )

        for joint in joints:
            parent_id = joint.get("parent_id", "")
            child_id = joint.get("child_id", "")
            jtype = joint.get("joint_type", "fixed")
            anchor = np.array(joint.get("anchor", [0, 0, 0]), dtype=np.float64)
            axis = np.array(joint.get("axis", [0, 1, 0]), dtype=np.float64)

            if jtype == "fixed" and child_id in enhanced_meshes and parent_id in enhanced_meshes:
                try:
                    p, c = add_snap_fit_to_parts(
                        enhanced_meshes[parent_id], enhanced_meshes[child_id],
                        anchor, axis,
                        clearance=fdm.standard_clearance,
                    )
                    enhanced_meshes[parent_id] = p
                    enhanced_meshes[child_id] = c
                except Exception:
                    pass

    if add_supports:
        print("    生成可折断支撑柱...")
        for part_id, mesh in enhanced_meshes.items():
            optimized, _, _ = optimize_orientation(mesh)
            enhanced_meshes[part_id] = optimized

    print("    写出增强 STL...")
    enhanced_stl_files = {}
    for part_id, mesh in enhanced_meshes.items():
        filepath = os.path.join(stl_enhanced_dir, f"{part_id}_enhanced.stl")
        mesh.export(filepath)
        enhanced_stl_files[part_id] = filepath

    assembly_enhanced = trimesh.util.concatenate(list(enhanced_meshes.values()))
    assembly_path = os.path.join(stl_enhanced_dir, "assembly_enhanced.stl")
    assembly_enhanced.export(assembly_path)
    enhanced_stl_files["assembly"] = assembly_path
    results["stl_enhanced"] = enhanced_stl_files

    if add_supports:
        print("    写出支撑结构...")
        all_supports = {}
        for part_id, mesh in enhanced_meshes.items():
            supports = generate_breakaway_supports(mesh)
            raft = generate_raft(mesh)
            if supports.vertices.size > 0:
                supp_path = os.path.join(support_dir, f"{part_id}_support.stl")
                supports.export(supp_path)
                all_supports[f"{part_id}_support"] = supp_path
            if raft is not None and raft.vertices.size > 0:
                raft_path = os.path.join(support_dir, f"{part_id}_raft.stl")
                raft.export(raft_path)
                all_supports[f"{part_id}_raft"] = raft_path
        results["supports"] = all_supports

        support_ratios = {}
        for part_id, mesh in enhanced_meshes.items():
            supports = generate_breakaway_supports(mesh)
            if supports.vertices.size > 0:
                support_ratios[part_id] = compute_support_volume_ratio(mesh, supports)
        results["support_ratios"] = support_ratios

    print("    写出 STEP 文件 (AP242)...")
    step_files = export_step_from_body_parts(
        parts, catalog_specs,
        output_path=os.path.join(step_dir, "assembly.step"),
    )
    results["step"] = step_files

    urdf_path = export_urdf(body_data, catalog_specs, output_dir)
    results["urdf"] = urdf_path

    print_report = compute_print_report(parts, fdm)
    print_path = os.path.join(output_dir, "print_report.json")
    with open(print_path, "w", encoding="utf-8") as f:
        json.dump(print_report, f, indent=2, ensure_ascii=False)
    results["print_report"] = print_path

    report = compute_manufacturability_report(body_data, catalog_specs)
    report_path = os.path.join(output_dir, "manufacturability.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    results["manufacturability"] = report_path

    if policy_state is not None and obs_dim > 0 and act_dim > 0:
        try:
            onnx_path = export_onnx(policy_state, obs_dim, act_dim, morph_dim, output_dir)
            results["onnx"] = onnx_path
            cpp_path = export_c_template(output_dir)
            results["c_template"] = cpp_path
        except Exception as e:
            print(f"  [警告] ONNX 导出失败: {e}")
            results["onnx"] = None
            results["c_template"] = None
    else:
        results["onnx"] = None
        results["c_template"] = None

    bom_path = export_bom(body_data, catalog_specs, output_dir)
    results["bom"] = bom_path

    body_json_path = os.path.join(output_dir, "body.json")
    with open(body_json_path, "w", encoding="utf-8") as f:
        json.dump(body_data, f, indent=2, ensure_ascii=False)
    results["body_json"] = body_json_path

    results["output_dir"] = output_dir
    results["grade"] = report.get("grade", "?")
    results["print_profile"] = fdm_profile

    # Phase 3: PBR 渲染 + glTF 导出
    try:
        print("\n  Phase 3: PBR 论文级渲染...")
        pres_results = export_presentation_suite(
            body_data, output_dir,
            prefix="gen79",
            quality="high",
            explode_distance=0.12,
        )
        results["glb"] = pres_results.get("glb", {})
        results["renders"] = pres_results.get("renders", {})
    except Exception as e:
        print(f"    Phase 3 跳过: {e}")

    # Phase 4: 生产包
    try:
        print("\n  Phase 4: Manufacturing package...")
        pkg_path = build_production_package(
            body_data, gen, output_dir,
            include_stl=True, include_dxf=True,
        )
        results["production_package"] = pkg_path
    except Exception as e:
        print(f"    Phase 4 跳过: {e}")

    return results


def export_presentation_suite(
    body_data: dict,
    output_dir: str,
    prefix: str = "assembly",
    quality: str = "high",
    explode_distance: float = 0.12,
) -> dict:
    """Phase 3: 论文级展示套件 — glTF + PBR渲染 + 爆炸图

    Returns:
        {
            'glb': {'normal': path, 'exploded': path},
            'renders': {'paper_main': path, ...},
        }
    """
    from forgecraft.geometry.parametric import ParametricGenerator
    from forgecraft.geometry.gltf_scene import export_gltf
    from forgecraft.geometry.pbr_renderer import render_paper_suite

    gen = ParametricGenerator(quality=quality)

    gltf_dir = os.path.join(output_dir, "gltf")
    render_dir = os.path.join(output_dir, "renders")

    results = {}

    # glTF 导出
    print("    导出 glTF 2.0 (GLB)...")
    try:
        glb = export_gltf(body_data, gen, gltf_dir, prefix=prefix,
                        explode_distance=explode_distance)
        results["glb"] = glb
        print(f"      装配体: {glb.get('normal', '?')}")
        print(f"      爆炸图: {glb.get('exploded', '?')}")
    except Exception as e:
        print(f"      glTF 导出跳过: {e}")

    # PBR 论文级渲染
    print("    PBR 论文级渲染 (250 DPI)...")
    try:
        fitness = body_data.get("fitness", 0)
        renders = render_paper_suite(
            body_data, gen, render_dir, prefix=prefix,
            explode_distance=explode_distance, dpi=250,
            title=f"ForgeCraft Assembly — Fitness {fitness:.4f}",
        )
        results["renders"] = renders
        for key, path in renders.items():
            print(f"      {key}: {path}")
    except Exception as e:
        print(f"      渲染跳过: {e}")

    return results



def export_from_evolution_loop(loop, output_dir: str = "design_output") -> dict:
    from forgecraft.core.loader import load_catalog

    if loop.best_body is None:
        raise ValueError("没有最佳形态可导出")

    body_data = {
        "name": loop.best_body.name,
        "fitness": loop.best_body.fitness,
        "fitness_components": loop.best_body.fitness_components,
        "num_parts": loop.best_body.num_parts(),
        "num_joints": loop.best_body.num_joints(),
        "parts": [p.to_dict() for p in loop.best_body.parts()],
        "joints": [j.to_dict() for j in loop.best_body.joints()],
    }

    catalog_specs = {}
    for pn, ps in loop.catalog.items():
        spec_dict = {
            "geometry": ps.shape,
            "color": ps.color,
            "mass": ps.mass,
            "actuated": ps.can_actuate,
        }
        catalog_specs[pn] = spec_dict

    policy_state = loop.best_trainer_state
    obs_dim, act_dim = 0, 0
    if policy_state is not None:
        try:
            from forgecraft.rl.env import ForgeCraftEnv
            env = ForgeCraftEnv(loop.best_body, loop.sim_config, loop.task_config, catalog=loop.catalog)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            env.close()
        except Exception:
            policy_state = None

    return export_all(
        body_data, catalog_specs, output_dir,
        policy_state=policy_state,
        obs_dim=obs_dim,
        act_dim=act_dim,
        morph_dim=loop.rl_config.morph_embed_dim,
    )


# ================================================================
# CLI
# ================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="ForgeCraft 制造导出")
    parser.add_argument("--body-file", type=str, required=True, help="进化结果 JSON (best_body.json)")
    parser.add_argument("--catalog", type=str, default="default", help="零件箱名称")
    parser.add_argument("--output", type=str, default="design_output", help="输出目录")
    parser.add_argument("--policy-file", type=str, default=None, help="策略权重文件 (.pt)")

    args = parser.parse_args()

    with open(args.body_file, "r", encoding="utf-8") as f:
        body_data = json.load(f)

    from forgecraft.core.loader import load_catalog
    catalog = load_catalog(name=args.catalog)
    catalog_specs = {}
    for pn, ps in catalog.parts.items():
        catalog_specs[pn] = ps.to_dict()

    policy_state = None
    obs_dim, act_dim = 0, 0
    if args.policy_file and os.path.exists(args.policy_file):
        import torch
        policy_state = torch.load(args.policy_file, map_location="cpu", weights_only=True)
        obs_dim = body_data.get("obs_dim", 10)
        act_dim = body_data.get("act_dim", 2)

    results = export_all(
        body_data, catalog_specs, args.output,
        policy_state=policy_state,
        obs_dim=obs_dim,
        act_dim=act_dim,
    )

    print("=" * 50)
    print("  ForgeCraft 制造导出完成")
    print("=" * 50)
    print(f"  输出目录: {results['output_dir']}")
    print(f"  制造评级: {results['grade']}")
    print()
    print("  导出文件:")
    for key, path in results.items():
        if key in ("manufacturability_report", "output_dir", "grade"):
            continue
        if isinstance(path, dict):
            for sub, spath in path.items():
                print(f"    stl/{sub}: {spath}")
        elif path:
            print(f"    {key}: {path}")

    report = results["manufacturability_report"]
    print(f"\n  可制造性报告:")
    print(f"    综合评分: {report['overall_score']:.2f} ({report['grade']})")
    print(f"    重心高度: {report['com_height']:.4f}m")
    for d in report.get("detail", []):
        if d["issues"]:
            print(f"    {d['part_type']}({d['part_id']}): {', '.join(d['issues'])}")


if __name__ == "__main__":
    main()
# ══════════════════════════════════════
# ══════════════════════════════════════════════
# 公共 API 导出
# ══════════════════════════════════════════════

from forgecraft.manufacturing.realtime_feedback import (
    RealtimeManufacturability,
    WallThicknessCheck,
    OverhangCheck,
    InterferenceCheck,
    JointStressCheck,
    MaterialEfficiencyCheck,
    ManufacturingCheck,
)
from forgecraft.manufacturing.onnx_deploy import (
    export_onnx_verified,
    verify_onnx_inference_speed,
)

__all__ = [
    "export_stl",
    "export_urdf",
    "export_onnx",
    "export_c_template",
    "export_bom",
    "export_all",
    "export_all_enhanced",
    "export_from_evolution_loop",
    "compute_manufacturability_report",
    # 实时制造性反馈
    "RealtimeManufacturability",
    "WallThicknessCheck",
    "OverhangCheck",
    "InterferenceCheck",
    "JointStressCheck",
    "MaterialEfficiencyCheck",
    "ManufacturingCheck",
    # ONNX 部署
    "export_onnx_verified",
    "verify_onnx_inference_speed",

    # Phase 3: PBR + glTF
    "export_presentation_suite",

    # Phase 4: 加工厂就绪
    "export_step",
    "export_step_parts",
    "generate_engineering_drawing",
    "export_dxf",
    "check_interferences",
    "generate_interference_report",
    "build_production_package",
]
