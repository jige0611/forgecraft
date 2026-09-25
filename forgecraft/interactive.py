"""
ForgeCraft 交互式 CLI

你可以用自然语言描述你想要什么样的机器人，系统会：
1. 自动匹配零件箱和任务目标
2. 让你确认和调整参数
3. 启动强化学习进化，找出最优机械形态

用法：
  python -m forgecraft.interactive                            # 交互式菜单
  python -m forgecraft.interactive --say "用轮子和马达做一个能跑的"   # 自然语言
  python -m forgecraft.interactive --say "六足爬行机器人"            # 自然语言
"""

__all__ = [
    "parse_natural_language",
    "part_display_name",
    "task_display_name",
    "build_custom_catalog_yaml",
    "run_interactive",
    "NL_PART_MAP",
    "NL_TASK_MAP",
]

# 自然语言 → 零件映射
NL_PART_MAP = {
    # 结构件
    "底座": "base", "底盘": "chassis", "底板": "base", "基座": "base",
    "车身": "chassis", "躯干": "body", "主体": "core", "核心": "core",
    "方块": "brick", "积木": "brick", "模块": "brick",

    # 连杆
    "杆": "link", "连杆": "link", "连接杆": "link", "棒": "link",
    "大腿": "upper_leg", "小腿": "lower_leg", "腿段": "limb_segment",
    "股": "upper_leg", "胫": "lower_leg",

    # 驱动关节
    "马达": "hinge_joint", "电机": "hinge_joint", "关节": "hinge_joint",
    "转动": "hinge_joint", "铰链": "hinge_joint",
    "髋关节": "hip", "胯": "hip", "髋": "hip",
    "膝关节": "knee", "膝盖": "knee", "膝": "knee",
    "髋马达": "hip_motor", "膝马达": "knee_motor",

    # 末端
    "脚": "foot", "足": "foot_pad", "脚掌": "foot_pad", "蹄": "foot",
    "轮子": "wheel", "车轮": "wheel", "轮": "wheel",
    "驱动轮": "wheel_drive", "主动轮": "wheel_drive",

    # 全零件库（直接用预设名）
    "robogrammar": "robogrammar", "ariel": "ariel",
    "walking": "walking", "primitives": "primitives",
}

# 自然语言 → 任务映射
NL_TASK_MAP = {
    "跑": "speed", "快": "speed", "速度": "speed",
    "加速": "speed", "冲刺": "speed", "竞速": "speed",
    "移动": "speed", "前进": "speed", "位移": "speed",
    "走路": "speed", "行走": "speed", "走": "speed",

    "爬": "climb", "爬升": "climb", "上升": "climb",
    "攀爬": "climb", "登高": "climb", "向上": "climb",
    "爬楼": "climb", "爬坡": "climb",

    "节能": "efficiency", "效率": "efficiency", "省电": "efficiency",
    "能耗": "efficiency", "省力": "efficiency", "经济": "efficiency",

    "多目标": "multi_objective", "平衡": "multi_objective",
    "综合": "multi_objective", "全能": "multi_objective",
    "帕累托": "multi_objective",

    "承重": "structure", "强度": "structure", "结构": "structure",
    "刚度": "structure", "支撑": "structure", "桥梁": "structure",
    "桁架": "structure", "支架": "structure",

    "轻量化": "structure", "轻量": "structure", "减重": "structure",
    "刚度比": "structure", "承载比": "structure",
}

# 自然语言 → 自定义指标
NL_METRIC_MAP = {
    "承重": "height", "强度": "height", "承载": "height",
    "高度": "height", "支撑力": "height",
    "刚": "height", "硬": "height",
    "轻": "manufacturability", "省材料": "manufacturability",
    "减重": "manufacturability", "用料": "manufacturability",
}

# 自然语言 → 数量/属性映射
NL_CONSTRAINT = {
    "四足": ("walking", 4), "四条腿": ("walking", 4),
    "六足": ("walking", 6), "六条腿": ("walking", 6),
    "八足": ("walking", 8), "八条腿": ("walking", 8),
    "蜘蛛": ("walking", 8), "蜘蛛": ("walking", 8), "多足": ("walking", 6),
    "双足": ("walking", 2), "两条腿": ("walking", 2),
    "两轮": (None, 2), "三轮": (None, 3),
    "履带": (None, 0),
}


def parse_natural_language(text: str) -> dict:
    text_lower = text.lower()

    task = None
    for kw, tk in sorted(NL_TASK_MAP.items(), key=lambda x: -len(x[0])):
        if kw in text_lower:
            task = tk
            break

    is_full_catalog = False
    for kw in ["primitives", "robogrammar", "ariel", "walking"]:
        if kw in text_lower:
            is_full_catalog = True
            catalog_name = kw
            part_names = []
            break

    if not is_full_catalog:
        catalog_name = "custom"
        part_names = []
        for kw, pt in sorted(NL_PART_MAP.items(), key=lambda x: -len(x[0])):
            if kw in text_lower and pt not in part_names:
                part_names.append(pt)

        is_walking_context = any(w in text_lower for w in ["四足", "六足", "八足", "腿", "蜘蛛"])
        if is_walking_context and not is_full_catalog and len(part_names) <= 2:
            part_names = ["main_body", "hip_motor", "knee_motor", "upper_leg", "lower_leg", "foot_pad", "wheel_drive"]
            catalog_name = "walking"

    constraint = None
    for kw, cc in sorted(NL_CONSTRAINT.items(), key=lambda x: -len(x[0])):
        if kw in text_lower:
            constraint = cc
            break

    metrics = []
    for kw, metric in sorted(NL_METRIC_MAP.items(), key=lambda x: -len(x[0])):
        if kw in text_lower and metric not in metrics:
            metrics.append(metric)

    return {
        "catalog_name": catalog_name,
        "parts": part_names,
        "task": task or "speed",
        "constraint": constraint,
        "metrics": metrics,
    }


PART_EMOJI = {
    "base": "🟦", "chassis": "🟦", "body": "🟦", "core": "🟦", "main_body": "🟦",
    "link": "🟩", "limb_segment": "🟩", "upper_leg": "🟩", "lower_leg": "🟩",
    "torso_segment": "🟩",
    "hinge_joint": "⚙️", "hip": "⚙️", "knee": "⚙️",
    "hip_joint": "⚙️", "hip_motor": "⚙️", "knee_motor": "⚙️",
    "hinge": "⚙️",
    "foot": "🥾", "leg_foot": "🥾", "foot_pad": "🥾",
    "wheel": "🛞", "robot_wheel": "🛞", "wheel_drive": "🛞",
    "brick": "🧱",
}

TASK_EMOJI = {
    "speed": "🏃", "climb": "🧗", "efficiency": "⚡", "multi_objective": "🎯",
    "structure": "🏗️",
}

PART_LABELS = {
    "base": "底座", "chassis": "底盘", "body": "躯干", "core": "核心模块",
    "main_body": "主机体",
    "link": "连杆", "limb_segment": "肢体段", "upper_leg": "大腿段",
    "lower_leg": "小腿段", "torso_segment": "躯干段",
    "hinge_joint": "铰链关节", "hip": "髋关节", "knee": "膝关节",
    "hip_joint": "髋关节", "hip_motor": "髋马达", "knee_motor": "膝马达",
    "hinge": "铰链",
    "foot": "脚", "leg_foot": "脚", "foot_pad": "脚掌",
    "wheel": "轮子", "robot_wheel": "轮子", "wheel_drive": "驱动轮",
    "brick": "积木块",
}

TASK_LABELS = {
    "speed": "速度最快", "climb": "爬升最高", "efficiency": "最节能",
    "multi_objective": "综合最优", "structure": "结构最强",
}

TASK_DESC = {
    "speed": "最大化水平位移速度",
    "climb": "最大化垂直高度",
    "efficiency": "最大化位移/能耗比",
    "multi_objective": "多目标帕累托优化",
    "structure": "最大化承载自重比",
}


def _colored(text: str, code: str = "96") -> str:
    try:
        return f"\033[{code}m{text}\033[0m"
    except Exception:
        return text


def part_display_name(name: str) -> str:
    emoji = PART_EMOJI.get(name, "  ")
    label = PART_LABELS.get(name, name)
    return f"  {emoji} {label:8s} ({name})"


def task_display_name(name: str) -> str:
    emoji = TASK_EMOJI.get(name, "  ")
    label = TASK_LABELS.get(name, name)
    desc = TASK_DESC.get(name, "")
    return f"  {emoji} {label:10s} - {desc}"


def build_custom_catalog_yaml(part_names: list, name: str = "custom") -> str:
    from forgecraft.core.loader import load_catalog
    all_name_to_spec = {}

    for cat_name in ["primitives", "robogrammar", "ariel", "walking"]:
        try:
            cat = load_catalog(name=cat_name)
            for pn, ps in cat.parts.items():
                if pn not in all_name_to_spec:
                    all_name_to_spec[pn] = ps.to_dict()
        except Exception:
            continue

    lines = [f"# 自定义零件箱: {name}"]
    lines.append("")
    lines.append("constraints:")
    lines.append("  max_depth: 5")
    lines.append("  max_children_per_node: 4")
    lines.append("  require_root: true")
    lines.append("  symmetry: none")
    lines.append("  max_parts: 25")
    lines.append("")
    lines.append("parts:")

    for pn in part_names:
        spec = all_name_to_spec.get(pn, {})
        lines.append(f"  {pn}:")
        geo = spec.get("geometry", "box")
        if isinstance(geo, str):
            lines.append(f"    geometry:")
            lines.append(f"      type: {geo}")
        color = spec.get("color", [0.5, 0.5, 0.5, 1.0])
        lines.append(f"      color: {color}")
        lines.append(f"    physics:")
        lines.append(f"      mass: {spec.get('mass', 0.1)}")
        friction = spec.get("friction", [0.6, 0.01, 0.01])
        lines.append(f"      friction: {friction}")
        size = spec.get("size", {})
        lines.append(f"    size:")
        for sk, sv in size.items():
            lines.append(f"      {sk}: {sv}")
        lines.append(f"    actuated: {'true' if spec.get('actuated') else 'false'}")
        joint = spec.get("joint")
        if joint and isinstance(joint, dict):
            lines.append(f"    joint:")
            lines.append(f"      type: {joint.get('type', 'hinge')}")
            lines.append(f"      axis: {joint.get('axis', [0, 1, 0])}")
            lines.append(f"      range: {joint.get('range')}")
            lines.append(f"      torque: {joint.get('torque', [0.5, 5.0])}")
            lines.append(f"      velocity: {joint.get('velocity', [5, 15])}")
            lines.append(f"      damping: {joint.get('damping', 0.5)}")
        faces = spec.get("faces", ["front"])
        lines.append(f"    faces: {faces}")

    return "\n".join(lines)


def run_interactive(nl_text: str = "", auto_confirm: bool = False):
    from forgecraft.core.loader import load_catalog, load_task, list_catalogs, list_tasks
    import torch

    print("\n" + "=" * 62)
    print(_colored("  🔧 ForgeCraft 交互式机械形态进化系统", "95"))
    print(_colored("  「你说要什么，我帮你找到最优形态」", "90"))
    print("=" * 62)

    catalog_name = None
    selected_parts = []
    selected_task = "speed"

    if nl_text:
        parsed = parse_natural_language(nl_text)
        catalog_name = parsed["catalog_name"]
        selected_parts = parsed["parts"]
        selected_task = parsed["task"]
        constraint = parsed.get("constraint")

        print(f"\n  📝 我理解你要的是:")
        print(f"     目标: {task_display_name(selected_task)}")

        if catalog_name in [c for c in list_catalogs() if c != "custom"]:
            cat = load_catalog(name=catalog_name)
            print(f"     零件箱: {catalog_name} ({len(cat.parts)} 种零件)")
            for name, spec in cat.parts.items():
                print(part_display_name(name))
        elif selected_parts:
            print(f"     零件 ({len(selected_parts)} 种):")
            for pn in selected_parts:
                print(part_display_name(pn))
        else:
            print(f"     零件箱: default (请从菜单选择)")

    if not nl_text or not selected_parts and catalog_name not in [c for c in list_catalogs() if c != "custom"]:
        if auto_confirm:
            if catalog_name == "custom" or not catalog_name:
                catalog_name = "primitives"
        else:
            _catalog_interactive_prompt(catalog_name, selected_parts)

    if not nl_text and not auto_confirm:
        _task_interactive_prompt(selected_task)

    if auto_confirm:
        generations, population, workers, use_cuda = 10, 8, 4, torch.cuda.is_available()
    else:
        print("\n  ── 第3步: 进化参数 ──")
        generations = _ask_int("  进化代数", 10)
        population = _ask_int("  种群大小", 8)
        workers = _ask_int("  并行进程数", 4)
        use_cuda = _ask_bool("  启用 GPU 加速 (RTX 5060)" if torch.cuda.is_available() else "  启用GPU加速(不可用)")

    print("\n" + "=" * 62)
    print(_colored("  🚀 确认配置，启动进化", "92"))
    print("=" * 62)
    print(f"  零件箱: {catalog_name}")
    print(f"  目标: {selected_task}")
    print(f"  代数: {generations} | 种群: {population} | 进程: {workers}")
    print(f"  GPU: {'是' if use_cuda else '否'}")
    print("-" * 62)

    confirm = "y" if auto_confirm else input("\n  开始进化? [Y/n]: ").strip().lower()
    if confirm and confirm not in ("y", "yes", ""):
        print("  已取消。")
        return

    _run_evolution(
        catalog_name=catalog_name,
        selected_parts=selected_parts,
        task_name=selected_task,
        generations=generations,
        population=population,
        workers=workers,
        use_cuda=use_cuda,
    )


def _catalog_interactive_prompt(current_name, selected_parts):
    from forgecraft.core.loader import load_catalog, list_catalogs
    print("\n  ── 第1步: 选择零件箱 ──")
    print("  可用的预设零件箱（你也可以自选零件组装）:")
    print()

    catalogs = list_catalogs()
    for i, cn in enumerate(catalogs):
        cat = load_catalog(name=cn)
        part_list = ", ".join(PART_LABELS.get(p, p) for p in cat.parts)
        print(f"  [{i+1}] {cn:16s} ({len(cat.parts)} 种: {part_list})")

    print("\n  [0] 自选零件（自由组合）")
    print("  [n] 自然语言描述")

    choice = input("\n  请选择 (默认=1): ").strip()
    if not choice:
        choice = "1"

    if choice == "0":
        current_name = "custom"
        selected_parts[:] = _interactive_part_selection()
        return current_name
    elif choice == "n":
        nl_text = input("\n  请描述你想要的机器人: ").strip()
        if nl_text:
            parsed = parse_natural_language(nl_text)
            current_name = parsed["catalog_name"]
            selected_parts[:] = parsed["parts"]
            return current_name
        return "primitives"
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(catalogs):
                return catalogs[idx]
        except ValueError:
            pass
    return "primitives"


def _task_interactive_prompt(current_task):
    from forgecraft.core.loader import list_tasks
    print("\n  ── 第2步: 选择目标 ──")
    print()
    tasks = list_tasks()
    for i, tn in enumerate(tasks):
        print(f"  [{i+1}] {task_display_name(tn)}")
    choice = input("\n  请选择 (默认=1, 速度): ").strip()
    if not choice:
        return "speed"
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(tasks):
            return tasks[idx]
    except ValueError:
        pass
    return "speed"


def _interactive_part_selection() -> list:
    from forgecraft.core.loader import load_catalog

    all_specs = {}
    seen = set()
    for cn in ["primitives", "robogrammar", "ariel", "walking"]:
        try:
            cat = load_catalog(name=cn)
            for pn, ps in cat.parts.items():
                if pn not in seen:
                    seen.add(pn)
                    all_specs[pn] = (cn, ps)
        except Exception:
            continue

    items = list(all_specs.items())
    for i, (pn, (src, ps)) in enumerate(items):
        act = "驱动" if ps.actuated else "固定"
        print(part_display_name(pn))

    print("\n  输入零件名（空格分隔），或 'all' 全选，或回车跳过：")
    print("  提示: 至少需要一个结构件(root)和一个驱动件(actuated)")
    sel = input("  > ").strip()
    if not sel:
        return list(all_specs.keys())[:6]
    if sel.lower() == "all":
        return list(all_specs.keys())
    return [s.strip() for s in sel.split() if s.strip() in all_specs]


def _ask_int(prompt: str, default: int) -> int:
    raw = input(f"  {prompt} (默认={default}): ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _ask_bool(prompt: str) -> bool:
    raw = input(f"  {prompt}? [y/N]: ").strip().lower()
    return raw in ("y", "yes")


def _run_evolution(
    catalog_name: str,
    selected_parts: list,
    task_name: str,
    generations: int,
    population: int,
    workers: int,
    use_cuda: bool,
):
    import json
    import os
    import tempfile
    import time

    import torch

    from forgecraft.core.loader import load_catalog, load_task
    from forgecraft.config import EvolutionConfig, RLConfig

    task_spec = load_task(name=task_name)
    sim_config = task_spec.to_sim_config()
    task_config = task_spec.to_task_config()

    if catalog_name == "custom" and selected_parts:
        yaml_content = build_custom_catalog_yaml(selected_parts, name="custom")
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        tmp.write(yaml_content)
        tmp.close()
        catalog = load_catalog(path=tmp.name)
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    else:
        catalog = load_catalog(name=catalog_name)

    has_root = any(not ps.actuated for ps in catalog.parts.values())
    has_actuator = any(ps.actuated for ps in catalog.parts.values())
    if not has_root:
        print("\n  ⚠️ 警告: 零件箱没有结构件（root），进化可能失败！")
    if not has_actuator:
        print("\n  ⚠️ 警告: 零件箱没有驱动件，机器人无法运动！")

    evo_config = EvolutionConfig(
        population_size=population,
        generations=generations,
        elite_count=max(2, population // 5),
    )

    rl_config = RLConfig(hidden_dim=64, gnn_hidden=32, morph_embed_dim=32)

    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    if use_cuda and device != "cuda":
        print("  警告: CUDA 不可用，退回 CPU 模式")
    if device == "cuda":
        print(f"\n  🖥️ GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n  零件箱: {catalog.name} ({len(catalog.parts)} 种零件)")
    print(f"  任务: {task_spec.name}")
    print(f"  进化配置: {generations}代 × {population}个体 × {workers}进程")
    print()

    from forgecraft.evolution.loop import EvolutionLoop

    loop = EvolutionLoop(
        evo_config=evo_config,
        sim_config=sim_config,
        rl_config=rl_config,
        task_config=task_config,
        catalog=catalog.to_part_specs(),
        device=device,
        seed=int(time.time() * 1000) % 10000,
        n_workers=workers,
    )

    total_start = time.time()
    try:
        loop.run(n_generations=generations)
    except KeyboardInterrupt:
        print("\n\n  用户中断。")

    total_time = time.time() - total_start

    print("\n" + "=" * 62)
    print(_colored("  🏆 进化结果", "93"))
    print("=" * 62)

    if loop.best_body:
        body = loop.best_body
        print(f"\n  最佳形态: {body.name}")
        print(f"  适应度: {body.fitness:.4f}")
        print(f"  零件数: {body.num_parts()}")
        print(f"  驱动关节: {len(body.actuated_joints())}")
        print(f"  总耗时: {total_time:.1f}s")
        print(f"\n  {body.describe()}")

        from forgecraft.simulation.builder import build_mjcf_model
        xml_str, _, _, _ = build_mjcf_model(body, catalog.to_part_specs())
        xml_path = "best_body.xml"
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
        print(f"\n  📄 MuJoCo 模型: {xml_path}")
        print(f"  💡 用 MuJoCo 查看: python -m mujoco.viewer --mjcf={xml_path}")

        # 制造导出
        from forgecraft.manufacturing import export_from_evolution_loop
        try:
            mfg_results = export_from_evolution_loop(loop, "design_output")
            print(f"\n  🏭 制造文件已导出: design_output/")
            print(f"     评级: {mfg_results['grade']}")
            print(f"     STL×{len(mfg_results['stl'])} | URDF | BOM")
            if mfg_results['onnx']:
                print(f"     ONNX策略 | C++推理模板")
        except Exception as e:
            print(f"\n  ⚠️ 制造导出失败: {e}")

        # Web 仪表盘
        print(f"\n  🌐 启动 Web 仪表盘查看结果...")
        print(f"  💡 运行: python -m forgecraft.dashboard --port 8080")
        print(f"  💡 然后打开: http://localhost:8080")

    print("\n" + "=" * 62)


def main():
    import sys

    auto = False
    if len(sys.argv) > 1 and sys.argv[1] == "--say":
        nl_text = sys.argv[2] if len(sys.argv) > 2 else ""
        if not nl_text:
            nl_text = input("  请描述你想要的机器人: ").strip()
        auto = "--auto" in sys.argv
        if "--quick" in sys.argv:
            auto = True
            nl_text = nl_text + " 快速测试"
        run_interactive(nl_text, auto_confirm=auto)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
