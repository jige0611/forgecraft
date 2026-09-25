"""
ForgeCraft YAML 配置管理

可用零件箱 (catalogs):
  primitives      — 3 类原语 (structure/actuator/contact)，零人类设计偏见
  default         — 经典机器人零件箱 (base/chassis/link/limb/wheel/foot)
  parameterized   — 连续参数空间 (4 模板: 连杆/执行器/箱体/足端)
  advanced        — 高级物理组件 (惯性轮/弹性关节/柔性连接等 8 种)
  biped           — 双足机器人 (STRIDE 风格)
  hexapod         — 六足机器人 (昆虫形态)
  ariel           — ARIEL 模块化框架
  trot            — TROT 可配置腿 (UMich EMBiRLab)
  stanford_pupper — Stanford Pupper 开源四足
  robogrammar     — RoboGrammar 图语法
  walking         — 多足行走专用

历史版本 (保留用于复现):
  parameterized_parts_v2 ~ v8 — 参数化零件箱版本迭代

可用任务 (tasks):
  speed           — 最大化前向位移 (默认)
  climb           — 最大化垂直高度
  efficiency      — 最大化位移/能耗比
  multi_objective — NSGA-II 多目标帕累托优化
  structure       — 承载自重比优化
  unconventional  — 反常识运动发现
  speed_density   — 速度密度联合优化
"""

import os
import yaml
from pathlib import Path

_CATALOG_DIR = Path(__file__).parent / "catalogs"
_TASK_DIR = Path(__file__).parent / "tasks"

def validate_all_configs():
    """验证所有 YAML 配置文件可正常解析"""
    errors = []
    for d, label in [(_CATALOG_DIR, "catalog"), (_TASK_DIR, "task")]:
        for f in sorted(d.glob("*.yaml")):
            try:
                with open(f, encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                name = data.get("name", f.stem)
            except Exception as e:
                errors.append(f"{label}/{f.name}: {e}")
    return errors


def list_configs():
    """列出所有可用配置"""
    catalogs = sorted([p.stem for p in _CATALOG_DIR.glob("*.yaml")])
    tasks = sorted([p.stem for p in _TASK_DIR.glob("*.yaml")])
    return {"catalogs": catalogs, "tasks": tasks}


if __name__ == "__main__":
    errs = validate_all_configs()
    if errs:
        print("YAML validation errors:")
        for e in errs:
            print(f"  - {e}")
    else:
        print("All YAML configs valid.")
    print()
    for k, v in list_configs().items():
        print(f"{k} ({len(v)}): {', '.join(v)}")
