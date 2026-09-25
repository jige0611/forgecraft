"""ForgeCraft CAD — 参数化机械几何引擎

将进化产生的抽象 Part 树转换为具有真实机械特征的 3D 装配体：
- 电机 → 圆筒体 + 轴 + 安装法兰 + 螺栓孔
- 轴承 → 外圈 + 内圈 + 滚道
- 底盘 → 带安装面的异形板
- 弹簧 → 螺旋线圈
- 管材 → 空心圆柱 + 端盖

通过装配求解器根据关节类型生成配合面（铰链→轴承座，固定→法兰对）。
"""

from .features import (
    create_bolt_hole,
    create_flange,
    create_bearing_seat,
    create_shaft,
    create_counterbore,
    create_threaded_hole,
    create_clearance_hole,
)

from .primitives import (
    generate_brushless_motor,
    generate_brushless_motor_compact,
    generate_micro_bearing,
    generate_carbon_tube,
    generate_alloy_chassis,
    generate_launch_spring,
    generate_sprint_foot,
    generate_battery,
    MECHANICAL_PART_GENERATORS,
)

from .assembler import (
    AssemblySolver,
    assemble_mechanical_body,
    assemble_and_export,
)

from .pipeline import CADPipeline

__all__ = [
    # Features
    "create_bolt_hole", "create_flange", "create_bearing_seat",
    "create_shaft", "create_counterbore", "create_threaded_hole",
    "create_clearance_hole",
    # Primitives
    "generate_brushless_motor", "generate_brushless_motor_compact",
    "generate_micro_bearing", "generate_carbon_tube",
    "generate_alloy_chassis", "generate_launch_spring",
    "generate_sprint_foot", "generate_battery",
    "MECHANICAL_PART_GENERATORS",
    # Assembly
    "AssemblySolver", "assemble_mechanical_body", "assemble_and_export",
    # Pipeline
    "CADPipeline",
]
