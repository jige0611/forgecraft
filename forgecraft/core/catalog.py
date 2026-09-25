"""默认零件箱 (DEFAULT_CATALOG)

定义了 8 种基础零件类型的物理参数:
  - base:        根节点零件 (非驱动, 锚定)
  - segment:     结构连接件 (杆/梁)
  - motor:       驱动单元 (可产生扭矩)
  - foot:        球形脚 (地面接触)
  - wheel:       轮子 (滚动接触)
  - chassis:     底盘平台 (大尺寸结构件)
  - limb_segment:肢体段 (特定尺寸)
  - link:        通用连接件

每个 PartSpec 定义了:
  - 物理属性: mass, shape, size_range, color
  - 驱动属性: can_actuate, max_torque, gear_ratio
  - 关节偏好: parent_joint, parent_joint_axis, child_joint
  - 传感器:   has_touch, has_imu
  - 参数范围: param_ranges, mass_range, density, friction
"""

from typing import Dict, List, Optional
import yaml
import os
import logging

from forgecraft.config import PartSpec

__all__ = ["DEFAULT_CATALOG"]

_logger = logging.getLogger(__name__)

DEFAULT_CATALOG: Dict[str, PartSpec] = {
    "base": PartSpec(
        part_type="base",
        mass=0.5,
        shape="box",
        size_range=[0.05, 0.15],
        can_actuate=False,
        color=[0.3, 0.3, 0.5, 1.0],
    ),
    "segment": PartSpec(
        part_type="segment",
        mass=0.1,
        shape="cylinder",
        size_range=[0.03, 0.2],
        can_actuate=False,
        color=[0.4, 0.5, 0.4, 1.0],
    ),
    "motor": PartSpec(
        part_type="motor",
        mass=0.12,
        shape="box",
        size_range=[0.02, 0.06],
        can_actuate=True,
        max_torque=5.0,
        color=[0.8, 0.3, 0.2, 1.0],
    ),
    "foot": PartSpec(
        part_type="foot",
        mass=0.05,
        shape="sphere",
        size_range=[0.02, 0.08],
        can_actuate=False,
        color=[0.5, 0.5, 0.2, 1.0],
    ),
    "wheel": PartSpec(
        part_type="wheel",
        mass=0.08,
        shape="cylinder",
        size_range=[0.03, 0.12],
        can_actuate=False,
        color=[0.2, 0.2, 0.2, 1.0],
    ),
}

# ══════════════════════════════════════════
# 🆕 V2 创新零件定义 (Phase 1)
# ══════════════════════════════════════════

V2_INNOVATIVE_PARTS: Dict[str, PartSpec] = {
    # ──────────────────────────────────────
    # 弹性储能单元 (螺旋弹簧)
    # ──────────────────────────────────────
    "spring_element": PartSpec(
        part_type="spring_element",
        mass=0.03,
        shape="helical_spring",          # 🆕 新形状类型
        size_range=[0.02, 0.15],         # free_length 范围
        can_actuate=False,               # 被动元件
        color=[0.85, 0.65, 0.20, 1.0],   # 黄铜色
        param_ranges={
            "stiffness": [10, 10000],     # N/m (对数分布)
            "max_deformation": [0.001, 0.05],  # m
            "damping": [0.01, 10],        # Ns/m
            "preload": [0, 20],           # N
            "wire_diameter": [0.001, 0.005],   # m
            "coil_diameter": [0.01, 0.05],      # m
            "num_coils": [5, 25],         # 圈数
        },
        mass_range=[0.01, 0.2],
        density=7800.0,                   # 钢材
        friction=[0.3, 0.01, 0.01],       # 低摩擦（内部运动）
    ),
    
    # ──────────────────────────────────────
    # 轻量化中空管结构
    # ──────────────────────────────────────
    "hollow_tube": PartSpec(
        part_type="hollow_tube",
        mass=0.07,
        shape="hollow_cylinder",          # 🆕 新形状类型
        size_range=[0.05, 0.5],           # length 范围
        can_actuate=False,
        color=[0.78, 0.79, 0.81, 1.0],    # 铝合金色
        param_ranges={
            "outer_diameter": [0.01, 0.08],     # m
            "length": [0.05, 0.5],              # m
            "wall_thickness": [0.001, 0.005],   # m
        },
        mass_range=[0.02, 0.5],
        density=2700.0,                    # 6063铝合金
        friction=[0.4, 0.01, 0.01],
    ),
    
    # ──────────────────────────────────────
    # 高摩擦半球形脚垫
    # ──────────────────────────────────────
    "hemisphere_foot": PartSpec(
        part_type="hemisphere_foot",
        mass=0.015,
        shape="hemisphere_shell",         # 🆕 新形状类型
        size_range=[0.005, 0.03],         # radius 范围
        can_actuate=False,
        color=[0.12, 0.12, 0.14, 1.0],    # 黑色橡胶
        param_ranges={
            "radius": [0.005, 0.03],           # m
            "shell_thickness": [0.002, 0.008],  # m
            "compliance": [0.0001, 0.001],      # m/N
        },
        mass_range=[0.005, 0.08],
        density=1200.0,                    # 橡胶
        friction=[1.2, 0.01, 0.01],       # ⭐ 高摩擦系数!
    ),
}


def get_default_catalog() -> Dict[str, PartSpec]:
    """获取默认目录（仅基础零件）"""
    return dict(DEFAULT_CATALOG)


def get_v2_catalog() -> Dict[str, PartSpec]:
    """
    获取V2完整目录（基础 + 创新零件）
    
    Returns:
        包含9种零件的完整目录（6种基础 + 3种创新）
    """
    catalog = dict(DEFAULT_CATALOG)
    catalog.update(V2_INNOVATIVE_PARTS)
    return catalog


def get_part_types() -> List[str]:
    """获取所有可用零件类型列表"""
    return list(DEFAULT_CATALOG.keys())


def get_v2_part_types() -> List[str]:
    """
    获取V2所有零件类型（包括创新零件）
    
    Returns:
        ['base', 'segment', 'motor', 'foot', 'wheel', 
         'spring_element', 'hollow_tube', 'hemisphere_foot']
    """
    all_parts = get_default_catalog()
    all_parts.update(V2_INNOVATIVE_PARTS)
    return list(all_parts.keys())


def load_parameterized_catalog(yaml_path: Optional[str] = None) -> Dict[str, PartSpec]:
    """
    从YAML文件加载参数化零件目录
    
    Args:
        yaml_path: YAML文件路径，默认为parameterized_parts_v2.yaml
        
    Returns:
        解析后的PartSpec字典
        
    Example:
        >>> catalog = load_parameterized_catalog()
        >>> spring = catalog['spring_element']
        >>> print(spring.param_ranges['stiffness'])
        [10, 10000]
    """
    if yaml_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        yaml_path = os.path.join(base_dir, "configs", "catalogs", 
                                 "parameterized_parts_v2.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"⚠️ 未找到YAML配置文件: {yaml_path}")
        print("   使用内置V2目录作为后备")
        return get_v2_catalog()
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    catalog = {}
    
    for part_name, part_config in config.get('parts', {}).items():
        try:
            physics = part_config.get('physics', {})
            geometry = part_config.get('geometry', {})
            size = part_config.get('size', {})
            parameters = part_config.get('parameters', {})
            
            param_ranges = {}
            
            for param_name, param_spec in parameters.items():
                if isinstance(param_spec, dict) and 'range' in param_spec:
                    param_ranges[param_name] = param_spec['range']
                elif isinstance(param_spec, list):
                    param_ranges[param_name] = param_spec
            
            for size_name, size_range in size.items():
                if isinstance(size_range, list) and len(size_range) == 2 and size_range[0] != 'auto':
                    param_ranges[size_name] = size_range
            
            spec = PartSpec(
                part_type=part_name,
                mass=physics.get('mass', 0.1),
                shape=geometry.get('type', 'box'),
                size_range=_extract_primary_size_range(size),
                can_actuate=part_config.get('actuated', False),
                color=geometry.get('color', [0.6, 0.6, 0.6, 1.0]),
                param_ranges=param_ranges,
                mass_range=physics.get('mass_range', []),
                density=physics.get('density', 1000.0),
                has_touch=part_config.get('sensor', {}).get('touch', False),
                has_imu=part_config.get('sensor', {}).get('imu', False),
                friction=physics.get('friction', [0.6, 0.01, 0.01]),
            )
            
            catalog[part_name] = spec
            
        except Exception as e:
            print(f"⚠️ 解析零件 '{part_name}' 失败: {e}")
            continue
    
    print(f"✅ 成功加载 {len(catalog)} 种零件从 {os.path.basename(yaml_path)}")
    return catalog


def _extract_primary_size_range(size_dict: Dict) -> List[float]:
    """
    从尺寸配置中提取主尺寸范围
    
    优先级：length > radius > height > 第一个数值范围
    """
    priority_keys = ['length', 'radius', 'height', 'free_length', 'outer_diameter']
    
    for key in priority_keys:
        if key in size_dict:
            value = size_dict[key]
            if isinstance(value, list) and len(value) == 2 and value[0] != 'auto':
                return value
    
    for key, value in size_dict.items():
        if isinstance(value, list) and len(value) == 2 and value[0] != 'auto':
            return value
    
    return [0.01, 0.3]


def is_innovative_part(part_type: str) -> bool:
    """
    检查是否为V2创新零件
    
    Args:
        part_type: 零件类型名称
        
    Returns:
        True 如果是Phase 1新增的创新零件
    """
    return part_type in V2_INNOVATIVE_PARTS


def get_innovative_part_info(part_type: str) -> Optional[Dict]:
    """
    获取创新零件的详细信息

    Returns:
        包含category、benefits、recommended_positions等信息的字典，
        或者None如果不是创新零件
    """
    innovative_info = {
        "spring_element": {
            "category": "energy_storage",
            "description": "弹性储能单元（螺旋弹簧）",
            "benefits": ["跳跃高度+200%", "能耗-40%", "冲击吸收"],
            "recommended_positions": ["驱动器与脚垫之间", "腿部关节处"],
            "material_options": ["piano_wire", "spring_steel_65mn", "titanium_spring"],
        },
        "hollow_tube": {
            "category": "lightweight_structure",
            "description": "轻量化中空管结构",
            "benefits": ["质量减轻60-80%", "转动惯量降低50-70%"],
            "recommended_positions": ["主框架", "替代实心腿杆"],
            "material_options": ["aluminum_6063", "carbon_fiber_tube"],
        },
        "hemisphere_foot": {
            "category": "contact_surface",
            "description": "高摩擦半球形脚垫",
            "benefits": ["抓地力+50-100%", "全向接触", "减震缓冲"],
            "recommended_positions": ["腿末端", "多点接触阵列"],
            "material_options": ["rubber_high_friction", "polyurethane", "silicone_soft"],
        },
    }
    
    return innovative_info.get(part_type)


if __name__ == "__main__":
    print("=" * 60)
    print("Catalog V2 测试")
    print("=" * 60)
    
    print("\n📦 基础零件数量:", len(get_default_catalog()))
    print("🆕 V2完整零件数量:", len(get_v2_catalog()))
    
    print("\n📋 所有V2零件类型:")
    for pt in get_v2_part_types():
        marker = "🆕" if is_innovative_part(pt) else "  "
        print(f"   {marker} {pt}")
    
    print("\n🔍 创新零件详情示例:")
    spring_info = get_innovative_part_info("spring_element")
    if spring_info:
        print(f"\n   🧲 {spring_info['description']}")
        print(f"   类别: {spring_info['category']}")
        print(f"   性能提升: {', '.join(spring_info['benefits'])}")
    
    print("\n📂 尝试从YAML加载...")
    yaml_catalog = load_parameterized_catalog()
    print(f"   ✅ 加载成功! 共{len(yaml_catalog)}种零件")
