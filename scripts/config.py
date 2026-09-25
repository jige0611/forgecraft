# ═══════════════════════════════════════════════════════════════
#  ForgeCraft 统一配置管理系统
#  ═══════════════════════════════════════════════════════════════
#
#  功能：
#  - 单一配置源（YAML/JSON/环境变量）
#  - 分层配置：默认 → 文件 → 环境变量 → 命令行参数
#  - 类型验证和范围检查
#  - 配置导出和序列化
#
#  使用：
#    from config import get_config, ConfigManager
#    cfg = get_config()
#    print(cfg.evolution.total_generations)
# ═══════════════════════════════════════════════════════════════

import os
import json
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List

logger = logging.getLogger("ForgeCraft.Config")


@dataclass
class EvolutionConfig:
    """进化算法配置"""
    total_generations: int = 100
    population_size: int = 40
    eval_episodes: int = 6
    sim_steps: int = 3000
    settling_steps: int = 200
    grounding_steps: int = 500
    
    # 选择和遗传
    tournament_size: int = 5
    elite_count: int = 5
    crossover_rate: float = 0.3
    mutation_rate: float = 0.8
    
    # 早停
    patience: int = 20
    min_improvement: float = 0.001


@dataclass
class SimulationConfig:
    """物理仿真配置"""
    dt: float = 0.002           # 时间步长 (s)
    gravity: tuple = (-9.81, 0, 0)  # 重力方向
    ground_friction: float = 1.0   # 地面摩擦系数
    
    # MuJoCo设置
    solver_iterations: int = 100
    solver_tolerance: float = 1e-6
    
    # 渲染
    render_width: int = 640
    render_height: int = 480
    render_offscreen: bool = False


@dataclass 
class ViewerConfig:
    """查看器配置"""
    auto_drive: bool = True
    drive_speed: float = 0.8
    show_hud: bool = True
    hud_refresh_rate: float = 10.0  # Hz
    camera_distance: float = 2.0
    camera_elevation: float = -30.0


@dataclass
class DirectDriveConfig:
    """直接驱动评估配置"""
    sim_steps: int = 1500
    n_frequencies: int = 3
    episodes_per_freq: int = 2
    
    # 驱动信号
    signal_amplitude: float = 1.0
    signal_frequencies: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    
    # 评分权重
    weight_displacement: float = 0.4
    weight_speed: float = 0.3
    weight_survival: float = 0.3


@dataclass
class TemplateConfig:
    """模板生成器配置"""
    seed: int = 42
    template_weights: Dict[str, float] = field(default_factory=lambda: {
        "differential_wheeled": 0.35,
        "quad_wheeled": 0.25,
        "bipedal": 0.15,
        "quadruped": 0.15,
        "crawler": 0.10,
    })
    min_parts: int = 4
    max_parts: int = 12
    size_variance: float = 0.15


@dataclass
class OutputConfig:
    """输出配置"""
    results_dir: str = "v11_results"
    log_level: str = "INFO"
    save_checkpoints: bool = True
    checkpoint_interval: int = 10  # 每N代保存
    export_format: str = "stl"     # stl, urdf, json


@dataclass
class ForgeCraftConfig:
    """ForgeCraft主配置类"""
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    direct_drive: DirectDriveConfig = field(default_factory=DirectDriveConfig)
    templates: TemplateConfig = field(default_factory=TemplateConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


class ConfigManager:
    """
    统一配置管理器
    
    支持分层配置加载：
    1. 默认值（代码中的默认）
    2. 配置文件（forgecraft.yaml / forgecraft.json）
    3. 环境变量（FORGECRAFT_前缀）
    4. 命令行参数（通过update方法）
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: ForgeCraftConfig = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._config = ForgeCraftConfig()
        self._config_file: Optional[Path] = None
    
    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @property
    def config(self) -> ForgeCraftConfig:
        return self._config
    
    def load_file(self, path: str | Path) -> 'ConfigManager':
        """从文件加载配置"""
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return self
        
        try:
            if path.suffix in ['.yaml', '.yml']:
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
            elif path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                logger.warning(f"Unsupported config format: {path.suffix}")
                return self
            
            self._apply_dict(data)
            self._config_file = path
            logger.info(f"Loaded config from: {path}")
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
        
        return self
    
    def load_env(self) -> 'ConfigManager':
        """从环境变量加载配置"""
        env_mapping = {
            # Evolution
            'FORGECRAFT_GENS': ('evolution', 'total_generations', int),
            'FORGECRAFT_POP': ('evolution', 'population_size', int),
            'FORGECRAFT_ELITE': ('evolution', 'elite_count', int),
            'FORGECRAFT_MUTATION': ('evolution', 'mutation_rate', float),
            
            # Simulation
            'FORGECRAFT_DT': ('simulation', 'dt', float),
            'FORGECRAFT_GRAVITY': ('simulation', 'gravity', str),  # 特殊处理
            
            # Viewer
            'FORGECRAFT_NO_HUD': ('viewer', 'show_hud', lambda x: not x.lower() == 'true'),
            
            # Output
            'FORGECRAFT_OUTPUT': ('output', 'results_dir', str),
            'FORGECRAFT_LOG_LEVEL': ('output', 'log_level', str),
        }
        
        for env_var, (section, attr, converter) in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    parsed_value = converter(value)
                    section_obj = getattr(self._config, section)
                    setattr(section_obj, attr, parsed_value)
                    logger.debug(f"Env override: {env_var}={parsed_value}")
                except Exception as e:
                    logger.warning(f"Invalid env var {env_var}={value}: {e}")
        
        return self
    
    def update(self, **kwargs) -> 'ConfigManager':
        """更新配置（命令行参数等）"""
        for key, value in kwargs.items():
            if '.' in key:
                section, attr = key.split('.', 1)
                if hasattr(self._config, section):
                    section_obj = getattr(self._config, section)
                    if hasattr(section_obj, attr):
                        setattr(section_obj, attr, value)
                        continue
            # 直接属性
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        return self
    
    def _apply_dict(self, data: Dict[str, Any]) -> None:
        """应用字典数据到配置"""
        for section_name, section_data in data.items():
            if isinstance(section_data, dict) and hasattr(self._config, section_name):
                section_obj = getattr(self._config, section_name)
                for attr, value in section_data.items():
                    if hasattr(section_obj, attr):
                        setattr(section_obj, attr, value)
            elif hasattr(self._config, section_name):
                setattr(self._config, section_name, value)
    
    def validate(self) -> List[str]:
        """验证配置有效性，返回错误列表"""
        errors = []
        
        # 进化参数验证
        if self.config.evolution.total_generations <= 0:
            errors.append("total_generations must be > 0")
        if self.config.evolution.population_size <= 0:
            errors.append("population_size must be > 0")
        if self.config.evolution.elite_count >= self.config.evolution.population_size:
            errors.append("elite_count must be < population_size")
        if not (0 <= self.config.evolution.mutation_rate <= 1):
            errors.append("mutation_rate must be in [0, 1]")
        
        # 仿真参数验证
        if self.config.simulation.dt <= 0:
            errors.append("dt must be > 0")
        if self.config.simulation.render_width <= 0 or self.config.simulation.render_height <= 0:
            errors.append("render dimensions must be positive")
        
        # 输出目录验证
        Path(self.config.output.results_dir).mkdir(parents=True, exist_ok=True)
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for section_name in ['evolution', 'simulation', 'viewer', 
                             'direct_drive', 'templates', 'output']:
            section_obj = getattr(self._config, section_name)
            result[section_name] = asdict(section_obj)
        return result
    
    def save(self, path: str | Path = None) -> Path:
        """保存配置到文件"""
        if path is None:
            path = Path(self.config.output.results_dir) / "config.yaml"
        else:
            path = Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Saved config to: {path}")
        return path
    
    def setup_logging(self) -> None:
        """根据配置设置日志"""
        level = getattr(logging, self.config.output.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def summary(self) -> str:
        """生成配置摘要"""
        lines = [
            "=" * 50,
            "  ForgeCraft Configuration Summary",
            "=" * 50,
            "",
            "  Evolution:",
            f"    Generations:      {self.config.evolution.total_generations}",
            f"    Population:       {self.config.evolution.population_size}",
            f"    Elite Count:      {self.config.evolution.elite_count}",
            f"    Mutation Rate:    {self.config.evolution.mutation_rate:.2f}",
            "",
            "  Simulation:",
            f"    Time Step:       {self.config.simulation.dt*1000:.1f}ms",
            f"    Steps/Eval:      {self.config.evolution.sim_steps}",
            f"    Render Size:      {self.config.simulation.render_width}x{self.config.simulation.render_height}",
            "",
            "  Output:",
            f"    Results Dir:     {self.config.output.results_dir}",
            f"    Log Level:       {self.config.output.log_level}",
            "",
            "=" * 50,
        ]
        return "\n".join(lines)


# 全局单例访问函数
_config_manager: Optional[ConfigManager] = None


def get_config() -> ForgeCraftConfig:
    """获取全局配置实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load_env()
        
        # 尝试自动加载配置文件
        for config_path in ['forgecraft.yaml', 'forgecraft.json', 'config.yaml']:
            if Path(config_path).exists():
                _config_manager.load_file(config_path)
                break
        
        # 验证并设置日志
        errors = _config_manager.validate()
        if errors:
            logger.warning(f"Config validation warnings: {errors}")
        _config_manager.setup_logging()
    
    return _config_manager.config


def init_config(config_file: str = None, **overrides) -> ForgeCraftConfig:
    """
    初始化配置（推荐入口点）
    
    Args:
        config_file: 配置文件路径
        **overrides: 命令行参数覆盖
        
    Returns:
        ForgeCraftConfig实例
    """
    global _config_manager
    _config_manager = ConfigManager()
    
    # 加载顺序：默认 → 文件 → 环境变量 → 覆盖
    _config_manager.load_env()
    
    if config_file:
        _config_manager.load_file(config_file)
    
    if overrides:
        _config_manager.update(**overrides)
    
    # 验证
    errors = _config_manager.validate()
    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
    
    _config_manager.setup_logging()
    
    return _config_manager.config


# 默认配置文件模板
DEFAULT_CONFIG_TEMPLATE = """\
# ForgeCraft Configuration File
# ============================

evolution:
  total_generations: 100
  population_size: 40
  eval_episodes: 6
  sim_steps: 3000
  settling_steps: 200
  grounding_steps: 500
  
  tournament_size: 5
  elite_count: 5
  crossover_rate: 0.3
  mutation_rate: 0.8
  
  patience: 20
  min_improvement: 0.001

simulation:
  dt: 0.002
  gravity: [-9.81, 0, 0]
  ground_friction: 1.0
  
  solver_iterations: 100
  solver_tolerance: 1.0e-6
  
  render_width: 640
  render_height: 480
  render_offscreen: false

viewer:
  auto_drive: true
  drive_speed: 0.8
  show_hud: true
  hud_refresh_rate: 10.0
  camera_distance: 2.0
  camera_elevation: -30.0

direct_drive:
  sim_steps: 1500
  n_frequencies: 3
  episodes_per_freq: 2
  
  signal_amplitude: 1.0
  signal_frequencies: [0.5, 1.0, 2.0]
  
  weight_displacement: 0.4
  weight_speed: 0.3
  weight_survival: 0.3

templates:
  seed: 42
  template_weights:
    differential_wheeled: 0.35
    quad_wheeled: 0.25
    bipedal: 0.15
    quadruped: 0.15
    crawler: 0.10
  min_parts: 4
  max_parts: 12
  size_variance: 0.15

output:
  results_dir: v11_results
  log_level: INFO
  save_checkpoints: true
  checkpoint_interval: 10
  export_format: stl
"""


if __name__ == "__main__":
    # 测试配置系统
    print(DEFAULT_CONFIG_TEMPLATE[:500])
    
    # 创建示例配置文件
    example_path = Path("example_forgecraft.yaml")
    with open(example_path, 'w') as f:
        f.write(DEFAULT_CONFIG_TEMPLATE)
    
    print(f"\nExample config saved to: {example_path}")
    
    # 测试加载
    cfg = init_config(str(example_path))
    print("\n" + ConfigManager().summary())
