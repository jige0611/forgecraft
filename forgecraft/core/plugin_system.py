"""插件系统模块

实现：
1. 插件注册机制
2. 插件生命周期管理
3. 依赖注入
4. 配置热更新
"""

import importlib
import inspect
import json
import os
from typing import Any, Callable, Dict, List, Optional, Type

from forgecraft.logging import get_logger

__all__ = [
    "Plugin",
    "PluginManager",
    "ConfigManager",
    "DependencyInjector",
    "get_plugin_manager",
    "get_config_manager",
    "get_dependency_injector",
]


class Plugin:
    """插件基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"plugin.{name}")
        self.enabled = True
    
    def initialize(self, config: dict):
        """初始化插件"""
        pass
    
    def start(self):
        """启动插件"""
        pass
    
    def stop(self):
        """停止插件"""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取插件元数据"""
        return {
            'name': self.name,
            'version': '1.0.0',
            'description': '',
            'dependencies': [],
        }


class PluginManager:
    """插件管理器"""
    
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_classes: Dict[str, Type[Plugin]] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.logger = get_logger("plugin.manager")
    
    def register_plugin_class(self, plugin_class: Type[Plugin]):
        """注册插件类"""
        plugin_name = plugin_class.__name__
        if plugin_name in self.plugin_classes:
            self.logger.warning(f"插件类 {plugin_name} 已存在，将被覆盖")
        
        self.plugin_classes[plugin_name] = plugin_class
        
        # 解析依赖
        metadata = plugin_class('temp').get_metadata()
        self.dependencies[plugin_name] = metadata.get('dependencies', [])
    
    def load_plugins_from_directory(self, directory: str):
        """从目录加载插件"""
        if not os.path.exists(directory):
            self.logger.warning(f"插件目录不存在: {directory}")
            return
        
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('_'):
                module_name = filename[:-3]
                try:
                    module_path = directory.replace('/', '.').replace('\\', '.')
                    module = importlib.import_module(f"{module_path}.{module_name}")
                    
                    # 查找插件类
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, Plugin) and obj != Plugin:
                            self.register_plugin_class(obj)
                            self.logger.info(f"加载插件类: {name}")
                except Exception as e:
                    self.logger.error(f"加载插件失败 {module_name}: {e}")
    
    def instantiate_plugin(self, plugin_name: str, config: dict) -> Optional[Plugin]:
        """实例化插件"""
        if plugin_name not in self.plugin_classes:
            self.logger.error(f"插件类不存在: {plugin_name}")
            return None
        
        # 检查依赖
        deps = self.dependencies.get(plugin_name, [])
        for dep in deps:
            if dep not in self.plugins:
                self.logger.error(f"插件 {plugin_name} 依赖 {dep} 未加载")
                return None
        
        try:
            plugin = self.plugin_classes[plugin_name](plugin_name)
            plugin.initialize(config)
            self.plugins[plugin_name] = plugin
            self.logger.info(f"实例化插件: {plugin_name}")
            return plugin
        except Exception as e:
            self.logger.error(f"实例化插件失败 {plugin_name}: {e}")
            return None
    
    def start_plugin(self, plugin_name: str) -> bool:
        """启动插件"""
        if plugin_name not in self.plugins:
            self.logger.error(f"插件未实例化: {plugin_name}")
            return False
        
        try:
            self.plugins[plugin_name].start()
            self.logger.info(f"启动插件: {plugin_name}")
            return True
        except Exception as e:
            self.logger.error(f"启动插件失败 {plugin_name}: {e}")
            return False
    
    def stop_plugin(self, plugin_name: str) -> bool:
        """停止插件"""
        if plugin_name not in self.plugins:
            self.logger.error(f"插件不存在: {plugin_name}")
            return False
        
        try:
            self.plugins[plugin_name].stop()
            self.logger.info(f"停止插件: {plugin_name}")
            return True
        except Exception as e:
            self.logger.error(f"停止插件失败 {plugin_name}: {e}")
            return False
    
    def start_all(self):
        """启动所有插件"""
        for plugin_name in self.plugins:
            self.start_plugin(plugin_name)
    
    def stop_all(self):
        """停止所有插件"""
        for plugin_name in reversed(list(self.plugins.keys())):
            self.stop_plugin(plugin_name)
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """获取插件实例"""
        return self.plugins.get(plugin_name)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.config = {}
        self.listeners: List[Callable[[dict], None]] = []
        self.config_files = []
        self.logger = get_logger("config.manager")
    
    def load_config(self, file_path: str):
        """加载配置文件"""
        if not os.path.exists(file_path):
            self.logger.warning(f"配置文件不存在: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.config.update(config)
                self.config_files.append(file_path)
                self.logger.info(f"加载配置: {file_path}")
                self._notify_listeners()
        except Exception as e:
            self.logger.error(f"加载配置失败 {file_path}: {e}")
    
    def load_configs_from_directory(self, directory: str):
        """从目录加载所有配置文件"""
        if not os.path.exists(directory):
            self.logger.warning(f"配置目录不存在: {directory}")
            return
        
        for filename in sorted(os.listdir(directory)):
            if filename.endswith('.json'):
                self.load_config(os.path.join(directory, filename))
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._notify_listeners()
    
    def register_listener(self, listener: Callable[[dict], None]):
        """注册配置变更监听器"""
        self.listeners.append(listener)
    
    def _notify_listeners(self):
        """通知所有监听器配置变更"""
        for listener in self.listeners:
            try:
                listener(self.config.copy())
            except Exception as e:
                self.logger.error(f"通知监听器失败: {e}")
    
    def save_config(self, file_path: str):
        """保存配置到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.logger.info(f"保存配置: {file_path}")
        except Exception as e:
            self.logger.error(f"保存配置失败 {file_path}: {e}")


class DependencyInjector:
    """依赖注入器"""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.factories: Dict[str, Callable] = {}
    
    def register(self, name: str, instance: Any):
        """注册服务实例"""
        self.services[name] = instance
    
    def register_factory(self, name: str, factory: Callable):
        """注册服务工厂"""
        self.factories[name] = factory
    
    def get(self, name: str) -> Any:
        """获取服务"""
        if name in self.services:
            return self.services[name]
        
        if name in self.factories:
            instance = self.factories[name]()
            self.services[name] = instance
            return instance
        
        raise ValueError(f"服务未注册: {name}")
    
    def inject(self, func: Callable) -> Callable:
        """装饰器：自动注入依赖"""
        def wrapper(*args, **kwargs):
            # 获取函数参数
            sig = inspect.signature(func)
            
            # 自动注入依赖
            for param_name, param in sig.parameters.items():
                if param_name not in kwargs and param_name in self.services:
                    kwargs[param_name] = self.get(param_name)
            
            return func(*args, **kwargs)
        
        return wrapper


# 全局实例
_global_plugin_manager = PluginManager()
_global_config_manager = ConfigManager()
_global_dependency_injector = DependencyInjector()


def get_plugin_manager() -> PluginManager:
    """获取全局插件管理器"""
    return _global_plugin_manager


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    return _global_config_manager


def get_dependency_injector() -> DependencyInjector:
    """获取全局依赖注入器"""
    return _global_dependency_injector


# 示例插件
class ExamplePlugin(Plugin):
    """示例插件"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.counter = 0
    
    def initialize(self, config: dict):
        self.counter = config.get('initial_counter', 0)
    
    def start(self):
        self.logger.info(f"示例插件启动，计数器: {self.counter}")
    
    def stop(self):
        self.logger.info(f"示例插件停止")
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': '1.0.0',
            'description': '示例插件',
            'dependencies': [],
        }