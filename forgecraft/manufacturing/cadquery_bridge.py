"""
CadQuery 桥接模块 — 主进程通过 uv + Python 3.12 子进程调用 cadquery

将 MechanicalBody 转为真 B-Rep STEP + 高精度 STL，无需降级 Python。

用法:
  >>> from forgecraft.manufacturing.cadquery_bridge import CadQueryBridge
  >>> bridge = CadQueryBridge()
  >>> result = bridge.export(body, output_dir, format="both")
  >>> print(result["files"])

依赖:
  - uv (pip install uv) — 自动管理 Python 3.12 环境
  - cadquery — uv 自动安装
"""

import os
import json
import sys
import subprocess
import logging
from typing import Dict, List, Any, Optional

from forgecraft.core.morphology import MechanicalBody
from forgecraft.config import PartSpec

_logger = logging.getLogger(__name__)

__all__ = ["CadQueryBridge", "cadquery_available"]

_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "_cq_worker.py")


def cadquery_available() -> bool:
    """检测 cadquery 是否可用 (通过 uv)"""
    try:
        # Check uv is installed
        import uv
        return True
    except ImportError:
        try:
            subprocess.run(
                [sys.executable, "-m", "uv", "--version"],
                capture_output=True, timeout=5
            )
            return True
        except Exception:
            return False


class CadQueryBridge:
    """cadquery 桥接器 — 通过 uv 子进程调用 cadquery
    
    quality: B-Rep 精度控制
      "medium" — 公差 0.1mm,  圆柱 32 段
      "high"   — 公差 0.01mm, 圆柱 64 段
      "ultra"  — 公差 0.005mm, 圆柱 128 段
    """
    
    def __init__(self, quality: str = "high"):
        self.quality = quality
        self._available = None
    
    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = cadquery_available()
        return self._available
    
    def _parts_to_payload(self, body: MechanicalBody,
                          catalog: Dict[str, Any] = None) -> List[Dict]:
        """将 MechanicalBody 转为 worker 可理解的 JSON"""
        parts = []
        for part in body.parts():
            spec = catalog.get(part.part_type) if catalog else None
            
            entry = {
                "part_id": part.part_id,
                "part_type": part.part_type,
                "params": dict(part.params),
                "position": part.position.tolist(),
                "actuated": False,
                "geometry_type": "cylinder",
                "color": [150, 150, 150],
            }
            
            if spec is not None:
                entry["geometry_type"] = getattr(spec, 'geometry_type',
                                                  getattr(spec, 'shape', 'cylinder'))
                entry["actuated"] = bool(getattr(spec, 'can_actuate',
                                                  getattr(spec, 'actuated', False)))
                color = getattr(spec, 'color', [0.6, 0.6, 0.6, 1.0])
                entry["color"] = [int(c * 255) for c in color[:3]]
            
            parts.append(entry)
        return parts
    
    def export(self, body: MechanicalBody,
                output_dir: str = "cadquery_output",
                catalog: Dict[str, Any] = None,
                fmt: str = "both") -> Dict[str, Any]:
        """导出 B-Rep STEP + 高精度 STL
        
        Args:
            body: 机械体
            output_dir: 输出目录
            catalog: 零件规格目录
            fmt: "step" | "stl" | "both"
        
        Returns:
            {"files": [...], "errors": [...]}
        """
        if not self.available:
            return {"files": [], "errors": ["cadquery not available (uv not found)"]}
        
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "parts": self._parts_to_payload(body, catalog),
            "output_dir": os.path.abspath(output_dir),
            "format": fmt,
            "name": body.name.replace(" ", "_"),
        }
        
        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "uv", "run",
                    "--python", "3.12",
                    "--with", "cadquery",
                    "python", _WORKER_SCRIPT,
                ],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=120,
                cwd=os.path.dirname(_WORKER_SCRIPT),
            )
            
            if proc.returncode != 0:
                err = proc.stderr.strip()[-300:] if proc.stderr else "unknown error"
                return {"files": [], "errors": [f"cadquery worker failed (rc={proc.returncode}): {err}"]}
            
            try:
                result = json.loads(proc.stdout.strip().split("\n")[-1])
                return result
            except (json.JSONDecodeError, IndexError):
                # Parse last JSON line from mixed output
                for line in reversed(proc.stdout.strip().split("\n")):
                    try:
                        return json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                return {"files": [], "errors": ["failed to parse worker output"]}
        
        except subprocess.TimeoutExpired:
            return {"files": [], "errors": ["cadquery worker timed out"]}
        except FileNotFoundError:
            return {"files": [], "errors": ["uv not found"]}
        except Exception as e:
            return {"files": [], "errors": [str(e)]}
    
    def export_step(self, body: MechanicalBody,
                     output_path: str,
                     catalog: Dict[str, Any] = None) -> str:
        """导出 STEP 文件 (真 B-Rep) → 返回文件路径"""
        out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        result = self.export(body, out_dir, catalog, fmt="step")
        
        # 查找生成的 step 文件并重命名到 output_path
        for f in result.get("files", []):
            if f.endswith(".step") and os.path.exists(f):
                if os.path.abspath(f) != os.path.abspath(output_path):
                    import shutil
                    shutil.move(f, output_path)
                    # 清理可能残留的同名文件
                    if os.path.exists(f):
                        os.remove(f)
                return output_path
        
        # 没找到文件，检查错误
        if result.get("errors"):
            _logger.warning(f"cadquery STEP failed: {result['errors'][:3]}")
        return ""
    
    def export_stl(self, body: MechanicalBody,
                    output_dir: str,
                    catalog: Dict[str, Any] = None) -> List[str]:
        """导出 STL 文件 → 返回文件路径列表"""
        result = self.export(body, output_dir, catalog, fmt="stl")
        return result.get("files", [])
