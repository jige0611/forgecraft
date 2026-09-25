# ═══════════════════════════════════════════════════════════════
#  ForgeCraft API 服务层 (FastAPI)
#  ═══════════════════════════════════════════════════════════════
#
#  端点：
#    GET  /api/health          健康检查
#    GET  /api/info            系统信息
#    POST /api/generate        生成机器人
#    POST /api/evaluate        评估机器人
#    POST /api/evolve/start    启动进化
#    GET  /api/evolve/status   进化状态
#    GET  /api/results         获取结果
#    GET  /api/export/{format} 导出模型
#
#  启动：python api.py 或 uvicorn api:app --port 8000
# ═══════════════════════════════════════════════════════════════

import sys
import os
import json
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

# 延迟导入重型模块
def get_catalog():
    from forgecraft.core.catalog import load_parameterized_catalog as load_catalog
    return load_catalog()


app = FastAPI(
    title="ForgeCraft Robot Evolution API",
    description="REST API for robot evolution system",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态
evolution_tasks: Dict[str, Dict] = {}


# ==================== 数据模型 ====================

class GenerateRequest(BaseModel):
    template_type: str = "random"
    count: int = 1
    seed: int = 42


class EvaluateRequest(BaseModel):
    body_data: Dict[str, Any]  # 序列化的body数据
    sim_steps: int = 1500


class EvolveRequest(BaseModel):
    generations: int = 100
    population_size: int = 40
    elite_count: int = 5
    mutation_rate: float = 0.8
    seed: int = 42


class ExportRequest(BaseModel):
    format: str = "stl"  # stl, urdf, json


# ==================== 端点实现 ====================

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/api/info")
async def system_info():
    """系统信息"""
    import platform
    import torch
    
    return {
        "system": platform.system(),
        "python": platform.python_version(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "results_exist": Path("v11_results").exists(),
    }


@app.post("/api/generate")
async def generate_robots(req: GenerateRequest):
    """生成机器人"""
    try:
        from locomotion_templates import LocomotionTemplateGenerator
        
        generator = LocomotionTemplateGenerator(seed=req.seed)
        
        if req.template_type != "random":
            for k in generator.template_weights:
                generator.template_weights[k] = 0.0
            if req.template_type in generator.template_weights:
                generator.template_weights[req.template_type] = 1.0
        
        population = generator.generate_population(size=req.count)
        
        results = []
        for body in population:
            results.append({
                "name": body.name,
                "type": getattr(body, 'robot_type', 'unknown'),
                "num_parts": body.num_parts(),
                "num_joints": len(body.joints()),
            })
        
        return {"success": True, "robots": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate")
async def evaluate_robot(req: EvaluateRequest):
    """评估单个机器人（简化版，需要完整body对象）"""
    raise HTTPException(
        status_code=501, 
        detail="Full evaluation requires MechanicalBody object. Use CLI instead."
    )


@app.post("/api/evolve/start")
async def start_evolution(req: EvolveRequest, background_tasks: BackgroundTasks):
    """启动后台进化任务"""
    task_id = str(uuid.uuid4())[:8]
    
    evolution_tasks[task_id] = {
        "status": "pending",
        "config": req.dict(),
        "started_at": datetime.now().isoformat(),
        "current_gen": 0,
        "best_fitness": 0.0,
    }
    
    # 后台运行进化
    background_tasks.add_task(run_evolution_background, task_id, req)
    
    return {
        "task_id": task_id,
        "status": "started",
        "message": f"Evolution started with ID: {task_id}"
    }


async def run_evolution_background(task_id: str, req: EvolveRequest):
    """后台进化任务"""
    try:
        from locomotion_templates import LocomotionTemplateGenerator
        from direct_drive_evaluator import DirectDriveEvaluator, DirectDriveConfig
        
        evolution_tasks[task_id]["status"] = "running"
        
        catalog = get_catalog()
        generator = LocomotionTemplateGenerator(seed=req.seed)
        evaluator = DirectDriveEvaluator(DirectDriveConfig(
            sim_steps=500, n_frequencies=1, episodes_per_freq=1
        ))
        
        best_fitness = 0.0
        
        for gen in range(req.generations):
            if task_id not in evolution_tasks:
                break
            
            pop = generator.generate_population(size=req.population_size)
            
            gen_best = 0
            for body in pop:
                try:
                    result = evaluator.evaluate_body(body, catalog, n_episodes=1)
                    if result and result.get('fitness', 0) > gen_best:
                        gen_best = result['fitness']
                        if gen_best > best_fitness:
                            best_fitness = gen_best
                except:
                    pass
            
            evolution_tasks[task_id].update({
                "current_gen": gen + 1,
                "best_fitness": best_fitness,
            })
            
            await asyncio.sleep(0.01)
        
        evolution_tasks[task_id]["status"] = "completed"
        evolution_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        evolution_tasks[task_id]["status"] = "failed"
        evolution_tasks[task_id]["error"] = str(e)


@app.get("/api/evolve/status/{task_id}")
async def get_evolution_status(task_id: str):
    """获取进化任务状态"""
    if task_id not in evolution_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return evolution_tasks[task_id]


@app.get("/api/results")
async def get_results(limit: int = 20):
    """获取实验结果"""
    results_file = Path("v11_results/v11_final_results.json")
    
    if not results_file.exists():
        return {"exists": False, "data": None}
    
    with open(results_file) as f:
        data = json.load(f)
    
    return {
        "exists": True,
        "best_result": data.get("best_result", {}),
        "total_generations": len(data.get("history", [])),
        "recent_history": data.get("history", [])[-limit:],
    }


@app.get("/api/templates")
async def list_templates():
    """列出可用模板类型"""
    return {
        "templates": [
            {"id": "differential_wheeled", "name": "Differential Wheeled", "desc": "Two-wheel drive robot"},
            {"id": "quad_wheeled", "name": "Quad Wheeled", "desc": "Four-wheel drive robot"},
            {"id": "bipedal", "name": "Bipedal", "desc": "Two-legged walker"},
            {"id": "quadruped", "name": "Quadruped", "desc": "Four-legged walker"},
            {"id": "crawler", "name": "Crawler", "desc": "Tracked/crawler robot"},
        ]
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("  ForgeCraft API Server Starting...")
    print("="*60)
    print(f"  URL: http://localhost:8000")
    print(f"  Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
