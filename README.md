# ForgeCraft

> 基于强化学习的机械形态协同进化框架  
> RL-based Mechanical Morphology Co-Evolution Framework

**ForgeCraft** 将强化学习 (PPO + GNN) 与进化算法 (NSGA-II + MAP-Elites) 结合，自动设计、评估并优化机械形态。流水线覆盖：从零件箱生成形态 → GNN 形态编码 → PPO 策略训练 → MuJoCo 物理评估 → STL / STEP / DXF / URDF / BOM 等制造文件导出。

> **当前状态 (v0.4.0，研究原型)**：默认仿真配置下，形态的根节点与世界坐标系固定连接（`forgecraft/simulation/builder.py` 未给根 body 生成 `<freejoint/>`，该关节只出现在降级路径 `_build_fallback_xml()` 中）。因此位移、跌落与速度类奖励在默认设置下不生效 —— 例如 `examples/demo_robot/best_body.json` 的适应度分量中只有 `upright` 非零。该快照展示的是几何与制造导出链路，**不是**运动学性能结果。详见文末[已知限制](#已知限制)。

---

## 快速开始

### 安装

```bash
# 1. 核心安装 (Python >= 3.10) — 仿真 / 形态建模 / 几何与网格处理
pip install -e .

# 2. 开发环境 (含 pytest / ruff)
pip install -e ".[dev]"

# 3. (可选) GPU 仿真 (Genesis World, Python < 3.14)
pip install -e ".[gpu]"

# 4. (可选) 按需启用其他能力
pip install -e ".[dashboard]"   # 实时 Web 看板 (fastapi + uvicorn)
pip install -e ".[hpo]"         # 超参数优化 (optuna)
pip install -e ".[dist]"        # Ray 分布式
pip install -e ".[mfg]"         # 展示套件: STL 工具 + 离线渲染 (numpy-stl, matplotlib)
pip install -e ".[deploy]"      # ONNX 跨平台部署
pip install -e ".[analysis]"    # 有限元 / 等几何分析

# 5. 一键安装全部
pip install -e ".[all]"
```

> GLB / glTF 导出依赖 `scipy`（trimesh 面着色路径），已含在核心依赖中。
> `renders/` 下的离线渲染图依赖 `matplotlib`（`.[mfg]` 或 `.[all]`）。

### 30 秒进化一把

```bash
# CPU 模式 (默认)
python -m forgecraft.main --population 20 --generations 10

# GPU 模式 (需要 CUDA)
python -m forgecraft.main --cuda --population 50 --generations 50

# 导出制造文件: STL / STEP / 3MF / URDF / BOM + 制造性报告
python -m forgecraft.main --generations 100 --export --export-dir design_output

# 列出可用的零件箱 / 任务
python -m forgecraft.main --list-catalogs
python -m forgecraft.main --list-tasks
```

### 从已有 best_body.json 单独导出

```bash
# 输出: STL / URDF / BOM / manufacturability.json / body.json
# --catalog 需与生成该形态时所用零件箱一致, 否则质量与材料估算会失真
python -m forgecraft.manufacturing --body-file design_output/best_body.json --catalog speedster

# 输出目录默认 design_output, 可用 --output 指定
python -m forgecraft.manufacturing --body-file design_output/best_body.json --catalog speedster --output out
```

需要 **STEP / DXF 工程图 / GLB / 渲染图 / 生产包** 时，走 Python API 的增强导出链路
（`examples/demo_robot/` 即由该链路生成）：

```python
from forgecraft.manufacturing import export_all_enhanced

export_all_enhanced(body_data, catalog_specs, "design_output")
```

---

## 示例产物

`examples/demo_robot/` 是一次进化（第 79 代）的交付物快照，用于展示**几何建模与制造导出链路**：

| 文件 | 说明 |
|------|------|
| `best_body.json` | 最佳形态（16 零件 / 15 关节；零件树 + 关节定义） |
| `evolution_history.json` | 逐代适应度记录 |
| `gen79_ind007_report.md` | 制造性报告（含可制造性评分与诊断告警） |
| `gltf/gen79.glb` | 装配体 glTF 2.0 二进制（任意 glTF 查看器可打开） |
| `gltf/gen79_exploded.glb` | 爆炸视图模型 |
| `renders/` | 13 张离线渲染图（7 视角 + 3 视角论文图，各有正常/爆炸两版） |
| `parts_showcase/` | 参数化零件特写图（**静态插图**：由仓库外的辅助脚本生成，无法由本仓库代码复现） |
| `bom.json` / `manufacturability.json` | 材料清单（$325.78）+ 可制造性评分（总分 0.9954 / 可制造性 0.70） |
| `3d_viewer.html` | 单文件网页查看器（浏览器直接打开，需外部 glTF 场景） |

> ⚠️ 该快照的适应度分量中只有 `upright` 非零（4.593 kg，**0 个驱动关节**，76 对零件碰撞，
> 4 个关节范围 `hi < lo`）。它是「流水线能跑通并产出可检查的交付物」的证据，
> **不是**「进化出了性能优异的机器人」的证据。参见文末[已知限制](#已知限制)。

![装配渲染](examples/demo_robot/renders/gen79_all_isometric.png)

![爆炸视图](examples/demo_robot/renders/gen79_exploded_paper_main_exploded.png)

---

## 核心能力

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 形态生成  │ →  │ GNN 编码 │ →  │ PPO 训练 │ →  │ 物理仿真 │ →  │ 制造导出  │
│ catalog   │    │ encoder  │    │ policy   │    │ MuJoCo   │    │ STL/URDF │
│ generator │    │ batch    │    │ rollout  │    │ Genesis  │    │ STEP/BOM │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
      ↑                               │                               │
      │         进化引擎 (NSGA-II + MAP-Elites + 课程学习)             │
      └───────────────────────────────┴───────────────────────────────┘
```

| 功能 | 描述 |
|------|------|
| **形态自动设计** | 从零件箱 (catalog) 组装任意拓扑的机械形态 |
| **GNN 形态编码** | 图神经网络将形态编码为固定维度向量 |
| **PPO 强化学习** | 形态条件 Actor/Critic 策略训练（每个个体在其评估预算内独立训练，见已知限制） |
| **物理仿真** | MuJoCo (CPU)；`genesis` 后端为可选依赖，本仓库未附带性能基准数据 |
| **多目标进化** | NSGA-II Pareto 前沿 + MAP-Elites 质量多样性 |
| **制造导出** | STL / STEP AP242（镶嵌几何）/ DXF 工程图 / URDF / ONNX / BOM / GLB / 生产包 |
| **错误恢复** | 原子断点 + 6 级降级保护 |
| **HPO 自动优化** | Optuna TPE + Bayesian GP 自动调参 |

---

## 架构

```
forgecraft/
├── core/              # 形态建模 + 零件箱 + 生成器
│   ├── morphology.py      MechanicalBody / Part / Joint 数据结构
│   ├── catalog.py         零件箱 (电机/杆材/轮子/脚...)
│   ├── generator.py       随机形态生成器
│   ├── loader.py          零件箱 & 任务加载
│   └── validation.py      运行时参数校验
│
├── rl/                # 强化学习
│   ├── encoder.py         GNN 形态编码器 (消息传递 + 注意力)
│   ├── ppo.py             PPO 训练器 (PPOBuffer + advantage 计算)
│   ├── morph_transfer.py  形态迁移学习 (MorphConditionalActor/Critic)
│   ├── pretrain.py        编码器预训练 (SimCLR 对比学习)
│   ├── batch_encoder.py   批量 GNN 编码器
│   ├── batch_ppo.py       批量 PPO rollout (借鉴 CleanRL)
│   ├── incremental_encoder.py 增量 GNN 编码 (变异局部重计算)
│   └── env.py             MuJoCo 强化学习环境
│
├── evolution/         # 进化引擎
│   ├── loop.py            进化主循环 (EvolutionLoop)
│   ├── breeder.py         种群繁殖器 (选择 + 交叉 + 变异)
│   ├── evaluator.py       种群评估器 (UCB 渐进式预算)
│   ├── operators.py       遗传算子 (变异/交叉 + 10+ 变异子)
│   ├── selection.py       选择算子 (精英/锦标赛/Pareto)
│   ├── adaptive_mutation.py 自适应变异 (多样性驱动)
│   ├── map_elites.py      MAP-Elites 质量多样性
│   ├── hyperparam_opt.py  超参数优化 (Optuna)
│   ├── recovery.py        错误恢复 (原子断点 + 6 级降级)
│   └── distributed.py     分布式评估 (Ray + 多进程)
│
├── simulation/        # 物理仿真
│   ├── builder.py         MuJoCo MJCF 场景构建
│   ├── terrain_curriculum.py 地形课程学习
│   ├── genesis_backend.py Genesis GPU 后端
│   └── backend_registry.py  多后端注册表
│
├── evaluation/        # 评估指标
│   ├── fitness.py         多目标适应度 (速度/能耗/稳定性/...)
│   ├── metrics.py         制造性评估指标
│   └── gpu_engine.py      GPU 加速评估引擎
│
├── geometry/          # 参数化几何 (产品级建模)
│   ├── parametric.py      9 种零件参数化生成器 (电机/轴承/底盘/弹簧/脚/电池...)
│   ├── materials.py       PBR 材质预设 (碳纤维 / 7075 铝 / 铬钢 / LiPo / TPU)
│   ├── gltf_scene.py      glTF 2.0 导出 (正常 / 爆炸双模式)
│   ├── exploded.py        爆炸图层级展开
│   └── pbr_renderer.py    离线渲染 (matplotlib 3D + 逐面 PBR 近似着色, 默认 250 DPI)
│
├── analysis/          # 有限元 / 等几何分析 (IGA)
│   ├── fea.py             结构有限元 (应力 / 模态)
│   ├── iga_*.py           等几何分析 (热 / 流体 / 断裂 / 多物理场)
│   └── topology.py        拓扑优化
│
├── cam/               # 增材制造前处理
│   ├── slicer.py          切片引擎
│   ├── infill/            填充图案 (gyroid / honeycomb / lightning ...)
│   └── gcode_writer.py    G-code 输出
│
├── manufacturing/     # 制造导出
│   ├── __init__.py           导出总入口 (export_all / export_all_enhanced)
│   ├── step_exporter.py      STEP AP242 装配体导出 (镶嵌三角面, 非解析 B-rep)
│   ├── drawing_generator.py  2D 工程图 (DXF + 公差标注)
│   ├── interference_checker.py 零件干涉 / 配合检查
│   ├── production_packager.py  一体化交付包 (ZIP)
│   ├── realtime_feedback.py  实时制造性反馈 (进化中嵌入)
│   └── onnx_deploy.py        ONNX 部署验证
│
├── knowledge/         # 知识存储 (关系 / 图 / 向量)
├── llm/               # LLM 辅助设计 (Agent + 子代理)
│
├── experiment/        # 实验管理
│   └── management.py      实验追踪 + 对比 + 实验清单
│
├── testing/           # 测试
│   ├── test_suite.py      核心单元测试
│   └── integration/       集成测试 (端到端进化 / 制造导出)
│
├── config.py          # 全局配置 (SimConfig / RLConfig / EvolutionConfig)
├── main.py            # CLI 入口
└── dashboard*.py      # Web 可视化仪表盘
```

---

## CLI 参数

常用参数（完整列表见 `python -m forgecraft.main --help`）：

```
python -m forgecraft.main [OPTIONS]

  --generations N         进化代数 (默认: 10)
  --population N          种群规模 (默认: 16)
  --elites N              精英保留数 (默认: 3)
  --cuda                  启用 GPU 加速
  --workers N             并行进程数 (默认: 4)
  --catalog NAME          零件箱 (默认: primitives)
  --task NAME             任务 (默认: speed)
  --seed SEED             随机种子 (默认: 42)
  --output DIR            结果输出目录
  --export                导出制造文件 (STL / STEP / 3MF / URDF / BOM)
  --export-dir DIR        导出目录 (默认: design_output)
  --quality LEVEL         几何精度 low|medium|high|ultra (默认: high)
  --resume PATH           从断点恢复
  --checkpoint-every N    每 N 代自动保存 (默认: 5)
  --curriculum            启用课程学习
  --pretrain              预训练编码器 (SimCLR)
  --map-elites            启用 MAP-Elites
  --hpo                   超参数优化
  --distributed           分布式计算 (Ray)
  --experiment-name NAME  实验追踪
  --turbo                 极速模式 (torch.compile)
  --quick                 快速测试
  --dashboard             Web 仪表盘
```

> `--catalog` 默认值为 `primitives`；`examples/demo_robot/` 使用的是 `speedster` 零件箱，
> 复现该快照时需显式指定。

---

## 配置

所有参数通过 dataclass 集中管理 (`forgecraft/config.py`)：

```python
from forgecraft.config import EvolutionConfig, RLConfig, SimConfig

evo = EvolutionConfig(
    population_size=50,
    generations=200,
    mutation_rate=0.35,
)

rl = RLConfig(
    hidden_dim=128,
    gnn_layers=3,
    actor_lr=3e-4,
    ppo_epochs=12,
)

sim = SimConfig(
    timestep=0.005,
    max_steps=500,
)
```

---

## 编程 API

```python
from forgecraft.evolution.loop import EvolutionLoop
from forgecraft.config import EvolutionConfig, TaskConfig
from forgecraft.core.loader import load_catalog, load_task

# 1. 加载零件箱和任务
catalog = load_catalog("primitives")
task = load_task("speed").to_task_config()

# 2. 创建进化循环
loop = EvolutionLoop(
    evo_config=EvolutionConfig(population_size=30),
    task_config=task,
    catalog=catalog.to_part_specs(),
    device="cuda",
    enable_map_elites=True,
)

# 3. 进化
loop.initialize_population()
loop.run(n_generations=100)

# 4. 导出
from forgecraft.manufacturing import export_from_evolution_loop
export_from_evolution_loop(loop, output_dir="design_output")
```

---

## 后端切换

```bash
# MuJoCo CPU (默认)
FORGECRAFT_BACKEND=mujoco python -m forgecraft.main

# Genesis GPU 后端 (可选, 需要 Python < 3.14)
pip install genesis-world
FORGECRAFT_BACKEND=genesis python -m forgecraft.main --cuda
```

代码切换：
```python
from forgecraft.simulation.backend_registry import BackendRegistry
print(BackendRegistry.get_status())
evaluator = BackendRegistry.create_evaluator("genesis", ...)
```

---

## 测试

测试分为两套：`forgecraft/testing/`（核心单元 + 集成）与 `tests/`（模块级单元测试）。

```bash
# 全量回归（两套一起，424 项）
pytest

# 仅核心单元测试
pytest forgecraft/testing/test_suite.py -v

# 仅集成测试
pytest forgecraft/testing/integration/ -v

# 跳过慢速用例
pytest -m "not slow"
```

---

## Docker

```bash
# CPU 版本
docker compose up forgecraft

# GPU 版本 (需要 nvidia-container-toolkit)
docker compose up gpu

# 快速跑一次
docker compose run forgecraft --quick
```

---

## 系统要求

| 组件 | 最低 | 推荐 |
|------|------|------|
| Python | 3.10+ | 3.12 |
| GPU | 无 (CPU) | RTX 3060+ / 8GB VRAM |
| RAM | 8 GB | 32 GB+ |
| MuJoCo | 3.0+ | 最新 |
| OS | Windows/Linux/Mac | Linux (CUDA) |

---

## 可选：标准件库

`scripts/` 下的 `extract_nopscad_dimensions.py` 等脚本可从 [NopSCADlib](https://github.com/nophead/NopSCADlib)
提取标准件（螺丝 / 螺母 / 轴承）尺寸并生成 STL 网格。

该库以 **GPL-3.0** 分发，与本项目的 MIT 许可不兼容，因此**未包含在本仓库内**。
如需使用请自行获取（生成网格还需安装 [OpenSCAD](https://openscad.org/)）：

```bash
git clone https://github.com/nophead/NopSCADlib.git
```

---

## 已知限制

本节如实列出当前版本的已知缺陷与范围边界。这些是**明确记录**的限制，而非未知问题。

### 1. 根节点未连接自由关节（影响运动类奖励）

`forgecraft/simulation/builder.py` 在构建 MJCF 时为根 body 只输出
`<body name="..." pos="...">`，**没有** `<freejoint/>`；该元素仅出现在
`_build_fallback_xml()` 降级路径中。后果：

- 根节点被焊接在世界坐标系，编译后的模型自由度等于内部关节数
  （实测 6 零件 / 5 关节的体 → `nq = nv = 5`，不含 6 个浮动基座自由度）；
- `framepos` 质心传感器的 z 分量恒定，`_com_z < fall_height` 形式的终止条件不会触发；
- `displacement` / `speed` / `velocity` 类奖励项恒为 0。

`examples/demo_robot/best_body.json` 的 `fitness_components` 只有 `upright = 5.2209`
非零，正是这一限制的直接体现。**这是一个待修的仿真建模缺陷**，不是有意设计；
修复会使既有演示快照与部分测试的数值失效，因此未在本版本中改动。

### 2. 每个个体独立训练策略，而非共享单一策略

进化循环中每次个体评估都会新建一个 `PPOTrainer`（可经 `prev_trainer_state`
继承上一代状态），并在该个体自身的评估预算内做 PPO 更新。
因此**不存在**一个跨拓扑泛化的单一策略网络；GNN 形态编码提供的是
条件输入，而非共享的通用控制器。跨拓扑零样本迁移属于未来工作。

### 3. 演示形态本身的缺陷

`examples/demo_robot/gen79_ind007_report.md` 记录的 gen79 个体：

| 项目 | 数值 |
|------|------|
| 零件 / 关节 | 16 / 15 |
| 驱动关节 | **0**（报告显式告警「没有驱动关节，机械体无法主动运动」） |
| 总质量 | 4.593 kg |
| 可制造性总分 / 可制造性得分 | 0.9954 / 0.70 |
| 零件间碰撞对 | 76 |
| 无效关节范围（`hi < lo`） | 4 个，另有 2 个为 `[0, 0]` |
| BOM 成本 | $325.78 |

### 4. STEP 导出为镶嵌几何

`manufacturing/step_exporter.py` 输出的是 AP242 文件（`FILE_SCHEMA`
`AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF`），几何实体为
`TESSELLATED_ITEM` + `COORDINATES_LIST` + `TRIANGULATED_FACE`，
即**三角网格镶嵌**，不是解析 B-rep（无 NURBS 曲面与拓扑边）。可被支持
AP242 镶嵌的查看器打开，但无法直接用于参数化 CAD 建模。

### 5. glTF / 渲染的依赖与保真度

- GLB 导出需要 `scipy`（trimesh 面着色 → `scipy.sparse`），已加入核心依赖；
- `renders/` 下的图由 matplotlib 3D 生成，采用**逐面** PBR 近似着色
  （Lambert 漫反射 + Blinn-Phong 高光 + Fresnel，视线方向固定为 +Z，三光源加权）；
  它不是光线追踪或 IBL，属于示意级渲染，不适用于材质对比。

### 6. 缺少性能基准

本仓库**不附带**任何 FPS / 加速比基准数据。`genesis` 后端为可选依赖，
需要单独安装并自行实测；此前文档中出现过的吞吐数字已移除。

### 7. `parts_showcase/` 中的插图不可复现

`examples/demo_robot/parts_showcase/` 下的三张 PNG 由仓库**外**的辅助脚本生成，
仓库内没有任何代码路径能重新产生它们。它们仅作为静态插图保留；
`renders/` 下的 13 张图则可由 `export_presentation_suite` / `render_paper_suite` 完整复现。

### 8. 分布式评估路径的验证范围

`forgecraft/evolution/distributed.py` 的 Ray 与 multiprocessing 两条路径均复用与
进程内路径**同一个**评估实现（`_evaluate_body_worker_ucb`，含真实 rollout 与 PPO 更新），
因此结果口径一致；但本仓库展示的所有结果都由进程内并行评估器产生，
**未附带**任何 Ray 集群的实测数据。此外 `RedisTaskQueue` 仅提供任务/结果队列原语，
未接入 `evaluate_population`，需自行实现消费端。

---

## 论文引用

本项目整合了以下领域的工作：

- **强化学习** — PPO (Schulman et al. 2017), GAE (Schulman et al. 2016)
- **图神经网络** — Message Passing GNN (Gilmer et al. 2017)
- **进化算法** — NSGA-II (Deb et al. 2002), MAP-Elites (Mouret & Clune 2015)
- **形态进化** — RoboGrammar (Zhao et al. 2020), EvoCraft
- **物理仿真** — MuJoCo (Todorov et al. 2012), Genesis World (2025)
- **制造性** — DfAM (Design for Additive Manufacturing)

---

## 许可证

MIT License

---

**ForgeCraft** — 从零到制造，自动设计最优机械形态。
