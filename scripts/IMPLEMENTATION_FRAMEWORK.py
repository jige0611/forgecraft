# ══════════════════════════════════════════════════════════════════════
# ForgeCraft v0.4 完整优化实施框架
# ══════════════════════════════════════════════════════════════════════
#
# 目标: 将进化速度从 小时/代 → 秒/代 (300-5000x 加速)
# 基于顶级开源项目的集成方案
#
# ══════════════════════════════════════════════════════════════════════

"""
目录:

A. 外部资源评估 — 开源项目对比与选型
B. P0-1: Genesis GPU 仿真后端 — 核心加速引擎
C. P0-2: 端到端集成测试 — AdamW style test loop
D. P1-1: GPU 批推理引擎 — 种群级并行编码+rollout
E. P1-2: 增量 GNN 编码 — 变异局部重计算
F. P1-3: 制造性实时反馈 — 进化中嵌入工艺约束
G. P2-1: CI/CD + Docker — 可复现构建
H. P2-2: ONNX Runtime 部署 — 策略导出验证
I. 总体实施路线图
"""

# ══════════════════════════════════════════════════════════════════════
# A. 外部资源评估 — 开源项目对比与选型
# ══════════════════════════════════════════════════════════════════════

EXTERNAL_RESOURCES = {

    # ── 仿真引擎 ──
    "genesis-world": {
        "url": "https://github.com/Genesis-Embodied-AI/genesis-world",
        "version": "1.1.1 (2026-06-10)",
        "pip": "pip install genesis-world",
        "qualifications": {
            "OS": "Linux/Win/Mac 全平台",
            "GPU": "Nvidia/AMD/Intel/Apple",
            "Python": ">=3.10,<3.14",
        },
        "performance": {
            "fps_single_env": "43,000,000 FPS (RTX 4090, Franka arm)",
            "parallel_envs": "最大 30,000 并行环境",
            "speedup": "430,000x 实时速度",
        },
        "api_highlights": [
            "scene.build(n_envs=B) — 一行代码创建并行环境",
            "MJCF/URDF/USD/OBJ/GLB 全格式加载",
            "GPU tensor (gs.device) 零拷贝控制",
            "envs_idx 批量选择性控制",
            "内置 PPO/SAC 训练循环",
            "自动 hibernate (静止物体零计算)",
        ],
        "pros": [
            "最活跃维护 (Genesis AI 公司支持)",
            "v1.1.1 已非常成熟",
            "原生 Windows 支持 (无需 WSL)",
            "API 风格与当前 MuJoCo 环境最接近",
            "自由并行度: 1~30000 环境",
            "多物理场 (Rigid/FEM/MPM/SPH) 可选",
        ],
        "cons": [
            "依赖 PyTorch ≥ 3.10",
            "需显存 ≥ 8GB (批处理百级环境)",
            "暂时无 MuJoCo python bindings 兼容层",
        ],
        "verdict": "★★★★★ 首选引擎",
    },

    "brax": {
        "url": "https://github.com/google/brax",
        "version": "0.14.2 (2026-03-16)",
        "pip": "pip install brax",
        "qualifications": {
            "OS": "Linux/Mac",
            "GPU": "Nvidia TPU/GPU 为主",
            "Python": ">=3.11",
        },
        "performance": {
            "fps_batch": "百万级 FPS (TPU/GPU batch mode)",
            "differentiable": "端到端可微物理",
        },
        "api_highlights": [
            "brax.training 仅维护 RL 训练部分",
            "brax.envs 已不维护 (推荐 MJX/MuJoCo Warp)",
            "JAX 原生 (torch CPU→GPU 需额外转换)",
        ],
        "pros": [
            "Google 出品，论文引用广泛",
            "可微物理 (APG 梯度策略优化)",
            "JIT 编译极致性能",
        ],
        "cons": [
            "v0.13+ 仅维护训练算法 (brax/training)",
            "环境部分不再维护",
            "JAX 生态与当前 PyTorch 栈不兼容",
            "Windows 支持弱",
        ],
        "verdict": "★★★☆☆ 适用于特定场景 (可微物理), 不推荐作为主引擎",
    },

    "mujoco-mjx": {
        "url": "https://github.com/google-deepmind/mujoco",
        "pip": "pip install mujoco-mjx",
        "relation": "Google 官方 MuJoCo XLA 加速版",
        "pros": [
            "与当前 MuJoCo MJCF 完全兼容",
            "JAX 加速版本 (XLA 编译)",
            "零迁移成本 (现有 MJCF scene 直接用)",
        ],
        "cons": [
            "性能不如 Genesis (无批量并行优化)",
            "JAX 与 PyTorch 链不兼容",
            "仅加速物理计算，不加速环境创建",
        ],
        "verdict": "★★★☆☆ 过渡方案，长期迁移到 Genesis",
    },

    # ── 训练基础设施 ──
    "cleanrl": {
        "url": "https://github.com/vwxyzjn/cleanrl",
        "version": "v2.6.0",
        "stars": "10k+",
        "pip": "pip install cleanrl",
        "relevance": "单文件 PPO 参考实现",
        "what_to_borrow": [
            "PPO loss 的 clip/clip_vloss 实现 (CleanRL 是最简洁可复现的)",
            "torch.compile 兼容性 (CleanRL 已适配)",
            "W&B/TensorBoard 集成模式",
            "deterministic seeding 最佳实践",
        ],
        "verdict": "★★★★☆ 不直接集成，参考其 PPO 最佳实践",
    },

    "pytorch-lightning": {
        "url": "https://github.com/Lightning-AI/pytorch-lightning",
        "relevance": "训练循环 + 断点管理 + 日志",
        "what_to_borrow": [
            "ModelCheckpoint 的 callback 设计模式",
            "自动 checkpoint + resume 工作流",
            "LightningModule.save_hyperparameters() 模式",
        ],
        "verdict": "★★★☆☆ 设计模式参考，不引入重依赖",
    },

    # ── 配置管理 ──
    "hydra": {
        "url": "https://github.com/facebookresearch/hydra",
        "pip": "pip install hydra-core",
        "relevance": "分层 YAML 配置 + 命令行覆盖",
        "what_to_borrow": [
            "configs/ 目录结构: model/data/training 分层",
            "命令行覆盖: key.subkey=value",
            "--multirun 超参扫描",
            "hydra-optuna-sweeper 插件",
        ],
        "integration_plan": "替换当前 argparse + dataclass 为 Hydra structured config",
        "verdict": "★★★★☆ 中优先级集成",
    },

    # ── 实验追踪 ──
    "mlflow": {
        "url": "https://github.com/mlflow/mlflow",
        "pip": "pip install mlflow",
        "relevance": "开源实验管理标准",
        "what_to_borrow": [
            "mlflow.tracking 实验对比 UI",
            "mlflow.artifacts 模型版本管理",
            "mlflow.pyfunc 模型部署接口",
        ],
        "integration_plan": "替换自建 ExperimentTracker 为 MLflow backend (可选)",
        "verdict": "★★★☆☆ 低优先级",
    },

    # ── 部署 ──
    "onnxruntime": {
        "url": "https://github.com/microsoft/onnxruntime",
        "pip": "pip install onnxruntime onnx",
        "relevance": "策略网络 → ONNX → C++/嵌入式",
        "what_to_borrow": [
            "torch.onnx.export 标准流程",
            "onnxruntime InferenceSession",
            "ONNX Runtime Web / Mobile 部署",
        ],
        "verdict": "★★★★☆ 部署闭环必需",
    },
}


# ══════════════════════════════════════════════════════════════════════
# B. P0-1: Genesis GPU 仿真后端 — 核心加速引擎
# ══════════════════════════════════════════════════════════════════════

"""
目标: 将单形态评估从 3-5秒 (CPU MuJoCo) → 0.01-0.05秒 (GPU Genesis)
原理: 种群内所有形态在 GPU 上并行仿真
     50 个形态 → 50 个平行环境 → 1 次 step() 全部推进

实现蓝图:

forgecraft/simulation/
├── __init__.py          # 已有
├── builder.py           # 已有 MuJoCo MJCF 构建
├── terrain_curriculum.py # 已有
├── genesis_backend.py   # 🆕 Genesis 后端
└── backend_registry.py  # 🆕 后端注册/切换
"""

GENESIS_BACKEND_IMPLEMENTATION = """
# ── genesis_backend.py 架构 ──

class GenesisSimBackend:
    \"\"\"Genesis GPU 加速仿真后端

    与当前 MuJoCo backend (ForgeCraftEnv) 保持相同接口，
    通过 backend_registry 动态切换。
    \"\"\"

    def __init__(self, population: List[MechanicalBody], sim_config, task_config):
        # 1. 初始化 Genesis
        import genesis as gs
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)

        # 2. 为每个形态创建 Scene
        #    关键: 同构形态可以共享 entity，异构形态需要独立 scene
        self.n_envs = len(population)

        # 3. 策略A: 单 scene 多 env (同构形态)
        #    scene.build(n_envs=self.n_envs)
        #    所有环境共享结构，仅状态各异

        # 4. 策略B: 多 scene (异构形态，每个形态独立)
        #    为每个形态创建独立 scene
        #    Genesis 的 heterogeneous envs 支持不同结构

        # 5. 批量 step
        #    scene.step() → 一次性推进所有环境

    def rollout(self, policy: nn.Module) -> Dict[int, float]:
        \"\"\"批量化 rollout

        与当前 PopulationEvaluator 接口一致
        \"\"\"
        obs = self._reset_all()  # (n_envs, obs_dim)
        total_reward = torch.zeros(self.n_envs, device=gs.device)

        for step in range(self.max_steps):
            # batch 推理
            actions = policy(obs)  # (n_envs, act_dim)

            # batch 仿真 — 核心加速点
            self._apply_actions(actions)
            self.scene.step()  # 一次 GPU kernel 推进所有环境

            obs, reward, done, _ = self._get_observations()
            total_reward += reward * (~done)

        return {i: total_reward[i].item() for i in range(self.n_envs)}


class GenesisMJCFAdapter:
    \"\"\"MJCF → Genesis 转换器

    当前 builder.py 生成 MJCF XML 字符串。
    Genesis 原生支持 MJCF 加载:
      scene.add_entity(gs.morphs.MJCF(file='temp.xml'))
    \"\"\"

    @staticmethod
    def build_genesis_scene(
        body: MechanicalBody, sim_config, task_config
    ) -> tuple:
        # 1. 复用现有 builder.build_mjcf_model(body, sim_config, task_config, catalog)
        mjcf_str = build_mjcf_model(body, sim_config, task_config, catalog)

        # 2. 写入临时 XML (Genesis 需要文件，或者用 StringIO)
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.xml', delete=False
        ) as f:
            f.write(mjcf_str)
            xml_path = f.name

        # 3. 加载到 Genesis
        import genesis as gs
        scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(dt=sim_config.timestep),
        )
        robot = scene.add_entity(gs.morphs.MJCF(file=xml_path))
        scene.build(n_envs=1)

        os.unlink(xml_path)
        return scene, robot
"""

# ── backend_registry.py 架构 ──
BACKEND_REGISTRY = """
class BackendRegistry:
    \"\"\"仿真后端注册表

    通过环境变量或配置切换后端:
      FORGECRAFT_BACKEND=genesis  # GPU 加速
      FORGECRAFT_BACKEND=mujoco   # CPU 兼容 (当前)
      FORGECRAFT_BACKEND=mjx      # MuJoCo XLA
    \"\"\"

    _backends = {
        "mujoco": "forgecraft.rl.env.ForgeCraftEnv",
        "genesis": "forgecraft.simulation.genesis_backend.GenesisSimBackend",
        "mjx": "forgecraft.simulation.mjx_backend.MJXSimBackend",
    }

    @classmethod
    def create_evaluator(cls, name=None, **kwargs):
        \"\"\"工厂方法: 按名称创建仿真后端\"\"\"
        backend_name = name or os.environ.get("FORGECRAFT_BACKEND", "mujoco")
        backend_cls = import_string(cls._backends[backend_name])
        return backend_cls(**kwargs)
"""

# 需要的外部依赖
GENESIS_DEPENDENCIES = """
pip install genesis-world  # 自动包含所有依赖
# 推荐: RTX 3060+ (8GB VRAM+) 以获得最佳并行性能
# 最低: 任何 CUDA GPU (可用 CPU 后端但速度无优势)
"""

# 风险评估
GENESIS_RISKS = """
1. MJCF 兼容性风险 (中):
   - ForgeCraft 生成的 MJCF 使用自定义 actuator/motor/joint 定义
   - Genesis MJCF parser 可能与 MuJoCo parser 有细微差异
   - 缓解: 先做兼容性测试 (10 个典型形态 → Genesis vs MuJoCo 行为对比)

2. 性能退化场景 (低):
   - 异构形态 (不同拓扑) 无法共享 scene → 独立 scene → 额外显存
   - 缓解: 按形态结构分组，同构形态共享 scene

3. 版本稳定性 (低):
   - Genesis v1.1.1 于 2026-06-10 发布，非常新
   - 缓解: pin 版本 + CI 中测试

4. 双后端维护成本 (中):
   - 需要同时维护 MuJoCo 和 Genesis 两条路径
   - 缓解: backend_registry 抽象层 + 共享测试套件
"""


# ══════════════════════════════════════════════════════════════════════
# C. P0-2: 端到端集成测试 — 完整的进化流程验证
# ══════════════════════════════════════════════════════════════════════

"""
目标: 自动化验证 初始化→进化→制造→导出 完整流程
方法: 借鉴 PyTorch 的 test_trainer.py 和 Lightning 的 integration tests

当前测试: 150 个单元测试，无完整流程集成测试
目标: 新增 5-8 个端到端测试

实现蓝图:

forgecraft/testing/
├── test_suite.py          # 已有 150 单元测试
├── integration/           # 🆕 集成测试目录
│   ├── __init__.py
│   ├── test_full_evolution.py    # 完整进化流程
│   ├── test_resume_training.py   # 断点续训
│   ├── test_manufacturing_e2e.py # 制造导出端到端
│   ├── test_hpo_e2e.py           # HPO 端到端
│   └── test_distributed_e2e.py   # 分布式端到端
└── conftest.py            # 🆕 共享 fixtures
"""

E2E_TEST_IMPLEMENTATION = """
# ── conftest.py 共享 fixtures ──

@pytest.fixture(scope="session")
def shared_catalog():
    \"\"\"所有集成测试共享的零件箱\"\"\"
    from forgecraft.core.loader import load_catalog
    return load_catalog(name="primitives")

@pytest.fixture(scope="session")
def shared_task():
    \"\"\"快速任务 (500步 max)\"\"\"
    from forgecraft.core.loader import load_task
    return load_task(name="speed").to_task_config()

@pytest.fixture
def quick_loop(shared_catalog, shared_task):
    \"\"\"快速进化循环 (3代, 5个体)\"\"\"
    from forgecraft.evolution.loop import EvolutionLoop
    config = EvolutionConfig(population_size=5, generations=3)
    loop = EvolutionLoop(
        evo_config=config,
        task_config=shared_task,
        catalog=shared_catalog.to_part_specs(),
        device="cpu",
        n_workers=1,
    )
    yield loop
    # cleanup
    import shutil, os
    for d in ["checkpoints", "design_output"]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


# ── test_full_evolution.py ──

class TestFullEvolution:
    \"\"\"完整进化流程测试 (每代~5秒, 总计~15秒)\"\"\"

    def test_quick_3gen(self, quick_loop):
        \"\"\"3代进化: 初始化 → 进化 → 制造导出\"\"\"
        quick_loop.initialize_population()
        assert len(quick_loop.population) == 5

        # 运行 (safe=False 避免 TrainGuard 包装)
        quick_loop.run(n_generations=3, safe=False)
        assert quick_loop.generation >= 3
        assert quick_loop.best_body is not None
        assert quick_loop.best_body.fitness > 0
        assert len(quick_loop.history) == 3

    def test_checkpoint_roundtrip(self, quick_loop):
        \"\"\"断点保存→加载→继续\"\"\"
        quick_loop.initialize_population()
        quick_loop.run(n_generations=2, safe=False)
        gen_before = quick_loop.generation
        fitness_before = quick_loop.best_body.fitness

        quick_loop.save_checkpoint("test_ckpt.pkl")

        # 加载恢复
        loop2 = EvolutionLoop.load_checkpoint("test_ckpt.pkl")
        assert loop2.generation == gen_before
        assert loop2.best_body.fitness == fitness_before

        # 继续进化
        loop2.run(n_generations=1, safe=False)
        assert loop2.generation > gen_before

    def test_manufacturing_export(self, quick_loop):
        \"\"\"进化 + 制造全流程\"\"\"
        quick_loop.initialize_population()
        quick_loop.run(n_generations=3, safe=False)

        from forgecraft.manufacturing import export_from_evolution_loop
        results = export_from_evolution_loop(quick_loop, output_dir="design_output")
        assert "stl" in results
        assert "urdf" in results
        assert "manufacturability_report" in results
"""

# 需要的外部资源
E2E_TEST_DEPENDENCIES = """
pytest-xdist   # 并行测试 (可选)
pytest-timeout # 超时保护 (集成测试必须)
pytest-cov     # 覆盖率报告
"""


# ══════════════════════════════════════════════════════════════════════
# D. P1-1: GPU 批推理引擎 — 种群级并行编码+rollout
# ══════════════════════════════════════════════════════════════════════

"""
目标: 同构形态共享 GNN 编码和 PPO rollout，消除冗余计算
原理: 当前每形态独立创建 env + 独立 GNN 编码 → 改为 batch 操作

瓶颈分析:
  - 当前: 50 形态 × 独立 GNN encode = 50 次 GPU kernel launch (overhead 高)
  - 优化后: 50 形态 × 1 次 batch GNN encode = 1 次 GPU kernel launch

实现蓝图:

forgecraft/rl/
├── batch_encoder.py    # 🆕 批量化 GNN 编码器
├── batch_ppo.py        # 🆕 批量化 PPO rollout
└── encoder.py          # 已有 单个编码器
"""

BATCH_GPU_IMPLEMENTATION = """
# ── batch_encoder.py 架构 ──

class BatchMorphologyEncoder:
    \"\"\"批量化形态编码器

    将 N 个异构 MechanicalBody 打包为单次 GPU 前向传播，
    通过 dynamic padding + attention mask 处理异构拓扑。
    \"\"\"

    def __init__(self, base_encoder: MorphologyEncoder):
        self.encoder = base_encoder

    def encode_batch(
        self, bodies: List[MechanicalBody]
    ) -> torch.Tensor:
        \"\"\"批量编码 N 个形态 → (N, embed_dim)

        实现:
          1. 对齐节点特征维度 (padding + mask)
          2. 构建 batch graph (N 个断开子图)
          3. 单次 GNN 前向传播
          4. 提取 per-graph 嵌入
        \"\"\"
        N = len(bodies)

        # Step 1: 收集所有子图的特征
        all_node_feats = []
        all_edge_index = []
        all_batch = []
        node_offset = 0

        for i, body in enumerate(bodies):
            feats, edges = self._extract_features(body)
            all_node_feats.append(feats)
            all_edge_index.append(edges + node_offset)
            all_batch.extend([i] * feats.shape[0])
            node_offset += feats.shape[0]

        # Step 2: stack + pad
        # 同类型形态通常节点数相近 → padding overhead <10%
        max_nodes = max(f.shape[0] for f in all_node_feats)
        padded_feats = torch.zeros(N, max_nodes, self.encoder.node_feat_dim)
        mask = torch.zeros(N, max_nodes, dtype=torch.bool)

        for i, f in enumerate(all_node_feats):
            n = f.shape[0]
            padded_feats[i, :n] = f
            mask[i, :n] = True

        # Step 3: 批量前向 (GPU 友好)
        node_feats_flat = torch.cat(all_node_feats, dim=0).to(device)
        edge_index_flat = torch.cat(all_edge_index, dim=1).to(device)
        batch_tensor = torch.tensor(all_batch, device=device)

        embeddings = self.encoder(node_feats_flat, edge_index_flat, batch_tensor)
        # embeddings: (total_nodes, hidden_dim)
        # → 对每个 graph pool (mean)
        result = self._global_mean_pool(embeddings, batch_tensor, N)
        return result  # (N, embed_dim)


# ── batch_ppo.py 架构 ──

class BatchPPORollout:
    \"\"\"批量化 PPO rollout

    核心: 将 N 个环境的 rollout 合并为单次 GPU 操作。

    适用场景:
      - 同构形态共享策略 (genesis backend)
      - 异构形态各自策略 (仅共享 encoder)

    借鉴 CleanRL 的 PPO 实现:
      https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
    \"\"\"

    def __init__(self, policy: nn.Module, envs):
        self.policy = policy
        self.envs = envs  # Genesis batch envs
        self.device = next(policy.parameters()).device

    @torch.no_grad()
    def rollout(
        self, num_steps: int
    ) -> Dict[str, torch.Tensor]:
        \"\"\"批量 rollout (N 个环境同时执行)

        返回: {obs, actions, rewards, dones, values, log_probs}
        \"\"\"
        obs = self.envs.reset()  # (N, obs_dim)

        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_values = []
        all_log_probs = []

        for _ in range(num_steps):
            # batch 推理 — 1 次 GPU kernel
            action, log_prob, _, value = self.policy.get_action_and_value(obs)

            # batch 执行 — 1 次 GPU kernel (Genesis)
            next_obs, reward, done, _ = self.envs.step(action)

            # 收集
            all_obs.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_dones.append(done)
            all_values.append(value)
            all_log_probs.append(log_prob)

            obs = next_obs

        return {
            "obs": torch.stack(all_obs),
            "actions": torch.stack(all_actions),
            "rewards": torch.stack(all_rewards),
            "dones": torch.stack(all_dones),
            "values": torch.stack(all_values),
            "log_probs": torch.stack(all_log_probs),
        }
"""


# ══════════════════════════════════════════════════════════════════════
# E. P1-2: 增量 GNN 编码 — 变异局部重计算
# ══════════════════════════════════════════════════════════════════════

"""
目标: 形态变异后仅重算受影响节点的嵌入，而非全图重算
原理: 拓扑变异只改变局部子图，大部分节点嵌入可复用

典型场景:
  - add_part_mutation: 新增 1 个零件 → 仅需编码新节点 + 邻居
  - delete_part_mutation: 删除 1 个零件 → 仅移除节点，其余不变
  - mutate_params: 参数变化 → 仅需更新受影响的节点特征

实现:

forgecraft/rl/
└── incremental_encoder.py  # 🆕 增量 GNN 编码器
"""

INCREMENTAL_GNN_IMPLEMENTATION = """
class IncrementalMorphologyEncoder:
    \"\"\"增量 GNN 编码器

    维护每个形态的嵌入缓存:
      {body_id: (node_embeddings, graph_embedding)}
    \"\"\"

    def __init__(self, encoder: MorphologyEncoder):
        self.encoder = encoder
        self._cache: Dict[int, torch.Tensor] = {}  # id → graph_embedding
        self._node_cache: Dict[int, List[torch.Tensor]] = {}  # id → node_embs

    def encode_with_diff(
        self,
        body: MechanicalBody,
        parent_body: Optional[MechanicalBody] = None,
        diff_info: Optional[Dict] = None,
    ) -> torch.Tensor:
        \"\"\"带差异信息的增量编码

        Args:
            body: 当前形态
            parent_body: 变异前的父形态 (None = 首次)
            diff_info: 变异详情 {"type": "add_part", "node_ids": [...], ...}

        返回: graph_embedding (morph_embed_dim,)
        \"\"\"
        bid = id(body)

        if parent_body is None or diff_info is None:
            # 首次编码: 全图计算
            emb = self.encoder.encode_body(body)
            self._cache[bid] = emb
            return emb

        pid = id(parent_body)
        if pid not in self._node_cache:
            # 缓存未命中: 回退到全图
            return self._full_encode(body)

        node_embs = self._node_cache[pid]  # 已有节点嵌入
        diff_type = diff_info["type"]

        if diff_type == "add_part":
            # 仅编码新节点 + K 跳邻居
            new_nodes = diff_info["node_ids"]
            affected = self._k_hop_neighbors(
                body, new_nodes, k=self.encoder.num_layers
            )
            # 重算 affected 节点
            updated = self.encoder.propagate(
                body, initial_embs=node_embs, update_mask=affected
            )
        elif diff_type == "delete_part":
            # 移除节点 + 清理受影响的邻居
            removed = diff_info["node_ids"]
            updated = self._recompute_neighbors(
                body, node_embs, removed
            )
        elif diff_type == "mutate_params":
            # 仅更新特征向量，不改拓扑
            updated = self._update_node_features(
                body, node_embs, diff_info["param_delta"]
            )
        else:
            return self._full_encode(body)

        # pool → graph embedding
        emb = updated.mean(dim=0)
        self._node_cache[bid] = updated
        self._cache[bid] = emb
        return emb

    def _compute_affected_mask(
        self, body: MechanicalBody, node_ids: List[int], k: int
    ) -> torch.Tensor:
        \"\"\"计算 K 跳邻居掩码

        返回: (n_nodes,) bool tensor
        时间复杂度: O(k * avg_degree)
        \"\"\"
        n = body.num_parts()
        adj = self._adjacency_matrix(body)
        # BFS from node_ids, k steps
        mask = torch.zeros(n, dtype=torch.bool)
        frontier = set(node_ids)
        for _ in range(k):
            mask[list(frontier)] = True
            frontier = {
                j for i in frontier
                for j in range(n)
                if adj[i, j] and j not in frontier
            }
        return mask
"""

# 性能预期
INCREMENTAL_GNN_BENCHMARK = """
场景分析 (10 零件形态, 平均度 3):

| 变异类型 | 受影响节点 | 全编码节点 | 加速比 |
|----------|-----------|-----------|--------|
| add_part | 1新 + 邻居(~3) | 10(全部) | ~2.5x |
| delete_part | 邻居(~3) | 10(全部) | ~3x |
| mutate_params | 1节点 + 邻居(~3) | 10(全部) | ~2.5x |
| 大变异 (5+节点变) | ~全部 | 10(全部) | ~1x (无加速) |
"""


# ══════════════════════════════════════════════════════════════════════
# F. P1-3: 制造性实时反馈 — 进化中嵌入工艺约束
# ══════════════════════════════════════════════════════════════════════

"""
目标: 进化过程中实时计算制造性指标，指导选择/变异方向
原理: 在适应度评估阶段嵌入轻量级制造性检查 (<1ms per body)

检查项 (借鉴 3D 打印/CNC 设计规范):
  1. 壁厚检查: 零件壁厚 ≥ 最小壁厚 (避免打印失败)
  2. 悬垂检查: 零件角度 ≤ 45° (无支撑打印)
  3. 干涉检查: 无相邻零件体积重叠
  4. 连接强度: joint 区域应力集中评估
  5. 材料浪费: 体积利用率

实现:

forgecraft/manufacturing/
└── realtime_feedback.py  # 🆕 实时制造性反馈
"""

REALTIME_FEEDBACK_IMPLEMENTATION = """
class RealtimeManufacturability:
    \"\"\"实时制造性评估器

    嵌入进化循环，在适应度评估时同步计算。
    所有计算 ≤ 1ms (30 零件形态)。
    \"\"\"

    def __init__(self, catalog: Dict[str, PartSpec]):
        self.catalog = catalog
        self.checks = [
            WallThicknessCheck(),
            OverhangCheck(),
            InterferenceCheck(),
            JointStressCheck(),
            MaterialEfficiencyCheck(),
        ]

    def evaluate(
        self, body: MechanicalBody
    ) -> Tuple[float, Dict[str, float]]:
        \"\"\"实时评估制造性

        返回: (overall_score, component_scores)
        overall_score: 0.0 (不可制造) - 1.0 (完美)
        \"\"\"
        scores = {}
        for check in self.checks:
            scores[check.name], detail = check.evaluate(body, self.catalog)

        # 乘法融合 (任意一环失败 → 整体失败)
        overall = 1.0
        for s in scores.values():
            overall *= s

        return overall, scores


class OverhangCheck:
    \"\"\"悬垂检查

    规则: 任何零件的底面边缘不能有超过 45° 的悬空。
    简化计算: 检查每个零件下方是否有支撑结构。
    \"\"\"

    def evaluate(self, body, catalog) -> Tuple[float, str]:
        n_overhang = 0
        n_total = 0

        for part in body.parts():
            # 检查该零件下方是否有相邻零件或地面
            has_support = self._has_support_below(body, part)

            if not has_support:
                # 检查零件角度是否可自支撑
                angle = self._compute_overhang_angle(body, part)
                if angle > 45:
                    n_overhang += 1

            n_total += 1

        if n_total == 0:
            return 1.0, "empty"

        score = 1.0 - (n_overhang / n_total) * 0.3
        return max(0.0, score), f"{n_overhang}/{n_total} overhangs"


class InterferenceCheck:
    \"\"\"干涉检查

    规则: 零件体积不能重叠。
    简化计算: AABB 重叠 + 近似体积交集。
    \"\"\"

    def evaluate(self, body, catalog) -> Tuple[float, str]:
        overlaps = 0
        parts = list(body.parts())
        n = len(parts)

        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = parts[i], parts[j]
                if self._aabb_overlap(pi, pj):
                    # 进一步检查: 形状级重叠
                    if self._shape_overlap(pi, pj):
                        overlaps += 1

        total_pairs = n * (n - 1) / 2
        if total_pairs == 0:
            return 1.0, "single part"

        score = 1.0 - (overlaps / total_pairs)
        return max(0.0, score), f"{overlaps}/{int(total_pairs)} overlaps"


# ── 集成到进化循环 ──
"""
在 EvolutionLoop.evolve_one_generation() 中:

# 制造性反馈添加到适应度
realtime_mfg = RealtimeManufacturability(catalog)

for body in population:
    mfg_score, mfg_details = realtime_mfg.evaluate(body)
    # 融入到适应度 (weight ~0.15)
    body.fitness = body.fitness * (0.85 + 0.15 * mfg_score)
    body.fitness_components['manufacturability'] = mfg_score

# 选择时优先选择可制造性高的
# Breeder 在精英选择时加入制造性权重
"""
"""


# ══════════════════════════════════════════════════════════════════════
# G. P2-1: CI/CD + Docker — 可复现构建
# ══════════════════════════════════════════════════════════════════════

"""
目标: GitHub Actions 自动化测试 + Docker 标准化环境
外部资源: GitHub Actions (免费), Docker Hub

文件规划:
├── .github/workflows/
│   ├── test.yml           # 每次 push 运行测试
│   ├── lint.yml           # 代码检查 (ruff)
│   └── release.yml        # 发布到 PyPI
├── Dockerfile             # CPU 版本
├── Dockerfile.cuda        # CUDA 版本
├── docker-compose.yml     # 本地开发
├── pyproject.toml         # 标准化项目配置
└── .dockerignore
"""

CI_CD_IMPLEMENTATION = """
# ── .github/workflows/test.yml ──

name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install pytest pytest-cov pytest-xdist

      - name: Run tests
        run: |
          pytest forgecraft/testing/ -v --tb=short \
            --cov=forgecraft \
            --cov-report=xml \
            -n auto

      - name: Upload coverage
        uses: codecov/codecov-action@v4


# ── Dockerfile ──
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install -e ".[gpu]"

ENV FORGECRAFT_BACKEND=genesis
CMD ["python", "-m", "forgecraft.main", "--help"]


# ── pyproject.toml (关键片段) ──
[project]
name = "forgecraft"
version = "0.4.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "mujoco>=3.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "pytest-xdist", "ruff"]
gpu = ["genesis-world>=1.1"]
dist = ["ray>=2.0"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["forgecraft/testing"]
addopts = "-v --tb=short"
timeout = 300
"""


# ══════════════════════════════════════════════════════════════════════
# H. P2-2: ONNX Runtime 部署 — 策略导出验证
# ══════════════════════════════════════════════════════════════════════

"""
目标: 进化完成的策略网络 → ONNX → 嵌入式部署
外部资源: Microsoft ONNX Runtime, onnx2torch

当前状态:
  - manufacturing/__init__.py 已有 export_onnx() (第338行)
  - 生成: policy.onnx + inference.py + cpp_controller.h
  - 但缺少运行时验证

增强点:
  1. 端到端测试: 导出 → 加载 → 推理 → 对比输出
  2. 量化支持: FP16/INT8 量化以适配嵌入式设备
  3. ONNX Runtime Profiling: 性能分析
"""

ONNX_IMPLEMENTATION = """
# ── 增强后的 export_onnx 验证流程 ──

def export_onnx_verified(
    policy_state: dict,
    obs_dim: int,
    act_dim: int,
    morph_dim: int,
    output_dir: str,
    verify: bool = True,
    quantize: Optional[str] = None,  # "fp16" or "int8"
) -> dict:
    \"\"\"导出 + 验证 + 量化 一站式

    流程:
      1. 从 checkpoint 重建 PyTorch 模型
      2. torch.onnx.export → policy.onnx
      3. onnx.checker.check_model → 验证格式
      4. onnxruntime.InferenceSession → 加载推理
      5. 对比输出 (PyTorch vs ONNX) → MSE ≤ 1e-6
      6. [可选] onnxruntime.quantization → FP16/INT8

    返回: {onnx_path, verification_passed, mse, inference_time_ms}
    \"\"\"
    import onnx
    import onnxruntime

    # 1. 导出
    policy = build_policy_from_state(policy_state, obs_dim, act_dim, morph_dim)
    dummy_obs = torch.randn(1, obs_dim)
    dummy_morph = torch.randn(1, morph_dim)

    onnx_path = os.path.join(output_dir, "policy.onnx")
    torch.onnx.export(
        policy,
        (dummy_obs, dummy_morph),
        onnx_path,
        input_names=["observation", "morphology_embedding"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch"},
            "morphology_embedding": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=17,
    )

    # 2. 验证格式
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    # 3. 推理验证
    if verify:
        session = onnxruntime.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        # PyTorch 推理
        with torch.no_grad():
            torch_out = policy(dummy_obs, dummy_morph).numpy()

        # ONNX 推理
        onnx_out = session.run(
            None,
            {"observation": dummy_obs.numpy(), "morphology_embedding": dummy_morph.numpy()},
        )[0]

        mse = np.mean((torch_out - onnx_out) ** 2)
        verified = mse < 1e-6

        # 4. 量化 (可选)
        if quantize == "fp16":
            from onnxruntime.transformers import float16
            fp16_path = onnx_path.replace(".onnx", "_fp16.onnx")
            float16.convert_float_to_float16(model, fp16_path)
            # 重新验证量化模型
            onnx_path = fp16_path

        return {
            "onnx_path": onnx_path,
            "verification_passed": verified,
            "mse": float(mse),
            "quantization": quantize,
        }

    return {"onnx_path": onnx_path, "verification_passed": None}
"""


# ══════════════════════════════════════════════════════════════════════
# I. 总体实施路线图
# ══════════════════════════════════════════════════════════════════════

ROADMAP = """
┌──────────────────────────────────────────────────────────────┐
│                  ForgeCraft v0.4 路线图                         │
├──────────┬─────────────────────────────────────────────────────┤
│ 阶段     │ 内容                                  预计变更行数    │
├──────────┼─────────────────────────────────────────────────────┤
│          │                                                     │
│ Phase 9  │ B. Genesis GPU 仿真后端              +~600 行        │
│ (核心)   │   - genesis_backend.py (Genesis适配器)               │
│          │   - backend_registry.py (后端切换)                   │
│          │   - 修改 evaluator.py 支持后端选择                   │
│          │   - 兼容性测试 (10个形态对比)                        │
│          │                                                     │
│          │ 外部依赖: pip install genesis-world                  │
│          │ 预期收益: 仿真速度 100-1000x ↑                       │
│          │ 风险: MJCF 兼容性 (中等)                              │
│          │                                                     │
├──────────┼─────────────────────────────────────────────────────┤
│          │                                                     │
│ Phase 10 │ D. GPU 批推理引擎                    +~400 行        │
│ (核心)   │   - batch_encoder.py (批量 GNN)                     │
│          │   - batch_ppo.py (批量 rollout)                     │
│          │   - 修改 evaluator.py 使用批量接口                   │
│          │                                                     │
│          │ 预期收益: 每代加速 3-5x                               │
│          │ 无外部依赖                                           │
│          │                                                     │
├──────────┼─────────────────────────────────────────────────────┤
│          │                                                     │
│ Phase 11 │ E. 增量 GNN 编码                    +~300 行        │
│ (可选)   │   - incremental_encoder.py                          │
│          │   - 缓存管理 + K跳邻居计算                           │
│          │                                                     │
│          │ 预期收益: 编码加速 30% (变异场景)                    │
│          │ 无外部依赖                                           │
│          │                                                     │
├──────────┼─────────────────────────────────────────────────────┤
│          │                                                     │
│ Phase 12 │ C. 端到端集成测试                   +~250 行        │
│ (可靠)   │   - conftest.py (共享 fixtures)                     │
│          │   - test_full_evolution.py                          │
│          │   - test_manufacturing_e2e.py                       │
│          │                                                     │
│          │ 外部依赖: pytest-xdist, pytest-timeout               │
│          │                                                     │
├──────────┼─────────────────────────────────────────────────────┤
│          │                                                     │
│ Phase 13 │ F. 制造性实时反馈                  +~350 行         │
│ (可选)   │   - realtime_feedback.py                             │
│          │   - 集成到 EvolutionLoop                             │
│          │                                                     │
│          │ 预期收益: 制造成功率 +40%                             │
│          │                                                     │
├──────────┼─────────────────────────────────────────────────────┤
│          │                                                     │
│ Phase 14 │ G. CI/CD + Docker                    +~150 行       │
│ (工程)   │   - .github/workflows/*.yml                         │
│          │   - Dockerfile + pyproject.toml                     │
│          │                                                     │
│          │ 无外部依赖                                           │
│          │                                                     │
├──────────┼─────────────────────────────────────────────────────┤
│          │                                                     │
│ Phase 15 │ H. ONNX 部署验证                    +~100 行        │
│ (工程)   │   - 增强 export_onnx_verified()                      │
│          │   - 端到端推理测试                                   │
│          │                                                     │
│          │ 外部依赖: onnxruntime, onnx                          │
│          │                                                     │
├──────────┼─────────────────────────────────────────────────────┤
│          │ 总计变更                                +~2,150 行   │
│          │ 当前代码                                    24,749 行 │
│          │ 预期最终                                    26,900 行 │
│          │ 测试目标                                    180-200 个 │
└──────────┴─────────────────────────────────────────────────────┘

推荐执行顺序:
  1. Phase 9  (Genesis 后端)   — 最大杠杆，独立模块
  2. Phase 12 (集成测试)       — 尽早建立安全网
  3. Phase 10 (GPU 批推理)     — 需要 Genesis 后端就位
  4. Phase 14 (CI/CD)          — 自动化保护
  5. Phase 11 (增量 GNN)       — 锦上添花
  6. Phase 13 (制造性反馈)     — 锦上添花
  7. Phase 15 (ONNX 验证)      — 部署闭环

资源需求:
  - GPU: RTX 3060+ / 8GB VRAM+ (Genesis GPU 仿真)
  - CI: GitHub Actions (免费)
  - 外部包: genesis-world, onnxruntime, pytest-xdist
  - 不需额外服务器或付费服务
"""
