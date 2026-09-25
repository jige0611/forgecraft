"""Few-shot 示例 — 帮助 LLM 理解设计推理模式"""

# ══════════════════════════════════════════════════════════════
#  Catalog 选择示例
# ══════════════════════════════════════════════════════════════

CATALOG_SELECTION_EXAMPLE = """
## 示例: 速度任务 Catalog 选择

### 用户
> 设计一个在平地上最快移动的机器人, 地形平坦, 无障碍物。

### 智能体分析
任务分析:
- 核心需求: 最大前向位移速度
- 物理约束: 需要轻量化 + 高效驱动
- 地形: 平坦 → 轮式最优 (滚动摩擦 << 滑动摩擦)
- 稳定需求: 低 (不会翻越障碍)

候选方案:
1. [default catalog + wheel + 2 motors]: 最简配置, 差速转向
2. [hoppers catalog]: 弹跳移动, 间歇式高速但不持续
3. [biped catalog]: 双足步态, 速度上限低

### 推荐
选择 default catalog (包含 wheel + motor + segment), 理由:
- 轮式在平地上的能量效率是腿式的 3-5 倍
- 2 motor 差速转向是最简驱动方案
- 不需要复杂拓扑, default 目录足够

### 配置
```yaml
catalog: default
evolution_strategy: map_elites
population_size: 80
bc_names: [speed, stability]
```
"""

# ══════════════════════════════════════════════════════════════
#  停滞处理示例
# ══════════════════════════════════════════════════════════════

STAGNATION_HANDLING_EXAMPLE = """
## 示例: QD-Score 停滞处理

### 当前状态
- 代数: 120/300
- QD-Score: 0.85 (停滞 25 代)
- 覆盖率: 42% (停滞 20 代)
- 种群多样性: 0.08 (临界)

### 智能体分析
信号检测:
- [coverage_stall]: 覆盖率连续 20 代无明显增长
- [diversity_collapse]: 种群多样性 < 0.1 阈值

候选方案:
1. [增加变异率 1.5x + 切换 50% random emitter]: 短期增加探索
2. [切换 CMA-ME]: 在已发现的精英附近局部精化
3. [保持现状]: 等待自然突破 (不推荐, 已停滞 25 代)

### 推荐
方案 1 → 方案 2 (两阶段), 理由:
- 先注入多样性 (3-5 代) 打破当前局部最优
- 再切换 CMA-ME 进行精化搜索
- 如果 10 代后无改善, 考虑切换 catalog

### 调整
```yaml
stage_1:
  generations: 3
  mutation_rate: 0.52  # x1.5
  random_emitter_ratio: 0.5

stage_2:
  strategy: cma_me
  sigma0: 0.2
```
"""

# ══════════════════════════════════════════════════════════════
#  Catalog 生成示例
# ══════════════════════════════════════════════════════════════

CATALOG_GENERATION_EXAMPLE = """
## 示例: 为新任务生成 Catalog

### 用户
> 设计一个能够在水下移动并抓取物体的机器人

### 智能体分析
任务分析:
- 核心需求: 水下移动 + 抓取
- 物理约束: 水阻力 → 需要流线型; 浮力 → 需要压载
- 特殊需求: 防水螺纹推进器 + 防腐蚀材料 + 水下抓取

候选零件:
- structural: [streamlined_body, ballast_tank, frame]
- actuation: [underwater_thruster, waterproof_servo]
- locomotion: [propeller, fin]
- manipulation: [waterproof_gripper, suction_cup]

### 生成
```yaml
name: underwater_manipulation
description: 水下移动与抓取
constraints:
  max_depth: 10
  max_parts: 30
  require_root: true

parts:
  streamlined_body:
    geometry: {type: capsule}
    physics: {mass: 0.5, drag_coefficient: 0.1}
    size: {length: [0.1, 0.5], radius: [0.02, 0.1]}

  underwater_thruster:
    geometry: {type: cylinder}
    actuated: true
    joint: {type: screw, axis: [0, 1, 0], torque: [0.1, 50]}
    physics: {waterproof: true}

  waterproof_gripper:
    geometry: {type: cylinder}
    actuated: true
    joint: {type: hinge, axis: [0, 0, 1], range: [0, 1.5]}
    physics: {friction: [0.8, 0.8, 0.8], waterproof: true}

  ballast_tank:
    geometry: {type: box}
    physics: {mass_range: [0.1, 2.0], buoyancy: adjustable}
    size: {length: [0.05, 0.2], width: [0.05, 0.2], height: [0.05, 0.2]}
```
"""
