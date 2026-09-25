系统架构
========

ForgeCraft 8 模块管线::

    config.py -- 全局配置中心
         |
    core -- 形态图/生成器/加载器/验证/缓存
         |
    evolution -- 进化引擎 (breeder/selection/operators)
         |                    |
         v                    v
    simulation -- MuJoCo物理仿真
         |                    |
         v                    v
    evaluation -- 适应度/GPU评估引擎
         |
    manufacturing -- 公差/支撑/连接/BOM/STEP/ONNX
         |
    rl -- PPO/GNN编码器/预训练/迁移学习

形态表示
--------

形态以 **有向无环图 (DAG)** 表示，基于 ``networkx.DiGraph``:

- **节点 (Part)**: 物理零件，含类型/参数/位置/四元数
- **边 (Joint)**: 关节连接，含类型/轴/范围/阻尼

基因型 = Part 参数 + 图拓扑，表型 = MuJoCo 仿真行为。

YAML 配置
---------

零件箱 (catalog) 和任务 (task) 均通过 YAML 文件定义:

- **Catalog**: 定义可用零件类型、物理属性、参数范围
- **Task**: 定义仿真参数、奖励分量、终止条件

内置 17 个零件箱 + 7 个任务配置，位于 ``configs/`` 目录。
