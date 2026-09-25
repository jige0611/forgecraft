ForgeCraft
==========

**基于强化学习的机械形态协同进化框架**

.. toctree::
   :maxdepth: 2
   :caption: 目录

   introduction
   api/core
   api/evolution
   api/rl
   api/simulation
   api/manufacturing
   api/evaluation
   api/config
   api/experiment

简介
----

ForgeCraft 将进化算法 (EA)、深度强化学习 (RL)、物理仿真 (MuJoCo) 结合，
自动设计、评估、进化出最优的机械结构形态。

核心流程
^^^^^^^^

1. **生成** — 从零件箱 (YAML catalog) 随机组装形态图
2. **编码** — GNN 编码器将异构形态图 → 固定维度嵌入
3. **仿真** — MuJoCo/Genesis 物理环境评估行为
4. **进化** — NSGA-II + MAP-Elites 多目标选择/交叉/变异
5. **制造** — 公差补偿 + 支撑生成 + STEP/ONNX 导出

快速开始
^^^^^^^^

.. code-block:: bash

   # 安装
   pip install -e ".[all]"

   # 快速进化
   python -m forgecraft run --quick

   # 完整进化 + 预训练 + 制造导出
   python -m forgecraft run \\
     --generations 100 --population 64 \\
     --pretrain --pretrain-epochs 200 \\
     --map-elites --export

   # 查看可用配置
   python -m forgecraft list-specs

   # 启动仪表盘
   python -m forgecraft dashboard

索引与搜索
^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
