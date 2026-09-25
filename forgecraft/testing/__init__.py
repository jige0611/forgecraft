# forgecraft.testing — 测试模块
#
# 用法:
#   单元测试:  pytest forgecraft/testing/test_suite.py -v
#   集成测试:  pytest forgecraft/testing/integration/ -v
#   性能基准:  python -m forgecraft.testing.benchmark --quick
#   全量回归:  pytest forgecraft/testing/ -v
#
# 测试结构:
#   conftest.py          — 共享 fixtures (session-scoped)
#   test_suite.py        — 单元测试 (181 个, 20 类)
#   integration/         — 端到端集成测试
#   benchmark.py         — 性能基准 (6 项)

__all__: list = []
