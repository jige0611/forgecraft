"""
单元测试: 进化选择 (NSGA-II / Tournament)
"""

import numpy as np
from forgecraft.core.morphology import MechanicalBody, Part, Joint
from forgecraft.evolution.selection import (
    tournament_select, non_dominated_sort, crowding_distance,
    pareto_elites, _dominates
)


def make_body(name, fitness, comps=None):
    body = MechanicalBody(name)
    root = Part("base", {"length": 0.1}, np.array([0, 0, 0.5]))
    body.add_part(root)
    body.fitness = fitness
    if comps:
        body.fitness_components = comps
    return body


class TestTournament:
    def test_basic_select(self):
        pop = [make_body(f"b{i}", float(i)) for i in range(5)]
        selected = tournament_select(pop, 3, tournament_size=3, rng=np.random.RandomState(42))
        assert len(selected) == 3

    def test_select_all_when_fewer(self):
        pop = [make_body(f"b{i}", float(i)) for i in range(3)]
        selected = tournament_select(pop, 10)
        assert len(selected) == 3


class TestDominance:
    def test_dominates_maximize(self):
        a = np.array([2.0, 3.0])
        b = np.array([1.0, 2.0])
        assert _dominates(a, b, ["maximize", "maximize"]) is True
        assert _dominates(b, a, ["maximize", "maximize"]) is False

    def test_no_dominance_equal(self):
        a = np.array([2.0, 3.0])
        b = np.array([2.0, 3.0])
        assert _dominates(a, b, ["maximize", "maximize"]) is False

    def test_dominates_with_minimize(self):
        a = np.array([1.0, 2.0])  # 更低能耗
        b = np.array([2.0, 1.0])  # 更低能耗但不同维度
        # a[1]=2 < b[1]=1? no. 所以a不支配b
        d = _dominates(a, b, ["maximize", "minimize"])
        # a在obj0 (max)上 1<2 差, 在obj1(min)上 2>1 差, 不支配
        assert d is False


class TestNonDominatedSort:
    def test_two_fronts(self):
        obj = np.array([
            [3.0, 3.0],  # Pareto front
            [5.0, 5.0],  # Pareto front
            [2.0, 2.0],  # dominated
        ])
        fronts = non_dominated_sort(obj, ["maximize", "maximize"])
        assert len(fronts) >= 1
        # 前两个是前沿(索引0,1), 第三个(索引2)应该被支配在后面
        fronts_flat = [i for f in fronts for i in f]
        assert 2 in fronts_flat
        assert 2 not in fronts[0]

    def test_all_dominated(self):
        obj = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        fronts = non_dominated_sort(obj, ["maximize", "maximize"])
        assert len(fronts) == 3


class TestCrowdingDistance:
    def test_infinity_for_extremes(self):
        obj = np.array([[1.0, 1.0], [2.0, 2.0], [10.0, 10.0]])
        idx = [0, 1, 2]
        cd = crowding_distance(obj, idx, ["maximize", "maximize"])
        assert cd[0] == float("inf")
        assert cd[2] == float("inf")


class TestParetoElites:
    def test_select_elites(self):
        pop = []
        obj = [
            {"speed": 3.0, "energy": 1.0},
            {"speed": 5.0, "energy": 2.0},
            {"speed": 2.0, "energy": 0.5},
        ]
        for i, c in enumerate(obj):
            b = make_body(f"b{i}", 0.0, comps=c)
            pop.append(b)
        elites = pareto_elites(pop, 2, ["speed", "energy"],
                                ["maximize", "maximize"])
        assert len(elites) == 2
