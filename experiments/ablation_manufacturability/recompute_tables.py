"""Recompute Tables 3, 5 and 6 of the manuscript from the stored run files.

The paper claims that every cell of these tables can be reproduced from the
repository. This script is that claim. It reads only the JSON files the runs
wrote, recomputes the statistics with an explicit tie rule, and writes both a
machine-readable summary and a printed table.

Statistics used in the paper
----------------------------
sigma(fitness)  population standard deviation of the stored fitness vector
dmfg            max(mfg) - min(mfg) within the final population
r               w * dmfg / sigma(fitness)
discordant      pairs (i, j) whose order reverses when w * mfg is subtracted
                from fitness; pairs tied on both values are excluded
at-risk         pairs (i, j) with |f_i - f_j| < w * |m_i - m_j|; by the bound in
                Section 7.5 the discordant count can never exceed it

Usage
-----
    .venv/Scripts/python.exe experiments/ablation_manufacturability/recompute_tables.py
"""

from __future__ import annotations

import json
import statistics
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHTS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)
CATALOGUES = ("primitives", "parameterized_parts_v7")
SEEDS = (7, 13, 42)


def _read(name: str) -> dict:
    with (HERE / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _vectors_v040(catalogue: str) -> tuple[list[float], list[float], float]:
    """Final population of the shipped build: in-the-loop arm, w = 0.05."""
    data = _read(f"results_{catalogue}.json")
    arm = data["arms"]["in-the-loop"]
    fitness = [row["fitness"] for row in arm["individuals"]]
    mfg = [row["manufacturability"] for row in arm["individuals"]]
    return fitness, mfg, float(arm["w_mfg_effective"])


def _vectors_repaired(name: str) -> tuple[list[float], list[float], float]:
    data = _read(name)
    return list(data["raw"]["fitness"]), list(data["raw"]["mfg"]), float(data["w_mfg"])


def counterfactual(fitness: list[float], mfg: list[float], w: float) -> dict[str, object]:
    n = len(fitness)
    total = n * (n - 1) // 2
    counter = [f - w * m for f, m in zip(fitness, mfg)]
    discordant = 0
    at_risk = 0
    for i, j in combinations(range(n), 2):
        df = fitness[i] - fitness[j]
        dg = counter[i] - counter[j]
        if df == 0.0 and dg == 0.0:
            continue
        if df * dg < 0:
            discordant += 1
        if abs(df) < w * abs(mfg[i] - mfg[j]):
            at_risk += 1

    def top(changed: int) -> int:
        before = sorted(range(n), key=lambda k: -fitness[k])[:changed]
        after = sorted(range(n), key=lambda k: -counter[k])[:changed]
        return changed - len(set(before) & set(after))

    sigma = statistics.pstdev(fitness)
    spread = max(mfg) - min(mfg)
    return {
        "n": n,
        "sigma": sigma,
        "dmfg": spread,
        "r": w * spread / sigma if sigma else float("inf"),
        "discordant_pairs": discordant,
        "at_risk_pairs": at_risk,
        "pairs_total": total,
        "top5_changes": top(5),
        "top10_changes": top(10),
    }


def table_3() -> list[dict[str, object]]:
    rows = []
    for catalogue in CATALOGUES:
        for build in ("v0.4.0", "repaired"):
            if build == "v0.4.0":
                fitness, mfg, w = _vectors_v040(catalogue)
            else:
                fitness, mfg, w = _vectors_repaired(f"results_v2_{catalogue}_s42.json")
            rows.append({"catalogue": catalogue, "build": build, "w": w,
                         **counterfactual(fitness, mfg, w)})
    return rows


def table_5() -> list[dict[str, object]]:
    rows = []
    for catalogue in CATALOGUES:
        fitness, mfg, _ = _vectors_repaired(f"results_v2_{catalogue}_s42.json")
        for w in WEIGHTS:
            rows.append({"catalogue": catalogue, "w": w,
                         **counterfactual(fitness, mfg, w)})
    return rows


def table_6() -> list[dict[str, object]]:
    rows = []
    for seed in SEEDS:
        base = _read(f"results_v2_parameterized_parts_v7_s{seed}.json")
        treat = _read(f"results_v2_parameterized_parts_v7_s{seed}_w0.5.json")
        base_mfg = list(base["raw"]["mfg"])
        treat_mfg = list(treat["raw"]["mfg"])
        rows.append({
            "seed": seed,
            "baseline_mean": statistics.mean(base_mfg),
            "treatment_mean": statistics.mean(treat_mfg),
            "delta_mean": statistics.mean(treat_mfg) - statistics.mean(base_mfg),
            "delta_median": statistics.median(treat_mfg) - statistics.median(base_mfg),
            "baseline_below_0.8": sum(1 for value in base_mfg if value < 0.8),
            "treatment_below_0.8": sum(1 for value in treat_mfg if value < 0.8),
            "treatment_r": counterfactual(treat["raw"]["fitness"], treat_mfg,
                                          float(treat["w_mfg"]))["r"],
        })
    return rows


def main() -> None:
    t3, t5, t6 = table_3(), table_5(), table_6()
    out = HERE / "recomputed_tables.json"
    out.write_text(json.dumps({"table_3": t3, "table_5": t5, "table_6": t6},
                              indent=2), encoding="utf-8")

    print("Table 3  (w as run)")
    for row in t3:
        print(f"  {row['catalogue']:<24} {row['build']:<9} sigma={row['sigma']:.4f} "
              f"dmfg={row['dmfg']:.4f} r={row['r']:.4f} disc={row['discordant_pairs']} "
              f"top5={row['top5_changes']} top10={row['top10_changes']}")

    print("\nTable 5  (repaired build, seed 42)")
    for row in t5:
        print(f"  {row['catalogue']:<24} w={row['w']:<5} r={row['r']:.3f} "
              f"at-risk={row['at_risk_pairs']:>3}/{row['pairs_total']} "
              f"disc={row['discordant_pairs']:>2} top5={row['top5_changes']} "
              f"top10={row['top10_changes']}")

    print("\nTable 6  (parameterized_parts_v7, w = 0.05 vs 0.50)")
    for row in t6:
        print(f"  seed {row['seed']:>2}: base={row['baseline_mean']:.4f} "
              f"treat={row['treatment_mean']:.4f} dmean={row['delta_mean']:+.4f} "
              f"dmed={row['delta_median']:+.4f} "
              f"<0.8: {row['baseline_below_0.8']}->{row['treatment_below_0.8']} "
              f"r_treat={row['treatment_r']:.3f}")

    print(f"\nwrote {out.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
