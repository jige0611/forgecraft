"""
ForgeCraft cProfile 性能基准

用法:
  python -m forgecraft.testing.benchmark_profile
  python -m forgecraft.testing.benchmark_profile --output profile.stats
"""

import cProfile
import pstats
import argparse
import os
import sys
import tempfile
from pathlib import Path

PROFILE_OUTPUT = os.environ.get("FORGECRAFT_PROFILE", "forgecraft_profile.stats")


def _benchmark_body_gen(n: int = 100):
    """Benchmark: body generation throughput"""
    from forgecraft.core.loader import load_catalog
    from forgecraft.core.generator import BodyGenerator
    catalog = load_catalog().to_part_specs()
    gen = BodyGenerator(catalog, seed=42)
    for _ in range(n):
        gen.generate_random_body(min_parts=3, max_parts=8)


def _benchmark_morph_encoder(n: int = 20):
    """Benchmark: morphology encoder forward pass"""
    import torch
    from forgecraft.core.loader import load_catalog
    from forgecraft.core.generator import BodyGenerator
    from forgecraft.rl.encoder import MorphologyEncoder

    catalog = load_catalog().to_part_specs()
    type_registry = {pt: i for i, pt in enumerate(catalog.keys())}
    gen = BodyGenerator(catalog, seed=42)
    encoder = MorphologyEncoder(
        node_feat_dim=32, edge_feat_dim=16, hidden_dim=64, output_dim=64,
        num_layers=3, part_type_registry=type_registry,
    )

    for _ in range(n):
        body = gen.generate_random_body(min_parts=4, max_parts=10)
        with torch.no_grad():
            encoder.encode_body(body)


def _benchmark_evolution_step(n: int = 3):
    """Benchmark: one evolution generation"""
    from forgecraft.core.loader import load_catalog
    from forgecraft.core.generator import BodyGenerator
    from forgecraft.evolution.breeder import PopulationBreeder
    from forgecraft.config import EvolutionConfig

    catalog = load_catalog().to_part_specs()
    gen = BodyGenerator(catalog, seed=42)
    config = EvolutionConfig()
    config.population_size = 20

    population = gen.generate_initial_population(size=20, min_parts=3, max_parts=8)
    breeder = PopulationBreeder(config, gen, seed=42)

    for _ in range(n):
        offspring = breeder.breed_next_generation(population)


def _benchmark_manufacturing(n: int = 10):
    """Benchmark: manufacturing pipeline"""
    import numpy as np
    from forgecraft.core.loader import load_catalog
    from forgecraft.core.generator import BodyGenerator
    from forgecraft.manufacturing.spec_filter import spec_body
    from forgecraft.manufacturing.tolerances import compute_joint_clearance, DEFAULT_FDM

    catalog = load_catalog().to_part_specs()
    gen = BodyGenerator(catalog, seed=42)

    for _ in range(n):
        body = gen.generate_random_body(min_parts=3, max_parts=6)
        spec_body(body)


def run_profile(output: str = PROFILE_OUTPUT):
    """Run all benchmarks under cProfile"""
    profiler = cProfile.Profile()

    print("Profiling body generation...")
    profiler.runcall(_benchmark_body_gen, 50)

    print("Profiling morph encoder...")
    profiler.runcall(_benchmark_morph_encoder, 10)

    print("Profiling evolution step...")
    profiler.runcall(_benchmark_evolution_step, 2)

    print("Profiling manufacturing...")
    profiler.runcall(_benchmark_manufacturing, 10)

    # Save stats
    profiler.dump_stats(output)
    print(f"\nProfile saved to: {output}")

    # Print top 20 hotspots
    stats = pstats.Stats(output)
    stats.strip_dirs().sort_stats("cumtime").print_stats(20)

    # Also show by internal time (most expensive functions)
    print("\n--- Top by internal time ---")
    stats.strip_dirs().sort_stats("tottime").print_stats(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForgeCraft cProfile performance benchmark")
    parser.add_argument("--output", "-o", default=PROFILE_OUTPUT, help="Profile output file")
    args = parser.parse_args()
    run_profile(args.output)
