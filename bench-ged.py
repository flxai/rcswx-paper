#!/usr/bin/env python3
import click
import json
import os
import time

import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from networkx.generators.trees import random_unlabeled_tree
from tqdm import tqdm


def run_trial(method, n_nodes, seed, cache_file):
    G1 = random_unlabeled_tree(n_nodes, seed=seed)
    G2 = random_unlabeled_tree(n_nodes, seed=seed + 1)
    if method == "sep":
        from sepx.similarity import graph_edit_distance
        t0 = time.time()
        graph_edit_distance(G1, G2)
    elif method == "ged4py":
        from ged4py import GraphEditDistance
        ged = GraphEditDistance(1, 1, 1, 1)
        t0 = time.time()
        ged.compare([G1, G2], None)
    else:
        raise ValueError("Unknown method")
    elapsed = time.time() - t0
    result = {"method": method, "n_nodes": n_nodes, "seed": seed, "time": elapsed}
    with open(cache_file, "w") as f:
        json.dump(result, f)
    return result

@click.command()
@click.argument("method", type=click.Choice(["sep", "ged4py"]))
@click.argument("min_nodes", type=int)
@click.argument("max_nodes", type=int)
@click.option("--runs", default=3, help="Number of runs per node count")
def main(method, min_nodes, max_nodes, runs):
    cache_dir = "results/benchmark"
    os.makedirs(cache_dir, exist_ok=True)
    futures = []
    with ProcessPoolExecutor() as executor:
        for seed in range(runs):
            for n in range(min_nodes, max_nodes + 1):
                cache_file = os.path.join(cache_dir, f"{method}_{n}_{seed}.json")
                if os.path.exists(cache_file):
                    click.echo(f"Skipping cached: {cache_file}")
                    continue
                futures.append(executor.submit(run_trial, method, n, seed, cache_file))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Benchmarking"):
            try:
                future.result()
            except Exception as e:
                click.echo(f"Error: {e}")
    click.echo("Benchmarking completed.")

if __name__ == "__main__":
    main()
