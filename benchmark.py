#!/usr/bin/env python3
import click
import json
import os
import time
import glob
from collections import defaultdict

import numpy as np
import pylab as plt
import seaborn as sns
from scipy.stats import gmean

from concurrent.futures import ProcessPoolExecutor, as_completed
from networkx.generators.trees import random_unlabeled_tree
from tqdm import tqdm

# Distance functions
# from sepx.similarity import graph_edit_distance
from search_state import DerivationTreeNode
from search_strategies.evolution import Individual
from search_strategies.utils import constrained_smith_waterman_crossover
# from ged4py import GraphEditDistance


# Benchmark trial function.
def run_trial(method, n_nodes, seed, cache_file):
    G1 = random_unlabeled_tree(n_nodes, seed=seed)
    G2 = random_unlabeled_tree(n_nodes, seed=seed + 1)
    if method == "sepx":
        t0 = time.time()
        graph_edit_distance(G1, G2)
    elif method == "cswx":
        G1_dt = nx_to_individual(G1, individual_id=f"cswx_{n_nodes}_{seed}_1", parent_id=None)
        G2_dt = nx_to_individual(G2, individual_id=f"cswx_{n_nodes}_{seed}_2", parent_id=None)
        t0 = time.time()
        constrained_smith_waterman_crossover(G1_dt, G2_dt)
    else:
        raise ValueError("Unknown method")
    elapsed = time.time() - t0
    result = {"method": method, "n_nodes": n_nodes, "seed": seed, "time": elapsed}
    with open(cache_file, "w") as f:
        json.dump(result, f)
    return result

@click.group()
def cli():
    pass

@cli.command()
@click.argument("method", type=click.Choice(["sepx", "cswx", "ged4py"]))
@click.argument("min_nodes", type=int)
@click.argument("max_nodes", type=int)
@click.option("--runs", default=3, help="Number of runs per node count")
def benchmark(method, min_nodes, max_nodes, runs):
    """Run benchmark trials."""
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

@cli.command()
def plot():
    """Plot benchmark results with publication quality."""
    import json, glob
    from collections import defaultdict
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gmean
    import matplotlib as mpl

    mpl.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 10,
    })
    # print(''.join([f'{a}\n' for a in plt.style.available]))
    plt.style.use('seaborn-v0_8-paper')

    files = glob.glob("results/benchmark/*.json")
    grouped = defaultdict(lambda: defaultdict(list))
    for f in files:
        with open(f) as fin:
            d = json.load(fin)
        grouped[d["method"]][d["n_nodes"]].append(d["time"])

    for method, subdict in grouped.items():
        x_vals, y_vals, y_err_lower, y_err_upper = [], [], [], []
        for n in sorted(subdict.keys()):
            arr = np.array(subdict[n])
            if np.any(arr <= 0):
                continue
            gm = gmean(arr)
            log_std = np.std(np.log(arr))
            gstd_factor = np.exp(log_std)
            x_vals.append(n)
            y_vals.append(gm)
            y_err_lower.append(gm - gm / gstd_factor)
            y_err_upper.append(gm * gstd_factor - gm)
        if x_vals:
            plt.errorbar(
                x_vals,
                y_vals,
                yerr=[y_err_lower, y_err_upper],
                label=method,
                marker='o',
                capsize=3
            )
    plt.yscale('log')
    plt.xlabel(r"\textbf{n\_nodes}")
    plt.ylabel(r"\textbf{Time (s, log scale)}")
    plt.legend()
    plt.savefig("results/benchmark/benchmark-cswx-sepx.svg", format='svg')
    plt.show()

if __name__ == "__main__":
    cli()
