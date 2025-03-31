#!/usr/bin/env python3
import click
import json
import os
import time
import glob
import pickle
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from networkx.generators.trees import random_unlabeled_tree
from tqdm import tqdm

from search_state import DerivationTreeNode
from search_strategies.evolution import Individual
from search_strategies.utils import constrained_smith_waterman_crossover

# Recursively count nodes in derivation tree
def count_nodes(node):
    children = getattr(node, 'children', [])
    return 1 + sum(count_nodes(child) for child in children)

# File wrapper shows tqdm progress bar during read
class TqdmFileWrapper:
    def __init__(self, file_obj, total):
        self.file = file_obj
        self.tqdm_bar = tqdm(total=total, desc="Loading pickle", unit="B", unit_scale=True)
    def read(self, size=-1):
        data = self.file.read(size)
        self.tqdm_bar.update(len(data))
        return data
    def readline(self, size=-1):
        data = self.file.readline(size)
        self.tqdm_bar.update(len(data))
        return data
    def __getattr__(self, attr):
        return getattr(self.file, attr)

# Database class loads benchmark pickle file on demand
class DerivationTreeDatabase:
    def __init__(self, pkl_path="data/benchmark/benchmark.pkl", show_progress=True):
        self.pkl_path = pkl_path
        self.show_progress = show_progress
        self.data = None
        self.groups = None

    def load(self):
        if self.data is None:
            if self.show_progress:
                click.echo("Warning: Loading pickle file may take while")
                file_size = os.path.getsize(self.pkl_path)
                with open(self.pkl_path, "rb") as f:
                    f_wrapped = TqdmFileWrapper(f, file_size)
                    self.data = pickle.load(f_wrapped)
            else:
                with open(self.pkl_path, "rb") as f:
                    self.data = pickle.load(f)
            self.groups = defaultdict(list)
            for bench_name, bench_dict in self.data.items():
                for seed, run in bench_dict.items():
                    for ind in run['population']:
                        n = count_nodes(ind.root)
                        self.groups[n].append(ind)
        return self

    def sample(self, n_nodes, samples=2, rng_seed=42):
        self.load()  # Ensure data loaded
        if len(self.groups[n_nodes]) < samples:
            raise ValueError(f"Not enough candidates with {n_nodes} nodes; found {len(self.groups[n_nodes])}")
        rng = random.Random(rng_seed)
        return tuple(rng.sample(self.groups[n_nodes], samples))

# Global database instance lazy loaded
DB = DerivationTreeDatabase(pkl_path="data/benchmark/benchmark.pkl", show_progress=True)

# Benchmark trial function
def run_trial(method, n_nodes, seed, cache_file):
    if method == "sepx":
        G1 = random_unlabeled_tree(n_nodes, seed=seed)
        G2 = random_unlabeled_tree(n_nodes, seed=seed + 1)
        t0 = time.time()
    elif method == "cswx":
        ind1, ind2 = DB.sample(n_nodes, samples=2, rng_seed=seed)
        t0 = time.time()
        constrained_smith_waterman_crossover(ind1.root, ind2.root)
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
@click.option("--runs", default=10, help="Number of runs per node count")
def benchmark(method, min_nodes, max_nodes, runs):
    """Run benchmark trials"""
    cache_dir = "results/benchmark"
    os.makedirs(cache_dir, exist_ok=True)
    # Prefetch database in main process before launching parallel tasks
    prefetch_database = True
    if prefetch_database:
        click.echo("Prefetching database...")
        DB.load()
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
    click.echo("Benchmarking completed")

@cli.command()
def plot():
    """Plot benchmark results with publication quality"""
    import json, glob
    from collections import defaultdict
    import matplotlib.pyplot as plt
    from scipy.stats import gmean
    import matplotlib as mpl

    mpl.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 10,
    })
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
                label=method.upper(),
                marker='o',
                capsize=3
            )
    plt.yscale('log')
    plt.xlabel(r"\textbf{\# of nodes}")
    plt.ylabel(r"\textbf{Time (s, log scale)}")
    plt.title("Runtime comparison between CSWX and SEPX")
    plt.legend()
    plt.savefig("results/benchmark/benchmark-cswx-sepx.svg", format='svg')
    plt.show()

@cli.command()
def plot_extended():
    """Plot SEPX results with log–log extension, vertical & horizontal reference lines, light grey grid, adjusted markers, and title"""
    import json, glob
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from scipy.stats import gmean
    import matplotlib as mpl

    # Global rcParams for fonts & sizes
    mpl.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })
    plt.style.use('seaborn-v0_8-paper')

    def compute_geom_stats(times_dict):
        keys = sorted(times_dict.keys())
        means, lows, highs = [], [], []
        for k in keys:
            vals = np.array(times_dict[k])
            if len(vals) == 0 or np.any(vals <= 0):
                continue
            lv = np.log(vals)
            mu, sd = lv.mean(), lv.std()
            means.append(np.exp(mu))
            lows.append(np.exp(mu - sd))
            highs.append(np.exp(mu + sd))
        return keys, means, lows, highs

    files = glob.glob("results/benchmark/*.json")
    sepx_data = defaultdict(list)
    cswx_data = defaultdict(list)
    for f in files:
        with open(f) as fin:
            d = json.load(fin)
        if d["method"] == "sepx":
            sepx_data[d["n_nodes"]].append(d["time"])
        elif d["method"] == "cswx":
            cswx_data[d["n_nodes"]].append(d["time"])

    # Add "1 second" reference by injecting a key if needed (optional)
    # Here we manually add it to our reference lines
    ref_yvals = [1, 60, 3600, 86400, 604800, 2592000, 31536000]
    ref_labels = ["1 second", "1 minute", "1 hour", "1 day", "1 week", "1 month", "1 year"]

    x_sepx, mean_sepx, low_sepx, high_sepx = compute_geom_stats(sepx_data)
    x_cswx, mean_cswx, low_cswx, high_cswx = compute_geom_stats(cswx_data)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.tick_params(axis='both', labelsize=14)
    
    if x_sepx:
        line_sepx, = ax.plot(x_sepx, mean_sepx, label='SEPX', color='tab:blue')
        ax.fill_between(x_sepx, low_sepx, high_sepx, alpha=0.3, color='tab:blue')
    if x_cswx:
        line_cswx, = ax.plot(x_cswx, mean_cswx, label='CSWX', color='tab:orange')
        ax.fill_between(x_cswx, low_cswx, high_cswx, alpha=0.3, color='tab:orange')

    ax.set_xlabel("Number of nodes", fontsize=14)
    ax.set_ylabel("Runtime (s)", fontsize=14)
    ax.set_yscale("log")
    ax.legend(loc='lower right', fontsize=14)

    def plot_log_extension(ax, xs, ys, color, x_max):
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
        x_ext = np.linspace(xs[-1], x_max, 50)
        y_ext = np.exp(intercept + slope * np.log(x_ext))
        ax.plot(x_ext, y_ext, linestyle=':', color=color, linewidth=2)
        y_min, y_max = np.min(y_ext), np.max(y_ext)
        ax.vlines(xs[-1], ymin=y_min, ymax=y_max, color=color, linestyle='--', alpha=0.7)
        ax.vlines(x_max, ymin=y_min, ymax=y_max, color=color, linestyle='--', alpha=0.7)
        mid_x = (xs[-1] + x_max) / 2
        mid_y = np.exp(intercept + slope * np.log(mid_x))
        # Extrapolation label offset increased by factor 3
        ax.text(mid_x, mid_y*3, "Extrapolation", color=color, fontsize=14, ha='center', va='bottom')

    if x_sepx:
        plot_log_extension(ax, x_sepx, mean_sepx, line_sepx.get_color(), 200)

    # Set x_limit to 105% of xmax for reference markers
    xmin, xmax = ax.get_xlim()
    x_limit = xmax * 1.05
    ax.set_xlim(xmin, x_limit)
    # Plot horizontal reference lines for ref_yvals
    for y, txt in zip(ref_yvals, ref_labels):
        ax.hlines(y, xmin, x_limit, color='grey', linestyle='--')
    # Place grey dot markers and text at 105% of xmax (further left than before)
    marker_x = x_limit * 0.95
    for y, txt in zip(ref_yvals, ref_labels):
        ax.plot([marker_x], [y], marker='o', color='grey')
        ax.text(marker_x * 0.99, y, txt, va='center', ha='right', color='grey', fontsize=14)

    ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
    ax.set_title("runtime comparison between csxw and sepx", fontsize=16)

    plt.savefig("results/benchmark/plot_extended_sepx.svg", format='svg')
    plt.show()

@cli.command()
def plot_extended():
    """Plot SEPX results with log–log extension, vertical & horizontal reference lines, light grey grid, adjusted markers, and title"""
    import json, glob
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from scipy.stats import gmean
    import matplotlib as mpl

    mpl.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })
    plt.style.use('seaborn-v0_8-paper')

    def compute_geom_stats(times_dict):
        keys = sorted(times_dict.keys())
        means, lows, highs = [], [], []
        for k in keys:
            vals = np.array(times_dict[k])
            if len(vals) == 0 or np.any(vals <= 0):
                continue
            lv = np.log(vals)
            mu, sd = lv.mean(), lv.std()
            means.append(np.exp(mu))
            lows.append(np.exp(mu - sd))
            highs.append(np.exp(mu + sd))
        return keys, means, lows, highs

    files = glob.glob("results/benchmark/*.json")
    sepx_data = defaultdict(list)
    cswx_data = defaultdict(list)
    for f in files:
        with open(f) as fin:
            d = json.load(fin)
        if d["method"] == "sepx":
            sepx_data[d["n_nodes"]].append(d["time"])
        elif d["method"] == "cswx":
            cswx_data[d["n_nodes"]].append(d["time"])

    x_sepx, mean_sepx, low_sepx, high_sepx = compute_geom_stats(sepx_data)
    x_cswx, mean_cswx, low_cswx, high_cswx = compute_geom_stats(cswx_data)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.tick_params(axis='both', labelsize=14)
    
    if x_sepx:
        line_sepx, = ax.plot(x_sepx, mean_sepx, label='SEPX', color='tab:blue')
        ax.fill_between(x_sepx, low_sepx, high_sepx, alpha=0.3, color='tab:blue')
    if x_cswx:
        line_cswx, = ax.plot(x_cswx, mean_cswx, label='CSWX', color='tab:orange')
        ax.fill_between(x_cswx, low_cswx, high_cswx, alpha=0.3, color='tab:orange')

    ax.set_xlabel("Number of nodes", fontsize=14)
    ax.set_ylabel("Runtime (s)", fontsize=14)
    ax.set_yscale("log")
    ax.legend(loc='lower right', fontsize=14)

    def plot_log_extension(ax, xs, ys, color, x_max):
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
        x_ext = np.linspace(xs[-1], x_max, 50)
        y_ext = np.exp(intercept + slope * np.log(x_ext))
        ax.plot(x_ext, y_ext, linestyle=':', color=color, linewidth=2)
        y_min, y_max = np.min(y_ext), np.max(y_ext)
        ax.vlines(xs[-1], ymin=y_min, ymax=y_max, color=color, linestyle='--', alpha=0.7)
        ax.vlines(x_max, ymin=y_min, ymax=y_max, color=color, linestyle='--', alpha=0.7)
        mid_x = (xs[-1] + x_max) / 2
        mid_y = np.exp(intercept + slope * np.log(mid_x))
        ax.text(mid_x, mid_y*3, "Extrapolation", color=color, fontsize=14, ha='center', va='bottom')

    if x_sepx:
        plot_log_extension(ax, x_sepx, mean_sepx, line_sepx.get_color(), 200)

    # Horizontal reference lines for specific runtimes (1 second, 1 minute, 1 hour, 1 day, 1 week, 1 month, 1 year)
    ref_yvals = [1, 60, 3600, 86400, 604800, 2592000, 31536000]
    ref_labels = ["1 second", "1 minute", "1 hour", "1 day", "1 week", "1 month", "1 year"]
    xmin, xmax = ax.get_xlim()
    x_limit = xmax * 1.05
    for y, txt in zip(ref_yvals, ref_labels):
        ax.hlines(y, xmin, x_limit, color='grey', linestyle='--')
    ax.set_xlim(xmin, x_limit)
    # Place grey dot markers and text as before (using x_limit*1.01)
    for y, txt in zip(ref_yvals, ref_labels):
        ax.plot([x_limit], [y], marker='o', color='grey')
        ax.text(x_limit * 1.01, y, txt, va='center', ha='left', color='grey', fontsize=14)

    ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
    ax.set_title("Runtime comparison between CSWX and SEPX", fontsize=16)

    plt.savefig("results/benchmark/plot_extended_sepx.svg", format='svg')
    plt.show()

if __name__ == "__main__":
    cli()

