#!/usr/bin/env python3
import click
import copy
import gc
import glob
import humanize
import json
import logging
import os
import pickle
import random
import sys
import time

import numpy as np

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from networkx.generators.trees import random_unlabeled_tree
from tqdm import tqdm

## from sepx.similarity import graph_edit_distance
from search_state import DerivationTreeNode
from search_strategies.evolution import Individual
from search_strategies.utils import constrained_smith_waterman_crossover

def count_nodes(node):
    children = getattr(node, 'children', [])
    return 1 + sum(count_nodes(child) for child in children)

class DatabaseNotLoadedError(Exception):
    pass

class InsufficientCandidatesError(Exception):
    pass

# Database class with two load modes.
class DerivationTreeDatabase:
    def __init__(self, pkl_path="data/benchmark/benchmark.pkl", load_mode="cswx1"):
        self.pkl_path = pkl_path
        self.load_mode = load_mode  # "cswx1" or "cswx2"
        self.data = None
        self.groups = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return self
        with open(self.pkl_path, "rb") as f:
            self.data = pickle.load(f)
        self.groups = defaultdict(list)
        if self.load_mode == "cswx1":
            for bench_name, bench_dict in self.data.items():
                for seed, run in bench_dict.items():
                    for ind in run['population']:
                        n = count_nodes(ind.root)
                        self.groups[n].append(ind)
        elif self.load_mode == "cswx2":
            for bench_name, bench_dict in self.data.items():
                for seed, run in bench_dict.items():
                    for arch, _, _, _ in run['rewards']:
                        n = count_nodes(arch[0])
                        self.groups[n].append(arch[0])
        self.loaded = True
        return self

    def sample(self, n_nodes, samples=2, rng_seed=42):
        if not self.loaded:
            click.echo("Prefetch DB first!")
            raise DatabaseNotLoadedError("Prefetch DB first")
        if len(self.groups[n_nodes]) < samples:
            raise InsufficientCandidatesError(
                f"Not enough candidates with {n_nodes} nodes; found {len(self.groups[n_nodes])}"
            )
        rng = random.Random(rng_seed)
        return tuple(rng.sample(self.groups[n_nodes], samples))

    def get_sample_counts(self):
        return {n_nodes: len(ind_list) for n_nodes, ind_list in self.groups.items()}

# Global database instance.
DB = None

def run_trial(method, n_nodes, seed, cache_file):
    try:
        if method == "sepx":
            G1 = random_unlabeled_tree(n_nodes, seed=seed)
            G2 = random_unlabeled_tree(n_nodes, seed=seed + 1)
            t0 = time.time()
            # graph_edit_distance(G1, G2)
        elif method in ["cswx1", "cswx2"]:
            ind1, ind2 = DB.sample(n_nodes, samples=2, rng_seed=seed)
            # For cswx1, extract the root to get the node with 'operation'
            if method == "cswx1":
                ind1, ind2 = ind1.root, ind2.root
            t0 = time.time()
            constrained_smith_waterman_crossover(ind1, ind2)
        else:
            raise ValueError("Unknown method")
        elapsed = time.time() - t0
        result = {"method": method, "n_nodes": n_nodes, "seed": seed, "time": elapsed}
        with open(cache_file, "w") as f:
            json.dump(result, f)
        return result
    except DatabaseNotLoadedError as e:
        logging.error(f'[DATABASE] {e}')
    except InsufficientCandidatesError as e:
        logging.error(f'[CANDIDATE] {e}')
    except Exception as e:
        logging.error(f'[GENERIC] {e}')
        raise

@click.group()
def cli():
    pass

@cli.command()
@click.argument("method", type=click.Choice(["sepx", "cswx1", "cswx2", "ged4py"]))
@click.argument("min_nodes", type=int)
@click.argument("max_nodes", type=int)
@click.option("--runs", default=10, help="Number of runs per node count")
@click.option("--shuffle/--no-shuffle", default=True, help="Shuffle order")
@click.option("-j", "--jobs", default=-1, help="Max number of jobs; -1 means all cores")
def benchmark(method, min_nodes, max_nodes, runs, shuffle, jobs):
    """Run benchmark trials."""
    global DB
    cache_dir = "results/benchmark"
    os.makedirs(cache_dir, exist_ok=True)
    if method in ['cswx1', 'cswx2']:
        click.echo(f"Prefetching database in {method} mode...")
        DB = DerivationTreeDatabase(pkl_path="data/benchmark/benchmark.pkl", load_mode=method)
        DB.load()
        click.echo("Prefetched database")
    max_workers = jobs if jobs > 0 else os.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for seed in range(runs):
            n_range = list(range(min_nodes, max_nodes + 1))
            if shuffle:
                random.shuffle(n_range)
            for n in n_range:
                cache_file = os.path.join(cache_dir, f"{method}_{n}_{seed}.json")
                if os.path.exists(cache_file):
                    click.echo(f"Skipping cached: {cache_file}")
                    continue
                futures.append(executor.submit(run_trial, method, n, seed, cache_file))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Benchmarking"):
            try:
                _ = future.result()
            except Exception as e:
                if "terminated abruptly" in str(e):
                    click.echo("Warning: A process was terminated abruptly.")
                else:
                    click.echo(f"Error: {e}")
            del future
            gc.collect()
    click.echo("Benchmarking completed")

@cli.command()
def plot():
    """
    Extended plot:
      - SEPX as before (blue line and fill).
      - CSWX1 in orange and CSWX2 in red:
          * Real data: line + fill.
          * Missing regions: dotted log-fit.
      - Bottom histogram & KDE as before.
    """
    import json, glob, pickle, os
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from scipy.stats import gmean, gaussian_kde
    import matplotlib as mpl

    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Linux Libertine O', 'Libertine']
    mpl.rcParams['text.latex.preamble'] = (
        r'\usepackage[nofontspec,semibold,lining]{libertine}'
        r'\usepackage[T1]{fontenc}'
        r'\usepackage[varqu,varl,scaled=0.96]{zi4}'
        r'\usepackage[libertine,vvarbb,upint]{newtxmath}'
        r'\usepackage[cal=cm,bb=ams,scr=boondoxo]{mathalpha}'
    )
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 14
    # mpl.rcParams.update({'font.size': 12})
    plt.style.use('seaborn-v0_8-paper')

    def compute_geom_stats(times_dict):
        xs, means, lows, highs = [], [], [], []
        for k in sorted(times_dict.keys()):
            vals = np.array(times_dict[k])
            if len(vals) == 0 or np.any(vals <= 0):
                continue
            log_vals = np.log(vals)
            mu, sd = log_vals.mean(), log_vals.std()
            gm = np.exp(mu)
            gstd_factor = np.exp(sd)
            xs.append(k)
            means.append(gm)
            lows.append(gm / gstd_factor)
            highs.append(gm * gstd_factor)
        return np.array(xs, dtype=float), np.array(means), np.array(lows), np.array(highs)

    def split_on_gaps(xs, means, lows, highs, gap=1.5):
        segments = []
        if len(xs) == 0:
            return segments
        start_idx = 0
        for i in range(1, len(xs)):
            if (xs[i] - xs[i - 1]) > gap:
                segments.append((xs[start_idx:i], means[start_idx:i],
                                 lows[start_idx:i], highs[start_idx:i]))
                start_idx = i
        segments.append((xs[start_idx:], means[start_idx:], lows[start_idx:], highs[start_idx:]))
        return segments

    def log_fit(x_data, y_data, x_vals):
        slope, intercept = np.polyfit(np.log(x_data), np.log(y_data), 1)
        return np.exp(intercept + slope * np.log(x_vals))

    files = glob.glob("results/benchmark/*.json")
    sepx_data = defaultdict(list)
    cswx1_data = defaultdict(list)
    cswx2_data = defaultdict(list)
    for f in files:
        with open(f) as fin:
            d = json.load(fin)
        if d["method"] == "sepx":
            sepx_data[d["n_nodes"]].append(d["time"])
        elif d["method"] == "cswx1":
            cswx1_data[d["n_nodes"]].append(d["time"])
        elif d["method"] == "cswx2":
            cswx1_data[d["n_nodes"]].append(d["time"])
            # Collect both
            # cswx2_data[d["n_nodes"]].append(d["time"])

    x_sepx, mean_sepx, low_sepx, high_sepx = compute_geom_stats(sepx_data)
    x_cswx1, mean_cswx1, low_cswx1, high_cswx1 = compute_geom_stats(cswx1_data)
    x_cswx2, mean_cswx2, low_cswx2, high_cswx2 = compute_geom_stats(cswx2_data)

    hist_file = 'data/benchmark-hist.pkl'
    if os.path.exists(hist_file):
        with open(hist_file, 'rb') as f:
            counts = pickle.load(f)
    else:
        click.echo("Loading DB to create hist file")
        temp_DB = DerivationTreeDatabase(pkl_path="data/benchmark/benchmark.pkl", load_mode="cswx1")
        temp_DB.load()
        counts = temp_DB.get_sample_counts()
        with open(hist_file, 'wb') as f:
            pickle.dump(counts, f)
    x_all = []
    for node_val, freq_val in counts.items():
        x_all.extend([node_val] * freq_val)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(6,4),
        gridspec_kw={'height_ratios': [4,1]}
    )

    # Plot SEPX
    if len(x_sepx) > 0:
        ax_top.plot(x_sepx, mean_sepx, label='SEPX', color='tab:red')
        ax_top.fill_between(x_sepx, low_sepx, high_sepx, alpha=0.3, color='tab:red')
    if len(x_sepx) >= 2:
        def sepx_log_extension(ax, xs, ys, color, x_max):
            slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
            x_ext = np.linspace(xs[-1], x_max, 50)
            y_ext = np.exp(intercept + slope * np.log(x_ext))
            ax.plot(x_ext, y_ext, linestyle=':', color=color, linewidth=1.2)
            y_min, y_max = np.min(y_ext), np.max(y_ext)
            ax.vlines(xs[-1], ymin=y_min, ymax=y_max, color=color, linestyle='--', linewidth=0.7)
            ax.vlines(x_max, ymin=y_min, ymax=y_max, color=color, linestyle='--', linewidth=0.7)
            mid_x = (xs[-1] + x_max) / 2
            mid_y = np.exp(intercept + slope * np.log(mid_x))
            ax.text(mid_x, mid_y * 3, "Extrapolation", color=color, fontsize=10,
                    ha='center', va='bottom')
        sepx_log_extension(ax_top, x_sepx, mean_sepx, 'tab:red', 200)

    # Plot CSWX1 (orange) and CSWX2 (red)
    def plot_data(segments, label, color):
        plotted = False
        for (xx, mm, ll, hh) in segments:
            lab = label if not plotted else None
            ax_top.plot(xx, mm, label=lab, color=color)
            ax_top.fill_between(xx, ll, hh, alpha=0.3, color=color)
            plotted = True

    cswx1_segments = split_on_gaps(x_cswx1, mean_cswx1, low_cswx1, high_cswx1, gap=1.5)
    cswx2_segments = split_on_gaps(x_cswx2, mean_cswx2, low_cswx2, high_cswx2, gap=1.5)
    plot_data(cswx1_segments, 'CSWX', 'tab:blue')
    plot_data(cswx2_segments, 'CSWX2', 'tab:orange')

    def plot_dotted_segments(ax, x_vals, y_vals, **kwargs):
        segments = []
        start = 0
        for i in range(1, len(x_vals)):
            if x_vals[i] - x_vals[i - 1] > 1:
                segments.append((x_vals[start:i], y_vals[start:i]))
                start = i
        segments.append((x_vals[start:], y_vals[start:]))
        for xs, ys in segments:
            ax.plot(xs, ys, **kwargs)

    # Dotted log-fit for missing regions (for both CSWX1 and CSWX2)
    for xs, means, color in [(x_cswx1, mean_cswx1, 'tab:blue'), (x_cswx2, mean_cswx2, 'tab:orange')]:
        if len(xs) >= 2:
            x_min_val = int(np.min(xs))
            x_max_val = 200
            x_fit_range = np.arange(x_min_val, x_max_val + 1, 1, dtype=float)
            y_fit = log_fit(xs, means, x_fit_range)
            mask_missing = ~np.isin(x_fit_range, xs)
            x_missing = x_fit_range[mask_missing]
            y_missing = y_fit[mask_missing]
            if len(x_missing) > 0:
                plot_dotted_segments(
                    ax_top, x_missing, y_missing,
                    linestyle=':', color=color, linewidth=1.2, label='_nolegend_'
                )

    ax_top.set_yscale("log")
    ax_top.set_ylabel(r"\textbf{Runtime (s)}")
    ax_top.set_title(r"\textbf{Runtime comparison between methods}")
    ax_top.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
    ax_top.set_xlim(left=0)
    ax_top.legend(loc='lower right')

    ref_yvals = [1, 60, 3600, 86400, 604800, 2592000, 31536000]
    ref_labels = ["1 second", "1 minute", "1 hour", "1 day", "1 week", "1 month", "1 year"]
    for y, label in zip(ref_yvals, ref_labels):
        ax_top.axhline(y, color='grey', linestyle='--', linewidth=.6)
        ax_top.text(
            1.01, y, label,
            va='center', ha='left', color='grey', fontsize=8,
            transform=ax_top.get_yaxis_transform()
        )

    ax_bot.set_xlabel(r"\textbf{Number of nodes}")
    ax_bot.set_ylabel(r"\textbf{Frequency}")
    ax_bot.set_title(r"\textbf{Distribution of number of nodes}")
    ax_bot.set_xlim(left=0)

    if x_all:
        x_min = 0
        x_max_val = max(x_all)
        bins = np.arange(x_min, x_max_val + 1, 1)
        ax_bot.hist(x_all, bins=bins, alpha=0.3,
                    color='green', edgecolor='none', label='Histogram')
        ax_kde = ax_bot.twinx()
        ax_kde.set_ylabel(r"\textbf{KDE Density}", color='black', fontsize=8)
        if len(set(x_all)) > 1:
            kde = gaussian_kde(x_all)
            def custom_cf():
                return 0.4 * kde.scotts_factor()
            kde.covariance_factor = custom_cf
            kde._compute_covariance()
            x_vals = np.linspace(x_min, x_max_val, 500)
            kde_pdf = kde(x_vals)
            ax_kde.plot(x_vals, kde_pdf, color='green', label='KDE')
        lines1, labels1 = ax_bot.get_legend_handles_labels()
        lines2, labels2 = ax_kde.get_legend_handles_labels()
        ax_bot.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)

    plt.tight_layout()
    plt.savefig("results/benchmark/benchmark-runtimes.svg", format='svg')
    plt.show()

if __name__ == "__main__":
    cli()
