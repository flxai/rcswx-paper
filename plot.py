from pickle import load
from glob import glob
from os.path import join

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks", context="paper")

import sys
sys.path.append('..')

import search_strategies
from arguments import parse_arguments


def load_results(path):
    full_path = join(path, "search_results_*.pkl")
    files = glob(full_path)
    if len(files) == 0:
        raise ValueError(f"No files found in {full_path}")
    results = []
    for i in range(0, len(files)):
        p = join(path, f"search_results_{i}.pkl")
        # print(f"Loading {path}")
        r = load(open(p, "rb"))
        # print(f"Reward = {results['reward']}")
        results.append(r)
    return results


def plot_results(args, results, key, save_path):
    plt.figure(figsize=(4, 3))
    plt.plot([r[key] for r in results])
    plt.xlabel("Iteration")
    plt.ylabel(key.capitalize())
    plt.title(args.grammar)
    sns.despine()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(save_path, f"plot_{key}.pdf"))


# parse the arguments
args = parse_arguments()

# load results
results = load_results(join(args.results_path, args.grammar))

# plot results
plot_results(args, results, "reward", join(args.figures_path, args.grammar))
