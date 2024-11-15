from pickle import load
from glob import glob
from os.path import join

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks", context="paper")

from tqdm import tqdm
import sys
sys.path.append('..')

import search_strategies
from arguments import parse_arguments
from visualise import visualise_derivation_tree
import utils


def load_results(path):
    full_path = join(path, "search_results.pkl")
    results = load(open(full_path, "rb"))
    print(f"Successfully loaded {results['iteration']} result iterations")
    return results


def plot_results(args, results, key, save_path):
    plt.figure(figsize=(6, 3))
    # set marker color based on the type of first node in the derivation tree
    colors = sns.color_palette("tab10")
    node_type = {"sequential": 0, "branching(2)": 1, "branching(4)": 2, "branching(8)": 3, "routing": 4, "computation": 5}
    data = [(i, reward, colors[node_type[arch[0].operation.name]], arch[0].operation.name) for i, (arch, reward) in enumerate(results[key])]
    for i, reward, color, label in tqdm(data, desc=f"Plotting {key}"):
        plt.scatter(i, reward, color=color, label=label, alpha=0.5)
    # plt.scatter(range(len(results[key])), [reward for arch, reward in results[key]])
    # only include unique labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1, 1))
    plt.xlabel("Iteration")
    plt.ylabel(key.capitalize())
    plt.title(args.grammar)
    sns.despine()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(save_path, f"plot_{key}.pdf"))
    print(f"Saved plot to {join(save_path, f'plot_{key}.pdf')}")


# parse the arguments
args = parse_arguments()

# load results
results = load_results(join(args.results_path, utils.get_exp_path(args)))

# plot results
plot_results(args, results, "rewards", join(args.figures_path, utils.get_exp_path(args)))

# visualise the best derivation tree
# best_arch, best_reward = max(results["rewards"], key=lambda x: x[1])
# print(f"Best architecture has reward: {best_reward}")
# visualise_derivation_tree(best_arch[0], iteration=results["iteration"], show=False, save_path=join(args.figures_path, utils.get_exp_path(args)))
# print(f"Visualised the best derivation tree at iteration {results['iteration']}")
