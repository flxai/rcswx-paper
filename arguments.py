import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run MCTS on a grammar.")
    parser.add_argument("--seed", type=int, default=0, help="The seed to use.")
    # search details
    parser.add_argument("--grammar", type=str, default="einspace", help="The grammar to use.")
    parser.add_argument("--search_strategy", type=str, default="mcts", help="The search strategy to use.")
    parser.add_argument("--steps", type=int, default=1000, help="The number of search steps.")
    parser.add_argument("--exploration_weight", type=float, default=1.0, help="The exploration weight to use.")
    parser.add_argument("--backtrack", action="store_true", help="Backtrack when out of options.")
    parser.add_argument("--mode", type=str, default="iterative", help="The mode to use.")
    parser.add_argument("--time_limit", type=int, default=60, help="The time limit to use.")
    parser.add_argument("--max_id_limit", type=int, default=1000, help="The maximum ID limit to use.")
    # evaluation details
    parser.add_argument("--dataset", type=str, default="mnist", help="The dataset to use.")
    parser.add_argument("--epochs", type=int, default=1, help="The number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size to use.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to use.")
    # logging and plotting
    parser.add_argument("--verbose_search", action="store_true", help="Print verbose output during search.")
    parser.add_argument("--verbose_eval", action="store_true", help="Print verbose output during evaluation.")
    parser.add_argument("--visualise", action="store_true", help="Visualise the derivation tree.")
    parser.add_argument("--visualise_scale", type=float, default=0.8, help="The scale of the visualisation.")
    parser.add_argument("--print_after", type=int, default=1000, help="Print after this many iterations.")
    # saving results and figures
    parser.add_argument("--figures_path", type=str, default="figures", help="The path to save the figures.")
    parser.add_argument("--results_path", type=str, default="results", help="The path to save the results.")
    # continue search
    parser.add_argument("--continue_search", action="store_true", help="Continue the search.")
    args = parser.parse_args()
    return args
