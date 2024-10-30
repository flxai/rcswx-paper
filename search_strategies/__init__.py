from os.path import join

from .random_search import RandomSearch
from .mcts import MCTS

__all__ = [
    "RandomSearch",
    "MCTS",
]


def create_search_strategy(args, grammar, evaluation_fn, input_params):
    # create the search strategy
    search_strategy = {
        "random_search": RandomSearch,
        "mcts": MCTS,
    }[args.search_strategy]

    # specific parameters for each search strategy
    search_specific_params = {
        "random_search": {
            "figures_path": join(
                args.figures_path,
                args.grammar,
                args.search_strategy,
                args.dataset,
                f"seed={args.seed}",
                f"backtrack={args.backtrack}",
                f"mode={args.mode}",
                f"time_limit={args.time_limit}",
                f"max_id_limit={args.max_id_limit}",
            ),
            "results_path": join(
                args.results_path,
                args.grammar,
                args.search_strategy,
                args.dataset,
                f"seed={args.seed}",
                f"backtrack={args.backtrack}",
                f"mode={args.mode}",
                f"time_limit={args.time_limit}",
                f"max_id_limit={args.max_id_limit}",
            ),
        },
        "mcts": {
            "figures_path": join(
                args.figures_path,
                args.grammar,
                args.search_strategy,
                args.dataset,
                f"seed={args.seed}",
                f"backtrack={args.backtrack}",
                f"mode={args.mode}",
                f"time_limit={args.time_limit}",
                f"max_id_limit={args.max_id_limit}",
                f"exploration_weight={args.exploration_weight}",
            ),
            "results_path": join(
                args.results_path,
                args.grammar,
                args.search_strategy,
                args.dataset,
                f"seed={args.seed}",
                f"backtrack={args.backtrack}",
                f"mode={args.mode}",
                f"time_limit={args.time_limit}",
                f"max_id_limit={args.max_id_limit}",
                f"exploration_weight={args.exploration_weight}",
            ),
            "exploration_weight": args.exploration_weight,
        }
    }

    # create the search
    search = search_strategy(
        # common parameters
        pcfg=grammar,
        evaluation_fn=evaluation_fn,
        input_params=input_params,
        seed=args.seed,
        mode=args.mode,
        backtrack=args.backtrack,
        max_id_limit=args.max_id_limit,
        time_limit=args.time_limit,
        verbose=args.verbose_search,
        verbose_after_iteration=args.print_after,
        visualise=args.visualise,
        visualise_after_iteration=args.print_after,
        visualise_scale=args.visualise_scale,
        continue_search=args.continue_search,
        # specific parameters
        **search_specific_params[args.search_strategy],
    )

    return search
