from os.path import join
from time import time


def get_exp_path(args):
    exp_path = join(
        args.grammar,
        args.search_strategy,
        args.dataset,
        f"seed={args.seed}",
        f"backtrack={args.backtrack}",
        f"mode={args.mode}",
        f"time_limit={args.time_limit}",
        f"max_id_limit={args.max_id_limit}",
        f"max_depth={args.max_depth}",
    )
    if args.search_strategy == "random_search":
        pass
    elif args.search_strategy == "mcts":
        exp_path = join(exp_path, f"exploration_weight={args.exploration_weight}")
    return exp_path


class Timer:
    def __init__(self):
        self.start = time()
        self.end = None

    def start(self):
        self.start = time()

    def stop(self):
        self.end = time()

    def __call__(self):
        return time() - self.start

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end = time()

    def __str__(self):
        self.stop()
        return str(self.end - self.start)
