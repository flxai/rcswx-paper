from os.path import join
from time import perf_counter as time
import psutil
import yaml


def load_config(args):
    # load yaml file and overwrite anything in it
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
        for key, value in config.items():
            if value == "None":
                config[key] = None
    # convert to args
    for key, value in config.items():
        setattr(args, key, value)
    # ensure device is set
    if args.device is None:
        raise ValueError("Please specify device")
    return args


def get_exp_path(args):
    exp_path = join(
        args.grammar,
        args.dataset,
        args.search_strategy,
        f"seed={args.seed}",
        f"backtrack={args.backtrack}",
        f"mode={args.mode}",
        f"time_limit={args.time_limit}",
        f"max_id_limit={args.max_id_limit}",
        f"depth_limit={args.depth_limit}",
        f"mem_limit={args.mem_limit}",
    )
    if args.search_strategy == "random_search":
        pass
    elif args.search_strategy == "mcts":
        exp_path = join(exp_path, f"aquisition_fn={args.aquisition_fn}")
        if args.aquisition_fn == "uct":
            exp_path = join(exp_path, f"exploration_weight={args.exploration_weight}")
        else:
            exp_path = join(exp_path, f"incubent_type={args.incubent_type}")
    return exp_path


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time()
        # print(f"Timer started at {self.start_time}")

    def stop(self):
        self.end_time = time()

    def __call__(self):
        current_time = time()
        duration = current_time - self.start_time
        # print(f"Timer called at {current_time}, start time: {self.start_time}, duration: {duration}")
        return duration


class Limiter:
    def __init__(self, limits):
        self.limits = limits
        self.timer = Timer()
        print(f"Limiter({self.limits})")

    def check(self, node, verbose=False):
        """
        Check if the limits have been reached.
        """
        # get the current duration
        duration = self.timer()
        # get the memory usage
        memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        if (
            node.depth >= self.limits["depth"] or
            duration >= self.limits["time"] or
            node.id >= self.limits["max_id"] or
            memory >= self.limits["memory"]
        ):
            # print which limit was reached
            if node.depth >= self.limits["depth"]:
                if verbose: print(f"Depth limit reached: {node.depth}, removing all but computation module")
            if duration >= self.limits["time"]:
                print(f"Time limit reached: {duration}, removing all but computation module")
            if node.id >= self.limits["max_id"]:
                if verbose: print(f"ID limit reached: {node.id}, removing all but computation module")
            if memory >= self.limits["memory"]:
                print(f"Memory limit reached: {memory}, removing all but computation module")
            return False
        else:
            return True

    def check_memory(self):
        """
        Check if the limits have been reached.
        """
        # get the memory usage
        self.memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        if self.memory >= self.limits["memory"]:
            return False
        else:
            return True
