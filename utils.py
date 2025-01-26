from os.path import join
from functools import reduce
from time import perf_counter as time
import psutil
import yaml

import sys
import gc
from pympler import asizeof, summary


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
        args.search_space,
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
        exp_path = join(exp_path, f"acquisition_fn={args.acquisition_fn}")
        if args.acquisition_fn == "uct":
            exp_path = join(exp_path, f"exploration_weight={args.exploration_weight}")
        else:
            exp_path = join(exp_path, f"incubent_type={args.incubent_type}")
        exp_path = join(exp_path, f"reward_mode={args.reward_mode}")
        exp_path = join(exp_path, f"add_full_paths={args.add_full_paths}")
    elif args.search_strategy == "evolution":
        exp_path = join(exp_path, f"regularised={args.regularised}")
        exp_path = join(exp_path, f"population_size={args.population_size}")
        exp_path = join(exp_path, f"architecture_seed={args.architecture_seed}")
        exp_path = join(exp_path, f"mutation_strategy={args.mutation_strategy}")
        exp_path = join(exp_path, f"mutation_rate={args.mutation_rate}")
        exp_path = join(exp_path, f"crossover_strategy={args.crossover_strategy}")
        exp_path = join(exp_path, f"crossover_rate={args.crossover_rate}")
        exp_path = join(exp_path, f"selection_strategy={args.selection_strategy}")
        exp_path = join(exp_path, f"tournament_size={args.tournament_size}")
        exp_path = join(exp_path, f"elitism={args.elitism}")
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

    def __str__(self):
        return f"Timer(start_time={self.start_time:.2f}, duration={self():.2f})" 


class Limiter:
    def __init__(self, limits, batch=None, compile_fn=None, n_batch_passes=5):
        self.limits = limits
        self.timer = Timer()
        self.memory_checkpoint = None

        self.batch = batch
        self.compile_fn = compile_fn
        self.n_batch_passes = n_batch_passes

        print(f"Limiter({self.limits})")

    def check(self, node, verbose=False):
        """
        Check if the limits have been reached.
        """
        # get the current duration
        duration = self.timer()
        # get the memory usage
        self.memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert bytes to MB

        # check if the diff between memory and memory checkpoint is over the limit
        self.diff = (self.memory - self.memory_checkpoint) if self.memory_checkpoint is not None else 0
        if (
            node.depth >= self.limits["depth"] or
            duration >= self.limits["time"] or
            node.id >= self.limits["max_id"] or
            self.memory >= self.limits["memory"] or 
            self.diff >= self.limits["individual_memory"]
        ):
            # print which limit was reached
            if node.depth >= self.limits["depth"]:
                limit_reached = "Depth"
            if duration >= self.limits["time"]:
                limit_reached = "Time"
            if node.id >= self.limits["max_id"]:
                limit_reached = "Max_id"
            if self.memory >= self.limits["memory"]:
                limit_reached = "Memory"
            if self.diff >= self.limits["individual_memory"]:
                limit_reached = "Individual Memory"
            # print(f"{limit_reached} limit reached for node {node.id}")
            # print(f"{self}")
            return False
        else:
            return True

    def set_memory_checkpoint(self):
        self.memory_checkpoint = psutil.Process().memory_info().rss / (1024 * 1024)

    def reset_memory_checkpoint(self):
        self.memory_checkpoint = None

    def check_memory(self):
        """
        Check if the limits have been reached.
        """
        # get the memory usage
        self.memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        self.diff = (self.memory - self.memory_checkpoint) if self.memory_checkpoint is not None else 0
        if self.diff >= self.limits["individual_memory"]:
            return False
        if self.memory >= self.limits["memory"]:
            return False
        return True

    def check_batch_pass_time(self, node, check_memory=False):
        """
        Check if the limits have been reached.
        """
        if self.batch is None:
            assert self.compile_fn is not None, "Compile function must be provided if limiting batch pass time"
            return True

        if check_memory:
            self.set_memory_checkpoint()

        # node to torch model
        model = self.compile_fn(node)

        # return false is the model is too large
        if check_memory:
            memcheck = self.check_memory()
            # self.reset_memory_checkpoint()
            if not memcheck:
                print(f"Model too large - {self.diff} MB")
                return False

        dur = 0
        timer = Timer()
        for _ in range(self.n_batch_passes):
            timer.start()
            model(self.batch)
            dur += timer()

        duration = dur / self.n_batch_passes

        print(f"Batch pass duration: {duration}")

        if duration >= self.limits["batch_pass_seconds"]:
            return False
        return True
    
    def check_build_safe(self, node):
        # compute the size of extra parameters
        param_size = node.input_params["num_params"] if "num_params" in node.input_params else 0
        # compute size of output tensor
        output_size = reduce(lambda x, y: x*y, node.output_params['shape']) * node.output_params["branching_factor"]
        # combine
        memory_size = param_size + output_size
        # print(f"Additional memory added by operation {node.operation.name}: {memory_size * 4 / (1024 * 1024):.2f} MB")
        return memory_size * 4 / (1024 * 1024) < self.limits["individual_memory"]

    # Function to summarize memory usage of all objects
    def summarise_memory(self, k=3):
        from search_state import DerivationTreeNode, Operation, Stack
        from search_strategies.mcts import SearchTreeNode
        from torch import Size

        print("---- Memory Usage Summary ----")
        # Using gc to inspect all objects
        objects = gc.get_objects()
        self.memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        print(f"Memory usage: {self.memory:.2f} MB")
        print(f"Objects tracked by garbage collector: {len(objects)}")
        
        # Display memory usage for each object type
        obj_types = {}
        for obj in objects:
            obj_type = type(obj).__name__
            obj_size = sys.getsizeof(obj)
            obj_types[obj_type] = obj_types.get(obj_type, 0) + obj_size
        
        # print("---- Memory by Object Type ----")
        for i, (obj_type, total_size) in enumerate(sorted(obj_types.items(), key=lambda x: -x[1])):
            print(f"{obj_type}: {total_size / (1024**2):.2f} MB")
            # self.analyse_instances(eval(obj_type))
            if i >= k:
                break
        # self.analyse_instances(DerivationTreeNode)
        # self.analyse_instances(SearchTreeNode)
        
        # print("\n---- Detailed Memory Summary (Pympler) ----")
        # # Generate a more detailed report using pympler
        # memory_summary = summary.summarize(objects)
        # summary.print_(memory_summary)

        # print("---- End of Memory Summary ----")

    def analyse_instances(self, obj_type):
        # Find all instances of the given object type
        instances = [obj for obj in gc.get_objects() if isinstance(obj, obj_type)]
        print(f"Found {len(instances)} instances of {obj_type.__name__}")
        
        # Analyze the memory usage of the first few instances
        for idx, instance in enumerate(instances[:5]):  # Limit to first 5 instances
            print(f"\nInstance {idx+1} of {obj_type.__name__}:")
            print(f"  Total size (recursive): {asizeof.asizeof(instance) / (1024**2):.2f} MB")
            for attr_name in dir(instance):
                if not attr_name.startswith("__"):
                    attr_value = getattr(instance, attr_name)
                    attr_size = asizeof.asizeof(attr_value)
                    print(f"  {attr_name}: {type(attr_value).__name__}, Size: {attr_size / (1024**2):.2f} MB")

    def __str__(self):
        repr = f"Limiter(\n"
        repr += f"\t{self.limits},\n"
        repr += f"\t{self.timer}\n"
        repr += f"\t{self.memory_checkpoint:.2f} MB (baseline memory)\n"
        repr += f"\t{self.memory:.2f} MB (total memory)\n"
        repr += f"\t{self.memory - self.memory_checkpoint:.2f} MB (individual memory)\n"
        repr += ")"
        return repr
