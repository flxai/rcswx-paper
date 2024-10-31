from glob import glob
from os.path import join
from os import makedirs
import pickle
import time
import random

from tqdm import tqdm

from visualise import visualise_derivation_tree
from search_state import DerivationTreeNode, Stack
from pcfg import OutOfOptionsError


class RandomSearch:
    def __init__(
            self,
            evaluation_fn,
            pcfg,
            input_params,
            seed=0,
            mode="iterative",
            backtrack=True,
            max_id_limit=1000,
            time_limit=300,
            verbose=False,
            verbose_after_iteration=None,
            visualise=False,
            visualise_after_iteration=None,
            visualise_scale=0.5,
            figures_path=None,
            results_path=None,
            continue_search=False,
        ):
        self.evaluation_fn = evaluation_fn
        self.pcfg = pcfg
        self.input_params = input_params
        self.seed = seed
        self.mode = mode
        self.backtrack = backtrack
        self.max_id_limit = max_id_limit
        self.time_limit = time_limit
        self.verbose = verbose
        self.verbose_after_iteration = verbose_after_iteration
        self.visualise = visualise
        self.visualise_after_iteration = visualise_after_iteration
        self.visualise_scale = visualise_scale
        self.figures_path = figures_path
        self.results_path = results_path
        self.continue_search = continue_search

        self.rewards = []
        self.iteration = 0

        self.set_rng_state(seed=self.seed)

        if self.continue_search:
            self.load_results()

        if mode == "iterative":
            self.sample_fn = self.sample_iterative
        elif mode == "recursive":
            self.sample_fn = self.sample_recursive

    def set_rng_state(self, seed=None, state=None):
        if state:
            random.setstate(state)
        elif seed:
            random.seed(seed)

    def learn(self, steps):
        print("-------------")
        print("Random Search")
        print(f"Steps: {steps}")
        print("--------------")

        for iteration in tqdm(range(self.iteration, steps), desc="RS", initial=self.iteration, total=steps):
            node, reward = self.sample(iteration)
            print(f"Step {iteration}, reward: {reward}")

    def sample(self, iteration):
        root = DerivationTreeNode(1, "network", input_params=self.input_params)

        max_id = 1
        stack = Stack([(root, False)])

        # sample the network, and keep going until we successfully sample a network
        try:
            success = False
            while not success:
                node, stack, max_id, duration = self.sample_fn(
                    root,
                    stack,
                    max_id,
                )
                if node is not None:
                    success = True
        except OutOfOptionsError:
            print("Reached dead end, returning reward 0")
            return root, 0

        # evaluate the network
        reward = self.evaluation_fn(node)
        self.rewards.append((node.get_root().serialise(), reward))

        # save the results
        self.save_results(iteration)

        return node, reward
        

    def sample_iterative(self, node, stack, max_id):
        start_time = time.time()
        i = 0
        while not stack.is_empty():
            i += 1
            if max_id > self.max_id_limit or time.time() - start_time > self.time_limit:
                if self.verbose: print(f"Breaking at max_id: {max_id}, time: {time.time() - start_time}")
                return None, stack, max_id, time.time() - start_time

            if self.verbose: print(f"Duration: {time.time() - start_time}, Time limit: {self.time_limit}")
            if self.verbose: print(f"Stack: {stack}, max_id: {max_id}")
            node, visited = stack.pop()

            if self.verbose: visualise_derivation_tree(node.get_root(), stack.stack, current_node_id=node.id)

            if node:
                if visited:
                    # Propagate the output params to the parent
                    node.give_back_output_params()
                    if not node.is_root():
                        if self.verbose: print(f"Propagated output params from node {node.id}({hex(id(node))}) to parent {node.parent.id}({hex(id(node.parent))})")
                        if self.verbose: print(f"Output params for node: {node.parent.id}, {node.parent.output_params}")
                else:
                    stack.append((node, True))
                    if not node.is_root():
                        # inherit the input params from the parent
                        node.inherit_input_params()
                        if self.verbose: print(f"Inherited input params from parent {node.parent.id}({hex(id(node.parent))}) to node {node.id}({hex(id(node))})")
                        if self.verbose: print(f"Input params for node: {node.id}, {node.input_params}")
                    try:
                        # select operation and initialise the node, children etc.
                        stack, max_id = node.initialise(self.pcfg.sample(node, verbose=self.verbose), stack, max_id)
                        if self.verbose: print(f"Memory of node {node.id}: {node.memory}")
                    except OutOfOptionsError:
                        if self.verbose: print(f"Out of options for node dtid={node.id}")
                        if self.backtrack:
                            # get the precursor node, and remove the previously chosen operation from its options
                            node = node.get_precursor()
                            if self.verbose: print(f"Backtracked to precursor node {node.id}")
                            # backtrack to the previous state of the stack
                            if self.verbose: print(f"Removing node {stack.stack[-1][0].id} operation {node.operation.name} from its available rules {stack.stack[-1][0].available_rules['options']}")
                            stack.restore(self.stack, node)
                            if self.verbose: print(f"Backtracked to node dtid={node.id}")
                            if self.verbose: print(f"Stack of node dtid={node.id}: {stack}")
                        else:
                            raise OutOfOptionsError
        return node.get_root(), stack, max_id, time.time() - start_time

    def sample_recursive(self, root, stack, max_id):
        """
        Recursive version of the sampling function
        """
        return None, None, None, None

    def save_results(self, iteration):
        if self.results_path:
            makedirs(self.results_path, exist_ok=True)
            with open(join(self.results_path, f"search_results_{iteration}.pkl"), "wb") as f:
                pickle.dump({
                    "rewards": self.rewards,
                    "iteration": iteration,
                    "rng_state": random.getstate(),
                }, f)

    def load_results(self):
        # find the latest search results
        latest_results = glob(join(self.results_path, "search_results_*.pkl"))
        if latest_results:
            latest_results = sorted(latest_results, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            latest_results = latest_results[-1]
            with open(latest_results, "rb") as f:
                data = pickle.load(f)
                self.rewards = data["rewards"]
                self.iteration = data["iteration"] + 1
                # set the random seed
                self.set_rng_state(state=data["rng_state"])
                print(f"Continuing search from iteration {self.iteration}")
        else:
            print("No previous search results found, starting from scratch")
