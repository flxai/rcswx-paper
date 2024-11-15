from os.path import join, exists
from os import makedirs
import pickle
import time
import random

from tqdm import tqdm

from utils import Timer
from visualise import visualise_derivation_tree
from search_state import DerivationTreeNode, Stack
from pcfg import OutOfOptionsError


class Sampler:
    def __init__(self, pcfg, mode, max_depth=20, time_limit=300, max_id_limit=1000, verbose=False):
        self.pcfg = pcfg
        self.mode = mode
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.max_id_limit = max_id_limit
        self.verbose = verbose

        self.nodes = {}

        if self.mode == "iterative":
            self.__call__ = self.sample_iterative
        elif self.mode == "recursive":
            raise NotImplementedError("Recursive mode not implemented")

    def sample_iterative(self, input_params, operations=None, timer=Timer()):
        root = DerivationTreeNode(id=1, level="network", input_params=input_params)
        self.nodes = {root.id: root}

        max_id = root.id
        stack = Stack([(root.id, False)])

        while not stack.is_empty():

            if self.verbose: print(f"Stack: {stack}")
            node_id, visited = stack.pop()
            node = self.nodes[node_id]
            if self.verbose: print(f"Node: {node.id}, visited: {visited}")
            if self.verbose: print(f"Node: {node}")

            if visited:
                # Propagate the output params to the parent
                node.give_back_output_params()
                if not node.is_root():
                    if self.verbose: print(f"Propagated output params from node {node.id} to parent {node.parent.id}")
                    if self.verbose: print(f"Output params for node: {node.parent.id}, {node.parent.output_params}")
            else:
                stack.append((node.id, True))
                if not node.is_root():
                    # inherit the input params from the parent
                    node.inherit_input_params()
                    if self.verbose: print(f"Inherited input params from parent {node.parent.id} to node {node.id}")
                    if self.verbose: print(f"Input params for node: {node.id}, {node.input_params}")
                try:
                    if self.verbose: print(f"Sampling node {node.id}")
                    # select operation and initialise the node, children etc.
                    operation = operations.pop(0) if operations else self.pcfg.sample(
                        node,
                        limits={
                            "max_depth": self.max_depth,
                            "time_limit": self.time_limit,
                            "max_id_limit": self.max_id_limit,
                        },
                        duration=timer(),
                        verbose=self.verbose,
                    )
                    stack, max_id = node.initialise(
                        operation,
                        stack,
                        max_id,
                    )
                    for child in node.children:
                        if child.id not in self.nodes:
                            self.nodes[child.id] = child
                except OutOfOptionsError:
                    # get the precursor node, and remove the previously chosen operation from its options
                    node = node.get_precursor()
                    # backtrack to the previous state of the stack
                    stack, _ = node.memory
                    stack.restore(stack, node)
                    if self.verbose: print(f"Backtracked to node {node.id}")
        self.nodes = {}
        return root


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
            max_depth=20,
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
        self.max_depth = max_depth
        self.verbose = verbose
        self.verbose_after_iteration = verbose_after_iteration
        self.visualise = visualise
        self.visualise_after_iteration = visualise_after_iteration
        self.visualise_scale = visualise_scale
        self.figures_path = figures_path
        self.results_path = results_path
        self.continue_search = continue_search

        self.sampler = Sampler(
            pcfg=self.pcfg,
            mode=self.mode,
            max_depth=self.max_depth,
            time_limit=self.time_limit,
            max_id_limit=self.max_id_limit,
            verbose=self.verbose
        )

        self.rewards = []
        self.iteration = 0

        self.set_rng_state(seed=self.seed)

        if self.continue_search:
            self.load_results()

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
            global timer
            timer = Timer()
            root = self.sampler(self.input_params)
            sample_duration = timer()

            # evaluate the network
            timer = Timer()
            reward = self.evaluation_fn(root)
            eval_duration = timer()
            self.rewards.append((root.serialise(), reward, sample_duration, eval_duration))
            print(f"Iteration {iteration}, reward: {reward}, sample duration: {sample_duration}, eval duration: {eval_duration}")
            print(f"Architecture:")
            for line in root.serialise():
                print(line)

            # visualise the derivation tree
            visualise_derivation_tree(
                root,
                scale=self.visualise_scale,
                iteration=iteration,
                save_path=self.figures_path,
                show=self.visualise,
            )

            # save the results
            self.save_results(iteration)

            timer.stop()

    def save_results(self, iteration):
        if self.results_path:
            makedirs(self.results_path, exist_ok=True)
            with open(join(self.results_path, f"search_results.pkl"), "wb") as f:
                pickle.dump({
                    "rewards": self.rewards,
                    "iteration": iteration,
                    "rng_state": random.getstate(),
                }, f)

    def load_results(self):
        # load the search results
        path = join(self.results_path, "search_results.pkl")
        if exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.rewards = data["rewards"]
                self.iteration = data["iteration"] + 1
                # set the random seed
                self.set_rng_state(state=data["rng_state"])
                print(f"Continuing search from iteration {self.iteration}")
        else:
            print("No previous search results found, starting from scratch")
