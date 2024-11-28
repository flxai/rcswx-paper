"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from collections import defaultdict
from copy import deepcopy
from functools import partial
from os.path import join, exists
from os import makedirs, rename, remove
import pickle
from search_strategies.random_search import Sampler
from pcfg import OutOfOptionsError
from search_state import Stack, DerivationTreeNode
from visualise import visualise_derivation_tree
from visualise import visualise_search_tree_2 as visualise_search_tree
from plot import Plotter

from rich import print
import math, random
from tqdm import tqdm
from scipy.stats import norm

ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])


class TimeLimitExceededError(Exception):
    pass


class SearchTreeNode:
    """
    A representation of a single search state.
    MCTS works by constructing a search tree of these Nodes.
    """
    def __init__(
            self,
            id,
            pcfg,
            node,
            operation,
            stack,
            max_id,
            time_limit,
            max_id_limit,
            depth_limit,
            mem_limit,
            verbose=False,
            backtrack=True
        ):
        self.id = id
        self.pcfg = pcfg
        self.node = node
        self.operation = operation
        self.stack = stack
        self.max_id = max_id
        self.time_limit = time_limit
        self.max_id_limit = max_id_limit
        self.depth_limit = depth_limit
        self.mem_limit = mem_limit
        self.verbose = verbose
        self.backtrack = backtrack

    def step(self, node, visited, stack, max_id, operation=None):
        """
        Step through the search tree
        """
        if self.verbose: print(f"Stepping node dtid={node.id}, visited={visited} with operation {node.operation}")
        if visited:
            # Propagate the output params to the parent
            node.give_back_output_params()
            if self.verbose and not node.is_root(): print(f"Propagated output params from node dtid={node.id}({hex(id(node))}) to parent dtid={node.parent.id}({hex(id(node.parent))})")
        else:
            stack.append((node, True))
            if not node.is_root():
                # inherit the input params from the parent
                node.inherit_input_params()
                if self.verbose: print(f"Inherited input params from parent dtid={node.parent.id}: {node.input_params}")
            try:
                # select operation and initialise the node, children etc.
                operation = operation if operation else self.pcfg.sample(
                    node,
                    verbose=self.verbose
                )
                stack, max_id = node.initialise(operation, stack, max_id, id_stack=False)
                if self.verbose: print(f"initialised node dtid={node.id} with operation {operation}")
            except OutOfOptionsError:
                if self.verbose: print(f"Out of options for node dtid={node.id}")
                if self.backtrack:
                    # get the precursor node, and remove the previously chosen operation from its options
                    node = node.get_precursor()
                    # backtrack to the previous state of the stack
                    stack.restore(self.stack, node)
                    if self.verbose: print(f"Backtracked to node dtid={node.id}")
                    if self.verbose: print(f"Stack of node dtid={node.id}: {stack}")
                else:
                    raise OutOfOptionsError
        return node, stack, max_id

    def find_children(self, search_tree_max_id):
        """
        Find all the children of this search tree node
        These are the possible operations of the next node in the stack
        """
        stack = deepcopy(self.stack)
        if self.verbose: print(f"Stack of node stid={self.id}: {stack}")
        node, visited = stack.pop()
        if self.verbose: print(f"After pop, stack of node stid={self.id}: {stack}")

        if not node.is_root(): node.inherit_input_params()
        if self.verbose: print(f"Inherited input_params: {node.input_params}")

        # find the available and filtered options for the current node
        options, probs = self.pcfg.get_available_options(node, verbose=self.verbose)
        options, _ = self.pcfg.filter_options(
            node,
            options,
            probs,
            verbose=self.verbose
        )
        
        # if self.verbose: print(f"Available options of node dtid={node.id}: {options}")
        children = []
        for i, operation in enumerate(options):
            child_node, child_stack, child_max_id = self.step(deepcopy(node), visited, deepcopy(stack), self.max_id, operation=operation)
            while not child_stack.is_empty() and child_stack.stack[-1][1]:
                # while the last element in the stack is visited
                # we want to pop it and go back to the parent
                _node, child_visited = child_stack.pop()
                _node, child_stack, child_max_id = self.step(_node, child_visited, child_stack, child_max_id)

            # extra fix for giving back output params
            def give_back_fix(node):
                if not node.is_root():
                    for n, _ in child_stack.stack:
                        if n == node.parent:
                            n.output_params = node.output_params
                            if self.verbose: print(f"Propagated output params from node dtid={node.id}({hex(id(node))}) to parent dtid={n.id}({hex(id(n))})")
                            if self.verbose: print(f"Output params: {n.output_params}")
                            for c in n.children:
                                if c.id == node.id:
                                    c.input_params = node.input_params
                                    c.operation = node.operation
                                    c.output_params = node.output_params
                    give_back_fix(node.parent)
            give_back_fix(child_node)

            child = SearchTreeNode(
                id=search_tree_max_id + i + 1,
                pcfg=self.pcfg,
                node=child_node,
                operation=child_node.operation,
                stack=child_stack,
                max_id=child_max_id,
                time_limit=self.time_limit,
                max_id_limit=self.max_id_limit,
                depth_limit=self.depth_limit,
                mem_limit=self.mem_limit,
                verbose=self.verbose,
                backtrack=self.backtrack
            )
            children.append(child)
            if self.verbose: print(f"Child node {child}\nwith parent {child.node.parent}")
            if self.verbose: print(f"Stack of child node {child.id}: {child.stack}")
        return children, search_tree_max_id + len(children)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return f"SearchTreeNode(id={self.id}, node={self.node})"

    def __repr__(self):
        return str(self)


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    def __init__(
            self,
            evaluation_fn,
            pcfg,
            limiter,
            input_params,
            seed=0,
            mode="iterative",
            backtrack=True,
            time_limit=300,
            max_id_limit=1000,
            depth_limit=20,
            mem_limit=4096,
            verbose=False,
            visualise=False,
            visualise_scale=0.5,
            vis_interval=10,
            figures_path=None,
            results_path=None,
            continue_search=False,
            # mcts specific parameters
            acquisition_fn="uct",
            exploration_weight=1.0,
            incubent_type="parent",
            reward_mode="sum",
        ):
        self.evaluation_fn = evaluation_fn
        self.pcfg = pcfg
        self.limiter = limiter
        self.input_params = input_params
        self.seed = seed
        self.mode = mode
        self.backtrack = backtrack
        self.time_limit = time_limit
        self.max_id_limit = max_id_limit
        self.depth_limit = depth_limit
        self.mem_limit = mem_limit
        self.verbose = verbose
        self.visualise = visualise
        self.visualise_scale = visualise_scale
        self.vis_interval = vis_interval
        self.figures_path = figures_path
        self.results_path = results_path
        self.continue_search = continue_search
        # mcts specific parameters
        self.exploration_weight = exploration_weight
        self.acquisition_fn = acquisition_fn
        self.incubent_type = incubent_type
        self.reward_mode = reward_mode

        initial_derivation_tree_max_id = 1
        initial_search_tree_max_id = 1

        # initialise the sampler
        self.sampler = Sampler(
            pcfg=self.pcfg,
            mode=self.mode,
            time_limit=self.time_limit,
            max_id_limit=self.max_id_limit,
            depth_limit=self.depth_limit,
            mem_limit=self.mem_limit,
            verbose=self.verbose
        )

        if self.acquisition_fn == "uct":
            self.acquisition_select = self._uct
        elif self.acquisition_fn == "ei":
            self.acquisition_select = self._ei
        else:
            raise ValueError("Invalid acquisition function")

        # default initialisation of the search
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.children = dict()
        self.rewards = []
        self.iteration = 0
        self.search_tree_max_id = initial_search_tree_max_id
        root = DerivationTreeNode(
            initial_derivation_tree_max_id,
            "network",
            input_params=self.input_params,
            limiter=self.pcfg.limiter,
        )
        self.search_tree_root = SearchTreeNode(
            id=self.search_tree_max_id,
            pcfg=self.pcfg,
            node=root,
            operation=None,
            stack=Stack([(root, False)]),
            max_id=1,
            time_limit=self.time_limit,
            max_id_limit=self.max_id_limit,
            depth_limit=self.depth_limit,
            mem_limit=self.mem_limit,
            verbose=self.verbose,
            backtrack=self.backtrack
        ) # The root of the search tree
        # set the random seed
        self.set_rng_state(seed=self.seed)

        # continue search from previous results
        if self.continue_search:
            self.load_results()

        # fix for the clock
        self.limiter.timer.start()
        print(f"Initialised MCTS at {self.limiter.timer.start_time}")

    def set_rng_state(self, seed=None, state=None):
        if state:
            random.setstate(state)
        elif seed:
            random.seed(seed)

    def learn(self, steps=10):
        """
        Run the Monte Carlo Tree Search algorithm for a number of steps
        """
        print("-----------------------")
        print("Monte Carlo Tree Search")
        print(f"Steps: {steps}")
        print("-----------------------")

        # expand the initial search tree from the root
        print("Initialising search tree: expanding children of root node")
        # start a timer for the first expansion
        self.limiter.timer.start()
        self._expand(self.search_tree_root)

        # iterate through the search tree, expanding and simulating
        for iteration in tqdm(range(self.iteration, steps), desc="MCTS", initial=self.iteration, total=steps):
            # do a single iteration of MCTS
            end_node, path, reward = self.do_rollout(self.search_tree_root, iteration)

            # save to results
            self.save_results(iteration)

            self.plot(end_node, path, reward, iteration)

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if self.is_terminal(node):
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child(self.search_tree_max_id)

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node, iteration):
        "Make the tree one layer bigger. (Train for one iteration.)"
        # select the node to expand
        path = self._select(node)
        if self.verbose: print("Path", path)

        # expand the node
        leaf = path[-1]
        # start timer
        self.limiter.timer.start()
        self._expand(leaf)

        success = False
        while not success:
            try:
                if self.verbose: print("Simulating architecture")
                simulation_path = deepcopy(path)
                # start timer
                self.limiter.timer.start()
                # simulate the architecture
                root = self._simulate(simulation_path)
                sample_duration = self.limiter.timer()

                # start timer
                self.limiter.timer.start()
                # evaluate the architecture
                reward = self._reward(root)
                eval_duration = self.limiter.timer()
                if self.verbose: print(f"Leaf node after simulate {leaf}")
                if self.verbose: print("Simulated architecture, with reward:", reward)
                success = True
            except (RuntimeError, MemoryError):
                print("GPU or RAM Memory error, trying again")

        # backpropagate the reward
        self._backpropagate(path, reward)
        if self.verbose: print("Backpropagated reward")

        # save the rewards
        serialised_architecture = root.serialise()
        self.rewards.append((serialised_architecture, reward, sample_duration, eval_duration))
        print(f"Iteration {iteration}, reward: {reward:.2f}, sample duration: {sample_duration:.2f}, eval duration: {eval_duration:.2f}")
        # print(f"Architecture:")
        # for line in serialised_architecture:
        #     print(line)

        return root, path, reward

    def _select(self, node):
        "Find an unexplored descendant of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            # All children of node should already be expanded:
            assert all(n in self.children for n in self.children[node])
            # descend a layer deeper
            node = max(self.children[node], key=partial(self.acquisition_select, parent=node))

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        if self.is_terminal(node):
            if self.verbose: print(f"Node stid={node.id} is at the end of a path")
            return # terminal node
        if self.verbose: print(f"Expanding node {node}")
        new_children, self.search_tree_max_id = node.find_children(self.search_tree_max_id)
        self.children[node] = new_children
        if self.verbose: print(f"Expanded node {node}\nwith children\n{self.children[node]}")

    def _simulate(self, path):
        "Returns the reward for a random simulation (to completion) of `node`"
        # keep track of time and stop if it exceeds the time limit
        operations = [node.node.operation for node in path if node.node.operation]
        root = self.sampler.sample(self.input_params, operations)
        return root

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            if self.reward_mode == "sum":
                self.Q[node] += reward
            elif self.reward_mode == "max":
                self.Q[node] = max(self.Q[node], reward)

    def _get_score(self, idx):
        if self.reward_mode == "sum":
            return self.Q[idx] / self.N[idx]
        elif self.reward_mode == "max":
            return self.Q[idx]

    def _uct(self, node, parent):
        "Upper confidence bound for trees"
        log_N_vertex = math.log(self.N[parent])
        mu = self._get_score(node)
        return mu + self.exploration_weight * math.sqrt(
            log_N_vertex / self.N[node]
        )

    def _ei(self, node, parent):
        "Expected Improvement for trees"
        log_N_vertex = math.log(self.N[parent])

        # what to compare it to
        if self.incubent_type == "global": # highest avg reward of all nodes
            y_star = max([self._get_score(n) for n in self.Q])
        elif self.incubent_type == "parent": # same as avg of children
            y_star = self._get_score(parent)
        elif self.incubent_type == "children":
            y_star = max([self._get_score(n) for n in self.children[parent]])

        h = lambda z, mu, sigma: norm.pdf(z, loc=0, scale=1) + z * norm.cdf(z, loc=0, scale=1)
        mu = self._get_score(node)
        sigma = math.sqrt(log_N_vertex / self.N[node]) + 0.0001

        return sigma * h((mu - y_star) / sigma, mu, sigma)

    def _reward(self, node):
        "Return the reward for the node"
        reward = self.evaluation_fn(node.get_root())
        return reward

    def is_terminal(self, node):
        return node.stack.is_empty()

    def plot(self, root, path, reward, iteration):
        # visualise the derivation and search trees
        visualise_derivation_tree(
            root.get_root(),
            scale=self.visualise_scale,
            iteration=iteration,
            save_path=self.figures_path,
            score=reward,
            show=self.visualise,
        )
        if self.verbose: print("Path", path)
        visualise_search_tree(
            self.search_tree_root,
            self.children,
            self.Q,
            self.N,
            path=[(a.id, b.id) for a, b in zip(path[0:], path[1:])],
            score_fn=lambda node, parent: self.Q[node] / self.N[node],
            scale=self.visualise_scale,
            iteration=iteration,
            save_path=self.figures_path,
            show=self.visualise,
        )
        visualise_search_tree(
            self.search_tree_root,
            self.children,
            self.Q,
            self.N,
            path=[(a.id, b.id) for a, b in zip(path[0:], path[1:])],
            score_fn=self.acquisition_select,
            scale=self.visualise_scale,
            iteration=f"acquisition_{iteration}",
            save_path=self.figures_path,
            show=self.visualise,
        )
        if iteration % self.vis_interval == 0:
            plotter = Plotter({"rewards": self.rewards})
            # find best architecture
            idx, best_arch, best_reward = plotter.find_best_architecture()
            # visualise it
            visualise_derivation_tree(
                best_arch[0], iteration=f"best_{idx}", score=best_reward, show=False,
                save_path=self.figures_path
            )
            # plot results
            plotter.plot_results("rewards", self.figures_path)
            # plot number of parameters
            plotter.plot_num_params(self.figures_path)
            # plot number of nodes
            plotter.plot_num_nodes(self.figures_path)

    def save_results(self, iteration):
        if self.results_path:
            makedirs(self.results_path, exist_ok=True)
            temp_path = join(self.results_path, f"search_results_temp.pkl")
            final_path = join(self.results_path, f"search_results.pkl")
            try:
                with open(temp_path, "wb") as f:
                    pickle.dump({
                        "rewards": self.rewards,
                        "Q": self.Q,
                        "N": self.N,
                        "children": self.children,
                        "iteration": iteration,
                        "search_tree_max_id": self.search_tree_max_id,
                        "search_tree_root": self.search_tree_root,
                        "rng_state": random.getstate(),
                    }, f)
                rename(temp_path, final_path)
            except KeyboardInterrupt:
                print("Saving interrupted. Partial results saved.")
                if exists(temp_path):
                    remove(temp_path)

    def load_results(self):
        # load the search results
        path = join(self.results_path, "search_results.pkl")
        if exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.Q = data["Q"]
                self.N = data["N"]
                self.children = data["children"]
                self.rewards = data["rewards"]
                self.iteration = data["iteration"] + 1
                self.search_tree_max_id = data["search_tree_max_id"]
                self.search_tree_root = data["search_tree_root"]
                # set the random seed
                self.set_rng_state(state=data["rng_state"])
                print(f"Continuing search from iteration {self.iteration}")
        else:
            print("No previous search results found, starting from scratch")
