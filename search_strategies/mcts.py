"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from collections import defaultdict
from copy import deepcopy
import pickle
from pcfg import OutOfOptionsError
from search_state import Stack, DerivationTreeNode
from visualise import visualise_derivation_tree
from visualise import visualise_search_tree_2 as visualise_search_tree

from rich import print
import math, random
from tqdm import tqdm
from time import time


class TimeLimitExceededError(Exception):
    pass


class SearchTreeNode:
    """
    A representation of a single search state.
    MCTS works by constructing a search tree of these Nodes.
    """
    def __init__(self, id, pcfg, node, operation, stack, max_id, verbose=False, backtrack=True):
        self.id = id
        self.pcfg = pcfg
        self.node = node
        self.operation = operation
        self.stack = stack
        self.max_id = max_id
        self.verbose = verbose
        self.backtrack = backtrack

    def step(self, node, visited, stack, max_id, operation=None):
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
                operation = operation if operation else self.pcfg.sample(node, verbose=self.verbose)
                stack, max_id = node.initialise(operation, stack, max_id)
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
        "All possible successors of this board state"
        # if self.id == 1:
        #     # if the current node is the root node
        #     stack = self.stack
        # else:
        #     stack = deepcopy(self.stack)
        stack = deepcopy(self.stack)
        if self.verbose: print(f"Stack of node stid={self.id}: {stack}")
        node, visited = stack.pop()
        if self.verbose: print(f"After pop, stack of node stid={self.id}: {stack}")
        if not node.is_root(): node.inherit_input_params()
        if self.verbose: print(f"Inherited input_params: {node.input_params}")
        # find the available and filtered options for the current node
        options, probs = self.pcfg.get_available_options(node)
        options, _ = self.pcfg.filter_options(node, options, probs)
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
                verbose=self.verbose,
                backtrack=self.backtrack
            )
            children.append(child)
            if self.verbose: print(f"Child node {child}\nwith parent {child.node.parent}")
            if self.verbose: print(f"Stack of child node {child.id}: {child.stack}")
        return children, search_tree_max_id + len(children)

    def find_random_child(self, search_tree_max_id, operation=None):
        "Random successor of this board state (for more efficient simulation)"
        if self.verbose: print(f"Finding random child of node stid={self.id} with operation {self.operation}")
        if self.verbose: print(f"Stack of node stid={self.id}: {self.stack}")
        node, visited = self.stack.pop()
        # find the available and filtered options for the current node
        node, stack, max_id = self.step(node, visited, self.stack, self.max_id, operation=operation)
        while not stack.is_empty() and stack.stack[-1][1]:
            # while the last element in the stack is visited
            # we want to pop it and go back to the parent
            _node, visited = stack.pop()
            _node, stack, max_id = self.step(_node, visited, stack, max_id)
        child = SearchTreeNode(
            search_tree_max_id + 1,
            self.pcfg,
            node,
            node.operation,
            stack,
            max_id,
            verbose=self.verbose,
            backtrack=self.backtrack
        )
        # if self.verbose: print(f"Child node {child}\nwith parent\n{child.node.parent}")
        return child, search_tree_max_id + 1

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
            input_params,
            seed=0,
            exploration_weight=1,
            mode="iterative",
            backtrack=True,
            max_id_limit=1000,
            time_limit=300,
            verbose=False,
            verbose_after_iteration=None,
            visualise=False,
            visualise_after_iteration=None,
            visualise_scale=0.5,
            save_fig_path=None,
            save_results_path=None,
        ):
        self.evaluation_fn = evaluation_fn
        self.pcfg = pcfg
        self.input_params = input_params
        self.seed = seed
        self.exploration_weight = exploration_weight
        self.mode = mode
        self.backtrack = backtrack
        self.max_id_limit = max_id_limit
        self.time_limit = time_limit
        self.verbose = verbose
        self.verbose_after_iteration = verbose_after_iteration
        self.visualise = visualise
        self.visualise_after_iteration = visualise_after_iteration
        self.visualise_scale = visualise_scale
        self.save_fig_path = save_fig_path
        self.save_results_path = save_results_path

        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node

        self.set_seed()

    def set_seed(self):
        random.seed(self.seed)

    def learn(self, rollouts=10):
        root = DerivationTreeNode(1, "network", input_params=self.input_params)
        node = SearchTreeNode(
            id=1,
            pcfg=self.pcfg,
            node=root,
            operation=None,
            stack=Stack([(root, False)]),
            max_id=1,
            verbose=self.verbose,
            backtrack=self.backtrack
        ) # The root of the search tree
        self.search_tree_max_id = 1
        self._expand(node)

        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing 10 rollouts each turn.
        for iteration in tqdm(range(rollouts)):
            if iteration == self.verbose_after_iteration:
                self.verbose = True
            if iteration == self.visualise_after_iteration:
                self.visualise = True

            end_node, path = self.do_rollout(node, iteration)
            if self.visualise: visualise_derivation_tree(end_node.node.get_root(), scale=self.visualise_scale, iteration=iteration)
            if self.verbose: print("Path", path)
            visualise_search_tree(
                node,
                self.children,
                self.Q,
                self.N,
                path=[(a.id, b.id) for a, b in zip(path[0:], path[1:])],
                scale=self.visualise_scale,
                iteration=iteration,
                save_path=self.save_fig_path,
                show=self.visualise,
            )
        # node = self.choose(node.get_root())
        # print(node)
        if self.verbose: print(self.Q)
        if self.verbose: print(self.N)
        # return self.get_best()

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
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        if self.verbose: print("Path", path)
        # path = self._make_path(path)
        # if self.verbose: print("Path", path)
        leaf = path[-1]
        # if self.verbose: print(f"Expanding children of node stid={leaf.id}")
        # if self.verbose: print(f"Stack of node stid={leaf.id}: {leaf.stack}")
        # if self.verbose: print(f"Leaf node before expand {leaf}")
        self._expand(leaf)
        # if self.verbose: print(f"Leaf node after expand {leaf}")
        if self.verbose: print("Simulating architecture")
        # changed so it runs the simulate function from the root node
        # and instantiates all nodes in the path to the leaf node
        # given the existing operations and params in these nodes
        # that will ensure that the deepcopy does not interfer in the simulation
        success = False
        while not success:
            simulation_path = deepcopy(path)
            final_leaf, reward = self._simulate(simulation_path)
            if final_leaf is not None:
                success = True
            else:
                print("Retrying simulation")
        if self.verbose: print(f"Leaf node after simulate {leaf}")
        if self.verbose: print("Simulated architecture, with reward:", reward)
        self._backpropagate(path, reward)
        if self.verbose: print("Backpropagated reward")
        # save to results
        if self.save_results_path:
            with open(f"{self.save_results_path}/results_{iteration}.pkl", "wb") as f:
                pickle.dump({
                    "leaf": final_leaf,
                    "reward": reward,
                    "path": path,
                    "Q": self.Q,
                    "N": self.N,
                    "children": self.children
                }, f)
        return final_leaf, path

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
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        if self.verbose: print(f"Expanding node {node}")
        new_children, self.search_tree_max_id = node.find_children(self.search_tree_max_id)
        self.children[node] = new_children
        if self.verbose: print(f"Expanded node {node}\nwith children\n{self.children[node]}")

    def _simulate(self, path):
        "Returns the reward for a random simulation (to completion) of `node`"
        # keep track of time and stop if it exceeds the time limit
        start_time = time()
        search_tree_max_id = self.search_tree_max_id
        try:
            node = None
            for (path_node, child) in zip(path, path[1:]):
                if node is None:
                    node = path_node
                if self.verbose:
                    node.verbose = True
                if self.verbose: visualise_derivation_tree(node.node.get_root(), scale=self.visualise_scale, iteration=0)
                if self.is_terminal(node):
                    # if self.visualise: visualise_derivation_tree(node.node.get_root(), scale=self.visualise_scale, iteration=0)
                    return node, self._reward(node)
                node, search_tree_max_id = node.find_random_child(search_tree_max_id, operation=child.operation)
                if self.verbose: print("Reconstructing derivation tree from path", node)
                if self.verbose: print(f"Stack of node stid={node.id}: {node.stack}")
            while True:
                if time() - start_time > self.time_limit:
                    print("Time limit exceeded")
                    return None, None
                if self.verbose:
                    node.verbose = True
                if self.verbose: visualise_derivation_tree(node.node.get_root(), scale=self.visualise_scale, iteration=0)
                if self.is_terminal(node):
                    # if self.visualise: visualise_derivation_tree(node.node.get_root(), scale=self.visualise_scale, iteration=0)
                    return node, self._reward(node)
                node, search_tree_max_id = node.find_random_child(search_tree_max_id)
                if self.verbose: print("Simulating", node)
                if self.verbose: print(f"Stack of node stid={node.id}: {node.stack}")
        except OutOfOptionsError:
            print("Reached dead end, returning reward 0")
            return node, 0


    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)

    def _reward(self, node):
        "Return the reward for the node"
        return self.evaluation_fn(node.node.get_root(), verbose=self.verbose)

    def is_terminal(self, node):
        return node.stack.is_empty()
