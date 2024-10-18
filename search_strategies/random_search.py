import time

from visualise import visualise_derivation_tree
from search_state import DerivationTreeNode, Stack
from pcfg import OutOfOptionsError


class RandomSearch:
    def __init__(
            self,
            pcfg,
            input_params,
            seed=0,
            mode="iterative",
            max_id_limit=1000,
            time_limit=300,
            verbose=False
        ):
        self.pcfg = pcfg
        self.input_params = input_params
        self.seed = seed
        self.mode = mode
        self.max_id_limit = max_id_limit
        self.time_limit = time_limit
        self.verbose = verbose

        self.set_seeds(seed)
        if verbose: print(f"Seed: {seed}")

        if mode == "iterative":
            self.sample_fn = self.sample_iterative
        elif mode == "recursive":
            self.sample_fn = self.sample_recursive

    def set_seeds(self, seed):
        # set random seeds
        import random
        random.seed(seed)

    def sample(self):
        root = DerivationTreeNode(1, "network", input_params=self.input_params)
        # root.initialise(self.pcfg.sample(root, self.verbose), [], 0)

        max_id = 1
        stack = Stack([(root, False)])

        success = False
        while not success:
            root, stack, max_id, duration, memory_list = self.sample_fn(
                root,
                stack,
                max_id,
                memory_list=[],
            )
            if root is not None:
                success = True

        return root, stack, max_id, duration, memory_list
        

    def sample_iterative(self, node, stack, max_id, memory_list=[], steps=None):
        start_time = time.time()
        i = 0
        while not stack.is_empty() and (steps is None or steps < i):
            i += 1
            if max_id > self.max_id_limit or time.time() - start_time > self.time_limit:
                if self.verbose: print(f"Breaking at max_id: {max_id}, time: {time.time() - start_time}")
                return None, stack, max_id, time.time() - start_time, memory_list

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
                        if self.verbose: print(f"Out of options for node {node.id}")
                        # get the precursor node, and remove the previously chosen operation from its options
                        node = node.get_precursor()
                        if self.verbose: print(f"Backtracked to precursor node {node.id}")
                        # backtrack to the previous state of the stack
                        if self.verbose: print(f"Removing node {stack.stack[-1][0].id} operation {node.operation.name} from its available rules {stack.stack[-1][0].available_rules['options']}")
                        stack.restore(stack, node)
                        if self.verbose: print(f"Updated stack: {stack}, max_id: {max_id}")
            mem = stack.__sizeof__() / 1e6
            memory_list.append(mem)
        return node.get_root(), stack, max_id, time.time() - start_time, memory_list

    def sample_recursive(self, root, stack, max_id, memory_list=[]):
        """
        Recursive version of the sampling function
        """
        return None, None, None, None, None
