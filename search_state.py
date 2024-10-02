from copy import deepcopy
from random import choice, choices
from rich import print


class SearchState:
    """A class to represent the state at an iteration of the search."""

    def __init__(
        self,
        search_space,
        evaluation_fn,
        # state variables
        operation,
        level,
        input_shape,
        other_shape,
        output_shape,
        input_mode,
        other_mode,
        output_mode,
        input_branching_factor,
        output_branching_factor,
        last_im_input_shape,
        module_depth,
        node_type,
        node_id,
    ):
        self.search_space = search_space
        self.evaluation_fn = evaluation_fn
        self.operation = operation
        self.level = level
        self.input_shape = input_shape
        self.other_shape = other_shape
        self.output_shape = output_shape
        self.input_mode = input_mode
        self.other_mode = other_mode
        self.output_mode = output_mode
        self.input_branching_factor = input_branching_factor
        self.output_branching_factor = output_branching_factor
        self.last_im_input_shape = last_im_input_shape
        self.module_depth = module_depth
        self.node_type = node_type
        self.node_id = node_id

    def available_operations(self):
        return self.search_space.get_available_options(self)

    def sample_operation(self):
        options = self.available_operations()
        print(options)
        # if the computation module is an available choice,
        # we give it a higher probability of being chosen
        # to balance the depth of sampled architectures
        # a computation_module_prob of over 50%
        # will lead to potentially infinite recursion
        if "computation_module" in [fn.__name__ for fn in options]:
            probs = [
                (
                    self.search_space.computation_module_prob
                    if fn.__name__ == "computation_module"
                    else (1 - self.search_space.computation_module_prob)
                    / (len(options) - 1)
                )
                for fn in options
            ]
            chosen = choices(options, weights=probs, k=1)[0]
        # otherwise we sample uniformly
        else:
            chosen = choice(options)
        return chosen

    def grow(self, operation):
        # update the state with the chosen operation
        new_search_state = self.search_space.grow(deepcopy(self), operation) # returns the new states of the children
        # print("New search state: ", new_search_state)
        return new_search_state

    def score(self):
        return self.evaluation_fn(self)

    def __str__(self):
        return f"SearchState(" \
        f"operation={self.operation}, " \
        f"level={self.level}, " \
        f"input_shape={self.input_shape}, " \
        f"other_shape={self.other_shape}, " \
        f"output_shape={self.output_shape}, " \
        f"input_mode={self.input_mode}, " \
        f"other_mode={self.other_mode}, " \
        f"output_mode={self.output_mode}, " \
        f"input_branching_factor={self.input_branching_factor}, " \
        f"output_branching_factor={self.output_branching_factor}, " \
        f"last_im_input_shape={self.last_im_input_shape}, " \
        f"module_depth={self.module_depth}, " \
        f"node_type={self.node_type}, " \
        f"node_id={self.node_id}, " \
        f")"
