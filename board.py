from collections import namedtuple, deque
from copy import deepcopy
from random import choice
from mcts import MCTS, Node
import torch


class Board(Node):
    def __init__(self, search_state, level_queue, stack):
        self.search_state = search_state
        self.level_queue = level_queue
        self.stack = stack

    def find_children(self):
        self.search_state.level = self.level_queue[-1] # we don't pop the level_queue here
        print("Level queue: ", self.level_queue)
        print("Level: ", self.search_state.level)

        self.search_state = self.state_queue[-1] # we don't pop the state_queue here

        # Otherwise, you can grow the architecture in all available options
        options = self.search_state.available_operations()
        # print("Available operations: ", options)
        moves = set()
        for operation in options:
            moves.add(self.make_move(operation))
        return moves

    def find_random_child(self):
        self.search_state.level = self.level_queue.pop()
        print("Level queue: ", self.level_queue)
        print("Level: ", self.search_state.level)

        self.search_state = self.state_queue.pop()

        operation = self.search_state.sample_operation()
        # populate the level_queue
        if operation.__name__ == "sequential_module":
            self.level_queue.append("second_fn")
            self.level_queue.append("first_fn")
        elif operation.__name__ == "branching_module":
            self.level_queue.append("aggregation_fn")
            self.level_queue.append("inner_fn")
            self.level_queue.append("branching_fn")
        elif operation.__name__ == "routing_module":
            self.level_queue.append("postrouting_fn")
            self.level_queue.append("inner_fn")
            self.level_queue.append("prerouting_fn")
        elif operation.__name__ == "computation_module":
            self.level_queue.append("computation_fn")
        return self.make_move(operation)

    def reward(self):
        # if not self.is_terminal():
        #     raise RuntimeError(f"reward called on nonterminal board {self}")
        # evaluate architecture score
        return self.search_state.score()

    def make_move(self, operation):
        search_state = self.search_state.grow(operation) # grow the current architecture with a chosen node


        # populate the level_queue
        if operation.__name__ == "sequential_module":
            first_fn_search_state = deepcopy(search_state)
            first_fn_search_state = SearchState(
                search_space=self.search_state.search_space,
                evaluation_fn=self.search_state.evaluation_fn,

            )
            first_fn_search_state.level = "first_fn"
            first_fn_search_state.input_shape = search_state.output_shape
            first_fn_search_state.input_mode = search_state.output_mode
            first_fn_search_state.input_branching_factor = search_state.output_branching_factor
            self.state_queue.append(search_state)

            self.level_queue.append("second_fn")
            self.level_queue.append("first_fn")
        elif operation.__name__ == "branching_module":
            self.level_queue.append("aggregation_fn")
            self.level_queue.append("inner_fn")
            self.level_queue.append("branching_fn")
        elif operation.__name__ == "routing_module":
            self.level_queue.append("postrouting_fn")
            self.level_queue.append("inner_fn")
            self.level_queue.append("prerouting_fn")
        elif operation.__name__ == "computation_module":
            self.level_queue.append("computation_fn")

        return EinspaceBoard(search_state, self.level_queue, self.state_queue) # return a new board with the updated architecture

    def to_pretty_string(self):
        return self.search_state.__str__()

    def __hash__(self):
        return hash(self.search_state)

    def __eq__(self, other):
        return self.search_state == other.search_state

    def __str__(self):
        return self.search_state.__str__()

    def __repr__(self):
        return self.search_state.__str__()


def new_einspace_board():
    search_space = EinSpace(
        input_shape=(1, 3, 32, 32),
        input_mode="im",
        num_repeated_cells=1,
        computation_module_prob=0.32,
        min_module_depth=0,
        max_module_depth=100,
        device="cpu",
    )
    evaluation_fn = lambda x: torch.randn(1).item()
    initial_search_state = SearchState.new_search_state(search_space, evaluation_fn)
    return EinspaceBoard(
        search_state=initial_search_state,
        level_queue=deque(["network"]),
        stack=deque([(initial_search_state, False)]),
    )


def learn_architecture():
    from rich import print
    rollouts = 10
    tree = MCTS()
    board = new_einspace_board()
    # print("Initial search state:")
    # print("\t", board.to_pretty_string())
    while True:
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(rollouts):
            tree.do_rollout(board)
        board = tree.choose(board)
        print("New search state:")
        print("\t", board.to_pretty_string())
        print(tree.Q)
        print(tree.N)
        if tree.is_terminal():
            break


if __name__ == "__main__":
    learn_architecture()
