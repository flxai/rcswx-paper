from copy import deepcopy
import sys
import time
import pickle

import dill
import json
import msgpack


class Operation:
    def __init__(self, name, build, infer, valid, inherit, give_back, type, child_levels=[]):
        self.name = name
        self.build = build
        self.infer = infer
        self.valid = valid
        self.inherit = inherit
        self.give_back = give_back
        self.type = type
        self.child_levels = child_levels

    def is_valid(self, node):
        return self.valid(node)

    def is_terminal(self):
        return self.type == "terminal"

    def __repr__(self):
        return f"Operation({self.name}, {self.type}, {self.child_levels})"

    def __sizeof__(self):
        # computes the total size of this object
        return sum(map(sys.getsizeof, self.__dict__.values()))

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class DerivationTreeNode:
    def __init__(self, id, level="network", parent=None, input_params={}, operation=None):
        self.id = id
        self.level = level
        self.parent = parent
        self.children = []
        self.input_params = input_params
        self.output_params = {}
        if operation:
            self.initialise(operation, id)
        else:
            self.operation = None
        self.available_rules = None

    def initialise(self, operation, stack, max_id):
        self.memory = (deepcopy(stack), max_id)
        # self.memory = (pickle.loads(pickle.dumps(stack)), max_id)
        # self.memory = (dill.loads(dill.dumps(stack)), max_id) # dill is slower than deepcopy
        # self.memory = (json.loads(json.dumps(stack)), max_id)
        # self.memory = (msgpack.loads(msgpack.dumps(stack)), max_id)

        self.operation = operation
        # print(f"Initializing node {self.id} with operation {self.operation}")

        # Compute the output params for the current node
        if self.operation.is_terminal():
            self.output_params = self.operation.infer(self)
        else:
            for i, child_level in enumerate(operation.child_levels):
                child = DerivationTreeNode(
                    id=max_id + i + 1,
                    level=child_level,
                    parent=self,
                )
                self.add_child(child)
        # print(f"initialised node {self.id} with operation {self.operation}")
        for child in reversed(self.children):
            # print(f"Adding child {child.id} to stack")
            stack.append((child, False))
            max_id = max(child.id, max_id)
        return stack, max_id

    def add_child(self, child):
        self.children.append(child)
        child.set_parent(self)

    def set_parent(self, parent):
        self.parent = parent

    def get_precursor(self):
        if self.is_root():
            self.precursor = None
        self_idx = self.parent.children.index(self)
        if self_idx == 0: # first child
            precursor = self.parent
        else: # not first child
            precursor = self.parent.children[self_idx - 1]
            while precursor.children: # find most recent 'cousin'
                precursor = precursor.children[-1]
        return precursor

    def inherit_input_params(self):
        child_idx = self.parent.children.index(self)
        self.parent.operation.inherit[child_idx](self)

    def give_back_output_params(self):
        if not self.is_root():
            child_idx = self.parent.children.index(self)
            self.parent.operation.give_back[child_idx](self)

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return self.children == []

    def is_first_child(self):
        return self.parent.children[0] == self

    def get_root(self):
        if self.is_root():
            return self
        return self.parent.get_root()

    def limit_options(self, operation):
        # self.parent.children.remove(self)
        # get index of the operation in the available rules
        op_names = [op.name for op in self.available_rules["options"]]
        idx = op_names.index(operation.name)
        self.available_rules["options"].pop(idx)
        self.available_rules["probs"].pop(idx)

    def __sizeof__(self):
        # computes the total size of this object
        return sum(map(sys.getsizeof, self.__dict__.values()))

    def __repr__(self):
        return (
            f"DerivationTreeNode(" \
            f"id={self.id}, level={self.level}, operation={self.operation}, input_params={self.input_params}, " \
            f"output_params={self.output_params}, address={hex(id(self))}, " \
            # f"memory={self.memory if hasattr(self, 'memory') else None}, " \
            f"size={round(self.__sizeof__() / 1e6, 2)} MB" \
            f")"
        )

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class Stack:
    def __init__(self, stack=[]):
        self.stack = stack

    def append(self, node):
        self.stack.append(node)
        # avoid saving too many memory states
        # for node, _ in self.stack[:-1]:
        #     node.memory = None

    def pop(self):
        return self.stack.pop()

    def restore(self, stack, node):
        self.stack = stack.stack
        self.stack[-1] = (self.stack[-1][0], False)
        new_node = self.stack[-1][0]
        new_node.limit_options(node.operation)

    def is_empty(self):
        return self.stack == []

    def __sizeof__(self):
        # computes the total size of this object
        return sum(map(sys.getsizeof, self.__dict__.values()))

    def __repr__(self):
        repr = "Stack(\n"
        if self.stack:
            for node in self.stack[:-1]:
                repr += f"\t{node},\n"
            repr += f"\t{self.stack[-1]}\n"
        repr += ")"
        return repr

    def __str__(self):
        repr = "Stack(\n"
        if self.stack:
            for node in self.stack[:-1]:
                repr += f"\t{node},\n"
            repr += f"\t{self.stack[-1]}\n"
        repr += ")"
        return repr
