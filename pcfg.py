from random import choices
import sys
from rich import print


class OutOfOptionsError(Exception):
    pass


class PCFG:
    def __init__(self, grammar):
        self.grammar = grammar

    def sample(self, node, limits, duration, verbose=False):
        available_options, available_probs = self.get_available_options(node, verbose)
        # if verbose: print(f"Sampled options for node {node.id} at level {node.level}: {available_options}, {available_probs}")
        options, probs = self.filter_options(node, available_options, available_probs, limits, duration, verbose)
        if verbose: print(f"Filtered options for node {node.id} at level {node.level}: {[op.name for op in options]}")
        node.available_rules = {"options": options, "probs": probs}
        if len(options) > 0:
            # if verbose: print(f"Sampled options for node {node.id} at level {node.level}: {options}, {probs}")
            operation = choices(
                options,
                weights=probs,
                k=1
            )[0]
            if verbose: print(f"Sampled operation {operation.name} for node {node.id} at level {node.level}")
        else:
            raise OutOfOptionsError(f"Out of options for node {node.id} at level {node.level}")
        return operation

    def get_available_options(self, node, verbose=False):
        if node.level not in self.grammar:
            return None
        if node.available_rules == None:
            available_options = self.grammar[node.level]["options"]
            available_probs = self.grammar[node.level]["probs"]
        else:
            available_options = node.available_rules["options"]
            available_probs = node.available_rules["probs"]
        return available_options, available_probs

    def filter_options(self, node, options, probs, limits, duration, verbose=False):
        indices = [i for i, op in enumerate(options) if op.valid(node)]
        # check if max depth has been reached
        if node.depth >= limits["max_depth"] or duration >= limits["time_limit"] or node.id >= limits["max_id_limit"]:
            # if we are choosing between modules,
            # remove all but the computation module option
            if any([op.name == "computation" for op in options]):
                indices = [i for i in indices if options[i].name == "computation"]
                # print which limit was reached
                if node.depth >= limits["max_depth"]:
                    if verbose: print(f"Depth limit reached: {node.depth}, removing all but computation module")
                if duration >= limits["time_limit"]:
                    if verbose: print(f"Time limit reached: {duration}, removing all but computation module")
                if node.id >= limits["max_id_limit"]:
                    if verbose: print(f"ID limit reached: {node.id}, removing all but computation module")
                if verbose: print(f"Filtered options for node {node.id} at level {node.level}: {[options[i].name for i in indices]}")
        options, probs = [options[i] for i in indices], [probs[i] for i in indices]
        # renormalize the probabilities
        probs = [p / sum(probs) for p in probs]
        return options, probs

    def __repr__(self):
        return f"Grammar({self.grammar})"

    def __str__(self):
        return f"Grammar({self.grammar})"

    def __sizeof__(self):
        # computes the total size of this object
        return sum(map(sys.getsizeof, self.__dict__.values()))
