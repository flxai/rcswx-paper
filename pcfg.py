from random import choices
import sys


class OutOfOptionsError(Exception):
    pass


class PCFG:
    def __init__(self, grammar):
        self.grammar = grammar

    def sample(self, node, verbose=False):
        options, probs = self.get_available_options(node, verbose)
        if len(options) > 0:
            if verbose: print(f"Sampled options for node {node.id} at level {node.level}: {options}, {probs}")
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
            if verbose: print(f"Sampled options for node {node.id} at level {node.level}: {available_options}, {available_probs}")
            options, probs = self.filter_options(node, available_options, available_probs)
            if verbose: print(f"Filtered options for node {node.id} at level {node.level}: {options}, {probs}")
            node.available_rules = {"options": options, "probs": probs}
        return node.available_rules["options"], node.available_rules["probs"]

    def filter_options(self, node, options, probs):
        indices = [i for i, op in enumerate(options) if op.valid(node)]
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
