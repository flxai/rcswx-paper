from copy import deepcopy
from random import choices
import sys
from rich import print
import psutil


class OutOfOptionsError(Exception):
    pass


class PCFG:
    def __init__(self, grammar, limiter):
        self.grammar = grammar
        self.limiter = limiter

    def sample(self, node, verbose=False):
        available_options, available_probs = self.get_available_options(node, verbose)
        node.available_rules = {"options": available_options, "probs": available_probs}
        # if verbose: print(f"Sampled options for node {node.id} at level {node.level}: {available_options}, {available_probs}")
        options, probs = self.filter_options(node, available_options, available_probs, verbose)
        if verbose: print(f"Filtered options for node {node.id} at level {node.level}: {[op.name for op in options]}")
        if verbose: print(f"Full list of options at node {node.id}: {[op.name for op in available_options]}")
        # node.available_rules = {"options": options, "probs": probs}
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
            available_options = deepcopy(self.grammar[node.level]["options"])
            available_probs = deepcopy(self.grammar[node.level]["probs"])
        else:
            available_options = node.available_rules["options"]
            available_probs = node.available_rules["probs"]
        # print(f"Available options for node {node.id} at level {node.level}: {[op.name for op in available_options]}")
        # print(f"Available probabilities for node {node.id} at level {node.level}: {available_probs}")
        return available_options, available_probs

    def filter_options(self, node, options, probs, verbose=False):
        indices = [i for i, op in enumerate(options) if op.valid(node)]
        # check if max depth has been reached
        # print(f"Node depth: {node.depth}, duration: {duration}, node id: {node.id}")
        success = self.limiter.check(node, verbose)
        if not success:
            # if we are choosing between modules,
            # remove all but the computation module option
            if any([op.name == "computation" for op in options]):
                indices = [i for i in indices if options[i].name == "computation"]
            if verbose: print(f"Filtered options for node {node.id} at level {node.level}: {[options[i].name for i in indices]}")
        options, probs = [options[i] for i in indices], [probs[i] for i in indices]
        # renormalize the probabilities
        probs = [p / sum(probs) for p in probs]
        return options, probs

    def __repr__(self):
        # return the object in a readable format
        result = ["Grammar:"]
        for level, rules in self.grammar.items():
            result.append(f"\t{level}:")
            for i, (rule, prob) in enumerate(zip(rules["options"], rules["probs"])):
                result.append(f"\t\t{rule.name:<20}(p={prob})")
        return "\n".join(result)

    def __str__(self):
        # return the object in a readable format
        result = ["Grammar:"]
        for level, rules in self.grammar.items():
            result.append(f"\t{level}:")
            for i, (rule, prob) in enumerate(zip(rules["options"], rules["probs"])):
                result.append(f"\t\t{rule.name:<20}(p={prob})")
        return "\n".join(result)

    def __sizeof__(self):
        # computes the total size of this object
        return sum(map(sys.getsizeof, self.__dict__.values()))
