"""CPU-only regressions; run with unittest discover -s tests.

Execute the real alignment, genotype and limiter classes without importing the
unrelated training, plotting and dataset entry points. Only NumPy and psutil are
needed; no Rust port or archived benchmark data is used.
"""

import ast
import copy
import gc
import io
import itertools
import sys
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[1]


def load_classes(relative, names, namespace):
    path = ROOT / relative
    source = ast.parse(path.read_text(), filename=str(path))
    body = [
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    namespace = {"__name__": __name__, **namespace}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


STATE = load_classes(
    "search_state.py",
    {"Operation", "DerivationTreeNode"},
    {"deepcopy": copy.deepcopy, "sys": sys},
)
UTILITY = load_classes(
    "utils.py",
    {"Timer", "Limiter"},
    {"time": time.perf_counter, "psutil": psutil},
)
ALGORITHM = load_classes(
    "search_strategies/utils/recursive_constrained_smith_waterman.py",
    {
        "MatrixCell",
        "MatrixOperation",
        "DecoyOperation",
        "DecoyNode",
        "AlignmentMatrixRecursive",
    },
    {
        "np": np,
        "copy": copy,
        "time": time,
        "psutil": psutil,
        "gc": gc,
        "Operation": STATE["Operation"],
        "DerivationTreeNode": STATE["DerivationTreeNode"],
    },
)


def tree(description):
    ids = itertools.count()

    def build(item, parent=None):
        name, children = (item, ()) if isinstance(item, str) else (item[0], item[1:])
        operation = STATE["Operation"](
            name,
            None,
            None,
            None,
            [],
            [],
            "nonterminal" if children else "terminal",
            [],
        )
        node = STATE["DerivationTreeNode"](
            next(ids), parent=parent, operation=operation
        )
        node.children = [build(child, node) for child in children]
        return node

    return build(description)


def computation(name):
    return "computation", name


def branch(first, second):
    return "branching(2)", "clone(2)", first, second, "add(2)"


def architecture(node):
    children = tuple(architecture(child) for child in node.children)
    if node.operation.name == "branching(2)":
        children = (children[0], *sorted(children[1:-1]), children[-1])
    return node.operation.name, children


class BranchBoundaryTests(unittest.TestCase):
    def align(self, first, second, collapse=False):
        with redirect_stdout(io.StringIO()):
            limiter = UTILITY["Limiter"]({"memory_crossover": 2048})
            return ALGORITHM["AlignmentMatrixRecursive"](
                tree(first),
                tree(second),
                limiter=limiter,
                collapse_corners=collapse,
            )

    def assert_history_costs(self, alignment):
        for path in alignment.matrix[-1][-1].paths:
            self.assertEqual(path[0].op_type, "start")
            self.assertEqual(
                sum(operation.value for operation in path), alignment.distance
            )
            self.assertEqual(path[-1].i, len(alignment.model_ops1) - 1)
            self.assertEqual(path[-1].j, len(alignment.model_ops2) - 1)

    def test_reversed_branches_preserve_zero_cost_histories(self):
        identity, relu, norm = map(computation, ("identity", "relu", "norm"))
        normal, reversed_ = branch(identity, relu), branch(relu, identity)
        longer = ("sequential", identity, norm)
        cases = {
            "flat": (normal, reversed_),
            "unequal_lengths": (branch(longer, relu), branch(relu, longer)),
            "prefix": (("sequential", norm, normal), ("sequential", norm, reversed_)),
            "routing": (
                ("routing", "identity", normal, "identity"),
                ("routing", "identity", reversed_, "identity"),
            ),
            "nested": (branch(normal, norm), branch(norm, reversed_)),
        }
        for name, pair in cases.items():
            for collapse in (False, True):
                with self.subTest(case=name, collapse=collapse):
                    result = self.align(*pair, collapse=collapse)
                    self.assertEqual(result.distance, 0)
                    self.assertEqual(result.nontrivial_ops, [])
                    self.assert_history_costs(result)

    def test_prefix_edit_survives_swapped_subproblem_and_applies(self):
        identity, relu, norm = map(computation, ("identity", "relu", "norm"))
        first = "sequential", identity, branch(identity, relu)
        second = "sequential", norm, branch(relu, identity)
        for collapse in (False, True):
            with self.subTest(collapse=collapse):
                result = self.align(first, second, collapse=collapse)
                self.assertEqual(result.distance, 0.5)
                self.assertEqual([op.op_type for op in result.nontrivial_ops], ["mut"])
                self.assert_history_costs(result)
                self.assertEqual(
                    architecture(result.generate_offspring([])),
                    architecture(tree(second)),
                )
                self.assertEqual(
                    architecture(result.generate_offspring()), architecture(tree(first))
                )


if __name__ == "__main__":
    unittest.main()
