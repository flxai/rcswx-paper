from functools import partial
from os.path import join

import torch

from search_strategies import create_search_strategy
from pcfg import PCFG
from grammars import grammars
from evaluation import evaluation_fn
from arguments import parse_arguments


# parse the arguments
args = parse_arguments()

# create the grammar
grammar = PCFG(grammars[args.grammar])

# create the evaluation function
evaluation_fn = partial(
    evaluation_fn,
    dataset=args.dataset,
    epochs=args.epochs,
    batch_size=args.batch_size,
    device=args.device,
    verbose=args.verbose_eval,
)

# create the input parameters
input_params = {
    "shape": torch.Size([1, 1, 28, 28]),
    "other_shape": None,
    "mode": "im",
    "other_mode": None,
    "branching_factor": 1,
    "last_im_shape": None,
}

# create the search strategy
search = create_search_strategy(args, grammar, evaluation_fn, input_params)

# run the search
search.learn(steps=args.steps)
