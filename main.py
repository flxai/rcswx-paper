from functools import partial
from pprint import pprint

import torch

from search_strategies import create_search_strategy
from pcfg import PCFG
from grammars import grammars
from evaluation import evaluation_fn
from arguments import parse_arguments
from data import get_data_loaders
from utils import load_config, Limiter


# parse the arguments
args = parse_arguments()
args = load_config(args)
pprint(vars(args))

# set the seed
torch.manual_seed(args.seed)

# create the limiter
# this makes sure that the search does not exceed
# time, memory (GPU and RAM), depth, or node limits during the search
limiter = Limiter(
    limits={
        "time": args.time_limit,
        "max_id": args.max_id_limit,
        "depth": args.depth_limit,
        "memory": args.mem_limit,
    }
)

# create the grammar
grammar = PCFG(
    grammar=grammars[args.search_space],
    limiter=limiter,
)
print(grammar)

train_loader, val_loader, _, _ = get_data_loaders(
    dataset=args.dataset,
    batch_size=args.batch_size,
    image_size=args.image_size,
    root="../einspace/data",
    load_in_gpu=args.load_in_gpu,
    device=args.device,
    log=args.verbose_eval,
)

eval_fn = partial(
    evaluation_fn,
    args=args,
    train_loader=train_loader,
    val_loader=val_loader,
)

# create the input parameters
input_params = {
    "shape": torch.Size([1, args.channels, *args.image_size]),
    "other_shape": None,
    "mode": "im",
    "other_mode": None,
    "branching_factor": 1,
    "last_im_shape": None,
}

# create the search strategy
search = create_search_strategy(args, grammar, eval_fn, limiter, input_params)

# run the search
search.learn(steps=args.steps)
