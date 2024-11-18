from functools import partial
from pprint import pprint

import torch

from search_strategies import create_search_strategy
from pcfg import PCFG
from grammars import grammars
from evaluation import evaluation_fn
from arguments import parse_arguments
from data import get_data_loaders
from trainers import Trainer
from network import Network
from utils import load_config


# parse the arguments
args = parse_arguments()
args = load_config(args)
pprint(vars(args))

# set the seed
torch.manual_seed(args.seed)

# create the grammar
grammar = PCFG(grammars[args.search_space])

train_loader, val_loader, trainval_loader, test_loader = get_data_loaders(
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
    val_loader=val_loader
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
search = create_search_strategy(args, grammar, eval_fn, input_params)

# run the search
search.learn(steps=args.steps)
