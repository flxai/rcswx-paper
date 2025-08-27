from functools import partial
import os
from pprint import pprint
from tqdm import tqdm
from contextlib import nullcontext
import random
from typing import Literal, Callable

import numpy as np
import torch

from .search_strategies import create_search_strategy
from .pcfg import PCFG
from .grammars import grammars
from .network import Network
from .evaluation import evaluation_fn
from .arguments import parse_arguments
from .data import get_data_loaders
from .utils import load_config, set_dataset_specific_args, Limiter
from functools import partial
from torch.utils.data import DataLoader, Dataset


def compile_fn(node, args):
    backbone = node.build(node, set_memory_checkpoint=True)
    return Network(
        backbone,
        node.output_params["shape"],
        args.num_classes,
        vars(args)
    ).to(args.device)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

dataset_names = ["csawm", "mnist", "cifar10", "cifar100", "addnist", "language", "multnist", "cifartile", "gutenberg", "isabella", "geoclassing", "chesseract", "spherical", "ninapro", "fsd50k", "darcyflow", "psicov", "cosmic", "ecg", "satellite", "deepsea"]

def run(args):
    # set the seed
    set_seed(args.seed)

    # Generate dataloaders if we have not passed any
    if args.train_loader == None:
        if args.verbose_setup: print(">>Generating dataloaders")
        if dataset not in dataset_names: raise Exception('Dataloader not specified. Please provide with valid training and validation PyTorch dataloaders or call this function with "dataset = dataset_name" (valid dataset names include '+str(dataset_names)+')')
        else:
            train_loader, val_loader, _, _ = get_data_loaders(
                dataset=args.dataset,
                batch_size=args.batch_size,
                image_size=args.image_size,
                root=args.data_path,
                load_in_gpu=args.load_in_gpu,
                device=args.device,
                log=args.verbose_eval,
                seed=args.seed,
            )
        if args.verbose_setup: print(">>Done\n")
    else:
        train_loader = args.train_loader
    # If we have passed a training but not a validation one, the training dataloader will be used for validation as well
    if (args.train_loader != None) and (args.val_loader == None):
        val_loader = train_loader
    else:
        val_loader = args.val_loader

    if "infer" in [args.num_classes, args.channels, args.data_channels, args.image_size, args.input_mode, args.score]:
        if args.verbose_setup: print(">>Running through dataloader to check for data corruption and to infer data properties")
        min_output = torch.Tensor([0])
        max_output = torch.Tensor([0])
        inp0, out0 = next(iter(val_loader))
        with tqdm(range(len(val_loader)), desc="Loading data", leave=True) if args.verbose_setup else nullcontext(range(len(val_loader))) as pbar:
            for b, batch in enumerate(val_loader):
                inp, out = batch
                if (inp0.shape != inp.shape): raise Exception("Input data presents different shapes. Only consistent dimensions are supported yet.")
                elif (out0.shape != out.shape): raise Exception("Output data presents different shapes. Only consistent dimensions are supported yet.")
                max_output = torch.max(max_output, torch.max(out))
                min_output = torch.min(max_output, torch.min(out))
                pbar.update()#inp.shape[0]
        
        # TODO: what if each input is a different size?
        if args.input_mode == "infer": args.input_mode = "im" if len(inp.shape) == 4 else "col"
        if args.image_size == "infer": 
            if args.input_mode == "im": args.image_size = (inp.shape[-2],inp.shape[-1])
            if args.input_mode == "col": args.image_size = tuple(inp.shape[-1])

        if args.input_mode == "im":
            if len(out.shape) == 4:
                num_classes = out.shape[1]
                mode = "im2im_multiclass"
            elif len(out.shape) == 3:
                num_classes = max_output.item()+1
                mode = "im2im_singleclass"
            elif len(out.shape) == 2:
                num_classes = out.shape[1]
                mode = "im2col_multiclass"
            elif len(out.shape) == 1:
                num_classes = max_output.item()+1
                mode = "im2scalar_singleclass"
                    
        elif args.input_mode == "col":
            if len(out.shape) == 3:
                num_classes = out.shape[1]
                mode = "col2col_multiclass"
            elif len(out.shape) == 2:
                num_classes = out.shape[1]
                mode = "col2single_multiclass"
            elif len(out.shape) == 1:
                num_classes = max_output.item()+1
                mode = "col2scalar_singleclass"

        if args.num_classes == "infer": args.num_classes = int(num_classes)
        if args.channels == "infer": args.channels = inp.shape[1]
        if args.data_channels == "infer": args.data_channels = inp.shape[1]
        if args.score == "infer":
            if torch.all(out == out.int()):
                if "singleclass" in mode: args.score = "xe"
                elif "multiclass" in mode: args.score = "multi_hot"
            else: args.score = "r2"
        del inp0, out0, inp, out, batch
        if args.verbose_setup: print(">>Done\n")
    if args.score_selection == "infer": args.score_selection = args.score
    
    # get batch for batch pass time limiting
    batch = next(iter(train_loader))[0].to(args.device)
    args.batch_size = batch.shape[0]

    # create the limiter
    # this makes sure that the search does not exceed
    # time, memory (GPU and RAM), depth, or node limits during the search
    if args.verbose_setup: print(">>Building limiter and grammar")
    limiter = Limiter(
        limits={
            "time": args.time_limit,
            "restart_time": args.restart_time_limit,
            "max_id": args.max_id_limit,
            "depth": args.depth_limit,
            "memory": args.mem_limit,
            "individual_memory": args.individual_mem_limit,
            "batch_pass_seconds": args.batch_pass_limit,
        },
        batch=batch,
        compile_fn=partial(compile_fn, args=args),
    )
    limiter.set_memory_checkpoint()
    print(f"Memory checkpoint: {limiter.memory_checkpoint} MB")

    # create the grammar
    grammar = PCFG(
        grammar=grammars[args.search_space],
        limiter=limiter,
    )
    #if args.verbose_setup: print(grammar)

    eval_fn = partial(
        evaluation_fn,
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    if args.verbose_setup: print(">>Done\n")
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
    if args.verbose_setup: print(">>Generating search strategy")
    search = create_search_strategy(args, grammar, eval_fn, limiter, input_params)
    if args.verbose_setup: print(">>Done\n")

    # run the search
    search.learn(steps=args.steps)


class Einargs():
    def __init__(self,
                 train_loader:DataLoader = None,
                 val_loader:DataLoader = None,
                 dataset:Literal["csawm", "mnist", "cifar10", "cifar100", "addnist", "language", "multnist", "cifartile", "gutenberg", "isabella", "geoclassing", "chesseract", "spherical", "ninapro", "fsd50k", "darcyflow", "psicov", "cosmic", "ecg", "satellite", "deepsea"] = "custom",
                 seed:int = 42,
                 search_space:str = "einspace",
                 search_strategy:Literal["mcts", "evolution"] = "evolution",
                 search_args:dict = {},
                 steps:int = 1000,
                 backtrack:bool = True,
                 mode:str = "iterative",
                 time_limit:int|float = 60,
                 restart_time_limit:int|float = 300,
                 max_id_limit:int = 10000,
                 depth_limit:int = 20,
                 mem_limit:int|float = 8192,
                 individual_mem_limit:int|float = 1024,
                 batch_pass_limit:int|float = 0.1,
                 population_size:int = 100,
                 device:str|torch.device = "cuda:0",
                 load_in_gpu:bool = False,
                 num_classes:int|str = "infer",
                 channels:int|str = "infer",
                 data_channels:int|str = "infer",
                 image_size:int|str = "infer",
                 input_mode:int|str = "infer",
                 score:Callable|Literal["infer","xe","multi_hot","r2","mse","relative_l2","mae8","map","cosmic","ecg","deepsea"] = "infer",
                 score_selection:Callable|Literal["infer","xe","multi_hot","r2","mse","relative_l2","mae8","map","cosmic","ecg","deepsea"] = "infer",
                 batch_size:int = 128,
                 epochs:int = 25,
                 patience:int = 25,
                 n_test_models:int = 1,
                 test_epochs:int = 200,
                 gradient_accumulation:int = 1,
                 hpo_runs:int = 1,
                 lr:float = 0.1,
                 momentum:float = 0.9,
                 weight_decay:float = 0.0005,
                 verbose_setup:bool = False,
                 verbose_search:bool = False,
                 verbose_eval:bool = False,
                 visualise:bool = False,
                 visualise_scale:float = 0.8,
                 vis_interval:int = 10,
                 data_path:str = "data",
                 figures_path:str = "figures",
                 results_path:str = "results",
                 continue_search:bool = True,
                 load_from:str = None):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset = dataset
        self.seed = seed
        self.search_space = search_space
        self.search_strategy = search_strategy
        for search_arg_name, search_arg in search_args.items(): setattr(self, search_arg_name, search_arg)
        self.steps = steps
        self.backtrack = backtrack
        self.mode = mode
        self.time_limit = time_limit
        self.restart_time_limit = restart_time_limit
        self.max_id_limit = max_id_limit
        self.depth_limit = depth_limit
        self.mem_limit = mem_limit
        self.individual_mem_limit = individual_mem_limit
        self.batch_pass_limit = batch_pass_limit
        self.population_size = population_size
        self.device = device
        self.load_in_gpu = load_in_gpu
        self.num_classes = num_classes
        self.channels = channels
        self.data_channels = data_channels
        self.image_size = image_size
        self.input_mode = input_mode
        self.score = score
        self.score_selection = score_selection
        self.epochs = epochs
        self.patience = patience
        self.n_test_models = n_test_models
        self.test_epochs = test_epochs
        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation
        self.hpo_runs = hpo_runs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.verbose_setup = verbose_setup
        self.verbose_search = verbose_search
        self.verbose_eval = verbose_eval
        self.visualise = visualise
        self.visualise_scale = visualise_scale
        self.vis_interval = vis_interval
        self.data_path = data_path 
        self.figures_path = figures_path
        self.results_path = results_path
        self.continue_search = continue_search
        self.load_from = load_from


def search(train_loader:DataLoader = None,
           val_loader:DataLoader = None,
           dataset:Literal["csawm", "mnist", "cifar10", "cifar100", "addnist", "language", "multnist", "cifartile", "gutenberg", "isabella", "geoclassing", "chesseract", "spherical", "ninapro", "fsd50k", "darcyflow", "psicov", "cosmic", "ecg", "satellite", "deepsea"] = "custom",
           seed:int = 42,
           search_space:str = "einspace",
           search_strategy:Literal["mcts", "evolution"] = "evolution",
           search_args:dict = {},
           steps:int = 1000,
           backtrack:bool = True,
           mode:str = "iterative",
           time_limit:int|float = 60,
           restart_time_limit:int|float = 300,
           max_id_limit:int = 10000,
           depth_limit:int = 20,
           mem_limit:int|float = 8192,
           individual_mem_limit:int|float = 1024,
           batch_pass_limit:int|float = 0.1,
           population_size:int = 100,
           device:str|torch.device = "cuda:0",
           load_in_gpu:bool = False,
           num_classes:int|str = "infer",
           channels:int|str = "infer",
           data_channels:int|str = "infer",
           image_size:int|str = "infer",
           input_mode:int|str = "infer",
           score:Callable|Literal["infer","xe","multi_hot","r2","mse","relative_l2","mae8","map","cosmic","ecg","deepsea"] = "infer",
           score_selection:Callable|Literal["infer","xe","multi_hot","r2","mse","relative_l2","mae8","map","cosmic","ecg","deepsea"] = "infer",
           batch_size:int = 128, # This will be overwritten is a dataloader is passed
           epochs:int = 25,
           patience:int = 25,
           n_test_models:int = 1, #TODO: allow for multiple model selection (ensembles, diversity, robustness, etc)
           test_epochs:int = 200,
           gradient_accumulation:int = 1,
           hpo_runs:int = 1,
           lr:float = 0.1, # TODO: give the option of "once_cycle"
           momentum:float = 0.9,
           weight_decay:float = 0.0005,
           verbose_setup:bool = False,
           verbose_search:bool = False,
           verbose_eval:bool = False,
           visualise:bool = False,
           visualise_scale:float = 0.8,
           vis_interval:int = 10,
           data_path:str = "data",
           figures_path:str = "figures",
           results_path:str = "results",
           continue_search:bool = True,
           load_from:str = None):

    #if search_strategy == "mcts":
    if "acquisition_fn" not in search_args: search_args["acquisition_fn"] = "uct"
    if "exploration_weight" not in search_args: search_args["exploration_weight"] = 1.0
    if "incubent_type" not in search_args: search_args["incubent_type"] = "parent"
    if "reward_mode" not in search_args: search_args["reward_mode"] = "sum"
    if "add_full_paths" not in search_args: search_args["add_full_paths"] = True
    #elif search_strategy == "evolution":
    if "generational" not in search_args: search_args["generational"] = True
    if "regularised" not in search_args: search_args["regularised"] = True
    if "population_size" not in search_args: search_args["population_size"] = 100
    if "architecture_seed" not in search_args: search_args["architecture_seed"] = None
    if "mutation_strategy" not in search_args: search_args["mutation_strategy"] = "random"
    if "mutation_rate" not in search_args: search_args["mutation_rate"] = 1.0
    if "crossover_strategy" not in search_args: search_args["crossover_strategy"] = "random"
    if "crossover_rate" not in search_args: search_args["crossover_rate"] = 1.0
    if "selection_strategy" not in search_args: search_args["selection_strategy"] = "tournament"
    if "tournament_size" not in search_args: search_args["tournament_size"] = 10
    if "elitism" not in search_args: search_args["elitism"] = None
    if "n_tries" not in search_args: search_args["n_tries"] = None

    args = Einargs(train_loader,val_loader,dataset,seed,search_space,search_strategy,search_args,steps,backtrack,mode,time_limit,restart_time_limit,max_id_limit,depth_limit,mem_limit,individual_mem_limit,batch_pass_limit,population_size,device,load_in_gpu,num_classes,channels,data_channels,image_size,input_mode,score,score_selection,batch_size,epochs,patience,n_test_models,test_epochs,gradient_accumulation,hpo_runs,lr,momentum,weight_decay,verbose_setup,verbose_search,verbose_eval,visualise,visualise_scale,vis_interval,data_path,figures_path,results_path,continue_search,load_from)
    run(args)


if __name__ == "__main__":
    # parse the arguments
    args = parse_arguments()
    args = load_config(args)
    args = set_dataset_specific_args(args)
    args.data_channels = args.channels
    args.channels = 16 if args.search_space == "hnasbench201" else args.channels
    args.train_loader = None
    args.val_loader = None
    pprint(vars(args))

    run(args)
