from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from network import Network
from trainers import Trainer


def random_evaluation_fn(node, args, train_loader, val_loader):
    return torch.rand(1).item()

def evaluation_fn(node, args, train_loader, val_loader):
    backbone = node.operation.build(node)
    model = Network(
        backbone,
        node.output_params["shape"],
        args.num_classes,
        config=vars(args),
    )
    # train and evaluate sampled network
    try:
        trainer = Trainer(
            model,
            device=args.device,
            train_dataloader=train_loader,
            valid_dataloader=val_loader,
            test_dataloader=None,
            config=vars(args),
            log=args.verbose_eval,
        )
        best = trainer.train()
        return best["val_score"]
    except RuntimeError as e:
        if args.verbose_eval: print(f"RuntimeError: {e}")
        if args.verbose_eval: print(f"Returning 0.0")
        return 0.0
