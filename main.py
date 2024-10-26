from tqdm import tqdm

import torch

from search_strategies.mcts import MCTS
from pcfg import PCFG
from grammars.einspace import grammar, quick_grammar


def evaluation_fn(node, epochs=1, batch_size=64, device="cuda:1", verbose=False):
    """
    Function that evaluates a node on MNIST,
    training for 1 epoch with SGD.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms

    from einspace.network import Network


    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Get model
    backbone = node.operation.build(node)
    model = Network(
        backbone,
        node.output_params["shape"], 10, {
            "search_space": "einspace",
            "dataset": "mnist",
        }
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train for 1 epoch
    for epoch in range(epochs):
        for data, targets in tqdm(trainloader, disable=not verbose):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    # evaluate on test set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, targets in testloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    if verbose: print(f"Test accuracy: {100 * correct / total}%")
    return correct / total


print_after = 1000

# set seed
torch.manual_seed(2)

mcts = MCTS(
    pcfg=PCFG(grammar),
    evaluation_fn=evaluation_fn, # lambda node, verbose: torch.rand(1).item(),
    input_params={
        "shape": torch.Size([1, 1, 28, 28]),
        "other_shape": None,
        "mode": "im",
        "other_mode": None,
        "branching_factor": 1,
        "last_im_shape": None,
    },
    seed=0,
    backtrack=True,
    time_limit=60,
    verbose=False,
    verbose_after_iteration=print_after,
    visualise=False,
    visualise_after_iteration=print_after,
    visualise_scale=0.8,
    save_fig_path="figures_grammar/search_tree",
    save_results_path="results_grammar",
)

mcts.learn(rollouts=1000)
