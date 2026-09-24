# Recursive Constrained Smith–Waterman crossover for neural architecture search

> [!IMPORTANT]
> **To use RCSWX in your own projects, see [rcswx](https://github.com/flxai/rcswx),
> the standalone Rust-backed implementation with reduced runtime and memory use.**
>
> This repository is dedicated to reproducing the paper’s results. It contains
> the original implementation, experiment configurations, benchmarks, and
> visualization code.

Research code accompanying
[**Evolutionary Architecture Search through Grammar-Based Sequence Alignment**](https://openreview.net/forum?id=YK8HlgLJd7).

## How RCSWX works

RCSWX compares and recombines neural network architectures represented as
**grammar derivation trees**. These trees record the rules used to construct
a network.

Recursive alignment identifies differences between two parent architectures
and computes a structural edit distance. Crossover applies a selection of
these edits to produce offspring architectures.

The experiments use
[**einspace**](https://arxiv.org/abs/2405.20838), a grammar-based neural
architecture search space. RCSWX provides the alignment and crossover
method; einspace defines the architectures explored during search.

## Reproducing the results

The experiment artifacts (`benchmark.pkl` and `models.pkl`) are available on
request. Please contact the authors using the email addresses listed in the
[publication](https://openreview.net/forum?id=YK8HlgLJd7).

Start with the [reproduction guide](REPRODUCIBILITY.md) for environment setup,
dataset and artifact preparation, experiment commands, and figure generation.

| Experiment or analysis | Entry point |
| --- | --- |
| Inspect alignment costs and edit paths | [Cost-matrix notebook](notebooks/cswx-cost-matrix.ipynb) |
| Benchmark crossover runtime | [`benchmark.py`](benchmark.py) |
| Run evolutionary searches and ablations | [Experiment configurations](configs/einspace/) and [`scripts/run.sh`](scripts/run.sh) |
| Visualize search results | [`exploration.py`](exploration.py) |

The guide separates individual runs from full experiment sweeps and includes
the cluster-specific submission instructions. It also maps experiments and
outputs to the corresponding paper figures.

Use the original implementation and the paper’s configurations when
reproducing its results. Evaluating the optimized implementation is a separate
comparison, not a replacement for the original experiments.

## Getting started

Clone this repository:

```sh
git clone https://github.com/flxai/rcswx-paper.git
cd rcswx-paper
```

Follow the [reproduction guide](REPRODUCIBILITY.md) before launching experiments.
The environment files are:

| File | Purpose |
| --- | --- |
| [`requirements.txt`](requirements.txt) | Original Python dependency pins |
| [`shell.nix`](shell.nix) | Nix development environment |
| [`scripts/juwels_modules.txt`](scripts/juwels_modules.txt) | JUWELS environment modules |

The [experiment launcher](scripts/run.sh) expects a Python virtual environment
named `venv` in the repository root.

Prepare the datasets and experiment artifacts required by the selected
workflow before running it. Full search sweeps involve training many
architectures; begin with an individual configuration to check the setup.

## Citation

Please cite the paper when using RCSWX or its experimental results:

```bibtex
@inproceedings{gomez2026evolutionary,
  author    = {Gómez Martín, Adri and Möller, Felix and McDonagh, Steven and
               Abella, Monica and Desco, Manuel and Crowley, Elliot J. and
               Klein, Aaron and Ericsson, Linus},
  title     = {Evolutionary Architecture Search through Grammar-Based Sequence Alignment},
  booktitle = {International Conference on Automated Machine Learning (AutoML)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=YK8HlgLJd7}
}
```

The experiments build on **einspace**, introduced by Ericsson et al. in
[**einspace: Searching for Neural Architectures from Fundamental Operations**](https://arxiv.org/abs/2405.20838).
Please also cite that work when using its search space.

## License

[MIT](LICENSE).
