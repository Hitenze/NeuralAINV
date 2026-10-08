# NeuralAINV

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![arXiv: 2510.13034](https://img.shields.io/badge/arXiv-2510.13034-b31b1b)](https://arxiv.org/abs/2510.13034)

NeuralAINV learns a Green kernel for `-Δu - 30u = f` on the unit square with
zero Dirichlet boundary conditions. Three examples use the same model:
finite differences, locally refined finite elements, and Green–Schur DD.

Related manuscript:
[Neural Approximate Inverse Preconditioners](https://arxiv.org/abs/2510.13034).
This demo illustrates the workflow; it does not reproduce the manuscript's experiments.

## Install

Requires Python 3.12 or newer.

```bash
git clone https://github.com/Hitenze/NeuralAINV.git
cd NeuralAINV
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[examples]"
```

The default installation uses CPU JAX. For the tested Linux/WSL CUDA 13
environment, install `requirements-cuda13.txt` before installing this package.

## Run with pretrained weights

Download the model from [Releases](https://github.com/Hitenze/NeuralAINV/releases)
and use the source code from the same release tag. These commands require a
published `neuralainv-shifted-poisson.zip` asset:

```bash
mkdir -p weights
curl --fail --location \
  https://github.com/Hitenze/NeuralAINV/releases/latest/download/neuralainv-shifted-poisson.zip \
  --output weights/neuralainv-shifted-poisson.zip
python -m zipfile -e weights/neuralainv-shifted-poisson.zip weights
```

Run the three examples from the repository root:

```bash
python examples/dense.py --weights weights/shifted_poisson --output results/dense
python examples/fem.py --weights weights/shifted_poisson --output results/fem
python examples/dd.py --weights weights/shifted_poisson --output results/dd
```

Use a new or empty output directory for each run. Each example saves
per-solve records and a summary in its output directory.

## Train

```bash
python examples/train.py --output runs/shifted_poisson --device gpu
```

For a short CPU test:

```bash
python examples/train.py --smoke --output runs/smoke --device cpu
```

Add `--resume` to the original command to resume an interrupted run.
To use your trained model, replace `weights/shifted_poisson` in the example
commands with `runs/shifted_poisson/weights`.

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
