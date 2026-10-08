"""Serialization and chunked continuous-kernel evaluation."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from .models import AlphaMSNN


def save_weights(directory, params, *, stage=4):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.json").write_text(
        json.dumps(
            {
                "architecture": "alphaMSNN",
                "stage": stage,
                "operator": "-Delta - 30",
                "domain": [[0.0, 1.0], [0.0, 1.0]],
                "boundary": "homogeneous Dirichlet",
                "dtype": "float32",
                "inputs": ["x1", "x2", "y1", "y2"],
                "output": "continuous Green kernel",
            },
            indent=2,
        )
        + "\n"
    )
    data = jax.tree.map(lambda x: np.asarray(x), params)
    (directory / "weights.msgpack").write_bytes(serialization.msgpack_serialize(data))


def load_weights(directory):
    directory = Path(directory)
    config = json.loads((directory / "model.json").read_text())
    if config["operator"] != "-Delta - 30":
        raise ValueError("weights do not describe the shifted Poisson operator")
    params = serialization.msgpack_restore((directory / "weights.msgpack").read_bytes())
    return GreenKernel(params, stage=config["stage"])


class GreenKernel:
    """Continuous G(x,y); no finite-difference or finite-element weighting."""

    def __init__(self, params, *, stage=4):
        self.params = jax.tree.map(jnp.asarray, params)
        self.model = AlphaMSNN(stage)

        def evaluate(pairs):
            with jax.default_matmul_precision("highest"):
                return self.model.apply({"params": self.params}, pairs).reshape(-1)

        self._evaluate = jax.jit(evaluate)

    def pairs(self, pairs, *, chunk_size=8192):
        pairs = np.asarray(pairs, dtype=np.float32)
        if pairs.ndim != 2 or pairs.shape[1] != 4:
            raise ValueError("pairs must have shape (n, 4)")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        out = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), chunk_size):
            block = pairs[start : start + chunk_size]
            count = len(block)
            block = np.pad(block, ((0, chunk_size - count), (0, 0)), mode="edge")
            out[start : start + count] = np.asarray(self._evaluate(jnp.asarray(block)))[
                :count
            ]
        return out

    def grid(self, x, y, *, chunk_size=8192):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != 2 or y.shape[1] != 2:
            raise ValueError("x and y must have shape (n, 2)")
        out = np.empty((len(x), len(y)), dtype=np.float64)
        flat = out.ravel()
        for start in range(0, flat.size, chunk_size):
            indices = np.arange(start, min(start + chunk_size, flat.size))
            pairs = np.column_stack((x[indices // len(y)], y[indices % len(y)]))
            flat[start : start + len(indices)] = self.pairs(
                pairs, chunk_size=chunk_size
            )
        return out
