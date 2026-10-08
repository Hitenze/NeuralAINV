"""Checks for adaptive scaling, stage composition, and independent experts."""

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralainv.models import (
    AdaptiveBranch,
    AlphaBeta,
    AlphaMSNN,
    FarField,
    Gate,
    SCALES,
    WIDTHS,
    apply_operator,
    ordered_exponents,
)
from neuralainv.inference import GreenKernel, load_weights, save_weights
from neuralainv.training import model_for_phase, start_phase


def mlp(params, value, first):
    for i in range(first, first + 3):
        layer = params[f"Dense_{i}"]
        value = jnp.tanh(value @ layer["kernel"] + layer["bias"])
    layer = params[f"Dense_{first + 3}"]
    return value @ layer["kernel"] + layer["bias"]


def test_adaptive_branch_matches_scaling_formula():
    pairs = jnp.array([[0.011, 0.012, 0.01, 0.01], [0.02, 0.015, 0.01, 0.01]])
    branch = AdaptiveBranch(10, 0.01)
    params = branch.init(jax.random.PRNGKey(42), pairs)["params"]
    source = pairs * jnp.array([0, 0, 1, 1])
    s = jax.nn.sigmoid(mlp(params, source, 0))
    alpha1 = 2 * s[:, 0] - 1
    alpha2 = alpha1 + s[:, 1] * (1 - alpha1)
    beta2 = s[:, 2]
    z = jnp.column_stack((pairs, pairs[:, :2] - pairs[:, 2:]))
    expected = 0.01 ** alpha1[:, None] * mlp(params, z / 0.01, 4) + 0.01 ** alpha2[
        :, None
    ] * mlp(params, z * 0.01 ** (-beta2[:, None]), 8)
    np.testing.assert_allclose(
        branch.apply({"params": params}, pairs), expected, rtol=2e-6
    )

    grad = jax.grad(lambda p: jnp.sum(branch.apply({"params": p}, pairs) ** 2))(params)
    for i in range(4):
        values = jax.tree.leaves(grad[f"Dense_{i}"])
        assert all(np.isfinite(v).all() for v in values)
        assert sum(float(jnp.linalg.norm(v)) for v in values) > 0


def test_exponents_and_gate_depend_on_source():
    pairs = jnp.array(
        [[0.2, 0.3, 0.4, 0.5], [0.8, 0.9, 0.4, 0.5], [0.2, 0.3, 0.7, 0.1]]
    )
    for model in (AlphaBeta(), Gate()):
        p = model.init(jax.random.PRNGKey(3), pairs)["params"]
        out = np.asarray(model.apply({"params": p}, pairs))
        np.testing.assert_array_equal(out[0], out[1])
        assert not np.allclose(out[0], out[2])
        if isinstance(model, Gate):
            assert np.all(out > 0)
            np.testing.assert_allclose(out.sum(axis=1), 1, atol=1e-7)
    logits = jnp.array([[-30.0, 30.0, -30.0], [0.0, 0.0, 0.0], [30.0, -30.0, 30.0]])
    a1, b1, a2, b2 = np.asarray(ordered_exponents(logits)).T
    assert np.all((-1 <= a1) & (a1 <= a2) & (a2 <= 1))
    np.testing.assert_array_equal(b1, np.ones(3))
    assert np.all((0 <= b2) & (b2 <= 1))


@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_multiscale_sum_and_roundtrip(stage, tmp_path):
    pairs = jnp.array([[0.2, 0.3, 0.22, 0.31], [0.6, 0.8, 0.5, 0.7]])
    model = AlphaMSNN(stage)
    params = model.init(jax.random.PRNGKey(42), pairs)["params"]
    prefix = "_globals" if stage == 4 else "_models"
    expected = FarField().apply({"params": params[f"{prefix}_0"]}, pairs)
    for i in range(2 if stage == 4 else stage):
        expected = expected + AdaptiveBranch(WIDTHS[i], SCALES[i]).apply(
            {"params": params[f"{prefix}_{i+1}"]}, pairs
        )
    if stage == 4:
        assert set(params) == {"_globals_0", "_globals_1", "_globals_2", "_ddnet"} | {
            f"_experts_{i}" for i in range(5)
        }
        gate = Gate().apply({"params": params["_ddnet"]}, pairs)
        outputs = [
            AdaptiveBranch(20, 0.002).apply({"params": params[f"_experts_{i}"]}, pairs)
            for i in range(5)
        ]
        expected = expected + jnp.sum(
            gate * jnp.concatenate(outputs, axis=1), axis=1, keepdims=True
        )
        changed = deepcopy(params)
        changed["_experts_0"]["Dense_7"]["bias"] = (
            changed["_experts_0"]["Dense_7"]["bias"] + 0.1
        )
        old_exponent_params = {
            f"Dense_{i}": params["_experts_0"][f"Dense_{i}"] for i in range(4)
        }
        alpha1 = AlphaBeta().apply({"params": old_exponent_params}, pairs)[:, 0]
        difference = model.apply({"params": changed}, pairs) - model.apply(
            {"params": params}, pairs
        )
        np.testing.assert_allclose(
            difference[:, 0], gate[:, 0] * 0.1 * 0.002**alpha1, rtol=2e-4, atol=2e-6
        )
    np.testing.assert_allclose(
        model.apply({"params": params}, pairs), expected, rtol=2e-6, atol=2e-6
    )
    save_weights(tmp_path, params, stage=stage)
    np.testing.assert_allclose(
        load_weights(tmp_path).pairs(pairs, chunk_size=2),
        np.asarray(expected).ravel(),
        rtol=2e-6,
        atol=2e-6,
    )


def test_phase_three_freezing_and_phase_four_initialization():
    params = AlphaMSNN(3).init(jax.random.PRNGKey(8), jnp.zeros((1, 4)))["params"]
    gate = Gate().init(jax.random.PRNGKey(9), jnp.zeros((1, 4)))["params"]
    state = dict(
        key=jax.random.PRNGKey(10),
        far=params["_models_0"],
        tops=[params[f"_models_{i}"] for i in (1, 2, 3)],
        gate=gate,
    )
    original = deepcopy(state)
    start_phase(state, 2)
    assert isinstance(model_for_phase(2), AdaptiveBranch)
    for a, b in zip(
        jax.tree.leaves(state["params"]), jax.tree.leaves(original["tops"][2])
    ):
        np.testing.assert_array_equal(a, b)
    start_phase(state, 3)
    for i, old in enumerate([original["far"], *original["tops"][:2]]):
        for a, b in zip(
            jax.tree.leaves(state["params"][f"_globals_{i}"]), jax.tree.leaves(old)
        ):
            np.testing.assert_array_equal(a, b)
    for i in range(5):
        expert = jax.tree.leaves(state["params"][f"_experts_{i}"])
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(expert, jax.tree.leaves(original["tops"][2]))
        )
        for j in range(i):
            other = jax.tree.leaves(state["params"][f"_experts_{j}"])
            assert any(not np.array_equal(a, b) for a, b in zip(expert, other))
    for a, b in zip(jax.tree.leaves(state["tops"]), jax.tree.leaves(original["tops"])):
        np.testing.assert_array_equal(a, b)


def test_operator_differentiates_targets_only():
    class Polynomial:
        def apply(self, variables, pairs):
            x1, x2, y1, y2 = pairs.T
            return ((x1**2 + 3 * x2**2) * (1 + y1 + y2**2))[:, None]

    model = Polynomial()
    pairs = jnp.array([[0.2, 0.3, 0.4, 0.5], [0.7, 0.8, 0.1, 0.9]])
    expected = (
        -8 * (1 + pairs[:, 2] + pairs[:, 3] ** 2) - 30 * model.apply({}, pairs)[:, 0]
    )
    np.testing.assert_allclose(apply_operator(model, {}, pairs), expected, rtol=1e-6)


@pytest.mark.parametrize("size", [0, -1])
def test_grid_rejects_nonpositive_chunk_size(size):
    p = AlphaMSNN(1).init(jax.random.PRNGKey(0), jnp.zeros((1, 4)))["params"]
    with pytest.raises(ValueError, match="chunk_size"):
        GreenKernel(p, stage=1).grid(
            np.zeros((1, 2)), np.zeros((1, 2)), chunk_size=size
        )
