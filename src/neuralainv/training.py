"""Four-stage Adam training for a full-domain shifted-Poisson Green kernel."""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from .inference import save_weights
from .models import (
    AlphaBeta,
    AlphaMSNN,
    AdaptiveBranch,
    FarField,
    Gate,
    SCALES,
    WIDTHS,
    apply_operator,
)
from .operators import fd_matrix
from .sampling import collocation


@dataclass
class TrainingConfig:
    seed: int = 42
    phase_iterations: tuple = (1000, 1000, 1000, 2000)
    n_batches: int = 20
    resample_every: int = 10
    n_far: int = 500
    n_near: int = 1500
    n_boundary: int = 500
    n_aux: int = 500
    batch_scale: int = 4
    coarse_n: int = 32
    alpha_warmup: int = 200
    gate_warmup: int = 1000
    lr_warmup: int = 50
    source_widths: tuple = (0.01, 0.005, 0.004, 0.004)
    base_lrs: tuple = (0.01, 0.001, 0.001, 0.0001)
    min_lrs: tuple = (0.001, 0.0001, 0.0001, 0.00001)
    warmup_lrs: tuple = (0.0001, 0.00001, 0.00001, 0.000001)
    save_every: int = 50

    @classmethod
    def smoke(cls):
        return cls(
            phase_iterations=(3, 3, 3, 3),
            n_batches=2,
            resample_every=2,
            n_far=8,
            n_near=24,
            n_boundary=8,
            n_aux=8,
            batch_scale=1,
            coarse_n=8,
            alpha_warmup=3,
            gate_warmup=3,
            lr_warmup=1,
            save_every=1,
        )


def adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return zeros, zeros, jnp.asarray(0, dtype=jnp.int32)


def adam_update(params, gradients, state, lr):
    m, v, step = state
    step = step + 1
    m = jax.tree.map(lambda a, g: 0.9 * a + 0.1 * g, m, gradients)
    v = jax.tree.map(lambda a, g: 0.999 * a + 0.001 * g**2, v, gradients)
    params = jax.tree.map(
        lambda p, a, b: p
        - lr * (a / (1 - 0.9**step)) / (jnp.sqrt(b / (1 - 0.999**step)) + 1e-7),
        params,
        m,
        v,
    )
    return params, (m, v, step)


def learning_rate(config, phase, outer):
    length = config.phase_iterations[phase]
    value = config.min_lrs[phase] + 0.5 * (
        config.base_lrs[phase] - config.min_lrs[phase]
    ) * (1 + np.cos(np.pi * outer / length))
    if outer < config.lr_warmup:
        value = (
            config.warmup_lrs[phase]
            + (value - config.warmup_lrs[phase]) * outer / config.lr_warmup
        )
    return float(value)


def model_for_phase(phase):
    return AdaptiveBranch(20, 0.002) if phase == 2 else AlphaMSNN(phase + 1)


def phase_weights(phase):
    return ((10.0, 100.0, 100.0, 100.0)[phase] * SCALES[min(phase, 2)] ** 2, 10.0, 10.0)


def loss_terms(model, params, batch, phase):
    xp, yp, xa, ya, xb, yb = batch
    pde = jnp.mean((apply_operator(model, params, xp) - yp) ** 2)
    boundary = jnp.mean((model.apply({"params": params}, xb).reshape(-1) - yb) ** 2)
    auxiliary = (
        jnp.mean((model.apply({"params": params}, xa).reshape(-1) - ya) ** 2)
        if phase < 3
        else jnp.asarray(0.0, dtype=pde.dtype)
    )
    return jnp.stack((pde, auxiliary, boundary))


def make_scan(model, phase):
    weights = jnp.asarray(phase_weights(phase))

    def objective(params, batch):
        terms = loss_terms(model, params, batch, phase)
        return jnp.sum(weights * terms), terms

    def step(carry, batch):
        params, opt, lr = carry
        (loss, terms), gradients = jax.value_and_grad(objective, has_aux=True)(
            params, batch
        )
        params, opt = adam_update(params, gradients, opt, lr)
        return (params, opt, lr), jnp.concatenate((loss[None], terms))

    return jax.jit(lambda carry, batches: jax.lax.scan(step, carry, batches))


def _warmup(model, params, x, target, steps):
    @jax.jit
    def step(params, opt):
        loss, grad = jax.value_and_grad(
            lambda p: jnp.mean((model.apply({"params": p}, x) - target) ** 2)
        )(params)
        params, opt = adam_update(params, grad, opt, jnp.float32(0.001))
        return params, opt, loss

    opt = adam_init(params)
    history = []
    for _ in range(steps):
        params, opt, loss = step(params, opt)
        history.append(float(loss))
    if not np.isfinite(history).all():
        raise FloatingPointError("non-finite initialization loss")
    return params, history


def initialize(config):
    key = jax.random.PRNGKey(config.seed)
    sample = jnp.zeros((1, 4), dtype=jnp.float32)
    modules = [AlphaBeta(), Gate(), FarField()] + [
        AdaptiveBranch(width, eps) for width, eps in zip(WIDTHS, SCALES)
    ]
    initial = []
    for model in modules:
        key, subkey = jax.random.split(key)
        initial.append(model.init(subkey, sample)["params"])
    alpha, gate, far, *tops = initial

    axis = jnp.linspace(1 / 33, 32 / 33, 32)
    grid = jnp.column_stack((jnp.tile(axis, 32), jnp.repeat(axis, 32)))
    x = jnp.column_stack((grid, grid))
    alpha, ah = _warmup(
        AlphaBeta(),
        alpha,
        x,
        jnp.tile(jnp.array([0.0, 1.0, 0.0, 1.0]), (1024, 1)),
        config.alpha_warmup,
    )
    anchors = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    x = jnp.column_stack((jnp.zeros_like(anchors), anchors))
    gate, gh = _warmup(Gate(), gate, x, jnp.eye(5), config.gate_warmup)
    for top in tops:
        for name in alpha:
            top[name] = alpha[name]

    n = config.coarse_n
    a = jnp.asarray(fd_matrix(n).toarray(), dtype=jnp.float32)
    labels = jnp.linalg.inv(a).reshape(-1)
    h = 1 / (n + 1)
    axis = jnp.linspace(h, 1 - h, n)
    grid = jnp.column_stack((jnp.tile(axis, n), jnp.repeat(axis, n)))
    pairs = jnp.column_stack(
        (jnp.repeat(grid, n * n, axis=0), jnp.tile(grid, (n * n, 1)))
    )
    multiplier = (config.n_aux * config.n_batches) // len(pairs) + 1
    pairs, labels = jnp.tile(pairs, (multiplier, 1)), jnp.tile(labels, multiplier)
    key, subkey = jax.random.split(key)
    order = jax.random.permutation(subkey, len(labels))
    pairs, labels = pairs[order], labels[order]
    key, subkey = jax.random.split(key)
    permutation = jax.random.permutation(subkey, len(labels))
    return dict(
        key=key,
        far=far,
        tops=tops,
        gate=gate,
        phase=-1,
        outer=0,
        updates=0,
        aux_pairs=pairs,
        aux_labels=labels,
        permutation=permutation,
        position=0,
        warmup={"alpha": ah, "gate": gh},
        train_seconds=0.0,
        phase_seconds=[0.0, 0.0, 0.0, 0.0],
    )


def start_phase(state, phase):
    key = state["key"]
    if phase < 2:
        key, _ = jax.random.split(key)
        params = {"_models_0": state["far"]}
        params.update(
            {f"_models_{i+1}": p for i, p in enumerate(state["tops"][: phase + 1])}
        )
    elif phase == 2:
        params = state["tops"][2]
    else:
        key, noise_key = jax.random.split(key)
        keys = jax.random.split(noise_key, 5)
        params = {
            "_ddnet": state["gate"],
            "_globals_0": state["far"],
            "_globals_1": state["tops"][0],
            "_globals_2": state["tops"][1],
        }
        for i, expert_key in enumerate(keys):
            leaves, structure = jax.tree_util.tree_flatten(state["tops"][2])
            leaf_keys = jax.random.split(expert_key, len(leaves))
            perturbed = [
                p + jax.random.normal(k, p.shape, p.dtype) * jnp.float32(0.001)
                for p, k in zip(leaves, leaf_keys)
            ]
            params[f"_experts_{i}"] = jax.tree_util.tree_unflatten(structure, perturbed)
        key, _ = jax.random.split(key)
    state.update(
        key=key,
        phase=phase,
        outer=0,
        params=params,
        optimizer=adam_init(params),
        pools=None,
    )


def finish_phase(state):
    phase, params = state["phase"], state["params"]
    if phase < 2:
        state["far"] = params["_models_0"]
        for i in range(phase + 1):
            state["tops"][i] = params[f"_models_{i+1}"]
    elif phase == 2:
        state["tops"][2] = params
        state["key"], _ = jax.random.split(state["key"])


def _resample(state, config, phase, prefix_eval):
    key = state["key"]
    if phase < 3:
        key, pde_key, aux_key = jax.random.split(key, 3)
    else:
        key, pde_key = jax.random.split(key)
    seed = int(jax.random.randint(pde_key, (), 0, 2**31 - 1))
    scale = config.batch_scale * config.n_batches
    xp, yp, xb, yb = collocation(
        seed,
        config.n_far * scale,
        config.n_near * scale,
        config.n_boundary * scale,
        config.source_widths[phase],
    )
    if phase < 3:
        count = len(state["aux_labels"])
        size, pos = config.n_aux * config.n_batches, state["position"]
        permutation = state["permutation"]
        if pos + size <= count:
            indices = permutation[pos : pos + size]
            pos += size
        else:
            head = permutation[pos:]
            permutation = jax.random.permutation(aux_key, count)
            pos = size - len(head)
            indices = jnp.concatenate((head, permutation[:pos]))
        state.update(position=pos, permutation=permutation)
        xa, ya = state["aux_pairs"][indices], state["aux_labels"][indices]
        if phase == 2:
            pp, pa, pb = prefix_eval(xp, xa, xb)
            yp, ya, yb = yp - pp, ya - pa, yb - pb
    else:
        xa = jnp.zeros((config.n_aux * config.n_batches, 4), dtype=jnp.float32)
        ya = jnp.zeros(config.n_aux * config.n_batches, dtype=jnp.float32)
    state.update(key=key, pools=(xp, yp, xa, ya, xb, yb))


def outer_update(state, config, scan, prefix_eval):
    phase, outer = state["phase"], state["outer"]
    if outer % config.resample_every == 0:
        _resample(state, config, phase, prefix_eval)
    batches = []
    key = state["key"]
    for x, y in zip(state["pools"][::2], state["pools"][1::2]):
        key, shuffle_key = jax.random.split(key)
        order = jax.random.permutation(shuffle_key, len(x))
        batches.extend(
            (
                x[order].reshape(config.n_batches, -1, 4),
                y[order].reshape(config.n_batches, -1),
            )
        )
    rate = jnp.float32(learning_rate(config, phase, outer))
    (params, opt, _), metrics = scan(
        (state["params"], state["optimizer"], rate), tuple(batches)
    )
    metrics = np.asarray(metrics)
    if not np.isfinite(metrics).all():
        raise FloatingPointError(f"non-finite loss in stage {phase+1}, outer {outer}")
    state.update(
        params=params,
        optimizer=opt,
        key=key,
        outer=outer + 1,
        updates=state["updates"] + config.n_batches,
    )
    return metrics, float(rate)


def prefix_evaluator(state):
    params = {
        "_models_0": state["far"],
        "_models_1": state["tops"][0],
        "_models_2": state["tops"][1],
    }
    model = AlphaMSNN(2)
    return jax.jit(
        lambda xp, xa, xb: (
            apply_operator(model, params, xp),
            model.apply({"params": params}, xa).reshape(-1),
            model.apply({"params": params}, xb).reshape(-1),
        )
    )


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    host = jax.tree.map(
        lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, state
    )
    with temporary.open("wb") as stream:
        pickle.dump(host, stream, protocol=5)
    temporary.replace(path)


def train(config, directory, *, resume=False, stop_after_updates=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state_file = directory / "resume.pkl"
    config_path = directory / "config.json"
    if resume:
        saved_config = json.loads(config_path.read_text())
        if saved_config != json.loads(json.dumps(asdict(config))):
            raise ValueError("resume configuration differs")
        with state_file.open("rb") as stream:
            state = pickle.load(stream)
        state = jax.tree.map(
            lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, state
        )
    else:
        if config_path.exists():
            raise FileExistsError(f"run already exists: {directory}")
        config_path.write_text(json.dumps(asdict(config), indent=2) + "\n")
        started = time.perf_counter()
        state = initialize(config)
        jax.block_until_ready(state)
        state["initialization_seconds"] = time.perf_counter() - started
        (directory / "initialization.json").write_text(
            json.dumps(
                {"seconds": state["initialization_seconds"], **state["warmup"]},
                indent=2,
            )
            + "\n"
        )
        save_state(state_file, state)

    if stop_after_updates is not None and state["updates"] >= stop_after_updates:
        return state

    history_path = directory / "history.csv"
    if resume and history_path.exists():
        with history_path.open() as stream:
            rows = list(csv.DictReader(stream))
        fields = (
            list(rows[0])
            if rows
            else [
                "updates",
                "stage",
                "loss",
                "pde",
                "aux",
                "boundary",
                "lr",
                "train_seconds",
            ]
        )
        rows = [row for row in rows if int(row["updates"]) <= state["updates"]]
        with history_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)

    with history_path.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(
                [
                    "updates",
                    "stage",
                    "loss",
                    "pde",
                    "aux",
                    "boundary",
                    "lr",
                    "train_seconds",
                ]
            )
        first_phase = max(0, state["phase"])
        for phase in range(first_phase, 4):
            if state["phase"] != phase:
                start_phase(state, phase)
            scan = make_scan(model_for_phase(phase), phase)
            prefix_eval = prefix_evaluator(state) if phase == 2 else None
            while state["outer"] < config.phase_iterations[phase]:
                started = time.perf_counter()
                metrics, rate = outer_update(state, config, scan, prefix_eval)
                elapsed = time.perf_counter() - started
                state["train_seconds"] += elapsed
                state["phase_seconds"][phase] += elapsed
                first_update = state["updates"] - config.n_batches
                for i, values in enumerate(metrics):
                    writer.writerow(
                        [
                            first_update + i + 1,
                            phase + 1,
                            *values.tolist(),
                            rate,
                            state["train_seconds"],
                        ]
                    )
                should_stop = (
                    stop_after_updates is not None
                    and state["updates"] >= stop_after_updates
                )
                if state["outer"] % config.save_every == 0 or should_stop:
                    stream.flush()
                    save_state(state_file, state)
                    status = dict(
                        stage=phase + 1,
                        updates=state["updates"],
                        loss=float(metrics[-1, 0]),
                        train_seconds=state["train_seconds"],
                        phase_seconds=state["phase_seconds"],
                    )
                    (directory / "progress.json").write_text(
                        json.dumps(status, indent=2) + "\n"
                    )
                    print(json.dumps(status), flush=True)
                if should_stop:
                    return state
            finish_phase(state)
            if phase < 3:
                start_phase(state, phase + 1)
            stream.flush()
            save_state(state_file, state)

    save_weights(directory / "weights", state["params"])
    count = sum(np.asarray(x).size for x in jax.tree.leaves(state["params"]))
    report = dict(
        status="training complete",
        updates=state["updates"],
        parameters=int(count),
        train_seconds=state["train_seconds"],
        phase_seconds=state["phase_seconds"],
        initialization_seconds=state["initialization_seconds"],
        timing="training loop includes JIT, resampling and synchronization; excludes checkpoint I/O",
    )
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_enable_x64", False)
    if args.device == "gpu" and jax.default_backend() != "gpu":
        raise RuntimeError("GPU requested but unavailable")
    if args.device == "cpu":
        jax.config.update("jax_platforms", "cpu")
    config = TrainingConfig.smoke() if args.smoke else TrainingConfig()
    train(
        config,
        args.output,
        resume=args.resume,
        stop_after_updates=args.stop_after_updates,
    )


if __name__ == "__main__":
    main()
