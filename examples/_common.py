"""Shared data and reporting for the Green-kernel examples."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from neuralainv import load_weights
from neuralainv.krylov import fgmres


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def prepare(args, example):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")

    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    kernel = load_weights(args.weights)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "example": example,
        "operator": "-Delta - 30",
        "weights": str(Path(args.weights).resolve()),
        "restart": 50,
        "max_iterations": 500,
        "relative_tolerance": 1e-8,
        "solver": "flexible right-GMRES",
        "initial_guess": "zero",
        "solver_dtype": "float64",
        "kernel_dtype": "float32",
        "kernel_matmul_precision": "highest",
        "RHS": "5 uniform(-1,1) seed 20260806, 5 standard-normal seed 20260807; unit norm",
    }
    write_json(out / "config.json", config)
    return kernel, out


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def solve_methods(matrix, rhs, methods, out, case):
    rows = []
    for name, apply, setup_seconds in methods:
        solutions = []
        for index, b in enumerate(rhs.T):
            residual_history = []
            started = time.perf_counter()
            x, info = fgmres(
                matrix,
                b,
                preconditioner=apply,
                relative_tolerance=1e-8,
                maximum_iterations=500,
                restart=50,
                callback=residual_history.append,
            )
            seconds = time.perf_counter() - started
            true_residual = float(np.linalg.norm(matrix @ x - b) / np.linalg.norm(b))
            row = dict(
                case=case,
                method=name,
                rhs=index,
                iterations=len(residual_history),
                accepted=bool(np.isfinite(true_residual) and true_residual <= 1e-8),
                true_residual=true_residual if np.isfinite(true_residual) else None,
                setup_seconds=setup_seconds,
                solve_seconds=seconds,
                solver_info=int(info),
            )
            with (out / "rows.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            solutions.append(x)
        np.savez_compressed(
            out / f"{case}_{name}_solutions.npz", x=np.column_stack(solutions)
        )
    np.savez_compressed(out / f"{case}_rhs.npz", rhs=rhs)
    return rows


def summarize(rows, out, metadata):
    groups = []
    for case, method in sorted({(r["case"], r["method"]) for r in rows}):
        selected = [r for r in rows if (r["case"], r["method"]) == (case, method)]
        groups.append(
            dict(
                case=case,
                method=method,
                total=len(selected),
                accepted=sum(r["accepted"] for r in selected),
                median_iterations=float(np.median([r["iterations"] for r in selected])),
                penalized_median=float(
                    np.median(
                        [r["iterations"] if r["accepted"] else 501 for r in selected]
                    )
                ),
                max_true_residual=max(
                    (
                        r["true_residual"]
                        for r in selected
                        if r["true_residual"] is not None
                    ),
                    default=None,
                ),
                median_setup_seconds=float(
                    np.median([r["setup_seconds"] for r in selected])
                ),
                median_solve_seconds=float(
                    np.median([r["solve_seconds"] for r in selected])
                ),
            )
        )
    report = dict(
        rows=len(rows),
        accepted=sum(r["accepted"] for r in rows),
        groups=groups,
        metadata=metadata,
    )
    write_json(out / "summary.json", report)
    write_json(out / "failures.json", [r for r in rows if not r["accepted"]])
    lines = [
        "| Case | Method | Accepted | Median iterations | Setup (s) | Solve (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in groups:
        lines.append(
            f"| {r['case']} | {r['method']} | {r['accepted']}/{r['total']} | "
            f"{r['median_iterations']:g} | {r['median_setup_seconds']:.6f} | "
            f"{r['median_solve_seconds']:.6f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2), flush=True)
