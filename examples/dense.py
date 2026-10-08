"""Dense Green preconditioning on three Cartesian grids."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import time
from scipy.sparse import save_npz
from neuralainv.operators import fd_matrix, grid_points, right_hand_sides
from _common import arguments, prepare, solve_methods, summarize


def main():
    args = arguments(__doc__)
    kernel, out = prepare(args, "dense")
    rows, metadata = [], []
    for n in (15, 31, 63):
        A = fd_matrix(n)
        points = grid_points(n)
        started = time.perf_counter()
        G = kernel.grid(points, points)
        materialize = time.perf_counter() - started
        save_npz(out / f"n{n}_matrix.npz", A)
        rhs = right_hand_sides(n * n)
        rows.extend(
            solve_methods(
                A,
                rhs,
                [("none", None, 0.0), ("dense_green", lambda r: G @ r, materialize)],
                out,
                f"n{n}",
            )
        )
        metadata.append(
            dict(
                n=n,
                unknowns=n * n,
                green_bytes=G.nbytes,
                materialize_seconds=materialize,
                timing="first measured materialization; includes query, synchronization and conversion",
            )
        )
    summarize(rows, out, metadata)


if __name__ == "__main__":
    main()
