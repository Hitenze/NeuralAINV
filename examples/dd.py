"""C-ILU and Green-Schur interface actions with shared interior factors."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import time
from scipy.sparse import save_npz
from neuralainv.dd import BlockElimination
from neuralainv.operators import fd_matrix, grid_points, right_hand_sides
from _common import arguments, prepare, solve_methods, summarize


def main():
    args = arguments(__doc__)
    kernel, out = prepare(args, "dd")
    rows, metadata = [], []
    for n in (15, 31, 63):
        A = fd_matrix(n)
        started = time.perf_counter()
        block = BlockElimination(A, n)
        interior_setup = time.perf_counter() - started
        started = time.perf_counter()
        coarse = block.c_ilu()
        c_setup = time.perf_counter() - started
        started = time.perf_counter()
        points = grid_points(n)[block.gamma]
        G = kernel.grid(points, points)
        green_setup = time.perf_counter() - started
        methods = [
            ("c_ilu", block.action(coarse.solve), interior_setup + c_setup),
            (
                "green_schur",
                block.action(lambda r: G @ r),
                interior_setup + green_setup,
            ),
        ]
        rows.extend(solve_methods(A, right_hand_sides(n * n), methods, out, f"n{n}"))
        save_npz(out / f"n{n}_matrix.npz", A)
        metadata.append(
            dict(
                n=n,
                unknowns=n * n,
                separator=len(block.gamma),
                interior_sizes=[len(g) for g in block.groups],
                interior_setup_seconds=interior_setup,
                c_ilu_setup_seconds=c_setup,
                green_materialize_seconds=green_setup,
                green_bytes=G.nbytes,
                interior_factor_nonzeros=[f.L.nnz + f.U.nnz for f in block.factors],
                c_factor_nonzeros=coarse.L.nnz + coarse.U.nnz,
                drop_tol=1e-6,
                fill_factor=10,
                ordering="COLAMD",
                drop_rule="basic,area",
                diagonal_pivot_threshold=1.0,
                inner_schur_iterations=0,
            )
        )
    summarize(rows, out, metadata)


if __name__ == "__main__":
    main()
