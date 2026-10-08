"""Mass-projected Green inverse on uniformly and locally refined P1 meshes."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import time
import numpy as np
from scipy.sparse import save_npz
from neuralainv.fem import assemble, project_kernel, square_mesh
from neuralainv.operators import right_hand_sides
from _common import arguments, prepare, solve_methods, summarize, write_json


def main():
    args = arguments(__doc__)
    kernel, out = prepare(args, "fem")
    rows, metadata = [], []
    meshes = []
    quadrature_reference = None
    for level in (0, 1, 2):
        mesh = square_mesh(level)
        meshes.append(mesh)
        A, M, q, D, free, error = assemble(mesh)
        started = time.perf_counter()
        G = kernel.grid(q, q)
        materialize = time.perf_counter() - started
        started = time.perf_counter()
        P = project_kernel(G, M, D)
        projection = time.perf_counter() - started
        del G
        if level == 0:
            quadrature_reference = P.copy()
        started = time.perf_counter()
        diagonal = A.diagonal()
        jacobi_setup = time.perf_counter() - started
        methods = [
            ("jacobi", lambda r: r / diagonal, jacobi_setup),
            ("green_mass", lambda r: P @ r, materialize + projection),
        ]
        rows.extend(
            solve_methods(A, right_hand_sides(len(free)), methods, out, f"level{level}")
        )
        save_npz(out / f"level{level}_matrix.npz", A)
        save_npz(out / f"level{level}_mass.npz", M)
        np.savez_compressed(
            out / f"level{level}_mesh.npz", points=mesh.p, cells=mesh.t, free=free
        )
        metadata.append(
            dict(
                level=level,
                unknowns=len(free),
                triangles=mesh.nelements,
                quadrature_order=2,
                quadrature_points=len(q),
                mass_identity_error=error,
                materialize_seconds=materialize,
                projection_seconds=projection,
                inverse_action_bytes=P.nbytes,
            )
        )
    mesh = meshes[0]
    A, M, q, D, free, error = assemble(mesh, quadrature_order=4)
    G = kernel.grid(q, q)
    alternative = project_kernel(G, M, D)
    relative = float(
        np.linalg.norm(alternative - quadrature_reference) / np.linalg.norm(alternative)
    )
    write_json(
        out / "quadrature_check.json",
        {
            "mesh": "uniform",
            "orders": [2, 4],
            "relative_action_matrix_difference": relative,
            "definition": "norm(P_order4-P_order2)/norm(P_order4)",
            "used_in_main_comparison": 2,
            "mass_identity_error_order4": error,
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, mesh, title in zip(
        axes, meshes, ("Uniform", "Mild refinement", "Moderate refinement")
    ):
        ax.triplot(mesh.p[0], mesh.p[1], mesh.t.T, lw=0.4, color="#246078")
        ax.set(
            aspect="equal",
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="x",
            ylabel="y",
            title=title,
        )
    fig.savefig(out / "meshes.svg")
    fig.savefig(out / "meshes.png", dpi=180)
    plt.close(fig)
    summarize(rows, out, metadata)


if __name__ == "__main__":
    main()
