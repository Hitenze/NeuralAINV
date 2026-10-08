"""Four-interior block elimination with an interchangeable interface action."""

import numpy as np
from scipy.sparse.linalg import spilu


def partition(n):
    if n < 3 or n % 2 != 1:
        raise ValueError("an odd number of interior grid points is required")
    j, i = np.indices((n, n))
    middle = n // 2
    groups = [
        np.flatnonzero((i < middle) & (j < middle)),
        np.flatnonzero((i > middle) & (j < middle)),
        np.flatnonzero((i < middle) & (j > middle)),
        np.flatnonzero((i > middle) & (j > middle)),
    ]
    gamma = np.flatnonzero((i == middle) | (j == middle))
    return groups, gamma


class BlockElimination:
    """Approximate interior elimination; no inner Schur iteration."""

    def __init__(self, matrix, n, *, drop_tol=1e-6, fill_factor=10):
        self.groups, self.gamma = partition(n)
        self.interior = np.concatenate(self.groups)
        self.offsets = np.cumsum([0] + [len(g) for g in self.groups])
        self.factors = [
            spilu(
                matrix[g][:, g].tocsc(),
                drop_tol=drop_tol,
                fill_factor=fill_factor,
                permc_spec="COLAMD",
                diag_pivot_thresh=1.0,
                drop_rule="basic,area",
            )
            for g in self.groups
        ]
        self.E = matrix[self.gamma][:, self.interior].tocsr()
        self.F = matrix[self.interior][:, self.gamma].tocsr()
        self.C = matrix[self.gamma][:, self.gamma].tocsc()
        self.size = matrix.shape[0]

    def interior_solve(self, rhs):
        return np.concatenate(
            [
                fac.solve(rhs[a:b])
                for fac, a, b in zip(self.factors, self.offsets[:-1], self.offsets[1:])
            ]
        )

    def action(self, terminal):
        def apply(rhs):
            z = self.interior_solve(rhs[self.interior])
            boundary = terminal(rhs[self.gamma] - self.E @ z)
            interior = z - self.interior_solve(self.F @ boundary)
            out = np.empty(self.size, dtype=np.float64)
            out[self.interior], out[self.gamma] = interior, boundary
            return out

        return apply

    def c_ilu(self, *, drop_tol=1e-6, fill_factor=10):
        return spilu(
            self.C,
            drop_tol=drop_tol,
            fill_factor=fill_factor,
            permc_spec="COLAMD",
            diag_pivot_thresh=1.0,
            drop_rule="basic,area",
        )
