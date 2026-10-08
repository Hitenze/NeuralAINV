"""Finite-difference matrices for the Dirichlet shifted Laplacian."""

import numpy as np
from scipy.sparse import diags, eye, kron


def fd_matrix(n):
    """Unit-scaled five-point matrix: T kron I + I kron T - 30 h^2 I."""
    if n < 1:
        raise ValueError("n must be positive")
    t = diags((-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)), (-1, 0, 1))
    unit = eye(n)
    return (kron(t, unit) + kron(unit, t) - 30 / (n + 1) ** 2 * eye(n * n)).tocsr()


def grid_points(n):
    axis = np.linspace(0.0, 1.0, n + 2)[1:-1]
    return np.column_stack((np.tile(axis, n), np.repeat(axis, n)))


def right_hand_sides(size):
    uniform = np.random.default_rng(20260806).uniform(-1.0, 1.0, (size, 5))
    normal = np.random.default_rng(20260807).normal(size=(size, 5))
    rhs = np.column_stack((uniform, normal))
    return rhs / np.linalg.norm(rhs, axis=0)
