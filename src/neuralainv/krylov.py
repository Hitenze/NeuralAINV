"""Real-arithmetic flexible GMRES with right preconditioning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def fgmres(
    matrix: Any,
    rhs: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    preconditioner: Callable[[np.ndarray], np.ndarray] | Any | None = None,
    relative_tolerance: float = 1.0e-6,
    maximum_iterations: int = 1000,
    restart: int = 20,
    absolute_tolerance: float | None = None,
    callback: Callable[[float], None] | None = None,
) -> tuple[np.ndarray, int]:
    """Solve with flexible right preconditioning.

    Return the iterate and zero on convergence, or the number of attempted
    iterations on failure. The callback receives one residual norm per
    attempted iteration, normally estimated from the Arnoldi relation.
    Both ``restart`` and ``maximum_iterations`` must be positive.
    """
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    if restart <= 0:
        raise ValueError("restart must be positive")
    n = len(rhs)
    if hasattr(matrix, "matvec"):
        matvec = matrix.matvec
    elif hasattr(matrix, "__matmul__"):
        matvec = lambda vector: matrix @ vector
    else:
        matvec = matrix

    if preconditioner is None:
        solve = lambda vector: vector.copy()
    elif hasattr(preconditioner, "solve"):
        solve = preconditioner.solve
    elif hasattr(preconditioner, "matvec"):
        solve = preconditioner.matvec
    elif callable(preconditioner):
        solve = preconditioner
    else:
        raise TypeError("preconditioner must be callable or expose solve/matvec")

    x = np.zeros(n, dtype=rhs.dtype) if x0 is None else x0.copy()
    residual = rhs - matvec(x)
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        rhs_norm = 1.0
    if absolute_tolerance is None:
        absolute_tolerance = relative_tolerance * rhs_norm

    total_iterations = 0
    while total_iterations < maximum_iterations:
        residual_norm = float(np.linalg.norm(residual))
        if (
            residual_norm <= absolute_tolerance
            or residual_norm <= relative_tolerance * rhs_norm
        ):
            return x, 0

        beta = residual_norm
        cycle = min(restart, maximum_iterations - total_iterations)
        basis = np.zeros((n, cycle + 1), dtype=rhs.dtype)
        preconditioned = np.zeros((n, cycle), dtype=rhs.dtype)
        hessenberg = np.zeros((cycle + 1, cycle), dtype=rhs.dtype)
        basis[:, 0] = residual / beta
        cosine = np.zeros(cycle, dtype=rhs.dtype)
        sine = np.zeros(cycle, dtype=rhs.dtype)
        rotated_rhs = np.zeros(cycle + 1, dtype=rhs.dtype)
        rotated_rhs[0] = beta

        inner = 0
        for inner in range(cycle):
            preconditioned[:, inner] = solve(basis[:, inner])
            work = matvec(preconditioned[:, inner])
            work_norm = float(np.linalg.norm(work))
            for basis_index in range(inner + 1):
                hessenberg[basis_index, inner] = np.dot(work, basis[:, basis_index])
                work = work - hessenberg[basis_index, inner] * basis[:, basis_index]
            hessenberg[inner + 1, inner] = np.linalg.norm(work)
            happy_breakdown = (
                hessenberg[inner + 1, inner] <= np.finfo(rhs.dtype).eps * work_norm
            )
            if not happy_breakdown:
                basis[:, inner + 1] = work / hessenberg[inner + 1, inner]

            for prior in range(inner):
                temporary = (
                    cosine[prior] * hessenberg[prior, inner]
                    + sine[prior] * hessenberg[prior + 1, inner]
                )
                hessenberg[prior + 1, inner] = (
                    -sine[prior] * hessenberg[prior, inner]
                    + cosine[prior] * hessenberg[prior + 1, inner]
                )
                hessenberg[prior, inner] = temporary

            denominator = np.hypot(
                hessenberg[inner, inner], hessenberg[inner + 1, inner]
            )
            if denominator == 0.0:
                if inner:
                    coefficients = _back_substitute(hessenberg, rotated_rhs, inner)
                    x = x + preconditioned[:, :inner] @ coefficients
                residual_norm = float(np.linalg.norm(rhs - matvec(x)))
                total_iterations += 1
                if callback is not None:
                    callback(residual_norm)
                if (
                    residual_norm <= absolute_tolerance
                    or residual_norm <= relative_tolerance * rhs_norm
                ):
                    return x, 0
                return x, total_iterations
            cosine[inner] = hessenberg[inner, inner] / denominator
            sine[inner] = hessenberg[inner + 1, inner] / denominator
            hessenberg[inner, inner] = (
                cosine[inner] * hessenberg[inner, inner]
                + sine[inner] * hessenberg[inner + 1, inner]
            )
            hessenberg[inner + 1, inner] = 0.0
            rotated_rhs[inner + 1] = -sine[inner] * rotated_rhs[inner]
            rotated_rhs[inner] = cosine[inner] * rotated_rhs[inner]
            total_iterations += 1
            if callback is not None:
                callback(abs(float(rotated_rhs[inner + 1])))

            if (
                abs(rotated_rhs[inner + 1]) <= absolute_tolerance
                or abs(rotated_rhs[inner + 1]) <= relative_tolerance * rhs_norm
            ):
                coefficients = _back_substitute(hessenberg, rotated_rhs, inner + 1)
                x = x + preconditioned[:, : inner + 1] @ coefficients
                return x, 0

            if happy_breakdown:
                # The rotated system is triangular; check the explicit residual
                # after updating the solution.
                break

        used = inner + 1
        coefficients = _back_substitute(hessenberg, rotated_rhs, used)
        x = x + preconditioned[:, :used] @ coefficients
        residual = rhs - matvec(x)

    return x, total_iterations


def _back_substitute(
    hessenberg: np.ndarray, rotated_rhs: np.ndarray, size: int
) -> np.ndarray:
    coefficients = np.zeros(size, dtype=rotated_rhs.dtype)
    for row in range(size - 1, -1, -1):
        value = rotated_rhs[row]
        for column in range(row + 1, size):
            value -= hessenberg[row, column] * coefficients[column]
        coefficients[row] = value / hessenberg[row, row]
    return coefficients
