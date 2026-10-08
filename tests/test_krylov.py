import numpy as np
import pytest

from neuralainv.krylov import fgmres


@pytest.mark.parametrize("parameter", ["maximum_iterations", "restart"])
@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_iteration_settings(parameter, value):
    with pytest.raises(ValueError, match=f"{parameter} must be positive"):
        fgmres(np.eye(2), np.ones(2), **{parameter: value})


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1.0, 1e-16, 1e16])
def test_matrix_scaling(dtype, scale):
    matrix = np.diag(np.array([1, 2], dtype=dtype)) * dtype(scale)
    rhs = np.ones(2, dtype=dtype)
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    history = []
    x, info = fgmres(
        matrix, rhs, restart=2, maximum_iterations=20,
        relative_tolerance=tolerance, callback=history.append,
    )
    assert info == 0
    assert len(history) == 2
    assert np.linalg.norm(matrix @ x - rhs) / np.linalg.norm(rhs) <= tolerance


def test_singular_step_preserves_previous_update():
    calls = 0

    def preconditioner(vector):
        nonlocal calls
        calls += 1
        return np.ones(2) if calls == 1 else np.zeros(2)

    history = []
    rhs = np.array([1.0, 0.0])
    x, info = fgmres(
        np.eye(2), rhs, preconditioner=preconditioner,
        restart=2, maximum_iterations=5, callback=history.append,
    )
    np.testing.assert_allclose(x, [0.5, 0.5])
    assert info == calls == len(history) == 2
    np.testing.assert_allclose(history[-1], np.linalg.norm(rhs - x))


def test_singular_first_step_reports_failure():
    rhs = np.array([1.0, 2.0])
    history = []
    x, info = fgmres(
        np.eye(2), rhs, preconditioner=lambda vector: np.zeros_like(vector),
        callback=history.append,
    )
    np.testing.assert_array_equal(x, np.zeros(2))
    assert info == len(history) == 1
    np.testing.assert_allclose(history[-1], np.linalg.norm(rhs))


def test_happy_breakdown_converges_in_one_step():
    rhs = np.array([1.0, -2.0, 3.0])
    history = []
    x, info = fgmres(np.eye(3), rhs, callback=history.append)
    assert info == 0 and len(history) == 1
    np.testing.assert_allclose(x, rhs, rtol=1e-14)


def test_restart_and_iteration_limit():
    matrix = np.diag([1.0, 2.0, 3.0])
    rhs = np.ones(3)
    for limit, expected_success in [(2, False), (100, True)]:
        history = []
        x, info = fgmres(
            matrix, rhs, restart=1, maximum_iterations=limit,
            relative_tolerance=1e-8, callback=history.append,
        )
        residual = np.linalg.norm(matrix @ x - rhs) / np.linalg.norm(rhs)
        assert (info == 0) == expected_success
        assert (residual <= 1e-8) == expected_success
        if not expected_success:
            assert info == len(history) == limit
