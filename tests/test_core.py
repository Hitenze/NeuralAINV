import jax
import jax.numpy as jnp
import numpy as np

from neuralainv.dd import BlockElimination, partition
from neuralainv.fem import assemble, project_kernel, square_mesh
from neuralainv.inference import GreenKernel, load_weights, save_weights
from neuralainv.krylov import fgmres
from neuralainv.models import AlphaMSNN
from neuralainv.operators import fd_matrix, grid_points


def test_weights_and_chunking(tmp_path):
    model = AlphaMSNN(4)
    params = model.init(jax.random.PRNGKey(42), jnp.zeros((1, 4)))["params"]
    assert len([k for k in params if k.startswith("_experts")]) == 5
    save_weights(tmp_path, params)
    restored = load_weights(tmp_path)
    x, y = grid_points(2), grid_points(3)
    expected = GreenKernel(params).grid(x, y, chunk_size=16)
    np.testing.assert_allclose(restored.grid(x, y, chunk_size=16), expected, atol=1e-6)
    pairs = np.column_stack((np.repeat(x, len(y), axis=0), np.tile(y, (len(x), 1))))
    np.testing.assert_allclose(
        restored.pairs(pairs, chunk_size=16).reshape(expected.shape), expected
    )


def test_fd_and_block_elimination():
    n = 7
    A = fd_matrix(n)
    axis = np.arange(1, n + 1) / (n + 1)
    u = np.outer(np.sin(np.pi * axis), np.sin(np.pi * axis)).ravel()
    eigenvalue = 8 * np.sin(np.pi / (2 * (n + 1))) ** 2 - 30 / (n + 1) ** 2
    np.testing.assert_allclose(A @ u, eigenvalue * u, atol=2e-15)
    groups, gamma = partition(n)
    np.testing.assert_array_equal(
        np.sort(np.concatenate(groups + [gamma])), np.arange(n * n)
    )
    for i, g in enumerate(groups):
        for other in groups[i + 1 :]:
            assert A[g][:, other].nnz == 0
    block = BlockElimination(A, n, drop_tol=0.0, fill_factor=100)
    inverse = np.linalg.inv(A.toarray())
    terminal = inverse[np.ix_(gamma, gamma)]
    rhs = np.random.default_rng(42).normal(size=n * n)
    action = block.action(lambda r: terminal @ r)
    np.testing.assert_allclose(
        action(rhs), np.linalg.solve(A.toarray(), rhs), rtol=1e-12, atol=1e-12
    )
    solution, _ = fgmres(A, rhs, preconditioner=action, relative_tolerance=1e-10)
    assert np.linalg.norm(A @ solution - rhs) / np.linalg.norm(rhs) < 1e-10


def test_fem_geometry_and_projection():
    for level, size in enumerate((225, 427, 639)):
        mesh = square_mesh(level)
        A, M, q, D, free, error = assemble(mesh)
        assert A.shape == (size, size)
        assert error < 1e-12
        assert np.all(M.diagonal() > 0)
    mesh = square_mesh(0)
    A, M, q, D, free, error = assemble(mesh)
    rng = np.random.default_rng(42)
    features = rng.normal(size=(len(q), 3))
    G = features @ features.T
    P = project_kernel(G, M, D)
    vector = rng.normal(size=len(free))
    direct = np.linalg.solve(
        M.toarray(), D.T @ (G @ (D @ np.linalg.solve(M.toarray(), vector)))
    )
    np.testing.assert_allclose(P @ vector, direct, rtol=1e-11, atol=1e-10)
