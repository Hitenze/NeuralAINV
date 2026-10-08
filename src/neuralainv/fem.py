"""P1 assembly and projection of a continuous Green operator."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix, csr_matrix
from skfem import Basis, BilinearForm, ElementTriP1, MeshTri, asm
from skfem.helpers import dot, grad


def square_mesh(level):
    if level not in (0, 1, 2):
        raise ValueError("level must be 0, 1, or 2")
    mesh = MeshTri.init_sqsymmetric().refined(3)
    centers = np.array([[0.25, 0.30], [0.75, 0.70]])
    for radius in (0.18, 0.09)[:level]:
        centroids = mesh.p[:, mesh.t].mean(axis=1).T
        distance = ((centroids[:, None] - centers[None, :]) ** 2).sum(axis=2)
        mesh = mesh.refined(np.flatnonzero((distance < radius**2).any(axis=1)))
    return mesh


@BilinearForm
def stiffness(u, v, w):
    return dot(grad(u), grad(v))


@BilinearForm
def mass(u, v, w):
    return u * v


def assemble(mesh, *, quadrature_order=2):
    basis = Basis(mesh, ElementTriP1(), intorder=quadrature_order)
    boundary = basis.get_dofs(facets=mesh.boundary_facets()).all()
    free = np.setdiff1d(np.arange(basis.N), boundary)
    K = csr_matrix(asm(stiffness, basis))[free][:, free]
    M = csr_matrix(asm(mass, basis))[free][:, free]
    q = np.asarray(basis.global_coordinates()).reshape(2, -1).T
    weights = basis.dx.ravel()
    nq = basis.dx.shape[1]
    values, columns = [], []
    for local in range(3):
        values.append(np.asarray(basis.basis[local][0]).ravel())
        columns.append(np.repeat(basis.element_dofs[local], nq))
    phi_all = coo_matrix(
        (
            np.concatenate(values),
            (np.tile(np.arange(len(q)), 3), np.concatenate(columns)),
        ),
        shape=(len(q), basis.N),
    ).tocsr()
    phi = phi_all[:, free]
    D = phi.multiply(weights[:, None]).tocsr()
    error = np.linalg.norm((phi.T @ D).toarray() - M.toarray()) / np.linalg.norm(
        M.toarray()
    )
    if error > 1e-12 or not np.isclose(weights.sum(), 1):
        raise ArithmeticError("quadrature and mass assembly disagree")
    return K - 30 * M, M, q, D, free, float(error)


def project_kernel(kernel, mass_matrix, weighted_basis):
    """Return M^-1 D^T G D M^-1 with D = W Phi."""
    projected = np.asarray(weighted_basis.T @ (kernel @ weighted_basis))
    factor = cho_factor(mass_matrix.toarray(), lower=True)
    left = cho_solve(factor, projected)
    return cho_solve(factor, left.T).T
