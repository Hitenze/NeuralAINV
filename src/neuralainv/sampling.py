"""Full-domain collocation pairs and Gaussian source regularization."""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("n_far", "n_near", "n_boundary"))
def collocation(seed, n_far, n_near, n_boundary, epsilon):
    key_i, key_n, key_b = jax.random.split(jax.random.PRNGKey(seed), 3)
    far = jax.random.uniform(key_i, (n_far, 4))
    ky, kr, kt = jax.random.split(key_n, 3)
    y = jax.random.uniform(ky, (n_near, 2))
    radius = jnp.sqrt(jax.random.uniform(kr, (n_near,))) * (3 * epsilon)
    theta = jax.random.uniform(kt, (n_near,)) * (2 * jnp.pi)
    x = jnp.clip(
        y + jnp.column_stack((radius * jnp.cos(theta), radius * jnp.sin(theta))), 0, 1
    )
    pairs = jnp.concatenate((far, jnp.column_stack((x, y))))
    r2 = jnp.sum((pairs[:, :2] - pairs[:, 2:]) ** 2, axis=1)
    log_delta = -jnp.log(2 * jnp.pi) - 2 * jnp.log(epsilon) - 0.5 * r2 / epsilon**2
    targets = jnp.exp(jnp.clip(log_delta, -60, 60))

    keys = jax.random.split(key_b, 5)
    counts = [n_boundary // 4 + int(i < n_boundary % 4) for i in range(4)]
    b, r, t, l = [jax.random.uniform(k, (n,)) for k, n in zip(keys, counts)]
    bx = jnp.concatenate((b, jnp.ones_like(r), t, jnp.zeros_like(l)))
    by = jnp.concatenate((jnp.zeros_like(b), r, jnp.ones_like(t), l))
    sources = jax.random.uniform(keys[4], (n_boundary, 2))
    boundary = jnp.column_stack((bx, by, sources))
    return pairs, targets, boundary, jnp.zeros(n_boundary)
