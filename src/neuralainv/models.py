"""Source-dependent scaling and mixtures of multiscale Green kernels."""

import jax
import jax.numpy as jnp
from flax import linen as nn

XAVIER = nn.initializers.xavier_uniform()
SCALES = (0.01, 0.005, 0.002)
WIDTHS = (10, 15, 20)


def features(xy):
    """Map (x1, x2, y1, y2) to (x1, x2, y1, y2, x1-y1, x2-y2)."""
    return jnp.concatenate((xy, xy[..., :2] - xy[..., 2:]), axis=-1)


def source_features(xy):
    return xy * jnp.array([0.0, 0.0, 1.0, 1.0], dtype=xy.dtype)


def ordered_exponents(logits):
    """Map three logits to (alpha1, beta1=1, alpha2, beta2)."""
    s = nn.sigmoid(logits)
    a1 = 2 * s[..., 0] - 1
    a2 = a1 + s[..., 1] * (1 - a1)
    return jnp.stack((a1, jnp.ones_like(a1), a2, s[..., 2]), axis=-1)


class AlphaBeta(nn.Module):
    """Learn alpha1(y), alpha2(y), beta2(y), with beta1=1 and ordered alphas."""

    @nn.compact
    def __call__(self, xy):
        z = source_features(xy)
        for width in (5, 5, 5):
            z = nn.tanh(nn.Dense(width, kernel_init=XAVIER)(z))
        return ordered_exponents(nn.Dense(3, kernel_init=XAVIER)(z))


class FarField(nn.Module):
    @nn.compact
    def __call__(self, xy):
        z = features(xy)
        for width in (50, 50, 50):
            z = nn.tanh(nn.Dense(width, kernel_init=XAVIER)(z))
        return nn.Dense(1, kernel_init=XAVIER)(z)


class AdaptiveBranch(nn.Module):
    width: int
    epsilon: float

    @nn.compact
    def __call__(self, xy):
        z = source_features(xy)
        for width in (5, 5, 5):
            z = nn.tanh(nn.Dense(width, kernel_init=XAVIER)(z))
        ab = ordered_exponents(nn.Dense(3, kernel_init=XAVIER)(z))
        transformed = features(xy)
        outputs = []
        for index in (0, 2):
            z = transformed * jnp.power(self.epsilon, -ab[..., index + 1])[..., None]
            for _ in range(3):
                z = nn.tanh(nn.Dense(self.width, kernel_init=XAVIER)(z))
            z = nn.Dense(1, kernel_init=XAVIER)(z)
            outputs.append(z * jnp.power(self.epsilon, ab[..., index])[..., None])
        return outputs[0] + outputs[1]


class Gate(nn.Module):
    """Five source-dependent softmax weights."""

    @nn.compact
    def __call__(self, xy):
        z = source_features(xy)
        for width in (5, 5, 5):
            z = nn.tanh(nn.Dense(width, kernel_init=XAVIER)(z))
        return nn.softmax(nn.Dense(5, kernel_init=XAVIER)(z), axis=-1)


class AlphaMSNN(nn.Module):
    """Sum of global scales, with five finest-scale experts in stage four."""

    stage: int = 4

    @nn.compact
    def __call__(self, xy):
        if self.stage not in (1, 2, 3, 4):
            raise ValueError("stage must be 1, 2, 3, or 4")
        prefix = "_globals" if self.stage == 4 else "_models"
        total = FarField(name=f"{prefix}_0")(xy)
        n_scales = 2 if self.stage == 4 else self.stage
        for i in range(n_scales):
            total = total + AdaptiveBranch(
                WIDTHS[i], SCALES[i], name=f"{prefix}_{i + 1}"
            )(xy)
        if self.stage == 4:
            weights = Gate(name="_ddnet")(xy)
            experts = [
                AdaptiveBranch(20, SCALES[2], name=f"_experts_{i}")(xy)
                for i in range(5)
            ]
            total = total + jnp.sum(
                weights * jnp.concatenate(experts, axis=-1), axis=-1, keepdims=True
            )
        return total


def apply_operator(model, params, pairs):
    """Evaluate (-Delta_x - 30) G(x,y), differentiating only target coordinates."""

    def scalar(x1, x2, y1, y2):
        p = jnp.array([[x1, x2, y1, y2]])
        return model.apply({"params": params}, p)[0, 0]

    d11 = jax.grad(jax.grad(scalar, 0), 0)
    d22 = jax.grad(jax.grad(scalar, 1), 1)

    def residual(p):
        return -d11(*p) - d22(*p) - 30 * scalar(*p)

    return jax.vmap(residual)(pairs)
