from jax_transformer import normalization, positional_encoding, residual

from jax import numpy as jnp


def test_positional_encoding() -> None:
    assert jnp.allclose(
        positional_encoding.encode_position(
            jnp.zeros([7], dtype=jnp.float32),  # odd number!
            jnp.array(1, dtype=jnp.float32),
        ),
        jnp.array(
            [
                0.84147096,
                0.54030228,
                0.07190644,
                0.99741137,
                0.00517945,
                0.99998659,
                0.00037275,
            ],
            dtype=jnp.float32,
        ),
    )


TEN_NORMALIZED = jnp.array(
    [
        -1.5666989,
        -1.2185436,
        -0.87038827,
        -0.52223295,
        -0.17407766,
        0.17407766,
        0.52223295,
        0.87038827,
        1.2185436,
        1.5666989,
    ],
    dtype=jnp.float32,
)


def test_layer_norm() -> None:
    d_model = 10
    p = normalization.init(d_model)
    x = jnp.arange(d_model, dtype=jnp.float32)
    y = normalization.layer_norm(p, x)
    assert jnp.allclose(y, TEN_NORMALIZED)


def test_residual() -> None:
    d_model = 10
    p = residual.init(jnp.array(1, dtype=jnp.float64), d_model)
    x = jnp.arange(d_model, dtype=jnp.float32)
    y = residual.residual(p, x, lambda p, z: z + p.astype(jnp.float32))
    assert jnp.allclose(y, TEN_NORMALIZED)
