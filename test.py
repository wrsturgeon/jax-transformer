from jax_transformer import positional_encoding

from jax import numpy as jnp


def test_positional_encoding() -> None:
    assert jnp.allclose(
        positional_encoding.encode_position(
            jnp.zeros([7], dtype=jnp.float32),  # odd number!
            jnp.array(1, dtype=jnp.float32),
        ),
        jnp.array(
            [
                8.4147096e-01,
                5.4030228e-01,
                7.1906447e-02,
                9.9741137e-01,
                5.1794504e-03,
                9.9998659e-01,
                3.7275933e-04,
            ],
            dtype=jnp.float32,
        ),
    )
