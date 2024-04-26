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
