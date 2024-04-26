from check_and_compile import check_and_compile
from jax import numpy as jnp
from jaxtyping import Array, Float32, UInt32


@check_and_compile()
def encode_position(
    input_embedding: Float32[Array, "d_model"],
    position: Float32[Array, ""],
    max_wavelength: Float32[Array, ""] = jnp.array(10000, dtype=jnp.float32),
) -> Float32[Array, "d_model"]:

    (d_model,) = input_embedding.shape

    indices: UInt32[Array, "d_model//2"] = jnp.arange(0, d_model, 2, dtype=jnp.uint32)
    zero_to_one: Float32[Array, "d_model//2"] = indices.astype(jnp.float32) / d_model
    exponentiated: Float32[Array, "d_model//2"] = jnp.power(max_wavelength, zero_to_one)
    clock_hands: Float32[Array, "d_model//2"] = position / exponentiated

    input_embedding = input_embedding.at[0::2].add(jnp.sin(clock_hands))
    input_embedding = input_embedding.at[1::2].add(
        jnp.cos(clock_hands if d_model & 1 == 0 else clock_hands[:-1])
    )

    return input_embedding
