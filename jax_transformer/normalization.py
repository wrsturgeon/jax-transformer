from beartype import beartype
from beartype.typing import NamedTuple
from check_and_compile import check_and_compile
from jax import numpy as jnp
from jaxtyping import jaxtyped, Array, Float32, Float64


class Parameters(NamedTuple):
    weights: Float64[Array, "d_model"]
    biases: Float64[Array, "d_model"]


@jaxtyped(typechecker=beartype)
def init(d_model: int) -> Parameters:
    return Parameters(
        weights=jnp.ones([d_model], dtype=jnp.float64),
        biases=jnp.zeros([d_model], dtype=jnp.float64),
    )


@check_and_compile()
def layer_norm(
    params: Parameters,
    unnormalized: Float32[Array, "*batch seq d_model"],
    epsilon: Float32[Array, ""] = jnp.array(1e-8, dtype=jnp.float32),
) -> Float32[Array, "*batch seq d_model"]:

    d_model = unnormalized.shape[-1]
    assert params.weights.shape == (d_model,)
    assert params.biases.shape == (d_model,)

    expectation = jnp.mean(unnormalized, axis=-1, keepdims=True)
    standard_deviation = jnp.std(unnormalized, axis=-1, keepdims=True)

    normalized = (unnormalized - expectation) / (standard_deviation + epsilon)

    weights = params.weights.reshape(*[1 for _ in range(1, unnormalized.ndim)], -1)
    biases = params.biases.reshape(*[1 for _ in range(1, unnormalized.ndim)], -1)

    # Element-wise multiplication (i.e. Hadamard product), not matrix multiplication
    return weights.astype(jnp.float32) * normalized + biases.astype(jnp.float32)
