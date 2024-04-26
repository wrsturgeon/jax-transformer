from beartype import beartype
from beartype.typing import Callable, NamedTuple
from check_and_compile import check_and_compile
from jax import nn as jnn, numpy as jnp, random as jrnd
from jaxtyping import jaxtyped, Array, Float32, Float64


class Parameters(NamedTuple):
    weights_1: Float64[Array, "d_model d_ff"]
    biases_1: Float64[Array, "d_ff"]
    weights_2: Float64[Array, "d_ff d_model"]
    biases_2: Float64[Array, "d_model"]


@jaxtyped(typechecker=beartype)
def init(key: Array, d_model: int, d_ff: int) -> Parameters:
    k1, k2 = jrnd.split(key)
    return Parameters(
        weights_1=jnn.initializers.he_normal()(k1, [d_model, d_ff], dtype=jnp.float64),
        biases_1=jnp.zeros([d_ff], dtype=jnp.float64),
        weights_2=jnn.initializers.he_normal()(k2, [d_ff, d_model], dtype=jnp.float64),
        biases_2=jnp.zeros([d_model], dtype=jnp.float64),
    )


@check_and_compile(2)
def feedforward(
    params: Parameters,
    x: Float32[Array, "*batch seq d_model"],
    activation: Callable[
        [Float32[Array, "*batch seq d_model"]],
        Float32[Array, "*batch seq d_model"],
    ] = jnn.gelu,
) -> Float32[Array, "*batch seq d_model"]:

    # Shape checks:
    *batch, seq, d_model = x.shape
    d_ff = params.weights_1.shape[-1]
    assert params.weights_1.shape == (
        d_model,
        d_ff,
    ), f"{params.weights_1.shape} =/= {(d_model, d_ff)}"
    assert params.biases_1.shape == (d_ff,), f"{params.biases_1.shape} =/= {(d_ff,)}"
    assert params.weights_2.shape == (
        d_ff,
        d_model,
    ), f"{params.weights_2.shape} =/= {(d_ff, d_model)}"
    assert params.biases_2.shape == (
        d_model,
    ), f"{params.biases_2.shape} =/= {(d_model,)}"
    assert params.biases_2.shape == (d_model,)

    # First affine transform:
    x = x @ params.weights_1.astype(jnp.float32) + params.biases_1.astype(jnp.float32)
    assert x.shape == (*batch, seq, d_ff), f"{x.shape} =/= {(*batch, seq, d_ff)}"

    # Nonlinearity:
    x = activation(x)
    assert x.shape == (*batch, seq, d_ff), f"{x.shape} =/= {(*batch, seq, d_ff)}"

    # Second affine transform:
    x = x @ params.weights_2.astype(jnp.float32) + params.biases_2.astype(jnp.float32)
    assert x.shape == (*batch, seq, d_model), f"{x.shape} =/= {(*batch, seq, d_model)}"

    return x
