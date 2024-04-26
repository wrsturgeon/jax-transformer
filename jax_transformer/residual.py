from jax_transformer import normalization

from beartype import beartype
from beartype.typing import Callable, NamedTuple
from check_and_compile import check_and_compile
from jax import numpy as jnp
from jaxtyping import jaxtyped, Array, Float32, Float64, PyTree


class Parameters(NamedTuple):
    sublayer: PyTree[Float64[Array, "..."]]
    normalization: normalization.Parameters


@jaxtyped(typechecker=beartype)
def init(sublayer: PyTree[Float64[Array, "..."]], d_model: int) -> Parameters:
    return Parameters(
        sublayer=sublayer,
        normalization=normalization.init(d_model),
    )


@check_and_compile(2)
def residual(
    params: Parameters,
    x: Float32[Array, "..."],
    f: Callable[
        [
            PyTree[Float64[Array, "..."]],
            Float32[Array, "..."],
        ],
        Float32[Array, "..."],
    ],
    epsilon: Float32[Array, ""] = jnp.array(1e-8, dtype=jnp.float32),
) -> Float32[Array, "..."]:

    # Run the sub-layer:
    y = f(params.sublayer, x)
    assert y.shape == x.shape, f"{y.shape} =/= {x.shape}"
    assert jnp.issubdtype(y.dtype, x.dtype), f"{y.dtype} is not {x.dtype}"

    # Add it to the original input:
    summed = x + y

    # Normalize:
    return normalization.layer_norm(params.normalization, summed, epsilon)
