from jax_transformer import normalization

from beartype import beartype
from beartype.typing import Callable, NamedTuple
from check_and_compile import check_and_compile
from jax import numpy as jnp
from jax.tree_util import tree_map, tree_structure
from jaxtyping import jaxtyped, Array, Float32, Float64, PyTree
import operator


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
    x: PyTree[Float32[Array, "..."], "X"],  # type: ignore[name-defined]
    f: Callable[  # type: ignore[name-defined]
        [
            PyTree[Float64[Array, "..."]],
            PyTree[Float32[Array, "..."], "X"],
            PyTree[Float32[Array, "..."], "A"],  # type: ignore[name-defined]
        ],
        PyTree[Float32[Array, "..."], "X"],
    ],
    aux: PyTree[Float32[Array, "..."], "A"] = (),  # type: ignore[name-defined]
    epsilon: Float32[Array, ""] = jnp.array(1e-8, dtype=jnp.float32),
) -> PyTree[Float32[Array, "..."], "X"]:  # type: ignore[name-defined]

    # Run the sub-layer:
    y = f(params.sublayer, x, aux)

    # Check that shapes & dtypes match:
    def match(a, b):
        assert a.shape == b.shape, f"{a.shape} =/= {b.shape}"
        assert jnp.issubdtype(a.dtype, b.dtype), f"{a.dtype} is not {b.dtype}"

    # Actually run the above:
    tsy, tsx = [tree_structure(z) for z in [y, x]]
    assert tsy == tsx, f"{tsy} =/= {tsx}"
    tree_map(match, y, x)

    # Add the sub-layer output to the original input:
    summed = tree_map(operator.add, x, y)

    # Normalize:
    return normalization.layer_norm(params.normalization, summed, epsilon)
