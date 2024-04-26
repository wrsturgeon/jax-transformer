from jax_transformer import feedforward, positional_encoding, residual

from beartype import beartype
from beartype.typing import Callable, List, NamedTuple
from check_and_compile import check_and_compile
from jax import nn as jnn, numpy as jnp, random as jrnd
import jax_attn
from jaxtyping import jaxtyped, Array, Float32, Float64


class Parameters(NamedTuple):
    attention: residual.Parameters
    feedforward: residual.Parameters


@jaxtyped(typechecker=beartype)
def init(
    key: Array,
    stack_depth: int = 6,
    d_model: int = 512,
    d_attn: int = 512,
    heads: int = 8,
    d_k: int = 64,
    d_v: int = 64,
    d_ff: int = 2048,
) -> List[Parameters]:
    k1, k2 = jrnd.split(key)
    keys = zip(jrnd.split(k1, num=stack_depth), jrnd.split(k2, num=stack_depth))
    params = [
        Parameters(
            attention=residual.init(
                sublayer=jax_attn.init(
                    k_attn,
                    embedding=d_model,
                    d_model=d_attn,
                    heads=heads,
                    d_k=d_k,
                    d_v=d_v,
                ),
                d_model=d_model,
            ),
            feedforward=residual.init(
                sublayer=feedforward.init(k_ffn, d_model=d_model, d_ff=d_ff),
                d_model=d_model,
            ),
        )
        for k_attn, k_ffn in keys
    ]
    assert len(params) == stack_depth
    return params


@check_and_compile(3, 4, 5)
def encode(
    params: List[Parameters],
    input_embedding: Float32[Array, "*batch seq d_model"],
    position: Float32[Array, "*batch seq"],
    encode_position: Callable[
        [Float32[Array, "*batch seq d_model"], Float32[Array, "*batch seq"]],
        Float32[Array, "*batch seq d_model"],
    ] = lambda x, p: positional_encoding.encode_position(
        x, p, jnp.array(10000, dtype=jnp.float32)
    ),
    attn_activation: Callable[
        [Float32[Array, "*batch seq d_model"]],
        Float32[Array, "*batch seq d_model"],
    ] = jnn.gelu,
    ffn_activation: Callable[
        [Float32[Array, "*batch seq d_model"]],
        Float32[Array, "*batch seq d_model"],
    ] = jnn.gelu,
) -> Float32[Array, "*batch seq d_model"]:

    # Set up attention as a function:
    run_attn = lambda p, z: jax_attn.run(p, z, False, attn_activation)

    # Set up our feedforward network as a function:
    run_ffn = lambda p, z: feedforward.feedforward(p, z, ffn_activation)

    # Encode input position:
    x = encode_position(input_embedding, position)

    # Run each layer:
    for p in params:

        # Self-attention:
        x = residual.residual(p.attention, x, run_attn)

        # Feedforward:
        x = residual.residual(p.feedforward, x, run_ffn)

    return x
