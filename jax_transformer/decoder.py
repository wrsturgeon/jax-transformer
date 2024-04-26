from jax_transformer import feedforward, positional_encoding, residual

from beartype import beartype
from beartype.typing import Callable, List, NamedTuple
from check_and_compile import check_and_compile
from jax import nn as jnn, numpy as jnp, random as jrnd
import jax_attn
from jaxtyping import jaxtyped, Array, Float32, Float64


class Parameters(NamedTuple):
    self_attention: residual.Parameters
    cross_attention: residual.Parameters
    feedforward: residual.Parameters


@jaxtyped(typechecker=beartype)
def init(
    key: Array,
    stack_depth: int,
    d_model: int,
    d_attn: int,
    heads: int,
    d_k: int,
    d_v: int,
    d_ff: int,
) -> List[Parameters]:
    k1, k2 = jrnd.split(key)
    keys = zip(jrnd.split(k1, num=stack_depth), jrnd.split(k2, num=stack_depth))
    params = [
        Parameters(
            self_attention=residual.init(
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
            cross_attention=residual.init(
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


@check_and_compile(4, 5, 6)
def decode(
    params: List[Parameters],
    encoder_output: Float32[Array, "*batch seq_e d_model"],
    output_embeddings: Float32[Array, "*batch seq_d d_model"],
    positions: Float32[Array, "*batch seq_d"],
    encode_positions: Callable[
        [Float32[Array, "*batch seq_d d_model"], Float32[Array, "*batch seq_d"]],
        Float32[Array, "*batch seq_d d_model"],
    ] = lambda x, p: positional_encoding.encode_positions(
        x, p, jnp.array(10000, dtype=jnp.float32)
    ),
    softmax: Callable[
        [Float32[Array, "*batch seq_d d_model"]],
        Float32[Array, "*batch seq_d d_model"],
    ] = jnn.softmax,
    ffn_activation: Callable[
        [Float32[Array, "*batch seq_d d_model"]],
        Float32[Array, "*batch seq_d d_model"],
    ] = jnn.gelu,
) -> Float32[Array, "*batch seq_d d_model"]:

    # Set up self-attention as a function:
    run_self_attn = lambda p, z, _: jax_attn.run(p, z, z, z, True, softmax)

    # Set up encoder-decoder attention as a function:
    run_cross_attn = lambda p, d, e: jax_attn.run(p, d, e, e, False, softmax)

    # Set up our feedforward network as a function:
    run_ffn = lambda p, z, _: feedforward.feedforward(p, z, ffn_activation)

    # Encode input position:
    x = encode_positions(output_embeddings, positions)

    # Run each layer:
    for p in params:

        # Self-attention:
        x = residual.residual(p.self_attention, x, run_self_attn)

        # Encoder-decoder attention:
        x = residual.residual(p.cross_attention, x, run_cross_attn, encoder_output)

        # Feedforward:
        x = residual.residual(p.feedforward, x, run_ffn)

    return x
