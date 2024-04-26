from jax_transformer import (
    decoder,
    encoder,
    feedforward,
    normalization,
    positional_encoding,
    residual,
)

from beartype import beartype
from beartype.typing import Callable, NamedTuple
from check_and_compile import check_and_compile
from jax import nn as jnn, numpy as jnp, random as jrnd
from jaxtyping import jaxtyped, Array, Float32, Float64


DEFAULT_D_MODEL: int = 512
DEFAULT_STACK_DEPTH: int = 6
DEFAULT_D_ATTN: int = 512
DEFAULT_HEADS: int = 8
DEFAULT_D_K: int = 64
DEFAULT_D_V: int = 64
DEFAULT_D_FF: int = 2048


class Parameters(NamedTuple):
    encoder: encoder.Parameters
    decoder: decoder.Parameters
    linear: Float64[Array, "d_model n_outputs"]


@jaxtyped(typechecker=beartype)
def init(
    key: Array,
    n_outputs: int,
    d_model: int = DEFAULT_D_MODEL,
    encoder_stack_depth: int = DEFAULT_STACK_DEPTH,
    encoder_d_attn: int = DEFAULT_D_ATTN,
    encoder_heads: int = DEFAULT_HEADS,
    encoder_d_k: int = DEFAULT_D_K,
    encoder_d_v: int = DEFAULT_D_V,
    encoder_d_ff: int = DEFAULT_D_FF,
    decoder_stack_depth: int = DEFAULT_STACK_DEPTH,
    decoder_d_attn: int = DEFAULT_D_ATTN,
    decoder_heads: int = DEFAULT_HEADS,
    decoder_d_k: int = DEFAULT_D_K,
    decoder_d_v: int = DEFAULT_D_V,
    decoder_d_ff: int = DEFAULT_D_FF,
) -> Parameters:
    k1, k2, k3 = jrnd.split(key, 3)
    return Parameters(
        encoder=encoder.init(
            key=k1,
            d_model=d_model,
            stack_depth=encoder_stack_depth,
            d_attn=encoder_d_attn,
            heads=encoder_heads,
            d_k=encoder_d_k,
            d_v=encoder_d_v,
            d_ff=encoder_d_ff,
        ),
        decoder=decoder.init(
            key=k2,
            d_model=d_model,
            stack_depth=decoder_stack_depth,
            d_attn=decoder_d_attn,
            heads=decoder_heads,
            d_k=decoder_d_k,
            d_v=decoder_d_v,
            d_ff=decoder_d_ff,
        ),
        linear=jnn.initializers.he_normal()(
            k3,
            [d_model, n_outputs],
            dtype=jnp.float64,
        ),
    )


@check_and_compile()
def run(
    params: Parameters,
    input_embeddings: Float32[Array, "*batch seq_in d_model"],
    input_positions: Float32[Array, "*batch seq_in"],
    output_embeddings: Float32[Array, "*batch seq_out d_model"],
    output_positions: Float32[Array, "*batch seq_out"],
    encode_positions: Callable[
        [Float32[Array, "*batch seq d_model"], Float32[Array, "*batch seq"]],
        Float32[Array, "*batch seq d_model"],
    ] = lambda x, p: positional_encoding.encode_positions(
        x, p, jnp.array(10000, dtype=jnp.float32)
    ),
    softmax: Callable[
        [Float32[Array, "*batch seq d_model"]],
        Float32[Array, "*batch seq d_model"],
    ] = jnn.softmax,
    ffn_activation: Callable[
        [Float32[Array, "*batch seq d_model"]],
        Float32[Array, "*batch seq d_model"],
    ] = jnn.gelu,
) -> Float32[Array, "*batch seq_out n_outputs"]:

    encoded: Float32[Array, "*batch seq_in d_model"] = encoder.encode(
        params.encoder,
        input_embeddings,
        input_positions,
        encode_positions,
        softmax,
        ffn_activation,
    )

    decoded: Float32[Array, "*batch seq_out d_model"] = decoder.decode(
        params.decoder,
        encoded,
        output_embeddings,
        output_positions,
        encode_positions,
        softmax,
        ffn_activation,
    )

    # Really doesn't make sense to use anything but softmax here
    return jnn.softmax(decoded @ params.linear.astype(jnp.float32))
