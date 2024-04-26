from jax_transformer import (
    decoder,
    encoder,
    feedforward,
    normalization,
    positional_encoding,
    residual,
)

from jax import numpy as jnp, random as jrnd


def test_positional_encoding() -> None:
    assert jnp.allclose(
        positional_encoding.encode_position(
            jnp.zeros([13, 7], dtype=jnp.float32),  # odd number!
            jnp.ones([13], dtype=jnp.float32),
        ),
        jnp.stack(
            [
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
                )
                for _ in range(13)
            ]
        ),
    )


TEN_NORMALIZED = jnp.array(
    [
        [
            -1.5666989,
            -1.2185436,
            -0.8703882,
            -0.5222329,
            -0.1740776,
            +0.1740776,
            +0.5222329,
            +0.8703882,
            +1.2185436,
            +1.5666989,
        ]
    ],
    dtype=jnp.float32,
)


def test_layer_norm() -> None:
    d_model = 10
    p = normalization.init(d_model)
    x = jnp.arange(d_model, dtype=jnp.float32)[jnp.newaxis]
    y = normalization.layer_norm(p, x)
    assert jnp.allclose(y, TEN_NORMALIZED)


def test_residual() -> None:
    d_model = 10
    p = residual.init(jnp.array(1, dtype=jnp.float64), d_model)
    x = jnp.arange(d_model, dtype=jnp.float32)[jnp.newaxis]
    y = residual.residual(p, x, lambda p, z, _: z + p.astype(jnp.float32))
    assert jnp.allclose(y, TEN_NORMALIZED)


def test_feedforward() -> None:
    d_model = 10
    d_ff = 20
    p = feedforward.init(jrnd.PRNGKey(42), d_model, d_ff)
    x = jnp.arange(d_model, dtype=jnp.float32)[jnp.newaxis]
    y = feedforward.feedforward(p, x)
    assert jnp.allclose(
        y,
        jnp.array(
            [
                [
                    -03.795891,
                    -06.430948,
                    -16.024843,
                    +06.165481,
                    -04.259849,
                    +06.478252,
                    +11.613619,
                    +09.119688,
                    -21.604553,
                    -02.048109,
                ]
            ],
            dtype=jnp.float32,
        ),
    )


def test_encoder() -> None:
    d_model = 10
    p = encoder.init(
        jrnd.PRNGKey(42),
        stack_depth=3,
        d_model=d_model,
        d_attn=11,
        heads=12,
        d_k=13,
        d_v=14,
        d_ff=15,
    )
    x = jnp.arange(3 * d_model, dtype=jnp.float32).reshape(3, d_model)
    y = encoder.encode(p, x, jnp.arange(3, dtype=jnp.float32))
    assert jnp.allclose(
        y,
        jnp.array(
            [
                [
                    +0.2543729,
                    +1.5914096,
                    +0.5870357,
                    -0.6957300,
                    -0.2853456,
                    +0.3008959,
                    -1.4158846,
                    +1.5442154,
                    -1.2895874,
                    -0.5913818,
                ],
                [
                    +0.4929160,
                    +1.7341815,
                    +0.5549921,
                    -0.2516452,
                    -0.2726935,
                    -0.4583319,
                    -1.3584229,
                    +1.4491484,
                    -1.3706355,
                    -0.5195089,
                ],
                [
                    +0.5733387,
                    +1.7064579,
                    +0.5736176,
                    -0.1685401,
                    -0.2333487,
                    -0.6158084,
                    -1.3775861,
                    +1.4050514,
                    -1.3561687,
                    -0.5070132,
                ],
            ],
            dtype=jnp.float32,
        ),
    )


def test_decoder() -> None:
    stack_depth = 3
    d_model = 10
    d_attn = 11
    heads = 8
    d_k = 12
    d_v = 13
    d_ff = 14
    seq = 5
    k1, k2 = jrnd.split(jrnd.PRNGKey(42))
    p_e = encoder.init(
        k1,
        stack_depth=stack_depth,
        d_model=d_model,
        d_attn=d_attn,
        heads=heads,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
    )
    p_d = decoder.init(
        k2,
        stack_depth=stack_depth,
        d_model=d_model,
        d_attn=d_attn,
        heads=heads,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
    )
    x = jnp.arange(seq * d_model, dtype=jnp.float32).reshape(seq, d_model)
    e = encoder.encode(p_e, x, jnp.arange(seq, dtype=jnp.float32))
    y = decoder.decode(p_d, e, x, seq - jnp.arange(seq, dtype=jnp.float32))
    assert jnp.allclose(
        y,
        jnp.array(
            [
                [
                    -1.7899036,
                    +1.4953532,
                    +0.2726643,
                    +0.1088306,
                    -0.0555641,
                    -0.1933813,
                    -1.3028834,
                    -0.3394138,
                    +1.6067894,
                    +0.1975088,
                ],
                [
                    -1.8521041,
                    +1.4276607,
                    +0.2484906,
                    +0.1483715,
                    -0.0863817,
                    -0.2316310,
                    -1.1107715,
                    -0.4508236,
                    +1.7055516,
                    +0.2016378,
                ],
                [
                    -1.8560345,
                    +1.4196646,
                    +0.2540690,
                    +0.1483856,
                    -0.1066695,
                    -0.2372217,
                    -1.0576694,
                    -0.4939037,
                    +1.7267950,
                    +0.2025838,
                ],
                [
                    -1.8437077,
                    +1.4322349,
                    +0.2666365,
                    +0.1372194,
                    -0.1147831,
                    -0.2428780,
                    -1.0501943,
                    -0.5109285,
                    +1.7273637,
                    +0.1990371,
                ],
                [
                    -1.8317733,
                    +1.4470986,
                    +0.2760431,
                    +0.1261256,
                    -0.1242620,
                    -0.2419722,
                    -1.0541505,
                    -0.5184302,
                    +1.7218025,
                    +0.1995178,
                ],
            ],
            dtype=jnp.float32,
        ),
    )
