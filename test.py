import chex
import haiku as hk
import jax
import jax.numpy as jnp
import rlax

from udance import (
    Config,
    GaussianAndValue,
    MusicPolicy,
    ObsPredictor,
    RolloutResult,
    RolloutWithMusic,
    _make_batch,
    batched_gae,
)

T, N, S, A, P = 5, 4, 3, 2, 6


def test_batch() -> None:
    initial_obs = jnp.ones((N, S))
    rollout = RolloutResult([initial_obs])
    for _ in range(T):
        rollout.append(
            observation=jnp.ones((N, S)),
            action=jnp.ones((N, A)),
            reward=jnp.ones((N,)),
            output=GaussianAndValue(jnp.ones((N, A)), jnp.ones((N, A)), jnp.ones((N,))),
            terminal=jnp.zeros((N,), dtype=bool),
        )

    batch = _make_batch(rollout, jnp.ones((N,)), Config())
    chex.assert_shape(batch.observation, (T * N, S))
    chex.assert_shape(batch.action, (T * N, A))
    chex.assert_shape(
        (batch.reward, batch.advantage, batch.value_target, batch.log_prob),
        (T * N,),
    )


def test_gae() -> None:
    rewards = jax.random.normal(key=jax.random.PRNGKey(43), shape=(T, N))
    values = jax.random.normal(key=jax.random.PRNGKey(44), shape=(T + 1, N))
    discount = jnp.ones_like(rewards) * 0.999
    our_result = batched_gae(rewards, discount, 0.95, values)
    their_gae = jax.vmap(
        rlax.truncated_generalized_advantage_estimation,
        in_axes=(1, 1, None, 1),
        out_axes=1,
    )
    their_result = their_gae(rewards, discount, 0.95, values)
    chex.assert_shape((our_result, their_result), (T, N))
    chex.assert_trees_all_close(our_result, their_result)


def test_music_policy() -> None:
    config = Config()
    init, music_policy = hk.transform(lambda x: MusicPolicy(A, S, P, config)(x))
    obs = jnp.ones((N, S))
    pitch = jnp.ones((N,), dtype=jnp.int32)
    mask = jnp.zeros((N, 1))
    params = init(jax.random.PRNGKey(43), (obs, pitch, mask))
    output, state = music_policy(params, jax.random.PRNGKey(44), (obs, pitch, mask))
    chex.assert_shape((output.policy.mean, output.policy.stddev), (N, A))
    chex.assert_shape((output.pred.mean, output.pred.stddev), (N, S))
    chex.assert_shape(output.value, (N, 1))


def test_obs_predictor() -> None:
    config = Config()
    init, music_policy = hk.transform(lambda x: ObsPredictor(S, config)(x))
    obs = jnp.ones((N, S))
    mask = jnp.zeros((N, 1))
    params = init(jax.random.PRNGKey(43), (obs, mask))
    output, state = music_policy(params, jax.random.PRNGKey(44), (obs, mask))
    chex.assert_shape((output.mean, output.stddev), (N, S))


def test_music_batch() -> None:
    config = Config()
    init, music_policy = hk.transform(lambda x: MusicPolicy(A, S, P, config)(x))
    obs = jnp.ones((N, S))
    pitch = jnp.ones((N,), dtype=jnp.int32)
    mask = jnp.zeros((N, 1))
    params = init(jax.random.PRNGKey(43), (obs, pitch, mask))
    rollout = RolloutWithMusic([obs])
    for _ in range(T):
        rollout.append(
            observation=jnp.ones((N, S)),
            music=jnp.ones((N, P), dtype=jnp.int32),
            action=jnp.ones((N, A)),
            output=GaussianAndValue(jnp.ones((N, A)), jnp.ones((N, A)), jnp.ones((N,))),
            terminal=jnp.zeros((N,), dtype=bool),
        )

    batch = _make_batch(rollout, jnp.ones((N,)), Config())
    chex.assert_shape(batch.observation, (T * N, S))
    chex.assert_shape(batch.action, (T * N, A))
    chex.assert_shape(
        (batch.reward, batch.advantage, batch.value_target, batch.log_prob),
        (T * N,),
    )
