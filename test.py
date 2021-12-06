import chex
import jax
import jax.numpy as jnp
import optax
import rlax

from udance import (
    Actor,
    Config,
    GaussianAndValue,
    Learner,
    RolloutResult,
    _make_batch,
    batched_gae,
)

T, N, S, A = 5, 4, 3, 2


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
