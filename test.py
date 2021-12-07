import chex
import haiku as hk
import jax
import jax.numpy as jnp
import rlax

from brax.envs import create as create_brax_env

from udance import (
    Config,
    MusicPolicy,
    ObsPredictor,
    RolloutWithMusic,
    _make_music_batch,
    batched_gae,
    make_onestep_fn,
)

T, N, S, A, P = 5, 4, 3, 2, 6


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
    init, obs_predictor = hk.transform(lambda x: ObsPredictor(S, config)(x))
    obs = jnp.ones((N, S))
    mask = jnp.zeros((N, 1))
    params = init(jax.random.PRNGKey(43), (obs, mask))
    output, state = obs_predictor(params, jax.random.PRNGKey(44), (obs, mask))
    chex.assert_shape((output.mean, output.stddev), (N, S))


def test_music_batch() -> None:
    config = Config()
    init_mp, music_policy = hk.transform(
        lambda x, state: MusicPolicy(A, S, P, config)(x, state)
    )
    init_op, obs_predictor = hk.without_apply_rng(
        hk.transform(lambda x, state: ObsPredictor(S, config)(x, state))
    )
    obs = jnp.ones((N, S))
    pitch = jnp.ones((N,), dtype=jnp.int32)
    mask = jnp.zeros((N, 1))
    params_m = init_mp(jax.random.PRNGKey(43), (obs, pitch, mask), None)
    params_o = init_op(jax.random.PRNGKey(44), (obs, mask), None)
    rollout = RolloutWithMusic(observations=[obs])
    policy_hidden, predictor_hidden = None, None
    for i in range(T):
        policy_out, policy_hidden = music_policy(
            params_m,
            jax.random.PRNGKey(45 + i),
            (obs, pitch, mask),
            policy_hidden,
        )
        predictor_out, predictor_hidden = obs_predictor(
            params_o,
            (obs, mask),
            predictor_hidden,
        )
        rollout.append(
            observation=obs,
            music=pitch,
            action=jnp.ones((N, A)),
            policy_out=policy_out,
            policy_hidden=policy_hidden,
            predictor_out=predictor_out,
            predictor_hidden=predictor_hidden,
            terminal=jnp.zeros((N,), dtype=bool),
        )

    batch = _make_music_batch(rollout, jnp.ones((N,)), config)
    chex.assert_shape(batch.observation, (T, N, S))
    chex.assert_shape(batch.action, (T, N, A))
    chex.assert_shape(
        (batch.advantage, batch.value_target, batch.log_prob, batch.music),
        (T, N),
    )


def test_onestep_fn() -> None:
    env = create_brax_env(
        env_name="ant",
        episode_length=1000,
        action_repeat=1,
        auto_reset=True,
        batch_size=N,
    )
    config = Config()
    init, onestep = make_onestep_fn(env=env, n_pitches=P, config=config)
    state = env.reset(jax.random.PRNGKey(44))
    music = jax.random.randint(jax.random.PRNGKey(45), shape=(4,), minval=0, maxval=P)
    params = init(
        jax.random.PRNGKey(46),
        state,
        None,
        None,
        music,
        jnp.zeros((4, 1), dtype=bool),
    )
    state, action, policy_out, policy_state, predictor_out, predictor_state = onestep(
        params,
        jax.random.PRNGKey(47),
        state,
        None,
        None,
        music,
        jnp.zeros((4, 1), dtype=bool),
    )
    S, A = env.observation_size, env.action_size
    chex.assert_shape((action, *jax.tree_flatten(policy_out.policy)[0]), (N, A))
    chex.assert_shape(
        jax.tree_flatten(policy_out.pred)[0] + jax.tree_flatten(predictor_out)[0],
        (N, S),
    )
