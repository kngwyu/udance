import chex
import haiku as hk
import jax
import jax.numpy as jnp
import pytest
import rlax

from brax.envs import create as create_brax_env

from udance import (
    Batch,
    Config,
    MusicDecoder,
    MusicEncoder,
    MusicIter,
    Policy,
    Rollout,
    batched_gae,
    batched_ppoclip_loss,
    load_emopia,
    make_batch,
    make_loss_fn,
    make_onestep_fn,
    make_rewardgen_fn,
)

T, N, S, A, E, Z = 5, 4, 3, 2, 6, 4


@pytest.fixture
def config() -> Config:
    return Config(hidden_dims=(8, 8), rnn_hidden_dims=(8, 8), music_latent_dim=Z)


def test_gae(config: Config) -> None:
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


def test_batched_ppo_loss(config: Config) -> None:
    prob_ratio = jax.random.normal(key=jax.random.PRNGKey(43), shape=(T, N))
    adv = jax.random.normal(key=jax.random.PRNGKey(44), shape=(T, N))
    policy_loss = batched_ppoclip_loss(prob_ratio, adv, 0.2)
    chex.assert_shape(policy_loss, (T,))


def test_music_encoder(config: Config) -> None:
    def forward(event, mask):
        encoder = MusicEncoder(E, config)
        latent, _ = encoder((event, mask), encoder.initial_state(N))
        return latent

    init, music_encoder = hk.transform(forward)
    event = jnp.ones((N,), dtype=jnp.int32)
    mask = jnp.zeros((N,))
    params = init(jax.random.PRNGKey(43), event, mask)
    latent = music_encoder(params, jax.random.PRNGKey(44), event, mask)
    chex.assert_shape(jax.tree_flatten(latent)[0], (N, Z))


def test_encode_and_decode(config: Config) -> None:
    def forward(event, mask):
        encoder = MusicEncoder(E, config)
        latent, _state = encoder((event, mask), encoder.initial_state(N))
        decoder = MusicDecoder(E, config)
        logits, _state = decoder((latent.latent, mask), decoder.initial_state(N))
        return logits

    init, encode_decode = hk.transform(forward)
    event = jnp.ones((N,), dtype=jnp.int32)
    mask = jnp.zeros((N,))
    params = init(jax.random.PRNGKey(43), event, mask)
    logits = encode_decode(params, jax.random.PRNGKey(44), event, mask)
    chex.assert_shape(logits, (N, E))


def test_reward_gen(config: Config) -> None:
    init, reward_gen = make_rewardgen_fn(S, config)
    obs = jnp.ones((T + 1, N, S))
    music_latent = jnp.ones((T, N, Z))
    params, state = init(jax.random.PRNGKey(43), obs, music_latent)
    rewards, new_state = reward_gen(params, state, obs, music_latent)
    chex.assert_shape(rewards, (T, N))


def test_batch(config: Config) -> Batch:
    obs = jnp.ones((N, S))
    event = jnp.ones((N,), dtype=jnp.int32)
    mask = jnp.zeros((N,))

    def forward():
        music_encoder = MusicEncoder(n_events=E, config=config)
        rnn_state = music_encoder.initial_state(N)
        music_latent, next_rnn_state = music_encoder((event, mask), rnn_state)
        policy = Policy(action_dim=A, config=config)
        policy_out = policy(obs, music_latent.latent)
        action = policy_out.policy.as_distrax().sample(seed=hk.next_rng_key())
        return action, policy_out, music_latent

    init, fwd = hk.transform(forward)
    params = init(jax.random.PRNGKey(43))
    rollout = Rollout(observations=[obs])
    for i in range(T):
        action, policy_out, music_latent = fwd(params, jax.random.PRNGKey(44 + i))
        rollout.append(
            observation=obs,
            music=event,
            action=action,
            terminal=jnp.zeros((N,), dtype=bool),
            output=policy_out,
            music_latent=music_latent.latent,
        )

    batch = make_batch(
        rollout,
        jnp.ones((N,)),
        jnp.ones((T, N)),
        config,
    )

    chex.assert_shape((batch.observation, batch.next_observation), (T, N, S))
    chex.assert_shape(batch.action, (T, N, A))
    chex.assert_shape(
        (batch.advantage, batch.value_target, batch.log_prob, batch.music),
        (T, N),
    )

    return batch


def test_onestep_fn(config: Config) -> None:
    env = create_brax_env(
        env_name="ant",
        episode_length=1000,
        action_repeat=1,
        auto_reset=True,
        batch_size=N,
    )
    init, onestep = make_onestep_fn(env=env, n_events=E, config=config)
    state = env.reset(jax.random.PRNGKey(44))
    music = jax.random.randint(jax.random.PRNGKey(45), shape=(4,), minval=0, maxval=E)
    params = init(
        jax.random.PRNGKey(46),
        state,
        music,
        None,
        jnp.zeros((4,), dtype=bool),
    )
    state, action, policy_out, music_latant, rnn_state = onestep(
        params,
        jax.random.PRNGKey(47),
        state,
        music,
        None,
        jnp.zeros((4,), dtype=bool),
    )
    A = env.action_size
    chex.assert_shape(
        (action, policy_out.policy.mean, policy_out.policy.stddev),
        (N, A),
    )
    chex.assert_shape(music_latant, (N, Z))


def test_music_iter() -> None:
    music, _ = load_emopia("/home/yuji/hpc-home/emopia", 500)
    music_iter = MusicIter(music[:8], n_workers=N, seed=0)
    first_music, first_ended = next(music_iter)
    chex.assert_shape(first_music, (N,))
    chex.assert_shape(first_ended, (N,))
    for ended in first_ended:
        assert not ended
    length = [len(music) for music in music[:8]]
    length.sort()
    for _ in range(min(length)):
        _, ended = next(music_iter)
    assert sum(ended) == 1


def test_loss_fn(config: Config) -> None:
    init, loss_fn = make_loss_fn(A, S, E, config)
    batch = test_batch(config)
    params = init(jax.random.PRNGKey(42), batch, 2.0)
    loss, metrics = loss_fn(params, jax.random.PRNGKey(41), batch, 2.0)
    chex.assert_shape([loss] + list(metrics.values()), ())
