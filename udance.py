"""Based on https://github.com/NTT123/wavernn-16bit
"""

import dataclasses
import functools
import typing as t

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import muspy
import numpy as np
import optax
import rlax

from brax.envs import Env, State as BraxState, create as create_brax_env
from brax.io import html as brax_html

np.Array = chex.Array
Array = t.Union[chex.Array, np.Array]

NetworkOutput = t.Any
Self = t.Any


@dataclasses.dataclass(frozen=True)
class Config:
    """Some hyper parameters required by PPO."""

    clip_epsilon: float = 0.2
    entropy_coef: float = 0.001
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_optim_epochs: int = 10
    n_minibatches: int = 1
    normalize_adv: bool = False
    reward_scaling: float = 1.0
    # Network config
    hidden_dims: t.Sequence[int] = (64, 64)
    rnn_hidden_dim: int = 64
    drop_prob: float = 0.5


class GaussianAndValue(t.NamedTuple):
    mu: chex.Array
    stddev: chex.Array
    value: chex.Array


def orthogonal(scale: float = 5.0 / 3.0) -> hk.initializers.Orthogonal:
    return hk.initializers.Orthogonal(scale=scale)


def mlp(
    hidden_dims: t.Sequence[int],
    last_dim: int,
    last_scale: float = 1.0,
) -> t.List[hk.Module]:
    layers = [
        hk.nets.MLP(
            hidden_dims,
            w_init=orthogonal(),
            activation=jax.nn.tanh,
            activate_final=True,
        ),
        hk.Linear(last_dim, w_init=orthogonal(scale=last_scale)),
    ]
    return hk.Sequential(layers)


class MLPPolicy(hk.Module):
    """MLP Gaussian policy with tunable stddev"""

    def __init__(self, action_dim: int, config: Config) -> None:
        super().__init__(name="diag_gaussian_pi_and_v")

        self._mu_net = mlp(config.hidden_dims, action_dim, 0.01)
        self._value_net = mlp(config.hidden_dims, 1, 1.0)
        self._logstd_param = hk.get_parameter(
            "logstd",
            (1, action_dim),
            init=lambda shape, dtype: jnp.zeros(shape, dtype),
        )

    def __call__(self, observation: Array) -> GaussianAndValue:
        mu = self._mu_net(observation)
        stddev = jnp.ones_like(mu) * jnp.exp(self._logstd_param)
        value = self._value_net(observation)
        return GaussianAndValue(mu, stddev, value)


class Normal(t.NamedTuple):
    mean: chex.Array
    stddev: chex.Array

    def as_distrax(self) -> distrax.MultivariateNormalDiag:
        return distrax.MultivariateNormalDiag(loc=self.mean, scale_diag=self.stddev)


class LogstdNormal(hk.Module):
    def __init__(
        self,
        output_dim: int,
        w_init: t.Optional[callable] = None,
        name: t.Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self._mean = hk.Linear(output_dim, w_init=w_init)
        self._logstd = hk.Linear(output_dim, w_init=w_init)

    def __call__(self, x: chex.Array) -> Normal:
        mean = self._mean(x)
        logstd = self._logstd(x)
        return Normal(mean=mean, stddev=jnp.exp(logstd * 0.5))


class MusicPolicyOutput(t.NamedTuple):
    policy: Normal
    value: chex.Array
    pred: Normal


class MusicPolicy(hk.RNNCore):
    """Recurrent policy conditioned by music"""

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        n_pitches: int,
        config: Config,
    ) -> None:
        super().__init__(name="music_policy")
        self._music_encoder = hk.Embed(
            vocab_size=n_pitches,
            embed_dim=config.rnn_hidden_dim,
            w_init=orthogonal(1.0),
        )
        self._obs_encoder = hk.Linear(config.rnn_hidden_dim, w_init=orthogonal(1.0))
        self._lstm = hk.LSTM(config.rnn_hidden_dim)
        self._pi_mean = mlp(config.hidden_dims, action_dim, 0.01)
        self._value = mlp(config.hidden_dims, 1, 1.0)
        self._logstd_param = hk.get_parameter(
            "logstd",
            (1, action_dim),
            init=lambda shape, dtype: jnp.zeros(shape, dtype),
        )
        self._pi_logstd = hk.get_parameter(
            "logstd",
            (1, action_dim),
            init=lambda shape, dtype: jnp.zeros(shape, dtype),
        )
        self._pre_pred = hk.nets.MLP(
            config.hidden_dims,
            w_init=orthogonal(),
            activation=jax.nn.tanh,
            activate_final=True,
        )
        self._pred = LogstdNormal(obs_dim, w_init=hk.initializers.Orthogonal(scale=1.0))
        self._drop_prob = config.drop_prob

    def initial_state(self, batch_size: t.Optional[int]) -> hk.LSTMState:
        return self._lstm.initial_state(batch_size)

    def __call__(
        self,
        inputs: t.Tuple[chex.Array, chex.Array, chex.Array],
        prev_state: t.Optional[hk.LSTMState] = None,
    ) -> t.Tuple[MusicPolicyOutput, hk.LSTMState]:
        obs, music, rnn_mask = inputs
        if prev_state is None:
            prev_state = self.initial_state(obs.shape[0])
        obs_encoded = self._obs_encoder(obs)
        music_encoded = self._music_encoder(music)
        music_encoded_dropped = hk.dropout(
            hk.next_rng_key(),
            self._drop_prob,
            music_encoded,
        )
        post_rnn, next_state = self._lstm(
            jnp.concatenate((obs_encoded, music_encoded_dropped), axis=1),
            jax.tree_map(lambda x: x * rnn_mask, prev_state),
        )
        post_rnn_dropped = hk.dropout(hk.next_rng_key(), self._drop_prob, post_rnn)
        pi_mean = self._pi_mean(post_rnn_dropped)
        pi_std = jnp.ones_like(pi_mean) * jnp.exp(self._logstd_param)
        value = self._value(post_rnn_dropped)
        pred = self._pred(self._pre_pred(post_rnn_dropped))
        return MusicPolicyOutput(Normal(pi_mean, pi_std), value, pred), next_state


class ObsPredictor(hk.RNNCore):
    """Predict the next state"""

    def __init__(self, obs_dim: int, config: Config) -> None:
        super().__init__(name="obs_predictor")
        self._obs_encoder = hk.Linear(config.rnn_hidden_dim, w_init=orthogonal(1.0))
        self._lstm = hk.LSTM(config.rnn_hidden_dim)
        self._mlp = hk.nets.MLP(
            config.hidden_dims,
            w_init=orthogonal(),
            activation=jax.nn.tanh,
            activate_final=True,
        )
        self._pred = LogstdNormal(obs_dim, w_init=hk.initializers.Orthogonal(scale=1.0))

    def initial_state(self, batch_size: t.Optional[int]) -> hk.LSTMState:
        return self._lstm.initial_state(batch_size)

    def __call__(
        self,
        inputs: t.Tuple[chex.Array, chex.Array, chex.Array],
        prev_state: t.Optional[hk.LSTMState] = None,
    ) -> t.Tuple[Normal, hk.LSTMState]:
        obs, rnn_mask = inputs
        if prev_state is None:
            prev_state = self.initial_state(obs.shape[0])
        obs_encoded = self._obs_encoder(obs)
        post_rnn, next_state = self._lstm(
            obs_encoded,
            jax.tree_map(lambda x: x * rnn_mask, prev_state),
        )
        pred = self._pred(self._mlp(post_rnn))
        return pred, next_state


def sample_minibatch_indices(
    n_instances: int,
    n_minibatches: int,
    rng_key: chex.Array,
) -> t.Iterable[chex.Array]:
    indices = jax.random.permutation(rng_key, n_instances)
    minibatch_size = n_instances // n_minibatches
    for start in range(0, n_instances, minibatch_size):
        yield indices[start : start + minibatch_size]


@functools.partial(jax.jit, static_argnums=2)
def _gae_impl(
    r_t: chex.Array,
    discount_t: chex.Array,
    lambda_: float,
    values: chex.Array,
) -> chex.Array:
    chex.assert_rank([r_t, values, discount_t], 1)
    chex.assert_type([r_t, values, discount_t], float)
    lambda_ = jnp.ones_like(discount_t) * lambda_
    delta_t = r_t + discount_t * values[1:] - values[:-1]
    n = delta_t.shape[0]

    def update(i: int, advantage_t: chex.Array) -> chex.Array:
        t_ = n - i - 1
        adv_t = delta_t[t_] + lambda_[t_] * discount_t[t_] * advantage_t[t_ + 1]
        return jax.ops.index_update(advantage_t, t_, adv_t)

    advantage_t = jax.lax.fori_loop(0, n, update, jnp.zeros(n + 1))
    return advantage_t[:-1]


batched_gae = jax.vmap(_gae_impl, in_axes=(1, 1, None, 1), out_axes=1)


@dataclasses.dataclass
class RolloutWithMusic:
    observations: t.List[chex.Array]
    musics: t.List[chex.Array] = dataclasses.field(default_factory=list)
    actions: t.List[chex.Array] = dataclasses.field(default_factory=list)
    terminals: t.List[chex.Array] = dataclasses.field(default_factory=list)
    policy_outputs: t.List[
        t.Tuple[MusicPolicyOutput, hk.LSTMState]
    ] = dataclasses.field(default_factory=list)
    predictor_outputs: t.List[t.Tuple[Normal, hk.LSTMState]] = dataclasses.field(
        default_factory=list
    )

    def append(
        self,
        *,
        observation: chex.Array,
        music: chex.Array,
        action: chex.Array,
        policy_out: MusicPolicyOutput,
        predictor_out: Normal,
        terminal: chex.Array,
    ) -> None:
        self.observations.append(observation)
        self.musics.append(music)
        self.actions.append(action)
        self.policy_outputs.append(policy_out)
        self.predictor_outputs.append(predictor_out)
        self.terminals.append(terminal)

    def last_obs(self) -> chex.Array:
        assert len(self.observations) == len(self.actions) + 1
        return self.observations[-1]

    def clear(self) -> None:
        self.observations = [self.last_obs()]
        self.musics.clear()
        self.actions.clear()
        self.policy_outputs.clear()
        self.predictor_outputs.clear()
        self.terminals.clear()


class MusicBatch(t.NamedTuple):
    observation: chex.Array
    next_observation: chex.Array
    music: chex.Array
    action: chex.Array
    reward: chex.Array
    advantage: chex.Array
    value_target: chex.Array
    log_prob: chex.Array

    def __getitem__(self, idx: Array) -> Self:
        return self.__class__(
            observation=self.observation[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            advantage=self.advantage[idx],
            value_target=self.value_target[idx],
            log_prob=self.log_prob[idx],
        )


def _make_music_batch(
    rollout: RolloutWithMusic,
    next_value: chex.Array,
    config: Config,
) -> MusicBatch:
    action = jnp.concatenate(rollout.actions)
    policy_outputs = jax.tree_map(lambda *x: jnp.concatenate(x), *rollout.outputs)
    log_prob = policy_outputs.as_distrax().log_prob(action)
    reward = jnp.stack(rollout.rewards) * config.reward_scaling
    mask = 1.0 - jnp.stack(rollout.terminals)
    value = jnp.concatenate(
        (value.reshape(reward.shape), next_value.reshape(1, -1)),
        axis=0,
    )
    advantage = batched_gae(reward, mask * config.gamma, config.gae_lambda, value)
    value_target = advantage + value[:-1]
    if config.normalize_adv:
        advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)
    return Batch(
        observation=jnp.concatenate(rollout.observations[:-1]),
        action=action,
        reward=jnp.ravel(reward),
        advantage=jnp.ravel(advantage),
        value_target=jnp.ravel(value_target),
        log_prob=log_prob,
    )


@dataclasses.dataclass
class RolloutResult:
    """
    Required experiences for PPO.
    """

    observations: t.List[chex.Array]
    actions: t.List[chex.Array] = dataclasses.field(default_factory=list)
    rewards: t.List[chex.Array] = dataclasses.field(default_factory=list)
    terminals: t.List[chex.Array] = dataclasses.field(default_factory=list)
    outputs: t.List[GaussianAndValue] = dataclasses.field(default_factory=list)

    def append(
        self,
        *,
        observation: chex.Array,
        action: chex.Array,
        reward: chex.Array,
        output: GaussianAndValue,
        terminal: chex.Array,
    ) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.outputs.append(output)
        self.terminals.append(terminal)

    def last_obs(self) -> chex.Array:
        assert len(self.observations) == len(self.actions) + 1
        return self.observations[-1]

    def clear(self) -> None:
        self.observations = [self.last_obs()]
        self.actions.clear()
        self.rewards.clear()
        self.outputs.clear()
        self.terminals.clear()


class Batch(t.NamedTuple):
    """Batch for PPO, also used as minibatch by indexing."""

    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    advantage: chex.Array
    value_target: chex.Array
    log_prob: chex.Array

    def __getitem__(self, idx: Array) -> Self:
        return self.__class__(
            observation=self.observation[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            advantage=self.advantage[idx],
            value_target=self.value_target[idx],
            log_prob=self.log_prob[idx],
        )


def _make_batch(
    rollout: RolloutResult,
    next_value: chex.Array,
    config: Config,
) -> Batch:
    action = jnp.concatenate(rollout.actions)
    mu, stddev, value = jax.tree_map(lambda *x: jnp.concatenate(x), *rollout.outputs)
    log_prob = distrax.MultivariateNormalDiag(mu, stddev).log_prob(action)
    reward = jnp.stack(rollout.rewards) * config.reward_scaling
    mask = 1.0 - jnp.stack(rollout.terminals)
    value = jnp.concatenate(
        (value.reshape(reward.shape), next_value.reshape(1, -1)),
        axis=0,
    )
    advantage = batched_gae(reward, mask * config.gamma, config.gae_lambda, value)
    value_target = advantage + value[:-1]
    if config.normalize_adv:
        advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)
    return Batch(
        observation=jnp.concatenate(rollout.observations[:-1]),
        action=action,
        reward=jnp.ravel(reward),
        advantage=jnp.ravel(advantage),
        value_target=jnp.ravel(value_target),
        log_prob=log_prob,
    )


class Actor:
    def __init__(
        self,
        *,
        env: Env,
        config: Config,
        net_init_key: chex.PRNGKey,
        initial_state: chex.Array,
    ) -> None:
        init, step_transformed = self.make_step_fn(env)
        self.params = init(net_init_key, initial_state)
        self._step_impl = jax.jit(step_transformed)

    @staticmethod
    def make_step_fn(env: Env) -> hk.Transformed:
        def step_impl(
            state: BraxState,
        ) -> t.Tuple[BraxState, chex.Array, GaussianAndValue]:
            output = MLPPolicy(action_dim=env.action_size, config=Config)(state.obs)
            policy = distrax.MultivariateNormalDiag(output.mu, output.stddev)
            action = policy.sample(seed=hk.next_rng_key())
            state = env.step(state, jnp.tanh(action))
            return state, action, output

        return hk.transform(step_impl)

    def step(
        self,
        prng_key: chex.PRNGKey,
        state: BraxState,
    ) -> t.Tuple[chex.Array, GaussianAndValue]:
        return self._step_impl(self.params, prng_key, state)


class Learner:
    def __init__(
        self,
        *,
        action_dim: int,
        config: Config,
        actor: Actor,
        optimizer: optax.GradientTransformation,
    ) -> None:
        self._config = config
        self._actor = actor
        self._opt_state = optimizer.init(actor.params)
        self._opt_update = optimizer.update
        fwd = lambda obs: MLPPolicy(action_dim=action_dim, config=config)(obs)  # noqa
        _, self._pi_and_v = hk.without_apply_rng(hk.transform(fwd))

    @functools.partial(jax.jit, static_argnums=0)
    def _next_value(self, params: hk.Params, last_obs: Array) -> chex.Array:
        _, _, next_value = jax.lax.stop_gradient(self._pi_and_v(params, last_obs))
        return next_value

    def learn(
        self,
        rollout_result: RolloutResult,
        prng_seq: hk.PRNGSequence,
    ) -> Batch:
        next_value = self._next_value(self._actor.params, rollout_result.last_obs())
        batch = _make_batch(rollout_result, next_value, self._config)
        n_batches = batch.reward.shape[0]
        for _ in range(self._config.n_optim_epochs):
            indices_iter = sample_minibatch_indices(
                n_batches,
                self._config.n_minibatches,
                next(prng_seq),
            )
            for indices in indices_iter:
                self._actor.params, self._opt_state = self._update(
                    self._actor.params,
                    self._opt_state,
                    batch[indices],
                )
        return batch

    @functools.partial(jax.jit, static_argnums=0)
    def _update(
        self,
        params: hk.Params,
        opt_state: optax.OptState,
        ppo_batch: Batch,
    ) -> t.Tuple[hk.Params, optax.OptState]:
        g = jax.grad(self._loss)(params, ppo_batch)
        updates, new_opt_state = self._opt_update(g, opt_state)
        return optax.apply_updates(params, updates), new_opt_state

    def _loss(self, params: hk.Params, batch: Batch) -> chex.Array:
        mu, stddev, value = self._pi_and_v(params, batch.observation)

        # Policy loss
        policy = distrax.Normal(mu, stddev)
        log_prob = jnp.sum(policy.log_prob(batch.action), axis=-1)
        prob_ratio = jnp.exp(log_prob - batch.log_prob)
        policy_loss = rlax.clipped_surrogate_pg_loss(
            prob_ratio,
            batch.advantage,
            self._config.clip_epsilon,
            use_stop_gradient=False,
        )

        # Value loss
        value_loss = jnp.mean(rlax.l2_loss(value, batch.value_target))

        # Entropy regularization
        entropy_mean = jnp.mean(jnp.sum(policy.entropy(), axis=-1))
        return policy_loss + value_loss - self._config.entropy_coef * entropy_mean


def _musics_to_one_hots(musics: t.List[muspy.Music]) -> t.List[chex.Array]:
    pitches = [muspy.to_pitch_representation(music) for music in musics]
    min_value = min([np.min(pitch) for pitch in pitches])
    return [jnp.array(pitch - min_value) for pitch in pitches]


def load_jsb(root: str) -> t.List[chex.Array]:
    jsb = muspy.JSBChoralesDataset(root, download_and_extract=True).convert()
    return _musics_to_one_hots(jsb)


def test_ppo() -> None:
    env = create_brax_env(
        env_name="ant",
        episode_length=1000,
        action_repeat=1,
        auto_reset=True,
        batch_size=16,
    )
    eval_env = create_brax_env(
        env_name="ant",
        episode_length=1000,
        action_repeat=1,
        auto_reset=True,
        batch_size=1,
    )
    prng_seq = hk.PRNGSequence(0)
    state = env.reset(rng=next(prng_seq))
    config = Config()
    actor = Actor(
        env=env,
        config=config,
        net_init_key=next(prng_seq),
        initial_state=state,
    )
    eval_step = jax.jit(actor.make_step_fn(eval_env)[1])
    learner = Learner(
        action_dim=env.action_size,
        config=config,
        actor=actor,
        optimizer=optax.adam(3e-4, eps=1e-4),
    )
    rollout = RolloutResult([state.obs])
    for i in range(1000):
        for _ in range(256):
            state, act, out = actor.step(next(prng_seq), state)
            rollout.append(
                observation=state.obs,
                action=act,
                reward=state.reward,
                output=out,
                terminal=state.done,
            )
        batch = learner.learn(rollout, prng_seq)
        print(f"Step {i} Avg. Reward: {jnp.mean(batch.reward)}")
        if i % 100 == 0:
            eval_state = eval_env.reset(next(prng_seq))
            qps = [jax.tree_map(lambda x: x.reshape(x.shape[1:]), eval_state.qp)]
            while eval_state.done[0] is False:
                eval_state, _, _ = eval_step(actor.params, next(prng_seq), eval_state)
                qps.append(
                    jax.tree_map(lambda x: x.reshape(x.shape[1:]), eval_state.qp)
                )
                with open("viewer.html", "w") as f:
                    f.write(brax_html.render(env.sys, qps))
            if i // 100 == 1:
                print("Open viewer.html for checking evaluated behavior!")
        rollout.clear()


if __name__ == "__main__":
    import typer

    app = typer.Typer()
    app.command()(test_audio_processing)
    app.command()(test_ppo)
    app()
