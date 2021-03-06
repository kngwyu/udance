import collections
import dataclasses
import functools
import json
import pathlib
import pickle
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
import tqdm

from brax import QP
from brax.envs import Env, State as BraxState, create as create_brax_env
from brax.io import html as brax_html

np.Array = chex.Array
Array = t.Union[chex.Array, np.Array]

NetworkOutput = t.Any
Self = t.Any

batched_ppoclip_loss = jax.vmap(
    functools.partial(rlax.clipped_surrogate_pg_loss, use_stop_gradient=False),
    in_axes=(0, 0, None),
    out_axes=0,
)


@dataclasses.dataclass(frozen=True)
class Config:
    """Some hyper parameters required by PPO."""

    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_optim_epochs: int = 10
    n_minibatches: int = 4
    reward_clip: float = 100.0
    # Network config
    hidden_dims: t.Sequence[int] = (64, 64)
    rnn_hidden_dims: t.Sequence[int] = (128, 64)
    music_latent_dim: int = 16


def orthogonal(scale: float = 5.0 / 3.0) -> hk.initializers.Orthogonal:
    return hk.initializers.Orthogonal(scale=scale)


def mlp(hidden_dims: t.Sequence[int], last_dim: int) -> t.List[hk.Module]:
    layers = [
        hk.nets.MLP(
            hidden_dims,
            w_init=orthogonal(),
            activation=jax.nn.tanh,
            activate_final=True,
        ),
        hk.Linear(last_dim, w_init=orthogonal(scale=1.0)),
    ]
    return hk.Sequential(layers)


class Normal(t.NamedTuple):
    mean: chex.Array
    stddev: chex.Array

    def as_distrax(self) -> distrax.MultivariateNormalDiag:
        return distrax.MultivariateNormalDiag(loc=self.mean, scale_diag=self.stddev)

    def sample(self) -> chex.Array:
        """Only callable from hk.transform"""
        return self.as_distrax().sample(seed=hk.next_rng_key())

    def kld_to_std(self) -> chex.Array:
        prior = distrax.MultivariateNormalDiag(
            loc=jnp.zeros_like(self.mean),
            scale_diag=jnp.ones_like(self.stddev),
        )
        return self.as_distrax().kl_divergence(prior)


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


class MusicLatent(t.NamedTuple):
    dist: Normal
    latent: chex.Array


MusicRNNState = t.Tuple[hk.LSTMState, hk.LSTMState]


class MusicEncoder(hk.RNNCore):
    """Encode a sequence of music to a latent vector z"""

    def __init__(self, n_events: int, config: Config) -> None:
        super().__init__(name="music_encoder")
        self._encoder = hk.Embed(
            vocab_size=n_events,
            embed_dim=config.rnn_hidden_dims[0],
            w_init=orthogonal(np.sqrt(2.0)),
        )
        self._lstm1 = hk.LSTM(config.rnn_hidden_dims[0])
        self._lstm2 = hk.LSTM(config.rnn_hidden_dims[1])
        self._latent = LogstdNormal(
            output_dim=config.music_latent_dim,
            w_init=orthogonal(1.0),
        )

    def initial_state(self, batch_size: t.Optional[int]) -> MusicRNNState:
        s1 = self._lstm1.initial_state(batch_size)
        s2 = self._lstm2.initial_state(batch_size)
        return s1, s2

    def __call__(
        self,
        inputs: t.Tuple[chex.Array, chex.Array],
        prev_state: MusicRNNState,
    ) -> t.Tuple[MusicLatent, MusicRNNState]:
        x, mask = inputs
        state1, state2 = jax.tree_map(lambda x: x * mask.reshape(-1, 1), prev_state)
        x = self._encoder(x)
        x = jax.nn.relu(x)
        x, next_state1 = self._lstm1(x, state1)
        x = jax.nn.relu(x)
        x, next_state2 = self._lstm2(x, state2)
        dist = self._latent(x)
        return MusicLatent(dist, dist.sample()), (next_state1, next_state2)


class MusicDecoder(hk.RNNCore):
    def __init__(self, n_events: int, config: Config) -> None:
        super().__init__(name="music_decoder")
        self._lstm = hk.LSTM(config.rnn_hidden_dims[0])
        self._decoder = hk.Linear(n_events, w_init=orthogonal(1.0))

    def initial_state(self, batch_size: t.Optional[int]) -> hk.LSTMState:
        return self._lstm.initial_state(batch_size)

    def __call__(
        self,
        inputs: t.Tuple[chex.Array, chex.Array],
        prev_state: hk.LSTMState,
    ) -> t.Tuple[MusicLatent, hk.LSTMState]:
        latent, mask = inputs
        state = jax.tree_map(lambda x: x * mask.reshape(-1, 1), prev_state)
        x, next_state = self._lstm(latent, state)
        logit = self._decoder(x)
        return logit, next_state


class PolicyOutput(t.NamedTuple):
    policy: Normal
    value: chex.Array


class Policy(hk.Module):
    """Gaussian policy conditioned by music"""

    def __init__(self, action_dim: int, config: Config) -> None:
        super().__init__(name="policy")
        self._pi_mean = mlp(config.hidden_dims, action_dim)
        self._pi_logstd = hk.get_parameter(
            "logstd",
            (1, action_dim),
            init=lambda shape, dtype: jnp.zeros(shape, dtype),
        )
        self._value = mlp(config.hidden_dims, 1)

    def __call__(self, obs: chex.Array, music_latent: chex.Array) -> PolicyOutput:
        x = jnp.concatenate((obs, music_latent), axis=-1)
        pi_mean = self._pi_mean(x)
        pi_std = jnp.ones_like(pi_mean) * jnp.exp(self._pi_logstd)
        policy = Normal(pi_mean, pi_std)
        value = self._value(x)
        return PolicyOutput(policy, value)


class Predictor(hk.Module):
    """Predict next state with Gaussian"""

    def __init__(self, observation_dim: int, config: Config) -> None:
        super().__init__(name="prediction")
        prediction = [
            hk.nets.MLP(
                config.hidden_dims,
                w_init=orthogonal(),
                activation=jax.nn.tanh,
                activate_final=True,
            ),
            LogstdNormal(observation_dim, w_init=orthogonal(1.0)),
        ]
        self._prediction = hk.Sequential(prediction)

    def __call__(self, obs: chex.Array, music_latent: chex.Array) -> Normal:
        x = jnp.concatenate((obs, music_latent), axis=-1)
        return self._prediction(x)


def sample_workers(
    n_workers: int,
    n_minibatches: int,
    rng_key: chex.Array,
) -> t.Iterable[chex.Array]:
    assert n_workers % n_minibatches == 0
    indices = jax.random.permutation(rng_key, n_workers)
    n_workers_per_batch = n_workers // n_minibatches
    for start in range(0, n_workers, n_workers_per_batch):
        yield indices[start : start + n_workers_per_batch]


@jax.jit
def _gae_impl(
    r_t: chex.Array,
    discount_t: chex.Array,
    lambda_: float,
    values: chex.Array,
) -> chex.Array:
    """A fast implementation of Generalized Advantage Estimator"""
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


# Here I assume that value and reward have (T, N) shape
batched_gae = jax.vmap(_gae_impl, in_axes=(1, 1, None, 1), out_axes=1)


@chex.dataclass
class Rollout:
    """A container class that holds the result of N-step rollout"""

    observations: t.List[chex.Array]
    musics: t.List[chex.Array] = dataclasses.field(default_factory=list)
    actions: t.List[chex.Array] = dataclasses.field(default_factory=list)
    terminals: t.List[chex.Array] = dataclasses.field(default_factory=list)
    outputs: t.List[PolicyOutput] = dataclasses.field(default_factory=list)
    music_latents: t.List[chex.Array] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.terminals.append(jnp.zeros(self.observations[0].shape[0], dtype=bool))

    def append(
        self,
        *,
        observation: chex.Array,
        music: chex.Array,
        action: chex.Array,
        terminal: chex.Array,
        output: chex.Array,
        music_latent: chex.Array,
    ) -> None:
        self.observations.append(observation)
        self.musics.append(music)
        self.actions.append(action)
        self.terminals.append(terminal)
        self.outputs.append(output)
        self.music_latents.append(music_latent)

    def clear(self) -> None:
        self.observations = [self.observations[-1]]
        self.terminals = [self.terminals[-1]]
        self.musics.clear()
        self.actions.clear()
        self.outputs.clear()
        self.music_latents.clear()


class Batch(t.NamedTuple):
    observation: chex.Array
    next_observation: chex.Array
    music: chex.Array
    action: chex.Array
    advantage: chex.Array
    value_target: chex.Array
    log_prob: chex.Array
    mask: chex.Array
    prev_mask: chex.Array

    def __getitem__(self, idx: Array) -> Self:
        return self.__class__(
            observation=self.observation[:, idx],
            next_observation=self.next_observation[:, idx],
            music=self.music[:, idx],
            action=self.action[:, idx],
            advantage=self.advantage[:, idx],
            value_target=self.value_target[:, idx],
            log_prob=self.log_prob[:, idx],
            mask=self.mask[:, idx],
            prev_mask=self.prev_mask[:, idx],
        )


def dummy_batch(observation_dim: int, action_dim: int) -> Batch:
    return Batch(
        observation=jnp.ones((2, 1, observation_dim)),
        next_observation=jnp.ones((2, 1, observation_dim)),
        music=jnp.ones((2, 1), dtype=jnp.int32),
        action=jnp.ones((2, 1, action_dim)),
        advantage=jnp.ones((2, 1)),
        value_target=jnp.ones((2, 1)),
        log_prob=jnp.ones((2, 1)),
        mask=jnp.ones((2, 1)),
        prev_mask=jnp.ones((2, 1)),
    )


def make_batch(
    rollout: Rollout,
    next_value: chex.Array,
    reward: chex.Array,
    config: Config,
) -> t.Tuple[Batch, chex.Array, chex.Array]:
    # Observation
    observation = jnp.stack(rollout.observations)  # T + 1, N, obs-dim
    # Compute advantage
    mask = 1.0 - jnp.stack(rollout.terminals[1:])
    policy_outputs = jax.tree_map(lambda *x: jnp.stack(x), *rollout.outputs)
    value_t = policy_outputs.value.reshape(reward.shape)
    value_t1 = jnp.concatenate((value_t, next_value.reshape(1, -1)), axis=0)
    advantage = batched_gae(reward, mask * config.gamma, config.gae_lambda, value_t1)
    value_target = advantage + value_t
    # Log ??
    action = jnp.stack(rollout.actions)
    log_prob = policy_outputs.policy.as_distrax().log_prob(action)
    batch = Batch(
        observation=observation[:-1],
        next_observation=observation[1:],
        music=jnp.stack(rollout.musics),
        action=action,
        advantage=advantage,
        value_target=value_target,
        log_prob=log_prob,
        mask=mask,
        prev_mask=1.0 - jnp.stack(rollout.terminals[:-1]),
    )
    return batch


def make_rewardgen_fn(observation_dim: int, config: Config) -> hk.Transformed:
    def _update_rms(
        mean: chex.Array,
        var: chex.Array,
        count: float,
        batch: chex.Array,
    ) -> t.Tuple[chex.Array, chex.Array, float]:
        batch_mean = jnp.mean(batch)
        batch_size = batch.shape[0]
        delta = batch_mean - mean
        total_count = count + batch_size
        new_mean = mean + delta * batch_size / total_count
        m_a = var * count
        m_b = jnp.var(batch) * batch_size
        m2 = m_a + m_b + jnp.square(delta) * count * batch_size / total_count
        new_var = m2 / total_count
        return new_mean, new_var, total_count

    def gen_rewards(
        observation: chex.Array,
        music_latent: chex.Array,
    ) -> t.Tuple[chex.Array, chex.Array]:
        """log q(x'|x, z) - ??z' log q(x'|x, z')"""
        chex.assert_rank((observation, music_latent), 3)  # T + 1, N, O/H
        n_steps, n_workers, _ = music_latent.shape
        predictor = Predictor(observation_dim=observation_dim, config=config)
        # First, compute ??z' log q(x'|x, z')
        obs_expanded = jnp.repeat(observation[:-1], n_workers, 1)
        music_expanded = jnp.tile(music_latent, (1, n_workers, 1))
        dist_expanded = predictor(obs_expanded, music_expanded)
        obs_next_expanded = jnp.repeat(observation[1:], n_workers, 1)
        logq_expanded = dist_expanded.as_distrax().log_prob(obs_next_expanded)
        logq_sum = jnp.sum(logq_expanded.reshape(n_steps, n_workers, n_workers), axis=2)
        # Then compute log q(x'|x, z)
        dist = predictor(observation[:-1], music_latent)
        logq = dist.as_distrax().log_prob(observation[1:])
        raw_reward = logq - logq_sum + jnp.log(n_workers)
        # Normalize reward
        reward_mean = hk.get_state("reward_mean", shape=(), init=jnp.zeros)
        reward_var = hk.get_state("reward_var", shape=(), init=jnp.ones)
        reward_count = hk.get_state(
            "reward_count",
            shape=(),
            init=lambda *args: jnp.ones(*args) * 1e-4,
        )
        new_mean, new_var, new_count = _update_rms(
            reward_mean,
            reward_var,
            reward_count,
            raw_reward,
        )
        hk.set_state("reward_mean", new_mean)
        hk.set_state("reward_var", new_var)
        hk.set_state("reward_count", new_count)
        std = jnp.sqrt(new_var + 1e-8)
        normalized_reward = (raw_reward - new_mean) / std
        clipped_reward = jnp.clip(
            normalized_reward,
            a_min=-config.reward_clip,
            a_max=config.reward_clip,
        )
        return clipped_reward, raw_reward

    return hk.without_apply_rng(hk.transform_with_state(gen_rewards))


StepFn = t.Callable[
    [BraxState, chex.Array, t.Optional[MusicRNNState], chex.Array],
    t.Tuple[BraxState, chex.Array, PolicyOutput, chex.Array, MusicRNNState],
]


def make_onestep_fn(
    env: Env,
    n_events: int,
    config: Config,
) -> hk.Transformed:
    def step_impl(
        state: BraxState,
        music: chex.Array,
        rnn_state: t.Optional[MusicRNNState],
        prev_terminal: chex.Array,
    ) -> t.Tuple[BraxState, chex.Array, PolicyOutput, chex.Array, MusicRNNState]:
        # 1. Encode music
        mask = 1.0 - prev_terminal
        music_encoder = MusicEncoder(n_events=n_events, config=config)
        if rnn_state is None:
            rnn_state = music_encoder.initial_state(music.shape[0])
        music_latent, next_rnn_state = music_encoder((music, mask), rnn_state)

        # 2. Get policy and value
        policy = Policy(action_dim=env.action_size, config=config)
        policy_out = policy(state.obs, music_latent.latent)
        action = policy_out.policy.as_distrax().sample(seed=hk.next_rng_key())
        resetted_state = env.reset(hk.next_rng_key())

        # 3. Call env.step and get the next state
        def reset_if(old: chex.Array, new: chex.Array) -> chex.Array:
            terminal = prev_terminal.reshape((old.shape[0],) + (1,) * (old.ndim - 1))
            return jnp.where(terminal, new, old)

        state = jax.tree_map(reset_if, state, resetted_state)
        state = env.step(state, action)
        return state, action, policy_out, music_latent.latent, next_rnn_state

    return jax.tree_map(jax.jit, hk.transform(step_impl))


@chex.dataclass
class Actor:
    """Actor holds network parameters and rnn states"""

    params: hk.Params
    rnn_state: MusicRNNState
    step_fn: StepFn

    def step(
        self,
        prng_key: chex.PRNGKey,
        state: BraxState,
        music: chex.Array,
        prev_terminal: chex.Array,
    ) -> t.Tuple[BraxState, chex.Array, PolicyOutput, chex.Array]:
        state, action, policy_out, music_latent, self.rnn_state = self.step_fn(
            self.params,
            prng_key,
            state,
            music,
            self.rnn_state,
            prev_terminal,
        )
        return state, action, policy_out, music_latent


def make_loss_fn(
    action_dim: int,
    observation_dim: int,
    n_events: int,
    config: Config,
) -> hk.Transformed:
    def loss_fn(
        batch: Batch,
        beta: float,
    ) -> t.Tuple[chex.Array, t.Dict[str, chex.Array]]:
        # 1. Encode music
        music_encoder = MusicEncoder(n_events=n_events, config=config)
        music_latents, _ = hk.dynamic_unroll(
            music_encoder,
            (batch.music, batch.mask),
            music_encoder.initial_state(batch.observation.shape[1]),
        )
        latent_kl = jnp.mean(music_latents.dist.kld_to_std())
        # 2. Decode music
        music_decoder = MusicDecoder(n_events=n_events, config=config)
        music_logits, _ = hk.dynamic_unroll(
            music_decoder,
            (music_latents.latent, batch.prev_mask),
            music_decoder.initial_state(batch.observation.shape[1]),
            reverse=True,
        )
        music_distrib = distrax.Categorical(logits=music_logits)
        music_nll = -jnp.mean(music_distrib.log_prob(batch.music))
        # 3. Compute poliy loss
        policy = Policy(action_dim=action_dim, config=config)
        policy_outputs = policy(batch.observation, music_latents.latent)
        policy = policy_outputs.policy.as_distrax()
        log_prob = policy.log_prob(batch.action)
        prob_ratio = jnp.exp(log_prob - batch.log_prob)
        policy_loss = batched_ppoclip_loss(
            prob_ratio,
            batch.advantage,
            config.clip_epsilon,
        )
        policy_loss = jnp.mean(policy_loss)
        # 4. Compute value loss
        value = policy_outputs.value.reshape(batch.value_target.shape)
        value_loss = jnp.mean(rlax.l2_loss(value, batch.value_target))
        # 5. Compute state prediction loss
        predictor = Predictor(observation_dim=observation_dim, config=config)
        state_dist = predictor(batch.observation, music_latents.latent)
        state_nll = -jnp.mean(state_dist.as_distrax().log_prob(batch.next_observation))

        loss = (
            policy_loss
            + value_loss
            - config.entropy_coef * jnp.mean(policy.entropy())
            + music_nll
            + beta * latent_kl
            + state_nll
        )

        metrics = {
            "latent_kl": latent_kl,
            "music_nll": music_nll,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "state_nll": state_nll,
        }

        return loss, metrics

    return hk.transform(loss_fn)


class Learner:
    def __init__(
        self,
        *,
        actor: Actor,
        loss_fn: t.Callable[..., t.Tuple[chex.Array, dict]],
        action_dim: int,
        optimizer: optax.GradientTransformation,
        gen_reward: t.Callable[
            [hk.Params, hk.State, chex.Array, chex.Array],
            chex.Array,
        ],
        gen_reward_state: hk.State,
        config: Config,
    ) -> None:
        self._actor = actor
        self._config = config
        self._gen_reward = jax.jit(gen_reward)
        self._gen_reward_state = gen_reward_state
        self._opt_state = optimizer.init(actor.params)
        self._opt_update = optimizer.update

        def next_value_fn(obs: chex.Array, music_latent: chex.Array) -> chex.Array:
            policy = Policy(action_dim=action_dim, config=config)
            return policy(obs, music_latent).value

        _, next_value = hk.without_apply_rng(hk.transform(next_value_fn))
        self._next_value = jax.jit(next_value)

        def update(
            params: hk.Params,
            prng_key: chex.PRNGKey,
            opt_state: optax.OptState,
            batch: Batch,
            beta: float,
        ) -> t.Tuple[hk.Params, optax.OptState, t.Dict[str, chex.Array]]:
            grad, metrics = jax.grad(loss_fn, has_aux=True)(
                params, prng_key, batch, beta
            )
            updates, new_opt_state = self._opt_update(grad, opt_state)
            return optax.apply_updates(params, updates), new_opt_state, metrics

        self._update = jax.jit(update)

    def learn(
        self,
        rollout: Rollout,
        beta: float,
        prng_seq: hk.PRNGSequence,
    ) -> t.Dict[str, float]:
        (reward, raw_reward), self._gen_reward_state = self._gen_reward(
            self._actor.params,
            self._gen_reward_state,
            jnp.stack(rollout.observations),
            jnp.stack(rollout.music_latents),
        )
        next_value = self._next_value(
            self._actor.params,
            rollout.observations[-1],
            rollout.music_latents[-1],
        )
        batch = make_batch(rollout, next_value, reward, config=self._config)
        metrics = collections.defaultdict(lambda: 0.0)

        for _ in range(self._config.n_optim_epochs):
            for workers in sample_workers(
                batch.observation.shape[1],
                self._config.n_minibatches,
                next(prng_seq),
            ):
                self._actor.params, self._opt_state, new_metrics = self._update(
                    self._actor.params,
                    next(prng_seq),
                    self._opt_state,
                    batch[workers],
                    beta,
                )
                for key, value in new_metrics.items():
                    metrics[key] += value.item()
        for key in metrics:
            metrics[key] /= self._config.n_optim_epochs

        metrics["raw-reward-min"] = jnp.min(raw_reward).item()
        metrics["raw-reward-max"] = jnp.max(raw_reward).item()
        metrics["raw-reward-avg"] = jnp.mean(raw_reward).item()
        metrics["reward-min"] = jnp.min(reward).item()
        metrics["reward-max"] = jnp.max(reward).item()
        metrics["reward-avg"] = jnp.mean(reward).item()
        return metrics


class MusicIter:
    def __init__(
        self,
        musics: t.List[chex.Array],
        n_workers: int,
        seed: int,
    ) -> None:
        self._musics = np.array(musics, dtype=object)
        self._n_workers = n_workers
        self._n_musics = len(musics)
        self._random_state = np.random.RandomState(seed)
        self._selected = [
            self._random_state.choice(self._n_musics) for _ in range(n_workers)
        ]
        self._indices = [0] * self._n_musics

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> t.Tuple[chex.Array, chex.Array]:
        ended = np.zeros(self._n_workers, dtype=bool)
        for i in range(self._n_workers):
            if len(self._musics[self._selected[i]]) <= self._indices[i]:
                self._indices[i] = 0
                self._selected[i] = self._random_state.choice(self._n_musics)
                ended[i] = True
        res = [self._musics[i][j] for i, j in zip(self._selected, self._indices)]
        for i in range(self._n_workers):
            self._indices[i] += 1
        return jnp.stack(res), jnp.array(ended)


def _to_events(
    musics: t.List[muspy.Music],
    min_event_length: int,
) -> t.Tuple[t.List[chex.Array], int]:
    events = [muspy.to_event_representation(music) for music in musics]
    events = [event[event < 356].astype(int) for event in events]
    events = [event for event in events if event.shape[0] > min_event_length]
    min_value = min([np.min(event) for event in events])
    max_value = max([np.max(event) for event in events])
    return [jnp.array(event - min_value) for event in events], max_value - min_value


def load_emopia(root: str, min_event_length: int) -> t.Tuple[t.List[chex.Array], int]:
    emopia = muspy.EMOPIADataset(root, True).convert()
    return _to_events(emopia, min_event_length)


@dataclasses.dataclass
class ModelSaver:
    logdir: pathlib.Path
    count: int = 0

    def save_params(self, params: hk.Params) -> None:
        filename = f"model-{self.count}.pickle"
        with self.logdir.joinpath(filename).open(mode="wb") as f:
            pickle.dump(params, f)
        self.count += 1


def train(
    log_dir: str,
    midi_dir: str,
    env_name: str = "ant",
    n_train_midi: int = 128,
    n_eval_midi: int = 16,
    n_workers: int = 32,
    n_rollout_steps: int = 128,
    n_training_steps: int = 1000,
    logging_freq: int = 10,
    eval_freq: int = 100,
    save_freq: int = 100,
    beta_min: float = 1.0,
    beta_max: float = 4.0,
    min_event_length: int = 500,
    seed: int = 0,
) -> None:
    # Prepare env and MusicIter
    env = create_brax_env(
        env_name=env_name,
        episode_length=100000,
        auto_reset=False,
        batch_size=n_workers,
    )
    eval_env = create_brax_env(
        env_name=env_name,
        episode_length=100000,
        auto_reset=False,
        batch_size=1,
    )
    prng_seq = hk.PRNGSequence(seed)
    events, n_events = load_emopia(midi_dir, min_event_length)
    train_music_iter = MusicIter(
        musics=events[:n_train_midi],
        n_workers=n_workers,
        seed=seed,
    )
    eval_music_iter = MusicIter(
        musics=events[n_train_midi : n_train_midi + n_eval_midi],
        n_workers=1,
        seed=seed,
    )
    eval_music, _ = next(eval_music_iter)
    # Actor and training states
    state = env.reset(rng=next(prng_seq))
    config = Config()
    _, train_one_step = make_onestep_fn(env=env, n_events=n_events, config=config)
    _, eval_one_step = make_onestep_fn(env=eval_env, n_events=n_events, config=config)
    prev_terminal = jnp.zeros((n_workers,), dtype=bool)
    init, loss_fn = make_loss_fn(
        action_dim=env.action_size,
        observation_dim=env.observation_size,
        n_events=n_events,
        config=config,
    )
    initial_params = init(
        next(prng_seq),
        dummy_batch(env.observation_size, env.action_size),
        1.0,
    )
    actor = Actor(params=initial_params, rnn_state=None, step_fn=train_one_step)
    init_genrew, gen_reward = make_rewardgen_fn(env.observation_size, config)
    _, gen_reward_state = init_genrew(
        next(prng_seq),
        jnp.ones((2, 1, env.observation_size)),
        jnp.ones((1, 1, config.music_latent_dim)),
    )
    learner = Learner(
        actor=actor,
        loss_fn=loss_fn,
        action_dim=env.action_size,
        optimizer=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(3e-4, eps=1e-4),
        ),
        gen_reward=gen_reward,
        gen_reward_state=gen_reward_state,
        config=config,
    )
    rollout = Rollout(observations=[state.obs])
    log_dir = pathlib.Path(log_dir)
    metrics_file = log_dir.joinpath("metrics.jsonl")
    metrics_file.touch()
    viewer_id = 1
    beta = beta_max
    delta_beta = (beta_max - beta_min) / n_training_steps
    model_saver = ModelSaver(log_dir)
    # Training loop
    for i in tqdm.tqdm(range(n_training_steps)):
        # Rollout
        for _ in range(n_rollout_steps):
            music, ended = next(train_music_iter)
            prev_terminal = jnp.logical_or(ended, state.done)
            state, action, policy_out, music_latent = actor.step(
                prng_key=next(prng_seq),
                state=state,
                music=music,
                prev_terminal=prev_terminal,
            )
            rollout.append(
                observation=state.obs,
                music=music,
                action=action,
                terminal=prev_terminal,
                output=policy_out,
                music_latent=music_latent,
            )
        metrics = learner.learn(rollout, beta, prng_seq)
        beta -= delta_beta
        rollout.clear()
        if (i + 1) % logging_freq == 0:
            with metrics_file.open(mode="a") as f:
                json.dump(metrics, f)
                f.write("\n")
        if (i + 1) % save_freq == 0:
            model_saver.save_params(actor.params)
        if (i + 1) % eval_freq == 0:
            eval_state = eval_env.reset(next(prng_seq))
            qps = [jax.tree_map(lambda x: x.reshape(x.shape[1:]), eval_state.qp)]
            eval_actor = Actor(
                params=actor.params,
                rnn_state=None,
                step_fn=eval_one_step,
            )
            for music in eval_music:
                eval_state, _, _, _ = eval_actor.step(
                    state=eval_state,
                    music=music.reshape((1,)),
                    prng_key=next(prng_seq),
                    prev_terminal=jnp.zeros((1,)),
                )
                qps.append(
                    jax.tree_map(lambda x: x.reshape(x.shape[1:]), eval_state.qp)
                )
                with log_dir.joinpath(f"viewer-{viewer_id}.html").open(mode="w") as f:
                    f.write(brax_html.render(eval_env.sys, qps))
                viewer_id += 1


@chex.dataclass
class EvalResult:
    last_latent: chex.Array
    qps: t.List[QP]
    music_idx: int
    is_eval: bool


def eval_agent(
    params: hk.Params,
    midi_dir: str,
    env_name: str = "ant",
    n_train_midi: int = 128,
    n_eval_midi: int = 8,
    min_event_length: int = 500,
    seed: int = 0,
) -> t.Tuple[Env, t.List[EvalResult]]:
    env = create_brax_env(
        env_name=env_name,
        episode_length=100000,
        auto_reset=False,
        batch_size=1,
    )
    prng_seq = hk.PRNGSequence(seed)
    events, n_events = load_emopia(midi_dir, min_event_length)
    config = Config()
    _, eval_one_step = make_onestep_fn(env=env, n_events=n_events, config=config)
    reset = jax.jit(env.reset)

    def eval_one_music(events: chex.Array, idx: int, is_eval: bool) -> EvalResult:
        eval_actor = Actor(
            params=params,
            rnn_state=None,
            step_fn=eval_one_step,
        )
        qps = []
        state = reset(next(prng_seq))

        for event in music:
            state, _, _, latent = eval_actor.step(
                state=state,
                music=event.reshape((1,)),
                prng_key=next(prng_seq),
                prev_terminal=jnp.zeros((1,)),
            )
            qps.append(jax.tree_map(lambda x: x.reshape(x.shape[1:]), state.qp))

        return EvalResult(last_latent=latent, qps=qps, music_idx=idx, is_eval=is_eval)

    results = []
    for i, music in enumerate(events[:n_eval_midi]):
        results.append(eval_one_music(music, i, is_eval=False))

    for i, music in enumerate(events[n_train_midi : n_train_midi + n_eval_midi]):
        results.append(eval_one_music(music, i + n_train_midi, is_eval=True))

    return env, results


if __name__ == "__main__":
    import typer

    typer.run(train)
