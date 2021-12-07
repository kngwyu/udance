import collections
import dataclasses
import functools
import json
import pathlib
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
    entropy_coef: float = 0.001
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_optim_epochs: int = 10
    n_minibatches: int = 1
    reward_scaling: float = 1.0
    # Network config
    hidden_dims: t.Sequence[int] = (64, 64)
    rnn_hidden_dim: int = 64
    drop_prob: float = 0.5
    # Unsupervised learning
    min_r: float = -1.0
    max_r: float = 1.0


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
    """π(Xt+1|Xt, Yt, Xt-1, ...) & P(Xt+1|Xt, Yt Xt-1, ...)"""

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
            jax.tree_map(lambda x: x * rnn_mask.reshape(-1, 1), prev_state),
        )
        post_rnn_dropped = hk.dropout(hk.next_rng_key(), self._drop_prob, post_rnn)
        pi_mean = self._pi_mean(post_rnn_dropped)
        pi_std = jnp.ones_like(pi_mean) * jnp.exp(self._logstd_param)
        value = self._value(post_rnn_dropped)
        pred = self._pred(self._pre_pred(post_rnn_dropped))
        return MusicPolicyOutput(Normal(pi_mean, pi_std), value, pred), next_state


class ObsPredictor(hk.RNNCore):
    """P(Xt+1|Xt, Xt-1, ...)"""

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
            jax.tree_map(lambda x: x * rnn_mask.reshape(-1, 1), prev_state),
        )
        pred = self._pred(self._mlp(post_rnn))
        return pred, next_state


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
class RolloutWithMusic:
    """A container class that holds the result of N-step rollout"""

    observations: t.List[chex.Array]
    musics: t.List[chex.Array] = dataclasses.field(default_factory=list)
    actions: t.List[chex.Array] = dataclasses.field(default_factory=list)
    terminals: t.List[chex.Array] = dataclasses.field(default_factory=list)
    policy_outputs: t.List[MusicPolicyOutput] = dataclasses.field(default_factory=list)
    predictor_outputs: t.List[Normal] = dataclasses.field(default_factory=list)

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
    advantage: chex.Array
    value_target: chex.Array
    log_prob: chex.Array
    mask: chex.Array

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
        )


@functools.partial(jax.jit, static_argnums=2)
def _make_music_batch(
    rollout: RolloutWithMusic,
    next_value: chex.Array,
    config: Config,
) -> MusicBatch:
    # Observation
    observation = jnp.stack(rollout.observations)  # T, N, obs-dim
    # Compute rewards
    predictor = jax.tree_map(lambda *x: jnp.stack(x), *rollout.predictor_outputs)
    marginal_logp = predictor.as_distrax().log_prob(observation[1:])
    policy_outputs = jax.tree_map(lambda *x: jnp.stack(x), *rollout.policy_outputs)
    conditional_logp = policy_outputs.pred.as_distrax().log_prob(observation[1:])
    raw_rewards = conditional_logp - marginal_logp
    reward = jnp.clip(raw_rewards, a_min=config.min_r, a_max=config.max_r)
    # Compute advantage
    mask = 1.0 - jnp.stack(rollout.terminals)
    value = jnp.concatenate(
        (policy_outputs.value.reshape(reward.shape), next_value.reshape(1, -1)),
        axis=0,
    )
    advantage = batched_gae(reward, mask * config.gamma, config.gae_lambda, value)
    value_target = advantage + value[:-1]
    # Log π
    action = jnp.stack(rollout.actions)
    log_prob = policy_outputs.policy.as_distrax().log_prob(action)
    return MusicBatch(
        observation=observation[:-1],
        next_observation=observation[1:],
        music=jnp.stack(rollout.musics),
        action=action,
        advantage=advantage,
        value_target=value_target,
        log_prob=log_prob,
        mask=mask,
    )


def make_onestep_fn(
    env: Env,
    n_pitches: int,
    config: Config,
) -> hk.Transformed:
    def step_impl(
        state: BraxState,
        policy_state: t.Optional[hk.LSTMState],
        predictor_state: t.Optional[hk.LSTMState],
        music: chex.Array,
        prev_terminal: chex.Array,
    ) -> t.Tuple[
        BraxState,
        chex.Array,
        MusicPolicyOutput,
        hk.LSTMState,
        Normal,
        hk.LSTMState,
    ]:
        mask = 1.0 - prev_terminal
        obs_predictor = ObsPredictor(obs_dim=env.observation_size, config=Config)
        predictor_out, predictor_state = obs_predictor(
            (state.obs, mask),
            predictor_state,
        )
        music_policy = MusicPolicy(
            action_dim=env.action_size,
            obs_dim=env.observation_size,
            n_pitches=n_pitches,
            config=Config,
        )
        policy_out, policy_state = music_policy(
            (state.obs, music, mask),
            policy_state,
        )
        action = policy_out.policy.as_distrax().sample(seed=hk.next_rng_key())
        resetted_state = env.reset(hk.next_rng_key())
        state = jax.tree_map(
            lambda old, new: jnp.where(
                prev_terminal.reshape((old.shape[0],) + (1,) * (old.ndim - 1)),
                new,
                old,
            ),
            state,
            resetted_state,
        )
        state = env.step(state, jnp.tanh(action))
        return state, action, policy_out, policy_state, predictor_out, predictor_state

    return jax.tree_map(jax.jit, hk.transform(step_impl))


@chex.dataclass
class Actor:
    params: hk.Params
    policy_state: t.Optional[hk.LSTMState]
    predictor_state: t.Optional[hk.LSTMState]

    def step(
        self,
        onestep_fn: t.Callable[..., t.Any],
        state: BraxState,
        music: chex.Array,
        prng_key: chex.PRNGKey,
        prev_terminal: chex.Array,
    ) -> t.Tuple[BraxState, chex.Array, MusicPolicyOutput, Normal]:
        (
            state,
            action,
            policy_out,
            self.policy_state,
            predictor_out,
            self.predictor_state,
        ) = onestep_fn(
            self.params,
            prng_key,
            state,
            self.policy_state,
            self.predictor_state,
            music,
            prev_terminal,
        )
        return state, action, policy_out, predictor_out


class Learner:
    def __init__(
        self,
        *,
        action_dim: int,
        observation_dim: int,
        n_pitches: int,
        config: Config,
        actor: Actor,
        optimizer: optax.GradientTransformation,
    ) -> None:
        self._config = config
        self._actor = actor
        self._opt_state = optimizer.init(actor.params)
        self._opt_update = optimizer.update

        def next_value_fn(
            x: t.Tuple[chex.Array, chex.Array, chex.Array],
            state: hk.LSTMState,
        ) -> chex.Array:
            policy = MusicPolicy(action_dim, observation_dim, n_pitches, config)
            return policy(x, state)[0].value

        _, next_value = hk.transform(next_value_fn)
        self._next_value = jax.jit(next_value)

        def loss_fn(
            batch: MusicBatch,
            policy_initial_state: t.Optional[hk.LSTMState],
            predictor_initial_state: t.Optional[hk.LSTMState],
        ) -> t.Tuple[chex.Array, t.Dict[str, chex.Array]]:
            policy = MusicPolicy(action_dim, observation_dim, n_pitches, config)
            batch_size = batch.observation.shape[1]
            if policy_initial_state is None:
                policy_initial_state = policy.initial_state(batch_size)
            policy_outputs, _ = hk.dynamic_unroll(
                policy,
                (batch.observation, batch.music, batch.mask),
                policy_initial_state,
            )
            predictor = ObsPredictor(observation_dim, config)
            if predictor_initial_state is None:
                predictor_initial_state = predictor.initial_state(batch_size)
            predictor_output, _ = hk.dynamic_unroll(
                predictor,
                (batch.observation, batch.mask),
                predictor_initial_state,
            )

            # Prediction losses
            conditional_nll = -policy_outputs.pred.as_distrax().log_prob(
                batch.next_observation
            )
            marginal_nll = -predictor_output.as_distrax().log_prob(
                batch.next_observation
            )
            conditional_nll = jnp.sum(jnp.mean(conditional_nll, axis=1))
            marginal_nll = jnp.sum(jnp.mean(marginal_nll, axis=1))

            # Policy loss
            policy = policy_outputs.policy.as_distrax()
            log_prob = policy.log_prob(batch.action)
            prob_ratio = jnp.exp(log_prob - batch.log_prob)

            policy_loss = batched_ppoclip_loss(
                prob_ratio,
                batch.advantage,
                self._config.clip_epsilon,
            )
            policy_loss = jnp.sum(policy_loss)

            # Value loss
            value = policy_outputs.value.reshape(batch.value_target.shape)
            value_loss = jnp.sum(
                jnp.mean(rlax.l2_loss(value, batch.value_target), axis=1)
            )

            loss = (
                policy_loss
                + value_loss
                - self._config.entropy_coef * jnp.sum(policy.entropy())
                + conditional_nll
                + marginal_nll
            )

            metrics = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "conditional_nll": conditional_nll,
                "marginal_nll": marginal_nll,
            }

            return loss, metrics

        _, loss = hk.transform(loss_fn)

        self._loss = loss

        def update(
            params: hk.Params,
            prng_key: chex.PRNGKey,
            opt_state: optax.OptState,
            batch: MusicBatch,
            policy_initial_state: t.Optional[hk.LSTMState],
            predictor_initial_state: t.Optional[hk.LSTMState],
        ) -> t.Tuple[hk.Params, optax.OptState, t.Dict[str, chex.Array]]:
            grad, metrics = jax.grad(loss, has_aux=True)(
                params,
                prng_key,
                batch,
                policy_initial_state,
                predictor_initial_state,
            )
            updates, new_opt_state = self._opt_update(grad, opt_state)
            return optax.apply_updates(params, updates), new_opt_state, metrics

        self._update = jax.jit(update)

    def learn(
        self,
        rollout: RolloutWithMusic,
        prng_seq: hk.PRNGSequence,
        policy_initial_states: t.Optional[hk.LSTMState],
        predictor_initial_states: t.Optional[hk.LSTMState],
    ) -> t.Dict[str, float]:
        next_value = self._next_value(
            self._actor.params,
            next(prng_seq),
            (rollout.last_obs(), rollout.musics[-1], 1.0 - rollout.terminals[-1]),
            self._actor.policy_state,
        )
        batch = _make_music_batch(rollout, next_value, self._config)
        metrics = collections.defaultdict(lambda: 0.0)

        def select_if(
            state: t.Optional[hk.LSTMState],
            idx: chex.Array,
        ) -> t.Optional[hk.LSTMState]:
            if state is None:
                return None
            else:
                return jax.tree_map(lambda x: x[idx], state)

        for _ in range(self._config.n_optim_epochs):
            for workers in sample_workers(
                batch.value_target.shape[1],
                self._config.n_minibatches,
                next(prng_seq),
            ):
                self._actor.params, self._opt_state, new_metrics = self._update(
                    self._actor.params,
                    next(prng_seq),
                    self._opt_state,
                    batch[workers],
                    select_if(policy_initial_states, workers),
                    select_if(predictor_initial_states, workers),
                )
                for key, value in new_metrics.items():
                    metrics[key] += value.item()

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
        return jnp.concatenate(res), jnp.array(ended)


def _to_pitches(musics: t.List[muspy.Music]) -> t.Tuple[t.List[chex.Array], int]:
    pitches = [muspy.to_pitch_representation(music) for music in musics]
    min_value = min([np.min(pitch) for pitch in pitches])
    max_value = max([np.max(pitch) for pitch in pitches])
    return [jnp.array(pitch - min_value) for pitch in pitches], max_value - min_value


def load_jsb(root: str) -> t.Tuple[t.List[chex.Array], int]:
    jsb = muspy.JSBChoralesDataset(root, download_and_extract=True).convert()
    return _to_pitches(jsb)


def train(
    log_dir: str,
    midi_dir: str,
    n_train_midi: int = 64,
    n_eval_midi: int = 16,
    n_workers: int = 16,
    n_rollout_steps: int = 128,
    n_training_steps: int = 1000,
    logging_freq: int = 10,
    eval_freq: float = 100,
    seed: int = 0,
) -> None:
    # Prepare env and MusicIter
    env = create_brax_env(
        env_name="ant",
        episode_length=100000,
        action_repeat=1,
        auto_reset=True,
        batch_size=n_workers,
    )
    eval_env = create_brax_env(
        env_name="ant",
        episode_length=100000,
        action_repeat=1,
        auto_reset=True,
        batch_size=1,
    )
    prng_seq = hk.PRNGSequence(seed)
    pitches, n_pitches = load_jsb(midi_dir)
    train_music_iter = MusicIter(
        musics=pitches[:n_train_midi],
        n_workers=n_workers,
        seed=seed,
    )
    eval_music_iter = MusicIter(
        musics=pitches[n_train_midi : n_train_midi + n_eval_midi],
        n_workers=1,
        seed=seed,
    )
    eval_music, _ = next(eval_music_iter)
    # Actor and training states
    state = env.reset(rng=next(prng_seq))
    config = Config()
    init, train_one_step = make_onestep_fn(env=env, n_pitches=n_pitches, config=config)
    _, eval_one_step = make_onestep_fn(env=eval_env, n_pitches=n_pitches, config=config)
    prev_terminal = jnp.zeros((n_workers,), dtype=bool)
    params = init(
        next(prng_seq),
        state,
        None,
        None,
        next(train_music_iter)[0],
        prev_terminal,
    )
    actor = Actor(params=params, policy_state=None, predictor_state=None)
    learner = Learner(
        action_dim=env.action_size,
        observation_dim=env.observation_size,
        n_pitches=n_pitches,
        config=config,
        actor=actor,
        optimizer=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(3e-4, eps=1e-4),
        ),
    )
    rollout = RolloutWithMusic(observations=[state.obs])
    metrics_id = 1
    viewer_id = 1
    log_dir = pathlib.Path(log_dir)
    # Training loop
    for i in tqdm.tqdm(range(n_training_steps)):
        policy_initial_state = actor.policy_state
        predictor_initial_state = actor.predictor_state
        # Rollout
        for _ in range(n_rollout_steps):
            music, ended = next(train_music_iter)
            prev_terminal = jnp.logical_or(ended, state.done)
            state, action, policy_out, predictor_out = actor.step(
                onestep_fn=train_one_step,
                state=state,
                music=music,
                prng_key=next(prng_seq),
                prev_terminal=prev_terminal,
            )
            rollout.append(
                observation=state.obs,
                music=music,
                action=action,
                policy_out=policy_out,
                predictor_out=predictor_out,
                terminal=prev_terminal,
            )
        metrics = learner.learn(
            rollout,
            prng_seq,
            policy_initial_state,
            predictor_initial_state,
        )
        rollout.clear()
        if (i + 1) % logging_freq == 0:
            with log_dir.joinpath(f"metrics-{metrics_id}.json").open(mode="w") as f:
                json.dump(metrics, f)
            metrics_id += 1
        if (i + 1) % eval_freq == 0:
            eval_state = eval_env.reset(next(prng_seq))
            qps = [jax.tree_map(lambda x: x.reshape(x.shape[1:]), eval_state.qp)]
            eval_actor = Actor(params=params, policy_state=None, predictor_state=None)
            for music in eval_music:
                state, _, _, _ = eval_actor.step(
                    onestep_fn=eval_one_step,
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


if __name__ == "__main__":
    import typer

    typer.run(train)
