"""Based on https://github.com/NTT123/wavernn-16bit
"""

import dataclasses
import functools
import pathlib
import typing as t

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import librosa
import numpy as np
import optax
import rlax
import soundfile as sf

from brax.envs import Env, State as BraxState, create as create_brax_env
from brax.io import html as brax_html
from tqdm.auto import tqdm

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
    mlp_hidden_units: t.Sequence[int] = (64, 64)


class GaussianAndValue(t.NamedTuple):
    mu: chex.Array
    stddev: chex.Array
    value: chex.Array


class MLPPolicy(hk.Module):
    def __init__(self, action_dim: int, config: Config) -> None:
        super().__init__(name="diag_gaussian_pi_and_v")

        def build(last_layer: hk.Linear) -> t.List[hk.Module]:
            return [
                hk.Flatten(),
                hk.nets.MLP(
                    config.mlp_hidden_units,
                    w_init=hk.initializers.Orthogonal(scale=jnp.sqrt(2.0)),
                    activation=jax.nn.tanh,
                    activate_final=True,
                ),
                last_layer,
            ]

        self._mu_net = hk.Sequential(
            build(hk.Linear(action_dim, w_init=hk.initializers.Orthogonal(scale=0.01)))
        )
        self._value_net = hk.Sequential(
            build(hk.Linear(1, w_init=hk.initializers.Orthogonal(scale=1.0)))
        )
        self._logstd_param = hk.get_parameter(
            "logstd", (1, action_dim), init=lambda shape, dtype: jnp.zeros(shape, dtype)
        )

    def __call__(self, observation: Array) -> GaussianAndValue:
        mu = self._mu_net(observation)
        stddev = jnp.ones_like(mu) * jnp.exp(self._logstd_param)
        value = self._value_net(observation)
        return GaussianAndValue(mu, stddev, value)


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


class WavData(t.NamedTuple):
    mel: chex.Array
    wav: chex.Array
    sample_rate: int


def wav2mel(
    wav: np.Array,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: int = 0,
    fmax: int = 8000,
) -> np.Array:
    mel = librosa.feature.melspectrogram(
        librosa.to_mono(wav),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=1,
    )
    mel = np.log(mel + 1e-5)
    return mel


def load_data_on_memory(
    wav_dir: pathlib.Path,
) -> t.List[WavData]:
    # load all data files on memory
    dataset = []
    for fp in tqdm(sorted(wav_dir.glob("*.wav"))):
        wav, sr = sf.read(fp, dtype="int16")
        mel = wav2mel(wav.astype(np.float32) / 2 ** 15, sample_rate=sr)
        dataset.append(WavData(mel=jnp.array(mel), wav=jnp.array(wav), sample_rate=sr))
    return dataset


def create_data_iter(
    dataset: t.List[WavData],
    prng_seq: hk.PRNGSequence,
    seq_len: int,
    batch_size: int,
    hop_length: int = 256,
) -> t.Iterable[t.Tuple[np.Array, np.Array]]:

    batch = []
    while True:
        indices = jax.random.permutation(next(prng_seq), len(dataset))
        for mel, wav, _sr in map(lambda idx: dataset[idx], indices):
            r_idx = jax.random.randint(
                next(prng_seq),
                shape=(),
                minval=seq_len,
                maxval=mel.shape[1] - 1,
            )
            l_idx = seq_len - r_idx
            batch.append(
                (mel[:, l_idx:r_idx], wav[l_idx * hop_length : r_idx * hop_length])
            )
            if len(batch) == batch_size:
                mel_batch, wav_batch = jax.tree_map(
                    lambda *array: jnp.stack(array),
                    *batch,
                )
                yield jnp.swapaxes(mel_batch, 1, 2), wav_batch
                batch.clear()


def test_audio_processing(wave_dir: str) -> None:
    data = load_data_on_memory(pathlib.Path(wave_dir))
    prng_seq = hk.PRNGSequence(0)
    data_iter = create_data_iter(data, prng_seq, 100, 64)
    for i in range(5):
        mel, wav = next(data_iter)
        print(mel.shape, wav.shape)


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
