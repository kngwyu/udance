import dataclasses
import functools
import pathlib
import typing as t

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from brax.envs import Env, State as BraxState

chex.Array = chex.Array
Array = t.Union[chex.Array, np.ndarray]

Observation = np.ndarray
MaybePath = t.Union[pathlib.Path, str]
NetworkOutput = t.Any
PRNGKey = jnp.ndarray
Actor = t.Callable[[Observation], t.Tuple[chex.Array, NetworkOutput]]
ActionWrapper = t.Callable[[Array], Array]
Self = t.Any


def chain(*functions) -> t.Callable[..., t.Any]:
    """Chain multiple functions"""

    def chained_fn(*args, **kwargs) -> t.Any:
        x = functions[0](*args, **kwargs)
        for fn in functions[1:]:
            x = fn(x)
        return x

    return chained_fn


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

    observations: t.List[Observation]
    actions: t.List[chex.Array] = dataclasses.field(default_factory=list)
    rewards: t.List[chex.Array] = dataclasses.field(default_factory=list)
    terminals: t.List[chex.Array] = dataclasses.field(default_factory=list)
    outputs: t.List[GaussianAndValue] = dataclasses.field(default_factory=list)

    def append(
        self,
        *,
        observation: Observation,
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

    def last_obs(self) -> Observation:
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
        net_init_key: PRNGKey,
        initial_obs: chex.Array,
    ) -> None:
        net = lambda obs: MLPPolicy(env.action_size, config)(obs)  # noqa
        init, self._pi_and_v = hk.without_apply_rng(hk.transform(net))
        self.env = env
        self.params = init(net_init_key, initial_obs)

    @functools.partial(jax.jit, static_argnums=0)
    def _step(
        self,
        params: hk.Params,
        state: BraxState,
        prng_key: chex.PRNGKey,
    ) -> t.Tuple[BraxState, chex.Array, GaussianAndValue]:
        output = self._pi_and_v(params, state.obs)
        policy = distrax.MultivariateNormalDiag(output.mu, output.stddev)
        action = policy.sample(seed=prng_key)
        state = self.env.step(state, action)
        return state, action, output

    def step(
        self,
        state: BraxState,
        prng_key: chex.PRNGKey,
    ) -> t.Tuple[BraxState, chex.Array, GaussianAndValue]:
        return self._step(self.params, state, prng_key)


class Learner:
    def __init__(
        self,
        *,
        actor: Actor,
        action_dim: int,
        config: Config,
        optimizer: optax.GradientTransformation,
    ) -> None:
        self._config = config
        self._actor = actor
        self._opt_state = optimizer.init(actor.params)
        self._opt_update = optimizer.update

    @functools.partial(jax.jit, static_argnums=0)
    def _next_value(self, params: hk.Params, last_obs: Array) -> chex.Array:
        _, _, next_value = jax.lax.stop_gradient(
            self._actor._pi_and_v(params, last_obs)
        )
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
        mu, stddev, value = self._actor._pi_and_v(params, batch.observation)

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


def test_ppo() -> None:
    from brax import envs

    env = envs.create(
        env_name="ant",
        episode_length=1000,
        action_repeat=1,
        auto_reset=True,
        batch_size=16,
    )
    prng_seq = hk.PRNGSequence(0)
    state = env.reset(rng=next(prng_seq))
    config = Config()
    actor = Actor(
        env=env,
        config=config,
        net_init_key=next(prng_seq),
        initial_obs=state.obs,
    )
    learner = Learner(
        action_dim=env.action_size,
        config=config,
        actor=actor,
        optimizer=optax.adam(3e-4, eps=1e-4),
    )
    rollout = RolloutResult([state.obs])
    for _ in range(100):
        for _ in range(256):
            state, act, out = actor.step(state, next(prng_seq))
            rollout.append(
                observation=state.obs,
                action=act,
                reward=state.reward,
                output=out,
                terminal=state.done,
            )
        batch = learner.learn(rollout, prng_seq)
        print(batch.reward.mean())


if __name__ == "__main__":
    import typer

    app = typer.Typer()
    app.command()(test_ppo)
    app()
