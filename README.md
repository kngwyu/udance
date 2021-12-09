# Unsupervised robot dancing

A tiny coursework project.

[Brax](https://github.com/google/brax) ants are trained via maximizing information transfer from
[Bach Chorales Data Set](https://archive.ics.uci.edu/ml/datasets/Bach+Chorales).


## Run

Due to the time constraint, implementation is done in a single file `udance.py`.

```
python udance.py train
```

## Test

Requires [pytest](https://docs.pytest.org/en/6.2.x/example/parametrize.html)

```
python -m pytest test.py
```

## Requirements
- [brax](https://github.com/google/brax)
- [chex](https://github.com/deepmind/chex)
- [distrax](https://github.com/deepmind/distrax)
- [haiku](https://github.com/deepmind/dm-haiku)
- [jax](https://github.com/google/jax)
- [muspy](https://github.com/salu133445/muspy)
- [numpy](https://numpy.org/)
- [optax](https://github.com/deepmind/optax)
- [rlax](https://github.com/deepmind/rlax)
- [typer](https://typer.tiangolo.com/)
