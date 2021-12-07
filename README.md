# Unsupervised robot dancing

A tiny coursework project.

[Brax](https://github.com/google/brax) ants are trained via maximizing information transfer from
[Bach Chorales Data Set](https://archive.ics.uci.edu/ml/datasets/Bach+Chorales).

## Test

```
python -m pytest test.py
```

## Run

```
python udance.py train
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
