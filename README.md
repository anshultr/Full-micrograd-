# Full Micrograd

A from-scratch implementation of reverse-mode automatic differentiation (backpropagation) — built up from raw scalar arithmetic all the way to a trained multi-layer perceptron. Written while following Andrej Karpathy's [*"The spelled-out intro to neural networks and backpropagation: building micrograd"*](https://www.youtube.com/watch?v=VMj-3S1tku0).

## About the project

The goal is to learn a tiny neural network that minimizes the error between its predictions and a set of ground-truth values, by building the machinery that makes that possible from first principles rather than importing it.

It starts with plain multiplication and addition between numbers to show what a "forward pass" and a numerical derivative actually are. From there, a `Value` class wraps every number in a node that remembers how it was computed, so that calling `.backward()` on the final result walks the computation graph in reverse and deposits a gradient on every node that contributed to it — that's backpropagation. Once the engine is trustworthy, it's assembled into `Neuron`, `Layer`, and `MLP` classes and trained for 20 epochs with plain stochastic gradient descent.

## What's implemented

- **`Value`** — a scalar autograd engine, with `__add__`, `__mul__`, `__pow__`, `__truediv__`, `__sub__`, `tanh`, `sigmoid`, and `exp`, each defining both its forward computation and its local backward rule.
- **`.backward()`** — builds a topological ordering of the graph and applies the chain rule in reverse, once, from the final output.
- A hand-derived vs. automatic vs. numerical (finite-difference) comparison on a small expression, to confirm all three methods agree.
- A single neuron (`o = tanh(x1*w1 + x2*w2 + b)`), backpropagated by hand and cross-checked against PyTorch's own autograd on the same numbers — a sanity check that the from-scratch engine matches a trusted reference.
- `Neuron` / `Layer` / `MLP` — the same `Value` engine composed into an actual multi-layer perceptron.
- A training loop: zero gradients → forward pass → `loss.backward()` → step each parameter by `-lr * grad`, for 20 epochs on a 4-example toy dataset.

## Running it

```bash
python micrograd.py
```

Requires `numpy`, `matplotlib`, and `torch` (torch is only used for the verification section — the network itself has no dependencies beyond the standard library).

## Example output

Loss over the 20 training epochs on the toy dataset:

```
epoch 0    loss = 3.082
epoch 5    loss = 0.946
epoch 10   loss = 0.091
epoch 19   loss = 0.037
```

## Reference

[karpathy/micrograd](https://github.com/karpathy/micrograd) — this project is a learning re-implementation built while following that video, not an original engine.
