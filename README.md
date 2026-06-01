# Robustness of In-Context Learning in Transformers

Code for my Oxford MSc dissertation, *Robustness of In-Context Learning in
Transformers: Linear vs Nonlinear Attention under Adversarial Prompt
Perturbations*. The full write-up is included as the PDF in this repository.

## Motivation

**In-context learning (ICL)** lets a transformer solve a new task purely from examples in
its prompt, with no weight updates, which is a capability now central to how large
language models are used. Because the prompt is the training signal, ICL is
only as trustworthy as the demonstrations it is given, so it is natural to ask
how it behaves when those demonstrations are corrupted or adversarially chosen.

Prior theory shows that **linear attention** can implement gradient-descent-like
learning in context, but real models rely on **softmax attention**. This raises
the question this dissertation investigates: does the choice of attention
mechanism change how robust in-context learning is to adversarial prompt
perturbations? By comparing linear and softmax attention-only transformers (and
a GPT-2 baseline) under controlled label-flipping and label-hijacking attacks,
we isolate the effect of the attention non-linearity on ICL robustness.

## Overview

We study in-context learning for binary classification. Each prompt is
a single task: a sequence of labelled demonstrations `(x_i, y_i)` followed by an
unlabelled query `x_q`, and the model must predict the query label `y_q ∈ {-1, +1}`
from the context alone (no gradient updates at test time).

The experiments compare three model families and probe how robust each one is to
adversarial corruption of the prompt:

- **Linear attention-only transformer** (1–4 layers)
- **Softmax attention-only transformer** (1–4 layers)
- **GPT-2-style transformer** (baseline)

Two prompt attacks are evaluated:

- **Label Flipping (LF)** — flip the labels of `N` demonstrations.
- **Label Hijacking (LH)** — move `N` demonstrations toward the decision
  boundary (features changed, labels untouched).

## Files

| File | Description |
|------|-------------|
| [`linearly_separable_task.py`](linearly_separable_task.py) | Linearly separable tasks with data generation, model definitions, training, testing and the LF / LH robustness experiments (incl. the GPT-2 baseline). |
| [`nonlinearly_separable_task.py`](nonlinearly_separable_task.py) | Non-linearly separable (parabola) tasks with the linear and softmax attention-only transformers. |
| [`gpt2_model.py`](gpt2_model.py) | Minimal GPT-2 architecture used as the benchmark model. |

## Data layout

Each task is packed into a single matrix of shape `(n + 1, d + 1)`:

- rows `0 … n-1` hold the `n` demonstrations — features in the first `d` columns,
  label `±1` in the last column;
- row `n` holds the query — features in the first `d` columns, last column `0`
  (the slot the model must fill).

A dataset stacks `B` such tasks into `(B, n + 1, d + 1)`. The default setting uses
feature dimension `d = 20` and context lengths `n = 40` / `n = 60`.

## Requirements

Python 3 with `numpy`, `torch`, `pandas`, and `matplotlib`. A CUDA GPU is
expected for training (the training loop asserts the model is on GPU).

## Notes

The scripts are organised top-to-bottom following the original research
notebooks: data generation → model definitions → training → evaluation →
attack experiments and plots. Training cells save checkpoints (`.pt`) that the
later evaluation and plotting sections load.
