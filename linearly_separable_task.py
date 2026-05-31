"""
Linearly separable in-context classification: linear / softmax attention-only transformers and a GPT-2 baseline, with label-flipping and label-hijacking robustness experiments.
"""


# Experiments

import numpy as np
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import math
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

# Check GPU

print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))

# Part 1. Non-Linear Attention-Only Transformer and Robustness in Adversarial Attack

# 1.1 Data Generation

# Generating w*

# w* function
def sample_w(d, mean=None, cov_scale=1.0):
    if mean is None:
        mean = np.zeros(d, dtype=np.float64)
    else:
        mean = np.asarray(mean, dtype=np.float64)

    cov = float(cov_scale) * np.eye(d, dtype=np.float64)
    w = np.random.multivariate_normal(mean, cov).astype(np.float32)
    w /= (np.linalg.norm(w) + 1e-8)
    return w

# Generating task matrix

# Task matrix function

def sample_ls_task_matrix(d, n, w, margin=0.0, apply_margin_to_query=False):
    assert n % 2 == 0
    half = n // 2

    w = np.asarray(w, dtype=np.float32)
    assert w.shape == (d,)

    # Positive samples
    x_pos = []
    while len(x_pos) < half:
        X = np.random.randn(half, d).astype(np.float32)
        s = X @ w
        pick = X[s > 0]
        need = half - len(x_pos)
        if pick.shape[0] > 0:
            x_pos.append(pick[:need])
    x_pos = np.vstack(x_pos)[:half]

    if margin > 0.0:
        s_pos = x_pos @ w
        delta = np.clip(np.float32(margin) - s_pos, 0.0, None)  # Ensure w·x >= margin
        x_pos = x_pos + delta[:, None] * w[None, :]

    # Negative samples
    x_neg = []
    while len(x_neg) < half:
        X = np.random.randn(half, d).astype(np.float32)
        s = X @ w
        pick = X[s < 0]
        need = half - len(x_neg)
        if pick.shape[0] > 0:
            x_neg.append(pick[:need])
    x_neg = np.vstack(x_neg)[:half]

    if margin > 0.0:
        s_neg = x_neg @ w
        delta = np.clip(s_neg + np.float32(margin), 0.0, None)  # Ensure w·x <= -margin
        x_neg = x_neg - delta[:, None] * w[None, :]

    # Combine positive and negative samples, shuffle
    x_ctx = np.vstack([x_pos, x_neg]).astype(np.float32)
    y_ctx = np.concatenate([np.ones(half, dtype=np.float32),
                            -np.ones(half, dtype=np.float32)])
    perm = np.random.permutation(n)
    x_ctx, y_ctx = x_ctx[perm], y_ctx[perm]

    # Query sample
    x_query = np.random.randn(d).astype(np.float32)
    y_query = np.sign(x_query @ w).astype(np.float32)
    if y_query == 0: 
        y_query = np.float32(1.0)
    
    # Ignore this as we don't change our query
    if apply_margin_to_query and margin > 0.0:
        s_q = float(x_query @ w)
        need = max(0.0, float(np.float32(margin)) - float(y_query) * s_q)  # Ensure y_query*(w·x) >= margin
        if need > 0.0:
            x_query = x_query + np.float32(y_query * need) * w

    # Assemble task matrix
    task_matrix = np.zeros((n+1, d+1), dtype=np.float32)
    task_matrix[:n, :d] = x_ctx
    task_matrix[:n,  d] = y_ctx
    task_matrix[n,  :d] = x_query
    task_matrix[n,   d] = 0.0
    return task_matrix, y_query

# Generating data

# Data function

def sample_ls_dataset(d, n, batch_size, mean=None, cov_scale=1.0, margin=0.0, apply_margin_to_query=False):
    x = np.zeros((batch_size, n+1, d+1), dtype=np.float32)
    y = np.zeros((batch_size,), dtype=np.float32)
    w_all = np.zeros((batch_size, d), dtype=np.float32)  # store w*
    for b in range(batch_size):
        w = sample_w(d, mean=mean, cov_scale=cov_scale)
        x[b], y[b] = sample_ls_task_matrix(d, n, w, margin=margin, apply_margin_to_query=apply_margin_to_query)
        w_all[b] = w
    return x, y, w_all

# Resampling negative query

def resample_ls_query_negative(dataset, y_query, w_all, margin=0.0, max_iter=5000):
    # Shapes
    B, n_plus_1, d_plus_1 = dataset.shape
    d = d_plus_1 - 1
    assert w_all.shape == (B, d), f"w_all shape mismatch: expected {(B, d)}, got {w_all.shape}"

    w_norm = np.linalg.norm(w_all, axis=1, keepdims=True) + 1e-8
    w_unit = (w_all / w_norm).astype(np.float32)

    for i in range(B):
        x_q = dataset[i, n_plus_1 - 1, :d]
        s_old = float(np.dot(x_q, w_unit[i]))

        if s_old >= -margin:
            found = False
            for _ in range(max_iter):
                x_cand = np.random.randn(d).astype(np.float32)
                if float(np.dot(x_cand, w_unit[i])) < -margin:
                    dataset[i, n_plus_1 - 1, :d] = x_cand
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    f"Failed to obtain a negative-side query for task {i} within max_iter. "
                    "Increase max_iter or reduce margin."
                )

    dataset[:, n_plus_1 - 1, d] = 0.0
    y_query[:] = -1.0
    return dataset, y_query

# Resampling positive query

def resample_ls_query_positive(dataset, y_query, w_all, margin=0.0, max_iter=5000):
    # Shapes
    B, n_plus_1, d_plus_1 = dataset.shape
    d = d_plus_1 - 1
    assert w_all.shape == (B, d), f"w_all shape mismatch: expected {(B, d)}, got {w_all.shape}"

    w_norm = np.linalg.norm(w_all, axis=1, keepdims=True) + 1e-8
    w_unit = (w_all / w_norm).astype(np.float32)

    for i in range(B):
        x_q = dataset[i, n_plus_1 - 1, :d]
        s_old = float(np.dot(x_q, w_unit[i]))

        if s_old <= margin:
            found = False
            for _ in range(max_iter):
                x_cand = np.random.randn(d).astype(np.float32)
                if float(np.dot(x_cand, w_unit[i])) > margin:
                    dataset[i, n_plus_1 - 1, :d] = x_cand
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    f"Failed to obtain a positive-side query for task {i} within max_iter. "
                    "Increase max_iter or reduce margin."
                )

    dataset[:, n_plus_1 - 1, d] = 0.0
    y_query[:] = 1.0
    return dataset, y_query

# Sanity check

def check_balance(dataset, y_query, n):
    """
    dataset: (B, n+1, d+1)
    y_query: (B,)
    n: number of context demos
    """
    y_ctx = dataset[:, :n, -1]
    pos_counts = (y_ctx == 1).sum(axis=1)
    neg_counts = (y_ctx == -1).sum(axis=1)

    total = y_ctx.shape[0]
    balanced = np.sum((pos_counts == n//2) & (neg_counts == n//2))

    print(f"Total tasks: {total}")
    print(f"Balanced tasks: {balanced} ({balanced/total*100:.2f}%)")
    print(f"Example counts (first 5 tasks):")
    for i in range(min(5, total)):
        print(f"Task {i}: +1={pos_counts[i]}, -1={neg_counts[i]}, query={y_query[i]}")

def check_query_balance(name: str, y_query: np.ndarray):
    y = np.asarray(y_query)
    total = y.size
    pos = int(np.sum(y > 0))
    neg = int(np.sum(y < 0))
    zero = int(np.sum(y == 0))
    print(f"{name}: total={total}, +1={pos} ({pos/total:.2%}), -1={neg} ({neg/total:.2%}), zero={zero}")

def check_query_all(name: str, y_query: np.ndarray, expected: float | None = None):

    y = np.asarray(y_query).astype(np.float32)
    total = y.size
    pos = int(np.sum(y == 1.0))
    neg = int(np.sum(y == -1.0))
    other = total - pos - neg

    if expected is None:
        all_pos = (pos == total)
        all_neg = (neg == total)
        ok = (other == 0) and (all_pos or all_neg)
        status = "ALL +1" if all_pos else ("ALL -1" if all_neg else "MIXED")
        print(f"{name}: {status}; +1={pos}, -1={neg}, other={other}")
        return ok
    else:
        exp = 1.0 if expected > 0 else -1.0
        ok = (pos == total) if exp == 1.0 else (neg == total)
        print(f"{name}: expected all {int(exp)} -> {ok}; +1={pos}, -1={neg}, other={other}")
        return ok


def check_query_all_positive(name: str, y_query: np.ndarray):
    return check_query_all(name, y_query, expected=1.0)

def check_query_all_negative(name: str, y_query: np.ndarray):
    return check_query_all(name, y_query, expected=-1.0)

# n = 60, training

np.random.seed(42)

d, n = 20, 60

batch_size_train = 30000
dataset_train_ls_n60, y_query_train_ls_n60, w_train_ls_n60 = sample_ls_dataset(d, n, batch_size_train)
assert dataset_train_ls_n60.shape == (batch_size_train, n+1, d+1)
assert y_query_train_ls_n60.shape == (batch_size_train,)
assert np.allclose(dataset_train_ls_n60[:, -1, -1], 0.0)

check_balance(dataset_train_ls_n60, y_query_train_ls_n60, n)
check_query_balance("train ls n=60", y_query_train_ls_n60)

# n = 60, validation on positive and negative query separately

np.random.seed(425)
d, n = 20, 60
batch_size_val = 3000

dataset_val_ls_n60, y_query_val_ls_n60, w_val_ls_n60 = sample_ls_dataset(
    d, n, batch_size_val, margin=0.0, apply_margin_to_query=False
)
assert dataset_val_ls_n60.shape == (batch_size_val, n+1, d+1)
assert y_query_val_ls_n60.shape == (batch_size_val,)
assert np.allclose(dataset_val_ls_n60[:, -1, -1], 0.0)

check_balance(dataset_val_ls_n60, y_query_val_ls_n60, n)
check_query_balance("val ls n=60 (mixed)", y_query_val_ls_n60)

# All positive query
dataset_val_ls_pos_n60 = dataset_val_ls_n60.copy()
y_query_val_ls_pos_n60 = y_query_val_ls_n60.copy()
dataset_val_ls_pos_n60, y_query_val_ls_pos_n60 = resample_ls_query_positive(
    dataset_val_ls_pos_n60, y_query_val_ls_pos_n60, w_val_ls_n60, margin=1e-4, max_iter=1000
)
assert np.allclose(dataset_val_ls_pos_n60[:, -1, -1], 0.0)
check_query_all_positive("val ls n=60 (pos-only)", y_query_val_ls_pos_n60)

# All negative query
dataset_val_ls_neg_n60 = dataset_val_ls_n60.copy()
y_query_val_ls_neg_n60 = y_query_val_ls_n60.copy()
dataset_val_ls_neg_n60, y_query_val_ls_neg_n60 = resample_ls_query_negative(
    dataset_val_ls_neg_n60, y_query_val_ls_neg_n60, w_val_ls_n60, margin=1e-4, max_iter=1000
)
assert np.allclose(dataset_val_ls_neg_n60[:, -1, -1], 0.0)
check_query_all_negative("val ls n=60 (neg-only)", y_query_val_ls_neg_n60)

check_balance(dataset_val_ls_pos_n60, y_query_val_ls_pos_n60, n)
check_balance(dataset_val_ls_neg_n60, y_query_val_ls_neg_n60, n)

# Illustration

np.random.seed(934)

d, n = 2, 40

w = sample_w(d)
task_matrix, y_query = sample_ls_task_matrix(d, n, w, margin=0.1)

x_ctx, y_ctx = task_matrix[:n, :d], task_matrix[:n, d]
x_query = task_matrix[n, :d]

plt.figure(figsize=(6, 6))
plt.scatter(x_ctx[y_ctx == 1, 0], x_ctx[y_ctx == 1, 1],
            c="#1f77b4", marker="o", s=60, edgecolors="k", label="Positive")
plt.scatter(x_ctx[y_ctx == -1, 0], x_ctx[y_ctx == -1, 1],
            c="#d62728", marker="s", s=60, edgecolors="k", label="Negative")
plt.scatter(x_query[0], x_query[1],
            c="#2ca02c", marker="*", s=250, edgecolors="k", label="Query")

xx = np.linspace(x_ctx[:,0].min()-1, x_ctx[:,0].max()+1, 100)
yy = -(w[0]/(w[1]+1e-8)) * xx
plt.plot(xx, yy, "k--", linewidth=2, label="Decision boundary")

plt.xlabel(r"$x_{1}$", fontsize=14)
plt.ylabel(r"$x_{2}$", fontsize=14)
plt.title("Linear Classification Task (d = 2)", fontsize=16, pad=12)
plt.legend(frameon=True, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# n = 40, training

np.random.seed(42)

d, n = 20, 40

# Training
batch_size_train = 30000
dataset_train_ls_n40, y_query_train_ls_n40, w_train_ls_n40 = sample_ls_dataset(d, n, batch_size_train)
assert dataset_train_ls_n40.shape == (batch_size_train, n+1, d+1)
assert y_query_train_ls_n40.shape == (batch_size_train,)
assert np.allclose(dataset_train_ls_n40[:, -1, -1], 0.0)

check_balance(dataset_train_ls_n40, y_query_train_ls_n40, n)
check_query_balance("train ls n=40", y_query_train_ls_n40)

# n = 40, validation on positive and negative query separately

np.random.seed(425)

d, n = 20, 40

batch_size_val = 3000
dataset_val_ls_n40, y_query_val_ls_n40, w_val_ls_n40 = sample_ls_dataset(
    d, n, batch_size_val, margin=0.0, apply_margin_to_query=False
)
assert dataset_val_ls_n40.shape == (batch_size_val, n+1, d+1)
assert y_query_val_ls_n40.shape == (batch_size_val,)
assert np.allclose(dataset_val_ls_n40[:, -1, -1], 0.0)

check_balance(dataset_val_ls_n40, y_query_val_ls_n40, n)
check_query_balance("val ls n=40 (mixed)", y_query_val_ls_n40)

# All positive queries
dataset_val_ls_pos_n40 = dataset_val_ls_n40.copy()
y_query_val_ls_pos_n40 = y_query_val_ls_n40.copy()
dataset_val_ls_pos_n40, y_query_val_ls_pos_n40 = resample_ls_query_positive(
    dataset_val_ls_pos_n40, y_query_val_ls_pos_n40, w_val_ls_n40, margin=1e-4, max_iter=1000
)
assert np.allclose(dataset_val_ls_pos_n40[:, -1, -1], 0.0)
check_query_all_positive("val ls n=40 (pos-only)", y_query_val_ls_pos_n40)

# All negative queries
dataset_val_ls_neg_n40 = dataset_val_ls_n40.copy()
y_query_val_ls_neg_n40 = y_query_val_ls_n40.copy()
dataset_val_ls_neg_n40, y_query_val_ls_neg_n40 = resample_ls_query_negative(
    dataset_val_ls_neg_n40, y_query_val_ls_neg_n40, w_val_ls_n40, margin=1e-4, max_iter=1000
)
assert np.allclose(dataset_val_ls_neg_n40[:, -1, -1], 0.0)
check_query_all_negative("val ls n=40 (neg-only)", y_query_val_ls_neg_n40)

check_balance(dataset_val_ls_pos_n40, y_query_val_ls_pos_n40, n)
check_balance(dataset_val_ls_neg_n40, y_query_val_ls_neg_n40, n)

# 1.2 Transformer Model

# Training model

# Training function (validation split into pos-only & neg-only)
def train_model(
    model: nn.Module,
    x: torch.Tensor, y: torch.Tensor,
    x_val_pos: torch.Tensor, y_val_pos: torch.Tensor,
    x_val_neg: torch.Tensor, y_val_neg: torch.Tensor,
    *,
    total_steps: int = 150000,
    batch_size: int = 256,
    max_lr: float = 1e-3,
    pct_start: float = 0.30,
    div_factor: float = 25.0,
    final_div_factor: float = 50.0,
    record_interval: int = 500,
    clip_grad: float = 1.0,
    weight_decay: float = 0.0,
    label: str = "exp",
    save_path: str | None = None
):
    device = next(model.parameters()).device
    assert device.type == "cuda", "Model is not on GPU! Use model.to('cuda') before training."

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos',
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )

    @torch.no_grad()
    def param_norm(model):
        total = 0.0
        for p in model.parameters():
            total += p.data.norm(2).item()**2
        return total ** 0.5

    record_steps, record_loss = [], []
    batch_train_acc, val_steps, val_acc = [], [], []      # val_acc = (pos+neg)/2
    grad_norms, param_norms = [], []
    best_val_acc, best_step = -1.0, -1

    # extra histories for pos/neg
    val_acc_pos_hist, val_acc_neg_hist = [], []
    best_val_acc_pos, best_step_pos = -1.0, -1
    best_val_acc_neg, best_step_neg = -1.0, -1

    for step in range(total_steps):
        idx = torch.randint(0, x.size(0), (batch_size,), device=device)
        xb, yb = x[idx], y[idx]

        optimizer.zero_grad()
        out = model(xb)
        pred = out[:, -1, -1]
        loss = 0.5 * (pred - yb).pow(2).mean()

        loss.backward()

        if clip_grad is not None:
            grad_norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        else:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    pn = p.grad.data.norm(2)
                    total_norm += pn.item() ** 2
            grad_norm_value = total_norm ** 0.5

        optimizer.step()
        scheduler.step()

        if (step % record_interval == 0) or (step == total_steps - 1):
            record_steps.append(step)
            record_loss.append(loss.item())
            grad_norms.append(grad_norm_value)
            param_norms.append(param_norm(model))

            acc_train_batch = ((pred * yb) > 0).float().mean().item()
            batch_train_acc.append(acc_train_batch)

            model.eval()
            with torch.no_grad():
                # pos-only
                out_pos = model(x_val_pos)
                pred_pos = out_pos[:, -1, -1]
                acc_pos = ((pred_pos * y_val_pos) > 0).float().mean().item()
                # neg-only
                out_neg = model(x_val_neg)
                pred_neg = out_neg[:, -1, -1]
                acc_neg = ((pred_neg * y_val_neg) > 0).float().mean().item()
                # average
                acc_v = 0.5 * (acc_pos + acc_neg)
            model.train()

            val_steps.append(step)
            val_acc.append(acc_v)
            val_acc_pos_hist.append(acc_pos)
            val_acc_neg_hist.append(acc_neg)

            # track per-split bests
            if acc_pos > best_val_acc_pos:
                best_val_acc_pos, best_step_pos = acc_pos, step
            if acc_neg > best_val_acc_neg:
                best_val_acc_neg, best_step_neg = acc_neg, step

            # checkpoint by avg acc
            if acc_v > best_val_acc:
                best_val_acc, best_step = acc_v, step
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)

            if step % 5000 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Step {step:6d}, Loss={loss.item():.4f}, "
                      f"TrainAcc={acc_train_batch:.3f}, "
                      f"ValAcc(avg)={acc_v:.3f}, ValPos={acc_pos:.3f}, ValNeg={acc_neg:.3f}, "
                      f"GradNorm={grad_norm_value:.3f}, ParamNorm={param_norms[-1]:.3f}, LR={lr:.6f}")

    return {
        "label": label,
        "record_steps": record_steps,
        "record_loss": record_loss,
        "batch_train_acc": batch_train_acc,
        "val_steps": val_steps,
        "val_acc": val_acc,                 # average of pos/neg per record
        "val_acc_pos": val_acc_pos_hist,    # pos-only history
        "val_acc_neg": val_acc_neg_hist,    # neg-only history
        "best_acc": best_val_acc,           # best of avg
        "best_step": best_step,
        "best_acc_pos": best_val_acc_pos,
        "best_step_pos": best_step_pos,
        "best_acc_neg": best_val_acc_neg,
        "best_step_neg": best_step_neg,
        "save_path": save_path,
        "grad_norms": grad_norms,
        "param_norms": param_norms
    }

# Ploting result

# Plot function (adds pos-only & neg-only validation figures)
def plot_results(results, title_suffix: str = ""):
    # ---- 1) Training Loss ----
    plt.figure()
    for res in results:
        plt.plot(res["record_steps"], res["record_loss"], label=res["label"])
    plt.xlabel("Step"); plt.ylabel("MSE Loss")
    plt.title(f"Training Loss vs Steps{title_suffix}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    # ---- 2) Training Accuracy ----
    plt.figure()
    for res in results:
        acc_train_pct = [a * 100.0 for a in res["batch_train_acc"]]
        plt.plot(res["record_steps"], acc_train_pct, label=res["label"])
    plt.xlabel("Step"); plt.ylabel("Training Accuracy (%)")
    plt.title(f"Training Accuracy vs Steps{title_suffix}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    # ---- 3) Validation Accuracy (avg of pos/neg) ----
    plt.figure()
    for res in results:
        acc_val_pct = [a * 100.0 for a in res["val_acc"]]
        best_avg = res.get("best_acc", None)
        if best_avg is not None:
            label = f'{res["label"]} (best={best_avg*100:.1f}%)'
        else:
            label = res["label"]
        plt.plot(res["val_steps"], acc_val_pct, label=label)
    plt.xlabel("Step"); plt.ylabel("Validation Accuracy (%)")
    plt.title(f"Validation Accuracy (avg) vs Steps{title_suffix}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    # ---- 4) Validation Accuracy on POS-ONLY ----
    has_any_pos = any("val_acc_pos" in r for r in results)
    if has_any_pos:
        plt.figure()
        for res in results:
            if "val_acc_pos" not in res:
                continue
            acc_pos_pct = [a * 100.0 for a in res["val_acc_pos"]]
            best_pos = res.get("best_acc_pos", None)
            if best_pos is not None:
                label = f'{res["label"]} (best pos={best_pos*100:.1f}%)'
            else:
                label = res["label"]
            plt.plot(res["val_steps"], acc_pos_pct, label=label)
        plt.xlabel("Step"); plt.ylabel("Pos-only Validation Accuracy (%)")
        plt.title(f"Validation Accuracy (pos-only) vs Steps{title_suffix}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.show()

    # ---- 5) Validation Accuracy on NEG-ONLY ----
    has_any_neg = any("val_acc_neg" in r for r in results)
    if has_any_neg:
        plt.figure()
        for res in results:
            if "val_acc_neg" not in res:
                continue
            acc_neg_pct = [a * 100.0 for a in res["val_acc_neg"]]
            best_neg = res.get("best_acc_neg", None)
            if best_neg is not None:
                label = f'{res["label"]} (best neg={best_neg*100:.1f}%)'
            else:
                label = res["label"]
            plt.plot(res["val_steps"], acc_neg_pct, label=label)
        plt.xlabel("Step"); plt.ylabel("Neg-only Validation Accuracy (%)")
        plt.title(f"Validation Accuracy (neg-only) vs Steps{title_suffix}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.show()

# (a) One-Layer Linear Attention-Only Transformer

class SingleLayerLinearAttentionTransformer(nn.Module):
    def __init__(self, 
                 d: int,
                 dropout: float = 0.0):
        
        super().__init__()
        self.d_model = d + 1

        # Projection-value merged projection P
        self.P = nn.Linear(self.d_model, self.d_model, bias=False)
        # Key-query merged projection Q
        self.Q = nn.Linear(self.d_model, self.d_model, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor):
        # x: (B, seq_len, d_model)
        # x = Z^T
        # Project to value and key-query spaces
        
        seq_len = x.shape[1]
        n = seq_len - 1

        # Masked value matrix
        M = x.new_zeros(seq_len, seq_len)
        if n > 0:
            M[:n, :n] = torch.eye(n, device=x.device, dtype=x.dtype)
    
        PV = self.P(x)                                  
        KQ = self.Q(x)                                  

        MPV = torch.matmul(M, PV)

        denom = max(n, 1) * math.sqrt(self.d_model - 1)
        SMPV = MPV / denom

        scores = torch.matmul(KQ, x.transpose(-2, -1))

        scores = self.dropout(scores)
        
        attn   = torch.matmul(scores, SMPV)            
        return x + attn

# L = 1, n = 60, zero mean

# One-layer Linear: n=60, zero-mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ls_linear_L1_n60 = SingleLayerLinearAttentionTransformer(d=20).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n60).float().to(device)
y = torch.from_numpy(y_query_train_ls_n60).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n60).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n60).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n60).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n60).float().to(device)

# ---- Train with dual validation (avg used for checkpointing) ----
ls_linear_L1_n60 = train_model(
    model_ls_linear_L1_n60,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=1e-4,
    pct_start=0.30,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    label="ls-linear-L1-n60",
    save_path="ls_linear_L1_n60.pt"
)

# Plots
plot_results([ls_linear_L1_n60], title_suffix=" (ls-linear-L1, n=60)")

# L = 1, n = 40, zero mean

# One-layer Linear: n=40, zero-mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ls_linear_L1_n40 = SingleLayerLinearAttentionTransformer(d=20).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n40).float().to(device)
y = torch.from_numpy(y_query_train_ls_n40).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n40).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n40).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n40).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n40).float().to(device)

# ---- Train with dual validation (avg used for checkpointing) ----
ls_linear_L1_n40 = train_model(
    model_ls_linear_L1_n40,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=1e-4,
    pct_start=0.30,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    label="ls-linear-L1-n40",
    save_path="ls_linear_L1_n40.pt"
)

# Plots
plot_results([ls_linear_L1_n40], title_suffix=" (ls-linear-L1, n=40)")

# (b) Multi-Layer Linear Attention-Only Transformer

class MultiLayerLinearAttentionTransformer(nn.Module):
    def __init__(self, 
                 d: int, 
                 L: int, 
                 dropout: float = 0.0):
        
        super().__init__()
        self.d_model = d + 1
        self.L = L

        self.P = nn.ModuleList([nn.Linear(self.d_model, self.d_model, bias=False) for _ in range(L)])
        self.Q = nn.ModuleList([nn.Linear(self.d_model, self.d_model, bias=False) for _ in range(L)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: (B, seq_len, d_model)
        B, seq_len, _ = x.shape
        n = seq_len - 1

        # dynamic mask
        M = x.new_zeros(seq_len, seq_len)
        if n > 0:
            M[:n, :n] = torch.eye(n, device=x.device, dtype=x.dtype)

        scale = max(n, 1) * math.sqrt(self.d_model - 1)

        for i in range(self.L):
            PV = self.P[i](x)                     
            KQ = self.Q[i](x)                     

            MPV = torch.matmul(M, PV)
            SMPV = MPV / scale

            scores = torch.matmul(KQ, x.transpose(-2, -1))
            scores = self.dropout(scores)

            attn = torch.matmul(scores, SMPV)
            x = x + attn

        return x

# L = 2, n = 60, zero mean

# Multi-layer Linear: n=60, zero-mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ls_linear_L2_n60 = MultiLayerLinearAttentionTransformer(d=20, L=2).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n60).float().to(device)
y = torch.from_numpy(y_query_train_ls_n60).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n60).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n60).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n60).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n60).float().to(device)

x = torch.from_numpy(dataset_train_ls_n60).float().to(device)
y = torch.from_numpy(y_query_train_ls_n60).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n60).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n60).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n60).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n60).float().to(device)

# ---- Train with dual validation (avg used for checkpointing) ----
ls_linear_L2_n60 = train_model(
    model_ls_linear_L2_n60,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=1e-4,
    pct_start=0.30,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    label="ls-linear-L2-n60",
    save_path="ls_linear_L2_n60.pt"
)

# Plots
plot_results([ls_linear_L2_n60], title_suffix=" (ls-linear-L2, n=60)")

# L = 2, n = 40, zero mean

# Multi-layer Linear: n=40, zero-mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ls_linear_L2_n40 = MultiLayerLinearAttentionTransformer(d=20, L=2).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n40).float().to(device)
y = torch.from_numpy(y_query_train_ls_n40).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n40).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n40).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n40).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n40).float().to(device)

# ---- Train with dual validation (avg used for checkpointing) ----
ls_linear_L2_n40 = train_model(
    model_ls_linear_L2_n40,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=1e-4,
    pct_start=0.30,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    label="ls-linear-L2-n40",
    save_path="ls_linear_L2_n40.pt"
)

# Plots
plot_results([ls_linear_L2_n40], title_suffix=" (ls-linear-L2, n=40)")

# L = 3, n = 60, zero mean

# Multi-layer Linear: n=60, zero-mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ls_linear_L3_n60 = MultiLayerLinearAttentionTransformer(d=20, L=3).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n60).float().to(device)
y = torch.from_numpy(y_query_train_ls_n60).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n60).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n60).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n60).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n60).float().to(device)

x = torch.from_numpy(dataset_train_ls_n60).float().to(device)
y = torch.from_numpy(y_query_train_ls_n60).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n60).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n60).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n60).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n60).float().to(device)

# ---- Train with dual validation (avg used for checkpointing) ----
ls_linear_L3_n60 = train_model(
    model_ls_linear_L3_n60,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=1e-4,
    pct_start=0.30,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    label="ls-linear-L3-n60",
    save_path="ls_linear_L3_n60.pt"
)

# Plots
plot_results([ls_linear_L3_n60], title_suffix=" (ls-linear-L3, n=60)")

# L = 3, n = 40, zero mean

# Multi-layer Linear: n=40, zero-mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ls_linear_L3_n40 = MultiLayerLinearAttentionTransformer(d=20, L=3).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n40).float().to(device)
y = torch.from_numpy(y_query_train_ls_n40).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n40).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n40).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n40).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n40).float().to(device)

x = torch.from_numpy(dataset_train_ls_n40).float().to(device)
y = torch.from_numpy(y_query_train_ls_n40).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n40).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n40).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n40).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n40).float().to(device)

# ---- Train with dual validation (avg used for checkpointing) ----
ls_linear_L3_n40 = train_model(
    model_ls_linear_L3_n40,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=1e-4,
    pct_start=0.30,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    label="ls-linear-L3-n40",
    save_path="ls_linear_L3_n40.pt"
)

# Plots
plot_results([ls_linear_L3_n40], title_suffix=" (ls-linear-L3, n=40)")

# L = 4, n = 60, zero mean

# Multi-layer Linear: n=60, zero-mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ls_linear_L4_n60 = MultiLayerLinearAttentionTransformer(d=20, L=4).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n60).float().to(device)
y = torch.from_numpy(y_query_train_ls_n60).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n60).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n60).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n60).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n60).float().to(device)

x = torch.from_numpy(dataset_train_ls_n60).float().to(device)
y = torch.from_numpy(y_query_train_ls_n60).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n60).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n60).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n60).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n60).float().to(device)

# ---- Train with dual validation (avg used for checkpointing) ----
ls_linear_L4_n60 = train_model(
    model_ls_linear_L4_n60,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=1e-4,
    pct_start=0.30,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    label="ls-linear-L4-n60",
    save_path="ls_linear_L4_n60.pt"
)

# Plots
plot_results([ls_linear_L4_n60], title_suffix=" (ls-linear-L4, n=60)")

# L = 4, n = 40, zero mean

# Multi-layer Linear: n=40, zero-mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ls_linear_L4_n40 = MultiLayerLinearAttentionTransformer(d=20, L=4).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n40).float().to(device)
y = torch.from_numpy(y_query_train_ls_n40).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n40).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n40).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n40).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n40).float().to(device)

x = torch.from_numpy(dataset_train_ls_n40).float().to(device)
y = torch.from_numpy(y_query_train_ls_n40).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n40).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n40).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n40).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n40).float().to(device)

# ---- Train with dual validation (avg used for checkpointing) ----
ls_linear_L4_n40 = train_model(
    model_ls_linear_L4_n40,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=1e-4,
    pct_start=0.30,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    label="ls-linear-L4-n40",
    save_path="ls_linear_L4_n40.pt"
)

# Plots
plot_results([ls_linear_L4_n40], title_suffix=" (ls-linear-L4, n=40)")

# (c) One-Layer Softmax Attention-Only Transformer

class SingleLayerSoftmaxAttentionTransformer(nn.Module):
    
    def __init__(self, 
                 d: int, 
                 dropout: float = 0.0):
        
        super().__init__()
        self.d_model = d + 1

        # Projection for value update
        self.P = nn.Linear(self.d_model, self.d_model, bias=False)

        # Projection for key-query interaction
        self.Q = nn.Linear(self.d_model, self.d_model, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: (B, seq_len, d_model)
        B, seq_len, _ = x.shape
        n = seq_len - 1

        # dynamic mask
        M = x.new_zeros(seq_len, seq_len)
        if n > 0:
            M[:n, :n] = torch.eye(n, device=x.device, dtype=x.dtype)

        scale = max(n, 1)

        PV = self.P(x)               
        KQ = self.Q(x)               

        MPV = torch.matmul(M, PV)     
        SMPV = MPV

        # Scores
        scores = torch.matmul(KQ, x.transpose(-2, -1))   # (B, L, L)
        scores = scores / math.sqrt(self.d_model - 1)

        # Mask the last column by using -inf
        if n > 0:
            scores[..., :, n] = float("-inf")

        # row-wise softmax
        scores = F.softmax(scores, dim=-1)

        scores = self.dropout(scores)

        attn = torch.matmul(scores, SMPV)

        return x + attn

# L = 1, n = 60, zero mean

# One-layer Softmax: n=60, zero-mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ls_softmax_L1_n60 = SingleLayerSoftmaxAttentionTransformer(d=20).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n60).float().to(device)
y = torch.from_numpy(y_query_train_ls_n60).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n60).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n60).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n60).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n60).float().to(device)

x = torch.from_numpy(dataset_train_ls_n60).float().to(device)
y = torch.from_numpy(y_query_train_ls_n60).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n60).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n60).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n60).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n60).float().to(device)

# ---- Train with dual validation (avg used for checkpointing) ----
ls_softmax_L1_n60 = train_model(
    model_ls_softmax_L1_n60,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=5e-3,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=1e-5,
    label="ls-softmax-L1-n60",
    save_path="ls_softmax_L1_n60.pt"
)

# Plots
plot_results([ls_softmax_L1_n60], title_suffix=" (ls-softmax-L1, n=60)")

# L = 1, n = 40, zero mean

# One-layer Softmax: n=40, zero mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ls_softmax_L1_n40 = SingleLayerSoftmaxAttentionTransformer(d=20).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n40).float().to(device)
y = torch.from_numpy(y_query_train_ls_n40).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n40).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n40).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n40).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n40).float().to(device)

# ---- Train with split validation (avg used for checkpointing) ----
ls_softmax_L1_n40 = train_model(
    model_ls_softmax_L1_n40,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=5e-3,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=1e-5,
    label="ls-softmax-L1-n40",
    save_path="ls_softmax_L1_n40.pt"
)

# Plot
plot_results([ls_softmax_L1_n40], title_suffix=" (ls-softmax-L1, n=40)")

# (d) Multi-Layer Softmax Attention-Only Transformer

class MultiLayerSoftmaxAttentionTransformer(nn.Module):
    
    def __init__(self, 
                 d: int, 
                 L: int, 
                 dropout: float = 0.0):
        
        super().__init__()
        self.d_model = d + 1
        self.L = L

        # One projection pair per layer
        self.P = nn.ModuleList([nn.Linear(self.d_model, self.d_model, bias=False) for _ in range(L)])
        self.Q = nn.ModuleList([nn.Linear(self.d_model, self.d_model, bias=False) for _ in range(L)])

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: (B, seq_len, d_model)
        B, seq_len, _ = x.shape
        n = seq_len - 1

        # dynamic mask
        M = x.new_zeros(seq_len, seq_len)
        if n > 0:
            M[:n, :n] = torch.eye(n, device=x.device, dtype=x.dtype)

        scale = max(n, 1)

        for i in range(self.L):
            PV = self.P[i](x)               
            KQ = self.Q[i](x)               

            MPV = torch.matmul(M, PV)     
            SMPV = MPV

            scores = torch.matmul(KQ, x.transpose(-2, -1))
            scores = scores / math.sqrt(self.d_model - 1)

            if n > 0:
                scores[..., :, n] = float("-inf")

            scores = F.softmax(scores, dim=-1)
            scores = self.dropout(scores)

            attn = torch.matmul(scores, SMPV)
            x = x + attn

        return x

# L = 2, n = 60, zero mean

# Multi-layer Softmax: n=60, L=2, zero mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ls_model_softmax_L2_n60 = MultiLayerSoftmaxAttentionTransformer(d=20, L=2).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n60).float().to(device)
y = torch.from_numpy(y_query_train_ls_n60).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n60).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n60).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n60).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n60).float().to(device)

# ---- Train with split validation (avg used for checkpointing) ----
ls_softmax_L2_n60 = train_model(
    ls_model_softmax_L2_n60,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=5e-3,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=5e-5,
    label="ls-softmax-L2-n60",
    save_path="ls_softmax_L2_n60.pt"
)

# Plot
plot_results([ls_softmax_L2_n60], title_suffix=" (ls-softmax-L2, n=60)")

# L = 2, n = 40, zero mean

# Multi-layer Softmax: n=40, L=2, zero mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_softmax_L2_n40 = MultiLayerSoftmaxAttentionTransformer(d=20, L=2).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n40).float().to(device)
y = torch.from_numpy(y_query_train_ls_n40).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n40).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n40).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n40).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n40).float().to(device)

# ---- Train with split validation (avg used for checkpointing) ----
ls_softmax_L2_n40 = train_model(
    model_softmax_L2_n40,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=5e-3,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=5e-5,
    label="ls-softmax-L2-n40",
    save_path="ls_softmax_L2_n40.pt"
)

# ---- Plot ----
plot_results([ls_softmax_L2_n40], title_suffix=" (ls-softmax-L2, n=40)")

# L = 3, n = 60, zero mean

# Multi-layer Softmax: n=60, L=3, zero mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ls_softmax_L3_n60 = MultiLayerSoftmaxAttentionTransformer(d=20, L=3, dropout=0.0).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n60).float().to(device)
y = torch.from_numpy(y_query_train_ls_n60).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n60).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n60).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n60).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n60).float().to(device)

# ---- Train with split validation (avg used for checkpointing) ----
ls_softmax_L3_n60 = train_model(
    model_ls_softmax_L3_n60,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=4e-4,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=2e-4,
    label="ls-softmax-L3-n60",
    save_path="ls_softmax_L3_n60.pt"
)

# Plot
plot_results([ls_softmax_L3_n60], title_suffix=" (ls-softmax-L3, n=60)")

# L = 3, n = 40, zero mean

# Multi-layer Softmax: n=40, L=3, zero mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ls_softmax_L3_n40 = MultiLayerSoftmaxAttentionTransformer(d=20, L=3, dropout=0.0).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n40).float().to(device)
y = torch.from_numpy(y_query_train_ls_n40).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n40).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n40).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n40).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n40).float().to(device)

# ---- Train with split validation (avg used for checkpointing) ----
ls_softmax_L3_n40 = train_model(
    model_ls_softmax_L3_n40,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=4e-4,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=2e-4,
    label="ls-softmax-L3-n40",
    save_path="ls_softmax_L3_n40.pt"
)

# Plot
plot_results([ls_softmax_L3_n40], title_suffix=" (ls-softmax-L3, n=40)")

# L = 4, n = 60, zero mean

# Multi-layer Softmax: n=60, L=4, zero mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ls_softmax_L4_n60 = MultiLayerSoftmaxAttentionTransformer(d=20, L=4).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n60).float().to(device)
y = torch.from_numpy(y_query_train_ls_n60).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n60).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n60).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n60).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n60).float().to(device)

# ---- Train with split validation (avg used for checkpointing) ----
ls_softmax_L4_n60 = train_model(
    model_ls_softmax_L4_n60,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=0.5e-4,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=70.0,
    record_interval=500,
    weight_decay=4e-4,
    label="ls-softmax-L4-n60",
    save_path="ls_softmax_L4_n60.pt"
)

# Plot
plot_results([ls_softmax_L4_n60], title_suffix=" (ls-softmax-L4, n=60)")

# L = 4, n = 40, zero mean

# Multi-layer Softmax: n=40, L=4, zero mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ls_softmax_L4_n40 = MultiLayerSoftmaxAttentionTransformer(d=20, L=4).to(device)

# ---- Training data ----
x = torch.from_numpy(dataset_train_ls_n40).float().to(device)
y = torch.from_numpy(y_query_train_ls_n40).float().to(device)

# ---- Validation data: pos-only / neg-only ----
x_val_pos = torch.from_numpy(dataset_val_ls_pos_n40).float().to(device)
y_val_pos = torch.from_numpy(y_query_val_ls_pos_n40).float().to(device)

x_val_neg = torch.from_numpy(dataset_val_ls_neg_n40).float().to(device)
y_val_neg = torch.from_numpy(y_query_val_ls_neg_n40).float().to(device)

# ---- Train with split validation (avg used for checkpointing) ----
ls_softmax_L4_n40 = train_model(
    model_ls_softmax_L4_n40,
    x, y,
    x_val_pos, y_val_pos,
    x_val_neg, y_val_neg,
    total_steps=150000,
    batch_size=1024,
    max_lr=1e-4,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=70.0,
    record_interval=500,
    weight_decay=3e-4,
    label="ls-softmax-L4-n40",
    save_path="ls_softmax_L4_n40.pt"
)

# Plot
plot_results([ls_softmax_L4_n40], title_suffix=" (ls-softmax-L4, n=40)")

# Baseline Model: GPT-2

# =======================
# GPT-2-style Causal ICL: train on your dataset, eval on your val_pos/val_neg
# =======================

# -------- device & seed --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed=42):
    import random
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# -------- adapter: (B, n+1, d+1) -> x:(B,n+1,d), y:(B,n+1) in {±1} --------
def pack_for_model(dataset_np: np.ndarray, y_query_np: np.ndarray, device=DEVICE):
    data = torch.from_numpy(dataset_np).to(device).float()
    x = data[:, :, :-1]                 # (B, n+1, d)
    y_ctx = data[:, :-1, -1]            # (B, n)
    yq = torch.from_numpy(y_query_np).to(device).float().unsqueeze(1)  # (B,1)
    y = torch.cat([y_ctx, yq], dim=1)   # (B, n+1)
    return x, y

@torch.no_grad()
def eval_query_acc(model: nn.Module, dataset_np: np.ndarray, y_query_np: np.ndarray,
                   batch_size=1024, device=DEVICE):
    model.eval()
    B = dataset_np.shape[0]; correct = 0
    for i in range(0, B, batch_size):
        x, y = pack_for_model(dataset_np[i:i+batch_size], y_query_np[i:i+batch_size], device)
        y_hat = model(x, y)               # (b, n+1)
        pred = torch.sign(y_hat[:, -1])
        pred[pred == 0] = 1.0
        correct += (pred == y[:, -1]).sum().item()
    return correct / B

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, drop: float):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop  = nn.Dropout(drop)
        self.resid_drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        B, T, E = x.shape
        H, Dh = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # (B,H,T,Dh)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)         # (B,H,T,T)
        att = att.masked_fill(~attn_mask, torch.finfo(att.dtype).min)
        att = self.attn_drop(att.softmax(dim=-1))
        y = att @ v                                           # (B,H,T,Dh)
        y = y.transpose(1, 2).contiguous().view(B, T, E)
        return self.resid_drop(self.o_proj(y))

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, drop: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x): return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))

class TransformerBlock(nn.Module):
    """Pre-LN: x = x + MHA(LN(x)); x = x + FFN(LN(x))"""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, drop: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim); self.attn = MultiHeadAttention(embed_dim, num_heads, drop)
        self.ln2 = nn.LayerNorm(embed_dim); self.ffn  = FeedForward(embed_dim, ffn_dim, drop)
    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x

class NumericEmbed(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim); self.proj = nn.Linear(in_dim, embed_dim)
    def forward(self, v): return self.proj(self.ln(v))  # (B,T,embed_dim)

class GPT2(nn.Module):
    def __init__(self, d_feature: int, embed_dim=256, num_layers=4, num_heads=8,
                 ffn_dim=512, max_len=256, drop=0.10):
        super().__init__()
        self.embed_dim = embed_dim
        self.ex = NumericEmbed(d_feature, embed_dim)
        self.ey = NumericEmbed(d_feature, embed_dim)
        self.pos = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ffn_dim, drop)
                                     for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    @staticmethod
    def causal_mask(T: int, device):
        return torch.ones(T, T, device=device, dtype=torch.bool).tril().unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        x: (B, n+1, d)   y: (B, n+1) in {±1}
        """
        B, Txy, d = x.shape
        L = 2 * Txy
        assert L <= self.pos.num_embeddings, "positional embedding too short; increase max_len"

        y_vec = torch.zeros(B, Txy, d, dtype=x.dtype, device=x.device)
        y_vec[..., 0] = y

        X = self.ex(x); Y = self.ey(y_vec)
        seq = torch.stack([X, Y], dim=2).reshape(B, L, self.embed_dim)
        seq = self.drop(seq + self.pos(torch.arange(L, device=x.device).unsqueeze(0)))

        mask = self.causal_mask(L, device=x.device)
        h = seq
        for blk in self.blocks: h = blk(h, mask)
        h = self.ln_f(h)

        idx_x = torch.arange(0, L, 2, device=x.device)
        y_hat = self.head(h.index_select(dim=1, index=idx_x)).squeeze(-1)  # (B,Txy)
        return y_hat

# GPT-2, n = 60, zero mean

def train_gpt2(
    model: nn.Module,
    train_ds: np.ndarray, train_yq: np.ndarray,
    val_pos: tuple | None = None,   # (dataset_pos, yq_pos)
    val_neg: tuple | None = None,   # (dataset_neg, yq_neg)
    *, steps=50000, batch_size=256, lr=3e-4, weight_decay=1e-2,
    clip_grad: float | None = 1.0, record_interval=500,
    label="gpt2icl", save_path: str | None = None, verbose=False
):
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    record_steps, record_loss = [], []
    batch_train_acc, val_steps, val_acc = [], [], []
    val_acc_pos_hist, val_acc_neg_hist = [], []
    best_val_acc, best_step = -1.0, -1
    best_val_acc_pos, best_step_pos = -1.0, -1
    best_val_acc_neg, best_step_neg = -1.0, -1

    N = train_ds.shape[0]
    for step in range(1, steps + 1):
        # minibatch
        idx = np.random.randint(0, N, size=(batch_size,))
        x_b, y_b = pack_for_model(train_ds[idx], train_yq[idx], device)

        y_hat = model(x_b, y_b)                 # (bs, n+1)
        loss = F.mse_loss(y_hat, y_b)

        # backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        opt.step()

        if (step % record_interval == 0) or (step == steps):
            model.eval()
            with torch.no_grad():
                pred_tr = torch.sign(y_hat[:, -1]); pred_tr[pred_tr == 0] = 1.0
                acc_tr = (pred_tr == y_b[:, -1]).float().mean().item()

                if (val_pos is not None) and (val_neg is not None):
                    acc_pos = eval_query_acc(model, val_pos[0], val_pos[1], device=device)
                    acc_neg = eval_query_acc(model, val_neg[0], val_neg[1], device=device)
                    acc_v = 0.5 * (acc_pos + acc_neg)
                    val_steps.append(step); val_acc.append(acc_v)
                    val_acc_pos_hist.append(acc_pos); val_acc_neg_hist.append(acc_neg)

                    if acc_pos > best_val_acc_pos: best_val_acc_pos, best_step_pos = acc_pos, step
                    if acc_neg > best_val_acc_neg: best_val_acc_neg, best_step_neg = acc_neg, step
                    if acc_v > best_val_acc:
                        best_val_acc, best_step = acc_v, step
                        if save_path is not None: torch.save(model.state_dict(), save_path)

                record_steps.append(step); record_loss.append(float(loss.item()))
                batch_train_acc.append(acc_tr)
            model.train()

            if verbose:
                if (val_pos is not None) and (val_neg is not None):
                    print(f"[{step:6d}] loss={loss.item():.4f} train_acc={acc_tr:.3f} "
                          f"val_pos={acc_pos:.3f} val_neg={acc_neg:.3f} avg={acc_v:.3f}")
                else:
                    print(f"[{step:6d}] loss={loss.item():.4f} train_acc={acc_tr:.3f}")

    return {
        "label": label,
        "record_steps": record_steps,
        "record_loss": record_loss,
        "batch_train_acc": batch_train_acc,
        "val_steps": val_steps,
        "val_acc": val_acc,                 # avg(pos,neg)
        "val_acc_pos": val_acc_pos_hist,
        "val_acc_neg": val_acc_neg_hist,
        "best_acc": best_val_acc,
        "best_step": best_step,
        "best_acc_pos": best_val_acc_pos,
        "best_step_pos": best_step_pos,
        "best_acc_neg": best_val_acc_neg,
        "best_step_neg": best_step_neg,
        "save_path": save_path,
    }

# =======================
# =======================
if __name__ == "__main__":
    set_seed(42)

    train_ds = dataset_train_ls_n60          # (B, n+1, d+1)
    train_yq = y_query_train_ls_n60          # (B,)
    val_pos_ds = dataset_val_ls_pos_n60      # (B, n+1, d+1)
    val_pos_yq = y_query_val_ls_pos_n60      # (B,)
    val_neg_ds = dataset_val_ls_neg_n60
    val_neg_yq = y_query_val_ls_neg_n60

    Txy = train_ds.shape[1]                  # = n+1
    d = train_ds.shape[2] - 1
    max_len = 2 * Txy

    model = GPT2(
        d_feature=d,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        ffn_dim=512,
        max_len=max_len,
        drop=0.10
    ).to(DEVICE)

    ls_gpt2_n60 = train_gpt2(
        model,
        train_ds, train_yq,
        val_pos=(val_pos_ds, val_pos_yq),
        val_neg=(val_neg_ds, val_neg_yq),
        steps=50000,
        batch_size=256,
        lr=1e-4,
        weight_decay=1.5e-4,
        clip_grad=2.0,
        record_interval=500,
        label="ls-gpt2-n60",
        save_path="ls_gpt2_n60.pt",
        verbose=False
    )

    with torch.no_grad():
        acc_pos = eval_query_acc(model, val_pos_ds, val_pos_yq, device=DEVICE)
        acc_neg = eval_query_acc(model, val_neg_ds, val_neg_yq, device=DEVICE)
        acc_avg = 0.5 * (acc_pos + acc_neg)
    print(f"[FINAL] val_pos={acc_pos:.4f}  val_neg={acc_neg:.4f}  avg={acc_avg:.4f}")
    print(f"[BEST(avg)] step={ls_gpt2_n60['best_step']}  acc={ls_gpt2_n60['best_acc']:.4f}")
    print("Checkpoint saved to:", ls_gpt2_n60["save_path"])

# Plot
plot_results([ls_gpt2_n60], title_suffix=" (ls-gpt2, n=60)")

# Illustration plot

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------- helpers ----------
def ema_smooth(y, alpha=0.2):
    y = np.asarray(y, dtype=float)
    out, m, started = np.empty_like(y), None, False
    for i, v in enumerate(y):
        if np.isnan(v):
            out[i] = np.nan if not started else m
            continue
        if not started:
            m, started = v, True
        else:
            m = alpha * v + (1 - alpha) * m
        out[i] = m
    return out

def _get(res, key):
    return np.asarray(res[key], dtype=float) if key in res else None

def pad_to_zero(steps, y):
    if steps is None or y is None or len(steps) == 0 or steps[0] <= 0:
        return steps, y
    return np.r_[0, steps], np.r_[y[0], y]

STYLE_HANDLES = [
    Line2D([], [], color='k', linestyle='-',  linewidth=1.8, label='Linear (solid)'),
    Line2D([], [], color='k', linestyle='--', linewidth=1.8, label='Softmax (dashed)'),
]
COLOR_HANDLES = [
    Line2D([], [], color="#000000", linestyle='-', linewidth=1.8, label='Loss (left axis)'),
    Line2D([], [], color="#F70606", linestyle='-', linewidth=1.8,
           label=r'$A^{+}$  ($y^{(\mathrm{query})}=+1$, right)'),
    Line2D([], [], color="#0431B7", linestyle='-', linewidth=1.8,
           label=r'$A^{-}$  ($y^{(\mathrm{query})}=-1$, right)'),
]

def plot_loss_pos_neg_panel(
    res_linear, res_softmax,
    title="Linear Classification Task",
    ema_alpha=0.25,
    xmax_show=100000,
    acc_top=85.0,
    acc_bottom=None,
    *,
    font_size=11,
    ytick_size=None,
    fig_width=7.0,
    lw_loss=1.8,
    lw_acc=1.8
):
    steps_loss_lin = _get(res_linear, "record_steps")
    loss_lin       = _get(res_linear, "record_loss")
    steps_val_lin  = _get(res_linear, "val_steps")
    pos_lin        = _get(res_linear, "val_acc_pos")
    neg_lin        = _get(res_linear, "val_acc_neg")

    steps_loss_smx = _get(res_softmax, "record_steps")
    loss_smx       = _get(res_softmax, "record_loss")
    steps_val_smx  = _get(res_softmax, "val_steps")
    pos_smx        = _get(res_softmax, "val_acc_pos")
    neg_smx        = _get(res_softmax, "val_acc_neg")

    if loss_lin is not None: loss_lin = ema_smooth(loss_lin, ema_alpha)
    if loss_smx is not None: loss_smx = ema_smooth(loss_smx, ema_alpha)
    if pos_lin  is not None: pos_lin  = ema_smooth(pos_lin,  ema_alpha) * 100
    if neg_lin  is not None: neg_lin  = ema_smooth(neg_lin,  ema_alpha) * 100
    if pos_smx  is not None: pos_smx  = ema_smooth(pos_smx,  ema_alpha) * 100
    if neg_smx  is not None: neg_smx  = ema_smooth(neg_smx,  ema_alpha) * 100

    steps_loss_lin, loss_lin = pad_to_zero(steps_loss_lin, loss_lin)
    steps_loss_smx, loss_smx = pad_to_zero(steps_loss_smx, loss_smx)
    steps_val_lin,  pos_lin  = pad_to_zero(steps_val_lin,  pos_lin)
    steps_val_lin,  neg_lin  = pad_to_zero(steps_val_lin,  neg_lin)
    steps_val_smx,  pos_smx  = pad_to_zero(steps_val_smx,  pos_smx)
    steps_val_smx,  neg_smx  = pad_to_zero(steps_val_smx,  neg_smx)

    aspect = 4.6 / 7.0
    fig_height = fig_width * aspect
    fig, ax_loss = plt.subplots(figsize=(fig_width, fig_height))
    ax_acc = ax_loss.twinx()
    fig.suptitle(title, fontsize=font_size + 1)

    COL_POS, COL_NEG, COL_LOSS = "#F70606", "#0431B7", "#000000"
    ls_lin, ls_smx = "-", "--"

    if steps_loss_lin is not None and loss_lin is not None:
        ax_loss.plot(steps_loss_lin, loss_lin, color=COL_LOSS, linestyle=ls_lin, linewidth=lw_loss)
    if steps_loss_smx is not None and loss_smx is not None:
        ax_loss.plot(steps_loss_smx, loss_smx, color=COL_LOSS, linestyle=ls_smx, linewidth=lw_loss)
    ax_loss.set_ylabel("Training Loss", fontsize=font_size)

    if steps_val_lin is not None and pos_lin is not None:
        ax_acc.plot(steps_val_lin, pos_lin, color=COL_POS, linestyle=ls_lin, linewidth=lw_acc)
    if steps_val_lin is not None and neg_lin is not None:
        ax_acc.plot(steps_val_lin, neg_lin, color=COL_NEG, linestyle=ls_lin, linewidth=lw_acc)
    if steps_val_smx is not None and pos_smx is not None:
        ax_acc.plot(steps_val_smx, pos_smx, color=COL_POS, linestyle=ls_smx, linewidth=lw_acc)
    if steps_val_smx is not None and neg_smx is not None:
        ax_acc.plot(steps_val_smx, neg_smx, color=COL_NEG, linestyle=ls_smx, linewidth=lw_acc)
    ax_acc.set_ylabel("Validation Accuracy (%)", fontsize=font_size)

    if acc_bottom is not None:
        y_low = float(acc_bottom)
    else:
        acc_arrays = [a for a in [pos_lin, neg_lin, pos_smx, neg_smx] if a is not None and len(a) > 0]
        if acc_arrays:
            acc_all = np.concatenate(acc_arrays)
            acc_all = acc_all[~np.isnan(acc_all)]
            y_low = max(0.0, 5.0 * np.floor(float(np.min(acc_all)) / 5.0)) if acc_all.size > 0 else 0.0
        else:
            y_low = 0.0
    y_low = min(y_low, acc_top - 1.0)
    ax_acc.set_ylim(y_low, acc_top)

    ax_loss.set_xlim(0, xmax_show)
    ax_loss.set_xlabel("Training step", fontsize=font_size)
    ax_loss.grid(True, alpha=0.3)

    if ytick_size is None:
        ytick_size = max(1, font_size - 1)
    ax_loss.tick_params(axis='x', labelsize=font_size - 1)
    ax_loss.tick_params(axis='y', labelsize=ytick_size)
    ax_acc.tick_params(axis='y', labelsize=ytick_size)

    fig.tight_layout()
    plt.show()
    return fig

fig_ls = plot_loss_pos_neg_panel(
    ls_linear_L1_n60, ls_softmax_L1_n60,
    title="Linear Classification Task",
    ema_alpha=0.25, xmax_show=100000,
    acc_top=83.0, acc_bottom=50.0,
    font_size=13, ytick_size=12, fig_width=7.5,
    lw_loss=2.0, lw_acc=2.0
)
fig_ls.savefig("panel_ls.png", dpi=300, bbox_inches="tight")

# Legend

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STYLE_HANDLES = [
    Line2D([], [], color='k', linestyle='-',  linewidth=1.8, label='linear-L1'),
    Line2D([], [], color='k', linestyle='--', linewidth=1.8, label='softmax-L1'),
]

COLOR_HANDLES = [
    Line2D([], [], color="#000000", linestyle='-', linewidth=1.8,
           label='Training loss'),
    Line2D([], [], color="#F70606", linestyle='-', linewidth=1.8,
           label=r'Accuracy of $y^{(\mathrm{query})}=+1$'),
    Line2D([], [], color="#0431B7", linestyle='-', linewidth=1.8,
           label=r'Accuracy of $y^{(\mathrm{query})}=-1$'),
]

fig = plt.figure(figsize=(3.2, 3.2))
ax = fig.add_axes([0,0,1,1]); ax.axis('off')

leg1 = fig.legend(handles=STYLE_HANDLES, loc='upper left',
                  bbox_to_anchor=(0.00, 1.00), frameon=False, fontsize=10, handlelength=2.8)
fig.canvas.draw()
leg2 = fig.legend(handles=COLOR_HANDLES, loc='upper left',
                  bbox_to_anchor=(0.00, 0.52), frameon=False, fontsize=10, handlelength=2.8)

plt.show()

fig.savefig("legend_preview.png", dpi=300, pad_inches=0.03)

# Together

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ls_path  = "panel_ls.png"
nls_path = "panel_nls.png"
leg_path = "legend_preview.png"

img_ls  = mpimg.imread(ls_path)
img_nls = mpimg.imread(nls_path)
img_leg = mpimg.imread(leg_path)

fig = plt.figure(figsize=(15, 5))
gs  = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.55], wspace=0.04)

ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(img_ls);  ax1.axis("off")
ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(img_nls); ax2.axis("off")
ax3 = fig.add_subplot(gs[0, 2]); ax3.imshow(img_leg); ax3.axis("off")

fig.savefig("panels_plus_legend.png", dpi=300, bbox_inches="tight")

plt.close(fig)

# Model Dictionary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_specs_ls = [
    # ls-linear (n=60)
    {"name":"ls-linear-L1-n60","arch":"ls-linear","L":1,"n":60,"path":"ls_linear_L1_n60.pt"},
    {"name":"ls-linear-L2-n60","arch":"ls-linear","L":2,"n":60,"path":"ls_linear_L2_n60.pt"},
    {"name":"ls-linear-L3-n60","arch":"ls-linear","L":3,"n":60,"path":"ls_linear_L3_n60.pt"},
    {"name":"ls-linear-L4-n60","arch":"ls-linear","L":4,"n":60,"path":"ls_linear_L4_n60.pt"},

    # ls-softmax (n=60)
    {"name":"ls-softmax-L1-n60","arch":"ls-softmax","L":1,"n":60,"path":"ls_softmax_L1_n60.pt"},
    {"name":"ls-softmax-L2-n60","arch":"ls-softmax","L":2,"n":60,"path":"ls_softmax_L2_n60.pt"},
    {"name":"ls-softmax-L3-n60","arch":"ls-softmax","L":3,"n":60,"path":"ls_softmax_L3_n60.pt"},
    {"name":"ls-softmax-L4-n60","arch":"ls-softmax","L":4,"n":60,"path":"ls_softmax_L4_n60.pt"},

    # ls-gpt2 (n=60)
    {"name":"ls-gpt2-n60","arch":"ls-gpt2","L":4,"n":60,"path":"ls_gpt2_n60.pt"},
]

# 1.3 Testing Trained Model

# Evaluating model

# Evaluation function

def evaluate_model(model, x_np, y_np, device="cuda"):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x_np).float().to(device)
        y = torch.from_numpy(y_np).float().to(device)
        out  = model(x)
        pred = out[:, -1, -1]
        acc  = ((pred * y) > 0).float().mean().item()
    return acc

def evaluate_model_gpt2(model, dataset_np, y_query_np, batch_size=1024, device="cuda"):
    model.eval()
    B = dataset_np.shape[0]
    correct = 0
    for i in range(0, B, batch_size):
        x, y = pack_for_model(dataset_np[i:i+batch_size], y_query_np[i:i+batch_size], device)
        y_hat = model(x, y)                 # (b, n+1)
        pred = torch.sign(y_hat[:, -1])
        pred[pred == 0] = 1.0
        correct += (pred == y[:, -1]).sum().item()
    return correct / B

# Testing data

np.random.seed(21)

d = 20
batch_size_test = 3000
ns = list(range(10, 31, 4))  # 10-30

dataset_test_ls_pos = {}
dataset_test_ls_neg = {}

for n in ns:
    ds_base, yq_base, w_all = sample_ls_dataset(d, n, batch_size_test, margin=0.0, apply_margin_to_query=False)

    ds_pos  = ds_base.copy()
    yq_pos  = yq_base.copy()
    ds_pos, yq_pos = resample_ls_query_positive(ds_pos, yq_pos, w_all, margin=1e-4, max_iter=1000)
    dataset_test_ls_pos[n] = (ds_pos, yq_pos, w_all)

    ds_neg  = ds_base.copy()
    yq_neg  = yq_base.copy()
    ds_neg, yq_neg = resample_ls_query_negative(ds_neg, yq_neg, w_all, margin=1e-4, max_iter=1000)
    dataset_test_ls_neg[n] = (ds_neg, yq_neg, w_all)

check_query_all_positive("test ls pos-only", dataset_test_ls_pos[30][1])
check_query_all_negative("test ls neg-only", dataset_test_ls_neg[30][1])

# Testing result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_saved_model_ls(path, arch, L, device=device):
    state = torch.load(path, map_location=device)

    if arch == "ls-softmax":
        model = (SingleLayerSoftmaxAttentionTransformer(d=20)
                 if L == 1 else MultiLayerSoftmaxAttentionTransformer(d=20, L=L))

    elif arch == "ls-linear":
        model = (SingleLayerLinearAttentionTransformer(d=20)
                 if L == 1 else MultiLayerLinearAttentionTransformer(d=20, L=L))

    elif arch == "ls-gpt2":
        pos_weight = state["pos.weight"]              # shape: [max_len, embed_dim]
        max_len, embed_dim = pos_weight.shape
        model = GPT2(
            d_feature=20,
            embed_dim=embed_dim,   # = 256
            num_layers=L,
            num_heads=8,
            ffn_dim=512,
            max_len=122,
            drop=0.10
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.to(device)
    model.load_state_dict(state)
    model.eval()
    return model

# ===== LS: Test models on pos-only / neg-only / avg (memory-safe) =====

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

def _eval_one_ls(model, arch, x_np, y_np, device):
    if arch == "ls-gpt2":
        return evaluate_model_gpt2(model, x_np, y_np, device=device)
    else:
        return evaluate_model(model, x_np, y_np, device=device)

def _eval_chunked(model, arch, x_np, y_np, device, bs=1024):
    model.eval()
    B = x_np.shape[0]
    tot = 0.0
    with torch.inference_mode():
        for i in range(0, B, bs):
            j = min(i + bs, B)
            acc = _eval_one_ls(model, arch, x_np[i:j], y_np[i:j], device=device)
            tot += float(acc) * (j - i)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    return tot / B

def _eval_chunked_safe(model, arch, x_np, y_np, device, bs_try=(2048, 1024, 512, 256)):
    for bs in bs_try:
        try:
            return _eval_chunked(model, arch, x_np, y_np, device=device, bs=bs)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return _eval_chunked(model.to("cpu"), arch, x_np, y_np, device=torch.device("cpu"), bs=256)

rows = []

ns_eval = [n for n in sorted(dataset_test_ls_pos.keys()) if 10 <= n <= 60]

for sp in model_specs_ls:
    name, arch, L, n_train, path = sp["name"], sp["arch"], sp["L"], sp["n"], sp["path"]
    try:
        model = load_saved_model_ls(path, arch, L, device=device_cpu)
    except FileNotFoundError:
        print(f"[skip] missing checkpoint: {path}")
        continue

    model.to(device)

    for n_test in ns_eval:
        x_pos, y_pos, _ = dataset_test_ls_pos[n_test]
        x_neg, y_neg, _ = dataset_test_ls_neg[n_test]

        acc_pos = _eval_chunked_safe(model, arch, x_pos, y_pos, device=device)
        acc_neg = _eval_chunked_safe(model, arch, x_neg, y_neg, device=device)
        acc_avg = 0.5 * (acc_pos + acc_neg)

        rows.append({
            "name": name, "arch": arch, "L": L, "n_train": n_train, "n_test": n_test,
            "acc_pos": float(acc_pos), "acc_neg": float(acc_neg), "acc_avg": float(acc_avg),
        })

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

df_all = (
    pd.DataFrame(rows)
      .sort_values(by=["arch","L","n_train","n_test"])
      .reset_index(drop=True)
)

print("\nTest Results (pos / neg / avg):")
print(df_all.to_string(index=False))

pivot_avg = df_all.pivot_table(index=["arch","L","n_train"], columns="n_test", values="acc_avg").sort_index()
pivot_pos = df_all.pivot_table(index=["arch","L","n_train"], columns="n_test", values="acc_pos").sort_index()
pivot_neg = df_all.pivot_table(index=["arch","L","n_train"], columns="n_test", values="acc_neg").sort_index()

# --- Build a shared color map: same color for the same (L, n_train) across arches ---
import itertools
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

CUSTOM_PALETTE = ["#F70606", "#0431B7", "#087a08", "#ed870b"]

pairs = (
    df_all[['L','n_train']]
    .drop_duplicates()
    .sort_values(['L','n_train'])
    .apply(lambda r: (r['L'], r['n_train']), axis=1)
    .tolist()
)
palette = list(itertools.islice(itertools.cycle(CUSTOM_PALETTE), len(pairs)))
color_map = dict(zip(pairs, palette))  # key: (L, n_train) -> color

STYLE = {
    "ls-linear":  dict(linestyle="-"),
    "ls-softmax": dict(linestyle="--"),
    "ls-gpt2":    dict(linestyle="-"),
}

def _plot_metric(
    metric_col: str,
    ylabel: str,
    *,
    title="Linear Classification Task",
    fig_side=6.0,
    font_size=11,
    title_size=None
):
    fig, ax = plt.subplots(figsize=(fig_side, fig_side))
    for (arch, L, n_train), g in df_all.groupby(["arch","L","n_train"], sort=False):
        g = g.sort_values("n_test")
        style = STYLE.get(arch, dict(linestyle="-"))
        color = color_map.get((L, n_train), None)
        if arch == "ls-gpt2":
            color = "#000000"  # black

        base = arch.replace("ls-", "")        # linear / softmax / gpt2
        label = "gpt2" if base == "gpt2" else f"{base}-L{L}"

        ax.plot(
            g["n_test"].values,
            g[metric_col].values,
            label=label,
            color=color,
            marker='o', markersize=5,
            markerfacecolor=color, markeredgecolor=color,
            **style
        )

    if title_size is None:
        title_size = font_size + 1
    ax.set_xlabel(r"$n_{\mathrm{test}}$", fontsize=font_size)
    ax.set_ylabel(f"{ylabel} (%)", fontsize=font_size)
    ax.set_title(title, fontsize=title_size)
    ax.set_xticks(sorted(df_all["n_test"].unique()))
    ax.grid(True)

    ymin, ymax = ax.get_ylim()
    xmax = 1.0 if ymax <= 1.5 else 100.0
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=xmax, decimals=0))

    ax.tick_params(axis='both', labelsize=font_size-1)

    fig.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    line_only_handles = [
        Line2D([], [], color=h.get_color(),
               linestyle=h.get_linestyle(),
               linewidth=h.get_linewidth())
        for h in handles
    ]
    ax.legend(line_only_handles, labels, loc='lower right', fontsize=10)

    plt.show()

_plot_metric(
    "acc_avg", "Average Accuracy",
    title="Linear Classification Task",
    fig_side=6.5, font_size=14, title_size=18
)

_plot_metric(
    "acc_pos", "Positive Accuracy",
    title="Linear Classification Task",
    fig_side=6.5, font_size=14, title_size=18
)

_plot_metric(
    "acc_neg", "Negative Accuracy",
    title="Linear Classification Task",
    fig_side=6.5, font_size=14, title_size=18
)

# 1.4 Label Flipping

# Generating flipped task matrices

# Flipped task matrix function (fixed clean data, flip N labels)

def apply_ls_label_flipping(dataset, y_query, N, *, rng=None, indices=None):
    """

    Parameters
    ----------
    dataset : np.ndarray, shape (B, n+1, d+1)
    y_query : np.ndarray, shape (B,)
    N : int
    rng : np.random.Generator, optional
    indices : np.ndarray, shape (B, N), optional

    Returns
    -------
    dataset_flipped : np.ndarray, shape (B, n+1, d+1)
    y_query_out : np.ndarray, shape (B,)
    flip_indices : np.ndarray, shape (B, N)
    """
    ds = np.ascontiguousarray(dataset.copy())
    B, T, Dp1 = ds.shape
    n = T - 1

    assert 0 <= N <= n, "N must satisfy 0 ≤ N ≤ n"
    assert np.allclose(ds[:, n, -1], 0.0), "query label slot must be 0"

    if rng is None:
        rng = np.random.default_rng()

    if indices is None:
        flip_indices = np.empty((B, N), dtype=np.int32)
        for b in range(B):
            if N > 0:
                flip_indices[b] = rng.choice(n, size=N, replace=False)
    else:
        flip_indices = np.asarray(indices, dtype=np.int32)
        assert flip_indices.shape == (B, N)
        assert np.all((0 <= flip_indices) & (flip_indices < n))

    if N > 0:
        for b in range(B):
            ds[b, flip_indices[b], -1] *= -1.0

    assert np.allclose(ds[:, n, -1], 0.0)

    return ds, y_query.copy(), flip_indices


def build_lf_attacks(dataset_test_ls_pos, dataset_test_ls_neg, N_values, *, ns=None, seed=2025):
    """
        dataset_test_ls_pos, dataset_test_ls_neg: {n: (ds, yq, w_all)}
    """
    
    if ns is None:
        ns_pos = set(dataset_test_ls_pos.keys())
        ns_neg = set(dataset_test_ls_neg.keys())
        ns = sorted(ns_pos & ns_neg)
    else:
        ns = list(ns)

    lf_pos, lf_neg = {}, {}
    for n in ns:
        assert n in dataset_test_ls_pos and n in dataset_test_ls_neg, f"missing test-set entry for n={n}"
        ds_pos, yq_pos, w_all_pos = dataset_test_ls_pos[n]
        ds_neg, yq_neg, w_all_neg = dataset_test_ls_neg[n]

        assert ds_pos.shape[1] == n + 1 and ds_neg.shape[1] == n + 1
        assert np.allclose(ds_pos[:, -1, -1], 0.0) and np.allclose(ds_neg[:, -1, -1], 0.0)

        lf_pos[n], lf_neg[n] = {}, {}
        for N in N_values:
            assert 0 <= N <= n, f"N={N} out of range (0..{n})"

            rng_pos = np.random.default_rng(seed + 7919 * int(n) + 37 * int(N) + 1)
            rng_neg = np.random.default_rng(seed + 7919 * int(n) + 37 * int(N) + 2)

            ds_pos_lf, yq_pos_lf, flip_idx_pos = apply_ls_label_flipping(ds_pos, yq_pos, N, rng=rng_pos)
            ds_neg_lf, yq_neg_lf, flip_idx_neg = apply_ls_label_flipping(ds_neg, yq_neg, N, rng=rng_neg)

            assert np.allclose(ds_pos_lf[:, -1, -1], 0.0) and np.allclose(ds_neg_lf[:, -1, -1], 0.0)

            lf_pos[n][N] = (ds_pos_lf, yq_pos_lf, w_all_pos, flip_idx_pos)
            lf_neg[n][N] = (ds_neg_lf, yq_neg_lf, w_all_neg, flip_idx_neg)

    return lf_pos, lf_neg

# Generating corrupted data on n = 30, N = 1-15, zero mean

## ===== Repeat LF attacks R times, compute mean±95% CI, and plot (N=0..15) =====
import gc, itertools, math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

def _eval_one_ls(model, arch, x_np, y_np, device):
    if arch == "ls-gpt2":
        return evaluate_model_gpt2(model, x_np, y_np, device=device)
    else:
        return evaluate_model(model, x_np, y_np, device=device)

def _eval_chunked(model, arch, x_np, y_np, device, bs=1024):
    model.eval()
    B = x_np.shape[0]
    tot = 0.0
    with torch.inference_mode():
        for i in range(0, B, bs):
            j = min(i + bs, B)
            acc = _eval_one_ls(model, arch, x_np[i:j], y_np[i:j], device=device)
            tot += float(acc) * (j - i)
            if torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.empty_cache()
    return tot / B

def _eval_chunked_safe(model, arch, x_np, y_np, device, bs_try=(2048,1024,512,256)):
    for bs in bs_try:
        try:
            return _eval_chunked(model, arch, x_np, y_np, device=device, bs=bs)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return _eval_chunked(model.to("cpu"), arch, x_np, y_np, device=torch.device("cpu"), bs=256)

N_values_full = list(range(0, 16))
N_values_attk = list(range(1, 16))
R = 10
n_eval = 30

rows = []
for rep in range(R):
    seed_rep = 2025 + rep
    lf_pos, lf_neg = build_lf_attacks(
        dataset_test_ls_pos, dataset_test_ls_neg,
        N_values=N_values_attk, ns=[n_eval], seed=seed_rep
    )

    for sp in model_specs_ls:
        name, arch, L, n_train, path = sp["name"], sp["arch"], sp["L"], sp["n"], sp["path"]
        try:
            mdl = load_saved_model_ls(path, arch, L, device=torch.device("cpu"))
        except FileNotFoundError:
            print(f"[skip] missing checkpoint: {path}")
            continue

        mdl.to(device)

        for N in N_values_full:
            if N == 0:
                x_pos, y_pos, _ = dataset_test_ls_pos[n_eval]
                x_neg, y_neg, _ = dataset_test_ls_neg[n_eval]
            else:
                x_pos, y_pos, _, _ = lf_pos[n_eval][N]
                x_neg, y_neg, _, _ = lf_neg[n_eval][N]

            acc_pos = _eval_chunked_safe(mdl, arch, x_pos, y_pos, device=device)
            acc_neg = _eval_chunked_safe(mdl, arch, x_neg, y_neg, device=device)
            acc_avg = 0.5 * (acc_pos + acc_neg)

            rows.append({
                "rep": rep,
                "name": name, "arch": arch, "L": L, "n_train": n_train,
                "n_test": n_eval, "N": N,
                "acc_pos": float(acc_pos), "acc_neg": float(acc_neg), "acc_avg": float(acc_avg),
            })

        del mdl
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        gc.collect()

df_lf_reps = (pd.DataFrame(rows)
              .sort_values(["arch","L","n_train","n_test","N","rep"])
              .reset_index(drop=True))

metrics = ["acc_pos", "acc_neg", "acc_avg"]
group_cols = ["name","arch","L","n_train","n_test","N"]

agg = (df_lf_reps
       .groupby(group_cols)[metrics]
       .agg(['mean','std','count'])
       .reset_index())

new_cols = []
for col in agg.columns:
    if isinstance(col, tuple):
        new_cols.append(col[0] if col[1] == '' else f"{col[0]}_{col[1]}")
    else:
        new_cols.append(col)
agg.columns = new_cols

z = 1.96
for m in metrics:
    mean_col = f"{m}_mean"
    std_col  = f"{m}_std"
    n_col    = f"{m}_count"
    se = agg[std_col] / np.sqrt(agg[n_col].clip(lower=1))
    agg[f"{m}_lo"] = (agg[mean_col] - z * se).clip(0.0, 1.0)
    agg[f"{m}_hi"] = (agg[mean_col] + z * se).clip(0.0, 1.0)

df_lf_stats = agg

# Plots

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

df_plot = df_lf_stats.copy()
if "N" not in df_plot.columns:
    raise ValueError("df_lf_stats must contain a 'N' column.")

CUSTOM_PALETTE = ["#F70606", "#0431B7", "#087a08", "#ed870b"]  # red, blue, green, orange
pairs = (
    df_plot[["L","n_train"]]
    .drop_duplicates()
    .sort_values(["L","n_train"])
    .apply(lambda r: (r["L"], r["n_train"]), axis=1)
    .tolist()
)
palette = list(itertools.islice(itertools.cycle(CUSTOM_PALETTE), len(pairs)))
color_map = dict(zip(pairs, palette))  # key: (L, n_train) -> color

STYLE = {
    "ls-linear":  dict(linestyle="-"),
    "ls-softmax": dict(linestyle="--"),
    "ls-gpt2":    dict(linestyle="-"),
}

def _legend_label(arch: str, L: int) -> str:
    base = arch.replace("ls-", "")  # linear / softmax / gpt2
    return "gpt2" if base == "gpt2" else f"{base}-L{L}"

def _plot_lf_mean_ci(metric: str, ylabel: str):
    # metric ∈ {"acc_avg","acc_pos","acc_neg"}
    mean_col, lo_col, hi_col = f"{metric}_mean", f"{metric}_lo", f"{metric}_hi"

    fig, ax = plt.subplots(figsize=(8, 6))
    for (arch, L, n_train), g in df_plot.groupby(["arch","L","n_train"], sort=False):
        g = g.sort_values("N")
        style = STYLE.get(arch, dict(linestyle="-"))
        color = "#000000" if arch == "ls-gpt2" else color_map.get((L, n_train), None)
        label = _legend_label(arch, L)

        ax.fill_between(
            g["N"].values, g[lo_col].values, g[hi_col].values,
            color=color, alpha=0.20, linewidth=0
        )
        ax.plot(
            g["N"].values, g[mean_col].values,
            label=label, color=color,
            marker="o", markersize=5,
            markerfacecolor=color, markeredgecolor=color,
            **style
        )

    ax.set_xlabel("Number of flipped labels (N)")
    ax.set_ylabel(ylabel)
    ax.set_title(r"Label Flipping on Linear Classification Task with $n_{\mathrm{test}}=30$")
    ax.set_xticks(list(range(0, 16)))
    ax.grid(True)

    ymin, ymax = ax.get_ylim()
    xmax = 1.0 if ymax <= 1.5 else 100.0
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=xmax, decimals=0))

    fig.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    line_only_handles = [
        Line2D([], [], color=h.get_color(),
               linestyle=h.get_linestyle(),
               linewidth=h.get_linewidth())
        for h in handles
    ]
    ax.legend(line_only_handles, labels, loc="upper right", fontsize=10)

    plt.show()

_plot_lf_mean_ci("acc_avg", "Average Accuracy")
_plot_lf_mean_ci("acc_pos", "Positive Accuracy")
_plot_lf_mean_ci("acc_neg", "Negative Accuracy")

def _plot_lf_avg_triplet_L1(
    n_train_sel: int | None = None,
    *,
    fig_size: tuple[float, float] = (8.0, 5.5),
    font_size: int = 12,
    title_size: int | None = None
):
    from matplotlib.ticker import PercentFormatter

    metric = "acc_avg"
    mean_col, lo_col, hi_col = f"{metric}_mean", f"{metric}_lo", f"{metric}_hi"

    df_L1 = df_plot[df_plot["L"] == 1]
    if n_train_sel is None:
        n_train_sel = int(df_L1["n_train"].mode().iloc[0])

    if title_size is None:
        title_size = font_size + 2

    fig, ax = plt.subplots(figsize=(float(fig_size[0]), float(fig_size[1])))

    for arch in ["ls-linear", "ls-softmax", "ls-gpt2"]:
        if arch == "ls-gpt2":
            g_cand = df_plot[df_plot["arch"] == "ls-gpt2"]
            g = g_cand[g_cand["L"] == 1] if (g_cand["L"] == 1).any() else g_cand
            if (g["n_train"] == n_train_sel).any():
                g = g[g["n_train"] == n_train_sel]
            color = "#000000"
            label = _legend_label(arch, 1)  # -> "gpt2"
        else:
            g = df_L1[(df_L1["arch"] == arch) & (df_L1["n_train"] == n_train_sel)]
            color = color_map.get((1, n_train_sel), "#000000")
            label = _legend_label(arch, 1)  # "linear-L1" / "softmax-L1"

        if g.empty:
            continue
        g = g.sort_values("N")
        style = STYLE.get(arch, dict(linestyle="-"))

        if lo_col in g.columns and hi_col in g.columns:
            ax.fill_between(
                g["N"].values, g[lo_col].values, g[hi_col].values,
                color=color, alpha=0.20, linewidth=0
            )
        ax.plot(
            g["N"].values, g[mean_col].values,
            label=label, color=color,
            marker="o", markersize=5,
            markerfacecolor=color, markeredgecolor=color,
            **style
        )

    ax.set_xlabel("Number of flipped labels (N)", fontsize=font_size)
    ax.set_ylabel("Average Accuracy", fontsize=font_size)
    ax.set_title(r"Label Flipping on Linear Classification Task with $n_{\mathrm{test}}=30$", fontsize=title_size)
    ax.set_xticks(list(range(0, 16)))
    ax.grid(True)

    ymin, ymax = ax.get_ylim()
    xmax = 1.0 if ymax <= 1.5 else 100.0
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=xmax, decimals=0))

    ax.tick_params(axis='both', labelsize=max(1, font_size - 1))

    fig.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    line_only_handles = [
        Line2D([], [], color=h.get_color(),
               linestyle=h.get_linestyle(),
               linewidth=h.get_linewidth())
        for h in handles
    ]
    ax.legend(line_only_handles, labels, loc="upper right", fontsize=font_size)

    plt.show()
    return fig

fig_triplet = _plot_lf_avg_triplet_L1(fig_size=(7.5, 6.0), font_size=13, title_size=16)
# fig_triplet.savefig("lf_triplet_L1.png", dpi=300, bbox_inches="tight")

def _plot_lf_avg_triplet_L1(
    n_train_sel: int | None = None,
    *,
    fig_size: tuple[float, float] = (8.0, 5.5),
    font_size: int = 12,
    title_size: int | None = None
):
    from matplotlib.ticker import PercentFormatter

    metric = "acc_avg"
    mean_col, lo_col, hi_col = f"{metric}_mean", f"{metric}_lo", f"{metric}_hi"

    df_L1 = df_plot[df_plot["L"] == 1]
    if n_train_sel is None:
        n_train_sel = int(df_L1["n_train"].mode().iloc[0])

    if title_size is None:
        title_size = font_size + 2

    fig, ax = plt.subplots(figsize=(float(fig_size[0]), float(fig_size[1])))

    for arch in ["ls-linear", "ls-softmax", "ls-gpt2"]:
        if arch == "ls-gpt2":
            g_cand = df_plot[df_plot["arch"] == "ls-gpt2"]
            g = g_cand[g_cand["L"] == 1] if (g_cand["L"] == 1).any() else g_cand
            if (g["n_train"] == n_train_sel).any():
                g = g[g["n_train"] == n_train_sel]
            color = "#000000"
            label = _legend_label(arch, 1)  # -> "gpt2"
        else:
            g = df_L1[(df_L1["arch"] == arch) & (df_L1["n_train"] == n_train_sel)]
            color = color_map.get((1, n_train_sel), "#000000")
            label = _legend_label(arch, 1)  # "linear-L1" / "softmax-L1"

        if g.empty:
            continue
        g = g.sort_values("N")
        style = STYLE.get(arch, dict(linestyle="-"))

        if lo_col in g.columns and hi_col in g.columns:
            ax.fill_between(
                g["N"].values, g[lo_col].values, g[hi_col].values,
                color=color, alpha=0.20, linewidth=0
            )
        ax.plot(
            g["N"].values, g[mean_col].values,
            label=label, color=color,
            marker="o", markersize=5,
            markerfacecolor=color, markeredgecolor=color,
            **style
        )

    ax.set_xlabel("Number of flipped labels (N)", fontsize=font_size)
    ax.set_ylabel("Average Accuracy", fontsize=font_size)
    ax.set_title(r"Label Flipping on Linear Classification Task with $n_{\mathrm{test}}=30$", fontsize=title_size)
    ax.set_xticks(list(range(0, 16)))
    ax.grid(True)

    ymin, ymax = ax.get_ylim()
    xmax = 1.0 if ymax <= 1.5 else 100.0
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=xmax, decimals=0))

    ax.tick_params(axis='both', labelsize=max(1, font_size - 1))

    fig.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    line_only_handles = [
        Line2D([], [], color=h.get_color(),
               linestyle=h.get_linestyle(),
               linewidth=h.get_linewidth())
        for h in handles
    ]
    ax.legend(line_only_handles, labels, loc="upper right", fontsize=font_size)

    plt.show()
    return fig

fig_triplet = _plot_lf_avg_triplet_L1(fig_size=(7.5, 6.0), font_size=13, title_size=16)
# fig_triplet.savefig("lf_triplet_L1.png", dpi=300, bbox_inches="tight")

# 1.5 Label Hijacking

import numpy as np

def apply_ls_label_hijacking(
    dataset: np.ndarray,
    y_query: np.ndarray,
    w_all: np.ndarray,
    N: int,
    delta: float = 0.0,
    seed: int | None = None,
    safety_eps: float = 1e-6,
    return_indices: bool = True,
):
    """
    - y_query: (B,) in {+1, -1}
    """
    assert dataset.ndim == 3
    B, L, Dp1 = dataset.shape
    n = L - 1
    d = Dp1 - 1
    assert n > 0 and d > 0
    assert y_query.shape[0] == B and w_all.shape == (B, d)
    assert np.allclose(dataset[:, -1, -1], 0.0), "query label slot must be 0"

    rng = np.random.default_rng(seed)
    ds_lh = dataset.copy()
    attacked_indices = []

    for b in range(B):
        c = int(y_query[b])                 # +1 / -1
        w = w_all[b].astype(np.float64)
        w_norm = np.linalg.norm(w)
        if w_norm == 0.0:
            attacked_indices.append(np.array([], dtype=int))
            continue
        w_hat = w / w_norm

        demo_labels = ds_lh[b, :n, -1]
        I_c = np.where(demo_labels == c)[0]
        if I_c.size == 0:
            attacked_indices.append(np.array([], dtype=int))
            continue

        N_b = min(N, I_c.size)
        pick = rng.choice(I_c, size=N_b, replace=False)
        attacked_indices.append(np.array(pick, dtype=int))

        for i in pick:
            x = ds_lh[b, i, :d].astype(np.float64)

            s = float(np.dot(w, x))
            m = abs(s) / w_norm
            target = max(delta, safety_eps)

            if m <= target + 1e-12:
                continue

            t = m - target
            x_new = x - c * t * w_hat
            s_new = float(np.dot(w, x_new))

            if (c == +1 and s_new <= 0) or (c == -1 and s_new >= 0):
                x_new = x - c * (m - (target + safety_eps)) * w_hat

            ds_lh[b, i, :d] = x_new.astype(dataset.dtype)

    if return_indices:
        return ds_lh, y_query, w_all, attacked_indices
    else:
        return ds_lh, y_query, w_all


def build_lh_attacks(
    dataset_pos_dict: dict,
    dataset_neg_dict: dict,
    N_values: list[int],
    ns: list[int],
    delta: float = 0.0,
    seed: int = 2025,
):
    """
        lh_pos[n][N] -> (ds_lh, yq, w_all, indices)
        lh_neg[n][N] -> (ds_lh, yq, w_all, indices)
    - dataset_pos_dict[n] / dataset_neg_dict[n] = (dataset, y_query, w_all)
    """
    lh_pos = {n: {} for n in ns}
    lh_neg = {n: {} for n in ns}

    for n in ns:
        ds_pos, yq_pos, w_pos = dataset_pos_dict[n]
        ds_neg, yq_neg, w_neg = dataset_neg_dict[n]

        for N in N_values:
            seed_here = int(seed + 1000 * n + N)

            ds_p, yq_p, w_p, idx_p = apply_ls_label_hijacking(
                ds_pos, yq_pos, w_pos, N=N, delta=delta, seed=seed_here, return_indices=True
            )
            lh_pos[n][N] = (ds_p, yq_p, w_p, idx_p)

            ds_n, yq_n, w_n, idx_n = apply_ls_label_hijacking(
                ds_neg, yq_neg, w_neg, N=N, delta=delta, seed=seed_here + 7, return_indices=True
            )
            lh_neg[n][N] = (ds_n, yq_n, w_n, idx_n)

    return lh_pos, lh_neg

# Generating corrupted data on n = 30, N = 1-15, zero mean

# ======================== Label Hijacking (LS) — Build, Eval, Plot ========================
import gc, itertools, math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def apply_ls_label_hijacking(
    dataset: np.ndarray,
    y_query: np.ndarray,
    w_all: np.ndarray,
    N: int,
    delta: float = 0.0,
    seed: int | None = None,
    safety_eps: float = 1e-6,
    return_indices: bool = True,
):
    """
    - y_query: (B,) in {+1, -1}
    """
    assert dataset.ndim == 3
    B, L, Dp1 = dataset.shape
    n = L - 1
    d = Dp1 - 1
    assert n > 0 and d > 0
    assert y_query.shape[0] == B and w_all.shape == (B, d)
    assert np.allclose(dataset[:, -1, -1], 0.0), "query label slot must be 0"

    rng = np.random.default_rng(seed)
    ds_lh = dataset.copy()
    attacked_indices = []

    for b in range(B):
        c = int(y_query[b])                 # +1 / -1
        w = w_all[b].astype(np.float64)
        w_norm = np.linalg.norm(w)
        if w_norm == 0.0:
            attacked_indices.append(np.array([], dtype=int))
            continue
        w_hat = w / w_norm

        demo_labels = ds_lh[b, :n, -1]
        I_c = np.where(demo_labels == c)[0]
        if I_c.size == 0:
            attacked_indices.append(np.array([], dtype=int))
            continue

        N_b = min(N, I_c.size)
        pick = rng.choice(I_c, size=N_b, replace=False)
        attacked_indices.append(np.array(pick, dtype=int))

        for i in pick:
            x = ds_lh[b, i, :d].astype(np.float64)

            s = float(np.dot(w, x))
            m = abs(s) / w_norm
            target = max(delta, safety_eps)

            if m <= target + 1e-12:
                continue

            t = m - target
            x_new = x - c * t * w_hat
            s_new = float(np.dot(w, x_new))

            if (c == +1 and s_new <= 0) or (c == -1 and s_new >= 0):
                x_new = x - c * (m - (target + safety_eps)) * w_hat

            ds_lh[b, i, :d] = x_new.astype(dataset.dtype)

    if return_indices:
        return ds_lh, y_query, w_all, attacked_indices
    else:
        return ds_lh, y_query, w_all

def build_lh_attacks(
    dataset_pos_dict: dict,
    dataset_neg_dict: dict,
    N_values: list[int],
    ns: list[int],
    delta: float = 0.0,
    seed: int = 2025,
):
    """
        lh_pos[n][N] -> (ds_lh, yq, w_all, indices)
        lh_neg[n][N] -> (ds_lh, yq, w_all, indices)
    - dataset_pos_dict[n] / dataset_neg_dict[n] = (dataset, y_query, w_all)
    """
    lh_pos = {n: {} for n in ns}
    lh_neg = {n: {} for n in ns}

    for n in ns:
        ds_pos, yq_pos, w_pos = dataset_pos_dict[n]
        ds_neg, yq_neg, w_neg = dataset_neg_dict[n]

        for N in N_values:
            seed_here = int(seed + 1000 * n + N)

            ds_p, yq_p, w_p, idx_p = apply_ls_label_hijacking(
                ds_pos, yq_pos, w_pos, N=N, delta=delta, seed=seed_here, return_indices=True
            )
            lh_pos[n][N] = (ds_p, yq_p, w_p, idx_p)

            ds_n, yq_n, w_n, idx_n = apply_ls_label_hijacking(
                ds_neg, yq_neg, w_neg, N=N, delta=delta, seed=seed_here + 7, return_indices=True
            )
            lh_neg[n][N] = (ds_n, yq_n, w_n, idx_n)

    return lh_pos, lh_neg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

def _eval_one_ls(model, arch, x_np, y_np, device):
    if arch == "ls-gpt2":
        return evaluate_model_gpt2(model, x_np, y_np, device=device)
    else:
        return evaluate_model(model, x_np, y_np, device=device)

def _eval_chunked(model, arch, x_np, y_np, device, bs=1024):
    model.eval()
    B = x_np.shape[0]
    tot = 0.0
    with torch.inference_mode():
        for i in range(0, B, bs):
            j = min(i + bs, B)
            acc = _eval_one_ls(model, arch, x_np[i:j], y_np[i:j], device=device)
            tot += float(acc) * (j - i)
            if torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.empty_cache()
    return tot / B

def _eval_chunked_safe(model, arch, x_np, y_np, device, bs_try=(2048,1024,512,256)):
    for bs in bs_try:
        try:
            return _eval_chunked(model, arch, x_np, y_np, device=device, bs=bs)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return _eval_chunked(model.to("cpu"), arch, x_np, y_np, device=torch.device("cpu"), bs=256)

N_values = list(range(0, 16))
R = 10
n_eval = 30
delta = 0.1

rows = []
for rep in range(R):
    seed_rep = 2025 + rep
    lh_pos, lh_neg = build_lh_attacks(
        dataset_test_ls_pos, dataset_test_ls_neg,
        N_values=N_values, ns=[n_eval], delta=delta, seed=seed_rep
    )

    for sp in model_specs_ls:
        name, arch, L, n_train, path = sp["name"], sp["arch"], sp["L"], sp["n"], sp["path"]
        try:
            mdl = load_saved_model_ls(path, arch, L, device=torch.device("cpu"))
        except FileNotFoundError:
            print(f"[skip] missing checkpoint: {path}")
            continue

        mdl.to(device)
        for N in N_values:
            x_pos, y_pos, _, _ = lh_pos[n_eval][N]
            x_neg, y_neg, _, _ = lh_neg[n_eval][N]

            acc_pos = _eval_chunked_safe(mdl, arch, x_pos, y_pos, device=device)
            acc_neg = _eval_chunked_safe(mdl, arch, x_neg, y_neg, device=device)
            acc_avg = 0.5 * (acc_pos + acc_neg)

            rows.append({
                "rep": rep,
                "name": name, "arch": arch, "L": L, "n_train": n_train,
                "n_test": n_eval, "N": N,
                "acc_pos": float(acc_pos), "acc_neg": float(acc_neg), "acc_avg": float(acc_avg),
            })

        del mdl
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        gc.collect()

df_lh_reps = (pd.DataFrame(rows)
              .sort_values(["arch","L","n_train","n_test","N","rep"])
              .reset_index(drop=True))

metrics = ["acc_pos", "acc_neg", "acc_avg"]
group_cols = ["name","arch","L","n_train","n_test","N"]

agg = (df_lh_reps
       .groupby(group_cols)[metrics]
       .agg(['mean','std','count'])
       .reset_index())

new_cols = []
for col in agg.columns:
    if isinstance(col, tuple):
        new_cols.append(col[0] if col[1] == '' else f"{col[0]}_{col[1]}")
    else:
        new_cols.append(col)
agg.columns = new_cols

z = 1.96
for m in metrics:
    mean_col = f"{m}_mean"
    std_col  = f"{m}_std"
    n_col    = f"{m}_count"
    se = agg[std_col] / np.sqrt(agg[n_col].clip(lower=1))
    agg[f"{m}_lo"] = (agg[mean_col] - z * se).clip(0.0, 1.0)
    agg[f"{m}_hi"] = (agg[mean_col] + z * se).clip(0.0, 1.0)

df_lh_stats = agg

# Plots

import itertools
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

CUSTOM_PALETTE = ["#F70606", "#0431B7", "#087a08", "#ed870b"]  # red, blue, green, orange
pairs = (
    df_lh_stats[['L','n_train']]
    .drop_duplicates()
    .sort_values(['L','n_train'])
    .apply(lambda r: (r['L'], r['n_train']), axis=1)
    .tolist()
)
palette = list(itertools.islice(itertools.cycle(CUSTOM_PALETTE), len(pairs)))
color_map = dict(zip(pairs, palette))  # key: (L, n_train) -> color

STYLE = {
    "ls-linear":  dict(linestyle="-"),
    "ls-softmax": dict(linestyle="--"),
    "ls-gpt2":    dict(linestyle="-"),
}

def _legend_label(arch: str, L: int) -> str:
    base = arch.replace("ls-", "")  # linear / softmax / gpt2
    return "gpt2" if base == "gpt2" else f"{base}-L{L}"

def _plot_lh_mean_ci(metric: str, ylabel: str):
    # metric ∈ {"acc_avg","acc_pos","acc_neg"}
    mean_col, lo_col, hi_col = f"{metric}_mean", f"{metric}_lo", f"{metric}_hi"

    fig, ax = plt.subplots(figsize=(8,6))
    for (arch, L, n_train), g in df_lh_stats.groupby(["arch","L","n_train"], sort=False):
        g = g.sort_values("N")
        style = STYLE.get(arch, dict(linestyle="-"))
        color = "#000000" if arch == "ls-gpt2" else color_map.get((L, n_train), None)
        label = _legend_label(arch, L)

        ax.fill_between(
            g["N"].values, g[lo_col].values, g[hi_col].values,
            color=color, alpha=0.20, linewidth=0
        )
        ax.plot(
            g["N"].values, g[mean_col].values,
            label=label, color=color,
            marker='o', markersize=5,
            markerfacecolor=color, markeredgecolor=color,
            **style
        )

    ax.set_xlabel("Number of hijacked demonstrations (N)")
    ax.set_ylabel(ylabel)
    ax.set_title(r"Label Hijacking on Linear Classification Task with $n_{\mathrm{test}}=30$")
    ax.set_xticks(list(range(0, 16)))  # N=0..15
    ax.grid(True)

    ymin, ymax = ax.get_ylim()
    xmax = 1.0 if ymax <= 1.5 else 100.0
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=xmax, decimals=0))

    fig.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    line_only_handles = [
        Line2D([], [], color=h.get_color(),
               linestyle=h.get_linestyle(),
               linewidth=h.get_linewidth())
        for h in handles
    ]
    ax.legend(line_only_handles, labels, loc='upper right', fontsize=10)

    plt.show()

_plot_lh_mean_ci("acc_avg", "Average Accuracy")
_plot_lh_mean_ci("acc_pos", "Positive Accuracy")
_plot_lh_mean_ci("acc_neg", "Negative Accuracy")

from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D

def _plot_lh_avg_triplet_L1(
    n_train_sel: int | None = None,
    *,
    fig_side: float = 6.5,
    fig_size: tuple[float, float] | None = None,
    font_size: int = 12,
    title_size: int | None = None
):
    metric = "acc_avg"
    mean_col, lo_col, hi_col = f"{metric}_mean", f"{metric}_lo", f"{metric}_hi"

    df_L1 = df_lh_stats[df_lh_stats["L"] == 1]
    if n_train_sel is None:
        if not df_L1.empty and not df_L1["n_train"].mode().empty:
            n_train_sel = int(df_L1["n_train"].mode().iloc[0])
        else:
            n_train_sel = int(df_lh_stats["n_train"].mode().iloc[0])

    if title_size is None:
        title_size = font_size + 2

    if fig_size is not None:
        if not (isinstance(fig_size, (tuple, list)) and len(fig_size) == 2):
            raise ValueError("fig_size must be a (width, height) tuple in inches.")
        fig, ax = plt.subplots(figsize=(float(fig_size[0]), float(fig_size[1])))
    else:
        fig, ax = plt.subplots(figsize=(fig_side, fig_side))

    for arch in ["ls-linear", "ls-softmax", "ls-gpt2"]:
        if arch == "ls-gpt2":
            g_all = df_lh_stats[df_lh_stats["arch"] == "ls-gpt2"]
            g = g_all[g_all["L"] == 1] if (g_all["L"] == 1).any() else g_all
            if (g["n_train"] == n_train_sel).any():
                g = g[g["n_train"] == n_train_sel]
            color = "#000000"
            label = _legend_label(arch, 1)  # -> "gpt2"
        else:
            g = df_L1[(df_L1["arch"] == arch) & (df_L1["n_train"] == n_train_sel)]
            color = color_map.get((1, n_train_sel), "#000000")
            label = _legend_label(arch, 1)  # "linear-L1" / "softmax-L1"

        if g.empty:
            continue

        g = g.sort_values("N")
        style = STYLE.get(arch, dict(linestyle="-"))

        if lo_col in g.columns and hi_col in g.columns:
            ax.fill_between(
                g["N"].values, g[lo_col].values, g[hi_col].values,
                color=color, alpha=0.20, linewidth=0
            )
        ax.plot(
            g["N"].values, g[mean_col].values,
            label=label, color=color,
            marker="o", markersize=5,
            markerfacecolor=color, markeredgecolor=color,
            **style
        )

    ax.set_xlabel("Number of corrupted label (N)", fontsize=font_size)
    ax.set_ylabel("Average Accuracy", fontsize=font_size)
    ax.set_title(r"Label Hijacking on Linear Classification Task with $n_{\mathrm{test}}=30$", fontsize=title_size)
    ax.set_xticks(list(range(0, 16)))  # N=0..15
    ax.grid(True)

    ymin, ymax = ax.get_ylim()
    xmax = 1.0 if ymax <= 1.5 else 100.0
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=xmax, decimals=0))

    ax.tick_params(axis='both', labelsize=max(1, font_size - 1))

    fig.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    line_only_handles = [
        Line2D([], [], color=h.get_color(),
               linestyle=h.get_linestyle(),
               linewidth=h.get_linewidth())
        for h in handles
    ]
    ax.legend(line_only_handles, labels, loc="upper right", fontsize=font_size)

    plt.show()
    return fig

fig_triplet = _plot_lh_avg_triplet_L1(fig_size=(7.5, 6.0), font_size=13, title_size=16)
