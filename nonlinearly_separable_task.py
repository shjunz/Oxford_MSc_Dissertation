"""
Non-linearly separable (parabola) in-context classification with linear and softmax attention-only transformers.
"""


import numpy as np
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

# Non-Linearly Separable Classification Problem (Parabola / Circle)

# Generating A* b*

# A*, b* function

def sample_A_b(
    d, 
    mode="circle", 
    *, 
    a_val: float = 1.0,
    k_val: float = 0.5,  
    radius_range=(0.5, 2.0), 
    eps=1e-8
):
    if mode == "circle":
        a = float(a_val)
        # radius
        r = float(np.random.uniform(radius_range[0], radius_range[1]))

        # direction
        u = np.random.randn(d).astype(np.float32)
        u /= (np.linalg.norm(u) + eps)

        # A, b
        b = (-2.0 * a * r) * u
        A = a * np.eye(d, dtype=np.float32)

    elif mode == "parabola":
        u = np.random.randn(d).astype(np.float32)
        u /= (np.linalg.norm(u) + eps)

        k = float(k_val)
        assert k > 0, "k_val must be positive"

        I = np.eye(d, dtype=np.float32)
        proj = I - np.outer(u, u)
        A = (-k) * proj
        b = u.copy()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    A = 0.5 * (A + A.T)
    return A.astype(np.float32), b.astype(np.float32)

# Generating parabola task matrices

# Parabola Score & Gradient & Margin

# Discriminant (parabola case):
# A = -k (I - u u^T),  b = u,  ||u|| = 1,  k > 0
# s(x) = <u, x> - k ( ||x||^2 - <u, x>^2 )

def parabola_quad_score(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    d = b.shape[0]
    u = b / (np.linalg.norm(b) + 1e-8)               # b ≈ u
    k = np.float32(-np.trace(A) / max(d - 1, 1))     # trace(A) = -k(d-1), k>0

    alpha = x @ u                                    # (...,)
    xx = np.einsum("...i,...i->...", x, x)           # (...,)
    return alpha - k * (xx - alpha * alpha)          # s(x)=<u,x>-k(||x||^2-<u,x>^2)

# Gradient (parabola): ∇s(x) = -2k (x - <u,x> u) + u

def parabola_quad_grad(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    d = b.shape[0]
    u = b / (np.linalg.norm(b) + 1e-8)
    k = float(-np.trace(np.asarray(A, np.float32)) / max(d - 1, 1))
    alpha = x @ u                           # (...,)
    return (-2.0 * k) * (x - alpha[..., None] * u) + u

def enforce_margin_point(x: np.ndarray, y: float, A: np.ndarray, b: np.ndarray, margin: float):
    """
    If y*s(x) < margin, take one gradient step along y*∇s to satisfy the margin.
    x: shape (d,)
    """
    x = np.asarray(x, dtype=np.float32)
    s0 = float(parabola_quad_score(x, A, b))
    if y * s0 >= margin:
        return x.astype(np.float32)

    b = np.asarray(b, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    d = b.shape[0]
    u = b / (np.linalg.norm(b) + 1e-8)
    k = float(-np.trace(A) / max(d - 1, 1))

    g = (-2.0 * k) * (x - (x @ u) * u) + u
    gg = float(np.dot(g, g))

    c = y * s0 - margin
    ug = float(np.dot(u, g))
    a_quad = -k * (gg - ug * ug)
    b_lin  = gg

    eps = 1e-12
    if abs(y * a_quad) < eps:
        t = -c / (b_lin + 1e-8)
    else:
        D = b_lin * b_lin - 4.0 * (y * a_quad) * c
        D = max(D, 0.0)
        t = (-b_lin + np.sqrt(D)) / (2.0 * y * a_quad)
        if not np.isfinite(t) or t <= 0:
            t = -c / (b_lin + 1e-8)

    x_new = x + y * t * g
    return x_new.astype(np.float32)

# Monte-Carlo estimation of the probability of +ve labels, k > 0 (k < 0 is the opposite)

def prob_s_positive(d=20, k=0.25, num_samples=200000):
    alpha = np.random.randn(num_samples)             
    beta = np.random.chisquare(df=d-1, size=num_samples)    
    s = alpha - k * beta
    return np.mean(s > 0)

p_positive = prob_s_positive(d=20, k=0.1, num_samples=200000)
print("d=20, P(s>0) ≈", p_positive)

# Parabola function

def sample_nls_parabola_task_matrix(d, n, A, b, margin=0.0, apply_margin_to_query=False, *, batch=4096):

    assert n % 2 == 0
    half = n // 2
    A = np.asarray(A, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    assert A.shape == (d, d)
    assert b.shape == (d,)

    p_hat = max(min(p_positive, 1.0 - 1e-12), 1e-12)

    need_pos = half
    need_neg = half
    curr_batch = max(
        int(np.ceil(max(need_pos / p_hat, need_neg / (1.0 - p_hat)))),
        int(batch)
    )

    max_once = 2_000_000
    chunk = 1_000_000
    curr_batch = min(curr_batch, max_once)

    x_pos, x_neg = [], []
    while need_pos > 0 or need_neg > 0:
        remain = curr_batch
        while remain > 0 and (need_pos > 0 or need_neg > 0):
            m = min(remain, chunk)
            x = np.random.randn(m, d).astype(np.float32)
            s = parabola_quad_score(x, A, b)              # <-- fix
            y = np.sign(s).astype(np.float32)
            y[y == 0.0] = 1.0

            if need_pos > 0:
                idx_pos = np.where(y > 0)[0]
                take_pos = min(need_pos, idx_pos.size)
                if take_pos > 0:
                    x_pos.append(x[idx_pos[:take_pos]])
                    need_pos -= take_pos

            if need_neg > 0:
                idx_neg = np.where(y < 0)[0]
                take_neg = min(need_neg, idx_neg.size)
                if take_neg > 0:
                    x_neg.append(x[idx_neg[:take_neg]])
                    need_neg -= take_neg

            remain -= m

        if need_pos > 0 or need_neg > 0:
            curr_batch = max(
                int(np.ceil(max(
                    (need_pos / p_hat) if need_pos > 0 else 0.0,
                    (need_neg / (1.0 - p_hat)) if need_neg > 0 else 0.0
                ))),
                32 * d,
                int(batch),
            )
            curr_batch = min(curr_batch, max_once)

    x_pos = np.vstack(x_pos)[:half] if len(x_pos) > 0 else np.empty((0, d), dtype=np.float32)
    x_neg = np.vstack(x_neg)[:half] if len(x_neg) > 0 else np.empty((0, d), dtype=np.float32)
    x_ctx = np.vstack([x_pos, x_neg]).astype(np.float32)
    y_ctx = np.concatenate([np.ones(half, np.float32), -np.ones(half, np.float32)])

    if margin > 0.0:
        for i in range(n):
            y_i = float(y_ctx[i])
            x_i = x_ctx[i]
            s_i = parabola_quad_score(x_i[None, :], A, b).item()
            if y_i * s_i < float(margin):
                x_ctx[i] = enforce_margin_point(x_i, y_i, A, b, float(margin))

    perm = np.random.permutation(n)
    x_ctx, y_ctx = x_ctx[perm], y_ctx[perm]

    x_query = np.random.randn(d).astype(np.float32)
    s_q = parabola_quad_score(x_query[None, :], A, b).item()  # <-- fix
    y_query = np.float32(1.0 if s_q >= 0.0 else -1.0)

    if apply_margin_to_query and margin > 0.0:
        if y_query * s_q < float(margin):
            x_query = enforce_margin_point(x_query, float(y_query), A, b, float(margin))
            s_q2 = parabola_quad_score(x_query[None, :], A, b).item()
            y_query = np.float32(1.0 if s_q2 >= 0.0 else -1.0)

    task_matrix = np.zeros((n + 1, d + 1), dtype=np.float32)
    task_matrix[:n, :d] = x_ctx
    task_matrix[:n,  d] = y_ctx
    task_matrix[n,  :d] = x_query
    task_matrix[n,   d] = 0.0
    return task_matrix, y_query

# Generating data

# Parabola data

# Ensure half queries are positive

def sample_nls_parabola_dataset(
    d: int,
    n: int,
    batch_size: int,
    *,
    k_val: float = 0.2,
    margin: float = 0.0,
    apply_margin_to_query: bool = False,
):
    assert batch_size % 2 == 0, "batch_size must be even to balance queries."

    half_bs = batch_size // 2
    x = np.zeros((batch_size, n + 1, d + 1), dtype=np.float32)
    y = np.zeros((batch_size,), dtype=np.float32)

    pos_idx = 0
    neg_idx = 0

    # Put positive samples in [0, half_bs), negative samples in [half_bs, batch_size)
    while (pos_idx < half_bs) or (neg_idx < half_bs):
        A, b = sample_A_b(d, mode="parabola", k_val=k_val)
        task_matrix, y_q = sample_nls_parabola_task_matrix(
            d, n, A, b,
            margin=margin,
            apply_margin_to_query=apply_margin_to_query,
        )

        if (y_q > 0) and (pos_idx < half_bs):
            x[pos_idx] = task_matrix
            y[pos_idx] = y_q
            pos_idx += 1
        elif (y_q < 0) and (neg_idx < half_bs):
            x[half_bs + neg_idx] = task_matrix
            y[half_bs + neg_idx] = y_q
            neg_idx += 1
        # If y_q == 0, just discard and resample

    perm = np.random.permutation(batch_size)
    x = x[perm]
    y = y[perm]

    return x, y

# Sanity Check

def check_balance(dataset, y_query, n):
    """
    dataset: (B, n+1, d+1)
    y_query: (B,)
    n: number of context demos
    """
    assert dataset.ndim == 3, "dataset should be (B, n+1, d+1)"
    B, nn1, _ = dataset.shape
    assert nn1 == n + 1, f"dataset second dim should be n+1, got {nn1}"
    assert y_query.shape[0] == B, "y_query size must match batch size"

    y_ctx = dataset[:, :n, -1]  # (B, n)
    pos_counts = np.sum(y_ctx > 0, axis=1)
    neg_counts = np.sum(y_ctx < 0, axis=1)
    zero_counts = np.sum(y_ctx == 0, axis=1)

    total = y_ctx.shape[0]
    balanced = int(np.sum((pos_counts == n // 2) & (neg_counts == n // 2)))

    print(f"Total tasks: {total}")
    print(f"Balanced tasks: {balanced} ({balanced / total * 100:.2f}%)")
    print(f"Zero labels in contexts (should be 0): total={int(np.sum(zero_counts))}")
    print("Example counts (first 5 tasks):")
    for i in range(min(5, total)):
        print(f"Task {i}: +1={int(pos_counts[i])}, -1={int(neg_counts[i])}, zero={int(zero_counts[i])}, query={y_query[i]}")

def check_query_balance(name: str, y_query: np.ndarray):
    y = np.asarray(y_query)
    total = y.size
    pos = int(np.sum(y > 0))
    neg = int(np.sum(y < 0))
    zero = int(np.sum(y == 0))
    print(f"{name}: total={total}, +1={pos} ({pos / total:.2%}), -1={neg} ({neg / total:.2%}), zero={zero}")

# n = 60

np.random.seed(42)

d, n = 20, 60

# ---------- Training ----------
batch_size_train = 30000
dataset_train_nls_parabola_n60, y_query_train_nls_parabola_n60 = sample_nls_parabola_dataset(
    d, n, batch_size_train,
    k_val=0.1,
    margin=0.0,
    apply_margin_to_query=False,
)

assert dataset_train_nls_parabola_n60.shape == (batch_size_train, n+1, d+1)
assert y_query_train_nls_parabola_n60.shape == (batch_size_train,)
assert np.allclose(dataset_train_nls_parabola_n60[:, -1, -1], 0.0)

# ---------- Validation ----------
batch_size_val = 3000
dataset_val_nls_parabola_n60, y_query_val_nls_parabola_n60 = sample_nls_parabola_dataset(
    d, n, batch_size_val,
    k_val=0.1,
    margin=0.0,
    apply_margin_to_query=False,
)

assert dataset_val_nls_parabola_n60.shape == (batch_size_val, n+1, d+1)
assert y_query_val_nls_parabola_n60.shape == (batch_size_val,)
assert np.allclose(dataset_val_nls_parabola_n60[:, -1, -1], 0.0)

check_balance(dataset_train_nls_parabola_n60, y_query_train_nls_parabola_n60, n)
check_balance(dataset_val_nls_parabola_n60, y_query_val_nls_parabola_n60, n)
check_query_balance("train queries", y_query_train_nls_parabola_n60)
check_query_balance("val queries", y_query_val_nls_parabola_n60)

# n = 40

np.random.seed(42)

d, n = 20, 40

# ---------- Training ----------
batch_size_train = 30000
dataset_train_nls_parabola_n40, y_query_train_nls_parabola_n40 = sample_nls_parabola_dataset(
    d, n, batch_size_train,
    k_val=0.1,
    margin=0.0,
    apply_margin_to_query=False,
)

assert dataset_train_nls_parabola_n40.shape == (batch_size_train, n+1, d+1)
assert y_query_train_nls_parabola_n40.shape == (batch_size_train,)
assert np.allclose(dataset_train_nls_parabola_n40[:, -1, -1], 0.0)

# ---------- Validation ----------
batch_size_val = 3000
dataset_val_nls_parabola_n40, y_query_val_nls_parabola_n40 = sample_nls_parabola_dataset(
    d, n, batch_size_val,
    k_val=0.1,
    margin=0.0,
    apply_margin_to_query=False,
)

assert dataset_val_nls_parabola_n40.shape == (batch_size_val, n+1, d+1)
assert y_query_val_nls_parabola_n40.shape == (batch_size_val,)
assert np.allclose(dataset_val_nls_parabola_n40[:, -1, -1], 0.0)

check_balance(dataset_train_nls_parabola_n40, y_query_train_nls_parabola_n40, n)
check_balance(dataset_val_nls_parabola_n40, y_query_val_nls_parabola_n40, n)
check_query_balance("train queries", y_query_train_nls_parabola_n40)
check_query_balance("val queries", y_query_val_nls_parabola_n40)

# Illustration

np.random.seed(934)

d, n = 2, 40
k_val = 3
A, b = sample_A_b(d=d, mode="parabola", k_val=k_val)

task_matrix, y_query = sample_nls_parabola_task_matrix(
    d=d, n=n, A=A, b=b,
    margin=0.2,
    apply_margin_to_query=False
)

x_ctx, y_ctx = task_matrix[:n, :d], task_matrix[:n, d]
x_query = task_matrix[n, :d]

u_dir = b / (np.linalg.norm(b) + 1e-8)         
beta_dir = np.array([-u_dir[1], u_dir[0]])      

beta_ctx = x_ctx @ beta_dir
beta_min, beta_max = beta_ctx.min(), beta_ctx.max()
pad = 0.5 * (beta_max - beta_min + 1e-6)
beta_grid = np.linspace(beta_min - pad, beta_max + pad, 400)

dm1 = d - 1
k_eff = -np.trace(A) / dm1          
alpha_curve = k_eff * (beta_grid**2 / dm1)

curve_xy = (alpha_curve[:, None] * u_dir[None, :]) + (beta_grid[:, None] * beta_dir[None, :])

all_x = np.vstack([x_ctx, x_query[None, :]])
x_min, x_max = all_x[:, 0].min(), all_x[:, 0].max()
y_min, y_max = all_x[:, 1].min(), all_x[:, 1].max()

pad_x = 0.2 * (x_max - x_min + 1e-6)
pad_y = 0.2 * (y_max - y_min + 1e-6)

plt.figure(figsize=(6, 6))

plt.scatter(x_ctx[y_ctx == 1, 0], x_ctx[y_ctx == 1, 1],
            c="#1f77b4", marker="o", s=60, edgecolors="k", label="Positive")
plt.scatter(x_ctx[y_ctx == -1, 0], x_ctx[y_ctx == -1, 1],
            c="#d62728", marker="s", s=60, edgecolors="k", label="Negative")
plt.scatter(x_query[0], x_query[1],
            c="#2ca02c", marker="*", s=250, edgecolors="k", label="Query")

plt.plot(curve_xy[:, 0], curve_xy[:, 1], "k--", linewidth=2, label="Parabola boundary")

plt.xlim(x_min - pad_x, x_max + pad_x)
plt.ylim(y_min - pad_y, y_max + pad_y)

plt.xlabel("Feature 1", fontsize=14)
plt.ylabel("Feature 2", fontsize=14)
plt.title("Nonlinearly Separable Task (Parabola)", fontsize=16, pad=12)
plt.legend(frameon=True, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# 1.2 Transformer Model

# Training

# Training function

def train_model(
    model: nn.Module,
    x: torch.Tensor, y: torch.Tensor,
    x_val: torch.Tensor, y_val: torch.Tensor,
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
    batch_train_acc, val_steps, val_acc = [], [], []
    grad_norms, param_norms = [], []
    best_val_acc, best_step = -1.0, -1

    for step in range(total_steps):
        idx = torch.randint(0, x.size(0), (batch_size,), device=device)
        xb, yb = x[idx], y[idx]
    
        optimizer.zero_grad()
        out = model(xb)
        pred = out[:, -1, -1]
        loss = 0.5 * (pred - yb).pow(2).mean()
       
        loss.backward()
        
        grad_norm_value = None
        if clip_grad is not None:
            grad_norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        else:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm_val = p.grad.data.norm(2)
                    total_norm += param_norm_val.item() ** 2
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
                out_v = model(x_val)
                pred_v = out_v[:, -1, -1]
                acc_v = ((pred_v * y_val) > 0).float().mean().item()
            model.train()

            val_steps.append(step)
            val_acc.append(acc_v)

            if acc_v > best_val_acc:
                best_val_acc, best_step = acc_v, step
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)

            if step % 5000 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Step {step:6d}, Loss={loss.item():.4f}, "
                      f"BatchTrainAcc={acc_train_batch:.3f}, ValAcc={acc_v:.3f}, "
                      f"GradNorm={grad_norm_value:.3f}, ParamNorm={param_norms[-1]:.3f}, LR={lr:.6f}")

    return {
        "label": label,
        "record_steps": record_steps,
        "record_loss": record_loss,
        "batch_train_acc": batch_train_acc,
        "val_steps": val_steps,
        "val_acc": val_acc,
        "best_acc": best_val_acc,
        "best_step": best_step,
        "save_path": save_path,
        "grad_norms": grad_norms,
        "param_norms": param_norms
    }

# Ploting result

# Plot function

def plot_results(results, title_suffix: str = ""):
    # Training Loss
    plt.figure()
    for res in results:
        plt.plot(res["record_steps"], res["record_loss"], label=res["label"])
    plt.xlabel("Step"); plt.ylabel("MSE Loss")
    plt.title(f"Training Loss vs Steps{title_suffix}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    # Training Accuracy (use 'batch_train_acc')
    plt.figure()
    for res in results:
        acc_train_pct = [a * 100.0 for a in res["batch_train_acc"]]
        plt.plot(res["record_steps"], acc_train_pct, label=res["label"])
    plt.xlabel("Step"); plt.ylabel("Training Accuracy (%)")
    plt.title(f"Training Accuracy vs Steps{title_suffix}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    # Validation Accuracy
    plt.figure()
    for res in results:
        acc_val_pct = [a * 100.0 for a in res["val_acc"]]
        plt.plot(res["val_steps"], acc_val_pct,
                 label=f'{res["label"]} (best={res["best_acc"]*100:.1f}%)')
    plt.xlabel("Step"); plt.ylabel("Validation Accuracy (%)")
    plt.title(f"Validation Accuracy vs Steps{title_suffix}")
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

# One-layer Linear: n=60, zero mean, parabola
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_linear_L1_n60 = SingleLayerLinearAttentionTransformer(d=20).to(device)

# Use NLS–parabola datasets (consistent naming)
x     = torch.from_numpy(dataset_train_nls_parabola_n60).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n60).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n60).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n60).float().to(device)

# Train
nls_parabola_linear_L1_n60 = train_model(
    model_nls_parabola_linear_L1_n60, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=5e-3,
    pct_start=0.30,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    label="nls-parabola-linear-L1-n60",
    save_path="nls_parabola_linear_L1_n60.pt"
)

# Plots
plot_results([nls_parabola_linear_L1_n60], title_suffix=" (nls-parabola-linear-L1, n=60)")

# L = 1, n = 40, zero mean

# One-layer Linear: n=40, zero mean, parabola
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_linear_L1_n40 = SingleLayerLinearAttentionTransformer(d=20).to(device)

# Use NLS–parabola datasets (n=40)
x     = torch.from_numpy(dataset_train_nls_parabola_n40).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n40).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n40).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n40).float().to(device)

# Train
nls_parabola_linear_L1_n40 = train_model(
    model_nls_parabola_linear_L1_n40, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=5e-3,
    pct_start=0.30,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    label="nls-parabola-linear-L1-n40",
    save_path="nls_parabola_linear_L1_n40.pt"
)

# Plots
plot_results([nls_parabola_linear_L1_n40], title_suffix=" (nls-parabola-linear-L1, n=40)")

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

# Multi-layer Linear: n=60, L=2, zero mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_linear_L2_n60 = MultiLayerLinearAttentionTransformer(d=20, L=2).to(device)

# Data
x     = torch.from_numpy(dataset_train_nls_parabola_n60).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n60).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n60).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n60).float().to(device)

# Train
nls_parabola_linear_L2_n60 = train_model(
    model_nls_parabola_linear_L2_n60, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=5e-4,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=2e-4,
    label="nls-parabola-linear-L2-n60",
    save_path="nls_parabola_linear_L2_n60.pt"
)   

# Plots
plot_results([nls_parabola_linear_L2_n60], title_suffix=" (nls-parabola-linear-L2, n=60)")

# L = 2, n = 40, zero mean

# Multi-layer Linear: n=40, L=2, zero mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_linear_L2_n40 = MultiLayerLinearAttentionTransformer(d=20, L=2).to(device)

# Data
x     = torch.from_numpy(dataset_train_nls_parabola_n40).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n40).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n40).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n40).float().to(device)

# Train
nls_parabola_linear_L2_n40 = train_model(
    model_nls_parabola_linear_L2_n40, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=3e-4,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=2e-4,
    label="nls-parabola-linear-L2-n40",
    save_path="nls_parabola_linear_L2_n40.pt"
)

# Plots
plot_results([nls_parabola_linear_L2_n40], title_suffix=" (nls-parabola-linear-L2, n=40)")

# L = 3, n = 60, zero mean

# Multi-layer Linear: n=60, L=3, zero mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_linear_L3_n60 = MultiLayerLinearAttentionTransformer(d=20, L=3).to(device)

# Data
x     = torch.from_numpy(dataset_train_nls_parabola_n60).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n60).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n60).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n60).float().to(device)

# Train
nls_parabola_linear_L3_n60 = train_model(
    model_nls_parabola_linear_L3_n60, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=2e-4,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=5e-4,
    label="nls-parabola-linear-L3-n60",
    save_path="nls_parabola_linear_L3_n60.pt"
)

# Plots
plot_results([nls_parabola_linear_L3_n60], title_suffix=" (nls-parabola-linear-L3, n=60)")

# L = 3, n = 40, zero mean

# Multi-layer Linear: n=40, L=3, zero mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_linear_L3_n40 = MultiLayerLinearAttentionTransformer(d=20, L=3).to(device)

# Data
x     = torch.from_numpy(dataset_train_nls_parabola_n40).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n40).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n40).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n40).float().to(device)

# Train
nls_parabola_linear_L3_n40 = train_model(
    model_nls_parabola_linear_L3_n40, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=2e-4,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=5e-4,
    label="nls-parabola-linear-L3-n40",
    save_path="nls_parabola_linear_L3_n40.pt"
)

# Plots
plot_results([nls_parabola_linear_L3_n40], title_suffix=" (nls-parabola-linear-L3, n=40)")

# L = 4, n = 60, zero mean

# Multi-layer Linear: n=60, L=4, zero mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_linear_L4_n60 = MultiLayerLinearAttentionTransformer(d=20, L=4).to(device)

# Data
x     = torch.from_numpy(dataset_train_nls_parabola_n60).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n60).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n60).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n60).float().to(device)

# Train
nls_parabola_linear_L4_n60 = train_model(
    model_nls_parabola_linear_L4_n60, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=1e-4,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=7e-4,
    label="nls-linear-L4-n60",
    save_path="nls_linear_L4_n60.pt"
)

# Plots
plot_results([nls_parabola_linear_L4_n60], title_suffix=" (nls-parabola-linear-L4, n=60)")

# L = 4, n = 40, zero mean

# Multi-layer Linear: n=40, L=4, zero mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_linear_L4_n40 = MultiLayerLinearAttentionTransformer(d=20, L=4).to(device)

# Data
x     = torch.from_numpy(dataset_train_nls_parabola_n40).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n40).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n40).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n40).float().to(device)

# Train
nls_parabola_linear_L4_n40 = train_model(
    model_nls_parabola_linear_L4_n40, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=3e-4,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=7e-4,
    label="nls-parabola-linear-L4-n40",
    save_path="nls_parabola_linear_L4_n40.pt"
)

# Plots
plot_results([nls_parabola_linear_L4_n40], title_suffix=" (nls-parabola-linear-L4, n=40)")

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

# One-layer Softmax: n=60, mean zero, parabola
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_softmax_L1_n60 = SingleLayerSoftmaxAttentionTransformer(d=20).to(device)

x     = torch.from_numpy(dataset_train_nls_parabola_n60).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n60).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n60).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n60).float().to(device)

# Train
nls_parabola_softmax_L1_n60 = train_model(
    model_nls_parabola_softmax_L1_n60, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=3e-3,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=1e-5,
    label="nls-parabola-softmax-L1-n60",
    save_path="nls_parabola_softmax_L1_n60.pt"
)

# Plot
plot_results([nls_parabola_softmax_L1_n60], title_suffix=" (nls-parabola-softmax-L1, n=60)")

# L = 1, n = 40, zero mean

# One-layer Softmax: n=40, mean zero, parabola
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_softmax_L1_n40 = SingleLayerSoftmaxAttentionTransformer(d=20).to(device)

x     = torch.from_numpy(dataset_train_nls_parabola_n40).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n40).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n40).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n40).float().to(device)

nls_parabola_softmax_L1_n40 = train_model(
    model_nls_parabola_softmax_L1_n40, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=3e-3,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=1e-5,
    label="nls-parabola-softmax-L1-n40",
    save_path="nls_parabola_softmax_L1_n40.pt"
)

# Plot
plot_results([nls_parabola_softmax_L1_n40], title_suffix=" (nls-parabola-softmax-L1, n=40)")

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

nls_parabola_model_softmax_L2_n60 = MultiLayerSoftmaxAttentionTransformer(d=20, L=2).to(device)

# Data
x     = torch.from_numpy(dataset_train_nls_parabola_n60).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n60).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n60).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n60).float().to(device)

# Train
nls_parabola_softmax_L2_n60 = train_model(
    nls_parabola_model_softmax_L2_n60, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=1e-3,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=6e-5,
    label="nls-parabola-softmax-L2-n60",
    save_path="nls_parabola_softmax_L2_n60.pt"
)

# Plot
plot_results([nls_parabola_softmax_L2_n60], title_suffix=" (nls-parabola-softmax-L2, n=60)")

# L = 2, n = 40, mean zero

# Multi-layer Softmax: n=40, L=2, mean zero, parabola
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_softmax_L2_n40 = MultiLayerSoftmaxAttentionTransformer(d=20, L=2, dropout=0.0).to(device)

x     = torch.from_numpy(dataset_train_nls_parabola_n40).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n40).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n40).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n40).float().to(device)

nls_parabola_softmax_L2_n40 = train_model(
    model_nls_parabola_softmax_L2_n40, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=1e-3,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=3e-5,
    label="nls-parabola-softmax-L2-n40",
    save_path="nls_parabola_softmax_L2_n40.pt"
)

# Plot
plot_results([nls_parabola_softmax_L2_n40], title_suffix=" (nls-parabola-softmax-L2, n=40)")

# L = 3, n = 60, zero mean, parabola

# Multi-layer Softmax: n=60, L=3, zero mean parabola
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_softmax_L3_n60 = MultiLayerSoftmaxAttentionTransformer(d=20, L=3, dropout=0.0).to(device)

x     = torch.from_numpy(dataset_train_nls_parabola_n60).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n60).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n60).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n60).float().to(device)

# Train
nls_parabola_softmax_L3_n60 = train_model(
    model_nls_parabola_softmax_L3_n60, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=5e-4,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=3e-4,
    label="nls-parabola-softmax-L3-n60",
    save_path="nls_parabola_softmax_L3_n60.pt"
)

# Plot
plot_results([nls_parabola_softmax_L3_n60], title_suffix=" (nls-parabola-softmax-L3, n=60)")

# L = 3, n = 40, zero mean

# Multi-layer Softmax: n=40, L=3, mean zero, parabola
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_softmax_L3_n40 = MultiLayerSoftmaxAttentionTransformer(d=20, L=3, dropout=0.0).to(device)

x     = torch.from_numpy(dataset_train_nls_parabola_n40).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n40).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n40).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n40).float().to(device)

nls_parabola_softmax_L3_n40 = train_model(
    model_nls_parabola_softmax_L3_n40, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=5e-4,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=3e-4,
    label="nls-parabola-softmax-L3-n40",
    save_path="nls_parabola_softmax_L3_n40.pt"
)

# Plot
plot_results([nls_parabola_softmax_L3_n40], title_suffix=" (nls-parabola-softmax-L3, n=40)")

# L = 4, n = 60, zero mean

# Multi-layer Softmax: n=60, L=4, zero mean parabola
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_softmax_L4_n60 = MultiLayerSoftmaxAttentionTransformer(d=20, L=4, dropout=0.0).to(device)

x     = torch.from_numpy(dataset_train_nls_parabola_n60).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n60).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n60).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n60).float().to(device)

# Train
nls_parabola_softmax_L4_n60 = train_model(
    model_nls_parabola_softmax_L4_n60, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=3e-4,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=6e-4,
    label="nls-parabola-softmax-L4-n60",
    save_path="nls_parabola_softmax_L4_n60.pt"
)

# Plot
plot_results([nls_parabola_softmax_L4_n60], title_suffix=" (nls-parabola-softmax-L4, n=60)")

# L = 4, n = 40, zero mean

# Multi-layer Softmax: n=40, L=4, mean zero, parabola
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nls_parabola_softmax_L4_n40 = MultiLayerSoftmaxAttentionTransformer(d=20, L=4, dropout=0.0).to(device)

x     = torch.from_numpy(dataset_train_nls_parabola_n40).float().to(device)
y     = torch.from_numpy(y_query_train_nls_parabola_n40).float().to(device)
x_val = torch.from_numpy(dataset_val_nls_parabola_n40).float().to(device)
y_val = torch.from_numpy(y_query_val_nls_parabola_n40).float().to(device)

nls_parabola_softmax_L4_n40 = train_model(
    model_nls_parabola_softmax_L4_n40, x, y, x_val, y_val,
    total_steps=150000,
    batch_size=1024,
    max_lr=3e-4,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=50.0,
    record_interval=500,
    weight_decay=4e-4,
    label="nls-parabola-softmax-L4-n40",
    save_path="nls_parabola_softmax_L4_n40.pt"
)

# Plot
plot_results([nls_parabola_softmax_L4_n40], title_suffix=" (nls-parabola-softmax-L4, n=40)")

# Model Dictionary

# nls–parabola model dictionary
model_specs_nls_parabola = [
    # nls-parabola-linear L1..L4
    {"name":"nls-parabola-linear-L1-n20","arch":"nls-parabola-linear","L":1,"n":20,"path":"nls_parabola_linear_L1_n20.pt"},
    {"name":"nls-parabola-linear-L1-n40","arch":"nls-parabola-linear","L":1,"n":40,"path":"nls_parabola_linear_L1_n40.pt"},
    {"name":"nls-parabola-linear-L1-n60","arch":"nls-parabola-linear","L":1,"n":60,"path":"nls_parabola_linear_L1_n60.pt"},
    {"name":"nls-parabola-linear-L2-n20","arch":"nls-parabola-linear","L":2,"n":20,"path":"nls_parabola_linear_L2_n20.pt"},
    {"name":"nls-parabola-linear-L2-n40","arch":"nls-parabola-linear","L":2,"n":40,"path":"nls_parabola_linear_L2_n40.pt"},
    {"name":"nls-parabola-linear-L2-n60","arch":"nls-parabola-linear","L":2,"n":60,"path":"nls_parabola_linear_L2_n60.pt"},
    {"name":"nls-parabola-linear-L3-n20","arch":"nls-parabola-linear","L":3,"n":20,"path":"nls_parabola_linear_L3_n20.pt"},
    {"name":"nls-parabola-linear-L3-n40","arch":"nls-parabola-linear","L":3,"n":40,"path":"nls_parabola_linear_L3_n40.pt"},
    {"name":"nls-parabola-linear-L3-n60","arch":"nls-parabola-linear","L":3,"n":60,"path":"nls_parabola_linear_L3_n60.pt"},
    {"name":"nls-parabola-linear-L4-n20","arch":"nls-parabola-linear","L":4,"n":20,"path":"nls_parabola_linear_L4_n20.pt"},
    {"name":"nls-parabola-linear-L4-n40","arch":"nls-parabola-linear","L":4,"n":40,"path":"nls_parabola_linear_L4_n40.pt"},
    {"name":"nls-parabola-linear-L4-n60","arch":"nls-parabola-linear","L":4,"n":60,"path":"nls_parabola_linear_L4_n60.pt"},

    # nls-parabola-softmax L1..L4
    {"name":"nls-parabola-softmax-L1-n20","arch":"nls-parabola-softmax","L":1,"n":20,"path":"nls_parabola_softmax_L1_n20.pt"},
    {"name":"nls-parabola-softmax-L1-n40","arch":"nls-parabola-softmax","L":1,"n":40,"path":"nls_parabola_softmax_L1_n40.pt"},
    {"name":"nls-parabola-softmax-L1-n60","arch":"nls-parabola-softmax","L":1,"n":60,"path":"nls_parabola_softmax_L1_n60.pt"},
    {"name":"nls-parabola-softmax-L2-n20","arch":"nls-parabola-softmax","L":2,"n":20,"path":"nls_parabola_softmax_L2_n20.pt"},
    {"name":"nls-parabola-softmax-L2-n40","arch":"nls-parabola-softmax","L":2,"n":40,"path":"nls_parabola_softmax_L2_n40.pt"},
    {"name":"nls-parabola-softmax-L2-n60","arch":"nls-parabola-softmax","L":2,"n":60,"path":"nls_parabola_softmax_L2_n60.pt"},
    {"name":"nls-parabola-softmax-L3-n20","arch":"nls-parabola-softmax","L":3,"n":20,"path":"nls_parabola_softmax_L3_n20.pt"},
    {"name":"nls-parabola-softmax-L3-n40","arch":"nls-parabola-softmax","L":3,"n":40,"path":"nls_parabola_softmax_L3_n40.pt"},
    {"name":"nls-parabola-softmax-L3-n60","arch":"nls-parabola-softmax","L":3,"n":60,"path":"nls_parabola_softmax_L3_n60.pt"},
    {"name":"nls-parabola-softmax-L4-n20","arch":"nls-parabola-softmax","L":4,"n":20,"path":"nls_parabola_softmax_L4_n20.pt"},
    {"name":"nls-parabola-softmax-L4-n40","arch":"nls-parabola-softmax","L":4,"n":40,"path":"nls_parabola_softmax_L4_n40.pt"},
    {"name":"nls-parabola-softmax-L4-n60","arch":"nls-parabola-softmax","L":4,"n":60,"path":"nls_parabola_softmax_L4_n60.pt"},
]

# 1.3 Testing Trained Model

# Evaluating model

# Evaluation function

def evaluate_model(model, x_np, y_np):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x_np).float().to(device)
        y = torch.from_numpy(y_np).float().to(device)
        out  = model(x)                  
        pred = out[:, -1, -1]            
        acc  = ((pred * y) > 0).float().mean().item()
    return acc

# Generating test data (predicting negative only)

def resample_test_query_parabola(dataset, y_query, u_all, k_val, margin=0.0, max_iter=1000):

    B, n_plus_1, d_plus_1 = dataset.shape
    d = d_plus_1 - 1
    assert u_all.shape == (B, d)

    u_norm = np.linalg.norm(u_all, axis=1, keepdims=True) + 1e-8
    u_all = u_all / u_norm

    def score(x, u):
        ux = float(np.dot(u, x))
        return ux - k_val * (float(np.dot(x, x)) - ux * ux)

    for i in range(B):
        x_q = dataset[i, n_plus_1 - 1, :d]
        s_old = score(x_q, u_all[i])

        if s_old >= -margin:
            found = False
            for _ in range(max_iter):
                x_cand = np.random.randn(d).astype(np.float32)
                if score(x_cand, u_all[i]) < -margin:
                    dataset[i, n_plus_1 - 1, :d] = x_cand
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    f"Task {i}: fail to find query with s(x) < -{margin}. "
                    "Increase max_iter or reduce margin."
                )

    dataset[:, n_plus_1 - 1, d] = 0.0
    y_query[:] = -1.0
    return dataset, y_query

# Testing data

np.random.seed(21)

d = 20
batch_size_test = 3000
ns = list(range(10, 31, 4))   # 10,14,18,22,26,30
k_val = 0.1

dataset_test_nls = {}  # dict[n] = (ds, yq, A_list, b_list)

for n in ns:
    ds = np.zeros((batch_size_test, n + 1, d + 1), dtype=np.float32)
    yq = np.full((batch_size_test,), -1.0, dtype=np.float32)
    A_list = np.zeros((batch_size_test, d, d), dtype=np.float32)
    b_list = np.zeros((batch_size_test, d), dtype=np.float32)

    for i in range(batch_size_test):
        # Sample A, b
        A, b = sample_A_b(d, mode="parabola", k_val=k_val)
        A_list[i] = A
        b_list[i] = b

        # Sample task matrix (unnormalized)
        task_matrix, _ = sample_nls_parabola_task_matrix(
            d, n, A, b, margin=0.0, apply_margin_to_query=False
        )

        # Force query to -1
        xq = task_matrix[n, :d]
        s_q = parabola_quad_score(xq[None, :], A, b).item()
        if s_q >= 0.0:
            xq = enforce_margin_point(xq, y=-1.0, A=A, b=b, margin=1e-4)
            task_matrix[n, :d] = xq

        # Set query label slot to 0
        task_matrix[n, d] = 0.0

        ds[i] = task_matrix
        yq[i] = -1.0

    # Sanity check
    assert np.all(yq == -1.0)
    assert np.allclose(ds[:, -1, -1], 0.0)

    dataset_test[n] = (ds, yq, A_list, b_list)

# Testing results

def load_saved_model_ls(path, arch, L, device=device):
    if arch in ("ls-softmax", "nls-parabola-softmax"):
        model = (SingleLayerSoftmaxAttentionTransformer(d=20)
                 if L == 1 else MultiLayerSoftmaxAttentionTransformer(d=20, L=L))
    elif arch in ("ls-linear", "nls-parabola-linear"):
        model = (SingleLayerLinearAttentionTransformer(d=20)
                 if L == 1 else MultiLayerLinearAttentionTransformer(d=20, L=L))
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

# Testing results on trained models (NLS–parabola test sets)
rows = []
for sp in model_specs_nls_parabola:
    arch, L, n_train, path = sp["arch"], sp["L"], sp["n"], sp["path"]
    try:
        model = load_saved_model_ls(path, arch, L, device=device)
    except FileNotFoundError:
        continue

    # dataset_test[n] = (x_np, y_np, A_list, b_list)
    for n_test, (x_np, y_np, _, _) in dataset_test.items():
        acc = evaluate_model(model, x_np, y_np, device=device)
        rows.append({
            "arch": arch,
            "L": L,
            "n_train": n_train,
            "n_test": n_test,
            "acc_clean": float(acc),
        })

df_all = (
    pd.DataFrame(rows)
      .sort_values(by=["arch","L","n_train","n_test"])
      .reset_index(drop=True)
)
print("\n Model Results")
print(df_all.to_string(index=False))

pivot = df_all.pivot_table(
    index=["arch","L","n_train"],
    columns="n_test",
    values="acc_clean"
).sort_index()

plt.figure(figsize=(8,6))

for (arch, L, n_train), group in df_all.groupby(["arch","L","n_train"], sort=False):
    g = group.sort_values("n_test")
    linestyle = "--" if "softmax" in arch else "-"
    label = f"{arch}-L{L}-n{n_train}"
    plt.plot(g["n_test"].values, g["acc_clean"].values,
             marker="o", linestyle=linestyle, label=label)

plt.xlabel("Test n")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Test Context Length")
plt.xticks(sorted(df_all["n_test"].unique()))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()
