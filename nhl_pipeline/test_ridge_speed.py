"""Quick test of the fixed X.multiply(sqrt_w) approach vs the old diags(w) approach."""
import numpy as np
from scipy.sparse import csr_matrix, diags as sparse_diags
from scipy.linalg import cho_factor, cho_solve
import time

np.random.seed(42)

# Build RAPM-scale problem: 282k × 1800 with ~10 nnz per row
print("Building 282k × 1800 sparse matrix...")
n, p = 282000, 1800
row_idx, col_idx, vals = [], [], []
for r in range(n):
    players = np.random.choice(p, size=10, replace=False)
    for pid in players:
        row_idx.append(r)
        col_idx.append(pid)
        vals.append(1.0)
X = csr_matrix((vals, (row_idx, col_idx)), shape=(n, p))
y = np.random.randn(n) * 0.001
w = np.random.uniform(5, 120, n)
alpha = 1e4
print(f"X shape: {X.shape}, nnz: {X.nnz}, nnz/row: {X.nnz/n:.0f}")

# Method 1: OLD — sparse_diags(w) + X.T @ W @ X
print("\n[OLD] X.T @ diags(w) @ X ...")
t0 = time.time()
W = sparse_diags(w)
XtWX_old = (X.T @ W @ X).toarray()
t_old = time.time() - t0
print(f"  Time: {t_old:.1f}s")

# Method 2: NEW — X.multiply(sqrt_w) + Xw.T @ Xw
print("\n[NEW] X.multiply(sqrt_w) then Xw.T @ Xw ...")
t0 = time.time()
sqrt_w = np.sqrt(w)
Xw = X.multiply(sqrt_w[:, np.newaxis])
XtWX_new = (Xw.T @ Xw).toarray()
t_new = time.time() - t0
print(f"  Time: {t_new:.1f}s")

# Compare
diff = np.abs(XtWX_old - XtWX_new).max()
print(f"\n  Max diff: {diff:.2e}")
print(f"  Speedup: {t_old/max(t_new,0.001):.1f}x")

# Full solve with new method
print("\n[FULL SOLVE] Normal equations with new method...")
t0 = time.time()
w_sum = w.sum()
y_mean = np.dot(w, y) / w_sum
x_mean = np.asarray(X.T @ w).ravel() / w_sum
y_c = y - y_mean
XtWX_new -= w_sum * np.outer(x_mean, x_mean)
XtWX_new += alpha * np.eye(p)
XtWy = np.asarray(X.T @ (w * y_c)).ravel()
c, low = cho_factor(XtWX_new)
beta = cho_solve((c, low), XtWy)
t_total = time.time() - t0
print(f"  Total solve time: {t_total:.1f}s")
print(f"  Coef range: [{beta.min():.6e}, {beta.max():.6e}]")
print("\n✅ Done!")
