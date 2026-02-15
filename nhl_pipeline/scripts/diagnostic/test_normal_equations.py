"""
Validate that the normal equations solver produces identical results to sklearn Ridge.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags as sparse_diags, random as sparse_random
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Ridge
import time
import sys


def ridge_sklearn(X, y, w, alpha):
    """sklearn reference implementation."""
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
    model.fit(X, y, sample_weight=w)
    return model.coef_


def ridge_normal_equations(X, y, w, alpha):
    """Normal equations implementation (from compute_corsi_apm.py)."""
    n, p = X.shape
    w_sum = w.sum()
    y_mean = np.dot(w, y) / w_sum
    x_mean = np.asarray(X.T @ w).ravel() / w_sum
    y_c = y - y_mean
    W = sparse_diags(w)
    XtWX = (X.T @ W @ X).toarray()
    XtWX -= w_sum * np.outer(x_mean, x_mean)
    XtWX += alpha * np.eye(p)
    XtWy = np.asarray(X.T @ (w * y_c)).ravel()
    c, low = cho_factor(XtWX)
    beta = cho_solve((c, low), XtWy)
    return beta


def main():
    np.random.seed(42)
    all_pass = True

    # Test 1: Small problem (exact comparison)
    print("[TEST 1] Small sparse problem (100 × 20)")
    n, p = 100, 20
    X = sparse_random(n, p, density=0.15, format='csr')
    y = np.random.randn(n)
    w = np.random.uniform(5, 120, n)
    alpha = 1e4

    t0 = time.time(); coef_sk = ridge_sklearn(X, y, w, alpha); t_sk = time.time() - t0
    t0 = time.time(); coef_ne = ridge_normal_equations(X, y, w, alpha); t_ne = time.time() - t0

    max_diff = np.max(np.abs(coef_sk - coef_ne))
    match = max_diff < 1e-8
    print(f"  sklearn: {t_sk:.4f}s | normal eq: {t_ne:.4f}s")
    print(f"  Max coef diff: {max_diff:.2e} | Match: {'✅' if match else '❌'}")
    all_pass &= match

    # Test 2: RAPM-like problem (similar to actual data)
    print("\n[TEST 2] RAPM-like sparse problem (10000 × 200, ~10 nnz/row)")
    n, p = 10000, 200
    # Build RAPM-like matrix: each row has +1 for 5 "offense" and +1 for 5 "defense" players
    row_idx, col_idx, vals = [], [], []
    for r in range(n):
        players = np.random.choice(p, size=10, replace=False)
        for pid in players:
            row_idx.append(r)
            col_idx.append(pid)
            vals.append(1.0)
    X = csr_matrix((vals, (row_idx, col_idx)), shape=(n, p))
    y = np.random.randn(n) * 0.01  # Small targets like real RAPM
    w = np.random.uniform(5, 120, n)

    t0 = time.time(); coef_sk = ridge_sklearn(X, y, w, alpha); t_sk = time.time() - t0
    t0 = time.time(); coef_ne = ridge_normal_equations(X, y, w, alpha); t_ne = time.time() - t0

    max_diff = np.max(np.abs(coef_sk - coef_ne))
    match = max_diff < 1e-8
    print(f"  sklearn: {t_sk:.4f}s | normal eq: {t_ne:.4f}s | Speedup: {t_sk/max(t_ne,0.001):.1f}x")
    print(f"  Max coef diff: {max_diff:.2e} | Match: {'✅' if match else '❌'}")
    all_pass &= match

    # Test 3: Full-scale offdef problem (realistic size)
    print("\n[TEST 3] Full-scale offdef (282000 × 1800, ~10 nnz/row)")
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

    print(f"  Running sklearn Ridge (lsqr)...")
    t0 = time.time(); coef_sk = ridge_sklearn(X, y, w, alpha); t_sk = time.time() - t0
    print(f"  sklearn: {t_sk:.1f}s")

    print(f"  Running normal equations (Cholesky)...")
    t0 = time.time(); coef_ne = ridge_normal_equations(X, y, w, alpha); t_ne = time.time() - t0
    print(f"  normal eq: {t_ne:.1f}s")

    max_diff = np.max(np.abs(coef_sk - coef_ne))
    rel_diff = max_diff / (np.max(np.abs(coef_sk)) + 1e-15)
    match = rel_diff < 1e-4  # Slightly more tolerance for large problem
    print(f"  Speedup: {t_sk/max(t_ne,0.001):.1f}x")
    print(f"  Max coef diff: {max_diff:.2e} (relative: {rel_diff:.2e}) | Match: {'✅' if match else '❌'}")
    all_pass &= match

    print("\n" + "=" * 60)
    if all_pass:
        print("✅ ALL TESTS PASSED — normal equations solver matches sklearn Ridge")
    else:
        print("❌ SOME TESTS FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
