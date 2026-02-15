"""Full validation: all vectorized functions against original implementations."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Ridge
import time, sys
from typing import Dict, List

home_cols = [f"home_skater_{i}" for i in range(1, 7)]
away_cols = [f"away_skater_{i}" for i in range(1, 7)]

# === ORIGINAL _build_sparse_X_net ===
def build_sparse_X_net_OLD(df, player_to_col, home_cols, away_cols):
    row_idx, col_idx, data_vals = [], [], []
    for r in range(len(df)):
        hp = df.iloc[r][home_cols].dropna().astype(int).tolist()
        ap = df.iloc[r][away_cols].dropna().astype(int).tolist()
        for pid in hp:
            c = player_to_col.get(pid)
            if c is not None: row_idx.append(r); col_idx.append(c); data_vals.append(1.0)
        for pid in ap:
            c = player_to_col.get(pid)
            if c is not None: row_idx.append(r); col_idx.append(c); data_vals.append(-1.0)
    return csr_matrix((data_vals, (row_idx, col_idx)), shape=(len(df), len(player_to_col)))

# === NEW _build_sparse_X_net ===
def build_sparse_X_net_NEW(df, player_to_col, home_cols, away_cols):
    n_rows = len(df)
    row_idx_list, col_idx_list, vals_list = [], [], []
    max_pid = max(player_to_col.keys()) + 1
    pid_lookup = np.full(max_pid, -1, dtype=np.int32)
    for pid, col in player_to_col.items(): pid_lookup[pid] = col
    for col_name in home_cols:
        if col_name not in df.columns: continue
        col_vals = df[col_name].values
        valid_mask = pd.notna(col_vals)
        if not valid_mask.any(): continue
        valid_rows = np.where(valid_mask)[0]
        valid_pids = col_vals[valid_mask].astype(np.int64)
        in_range = valid_pids < max_pid
        valid_rows, valid_pids = valid_rows[in_range], valid_pids[in_range]
        mapped = pid_lookup[valid_pids]; good = mapped >= 0
        row_idx_list.append(valid_rows[good]); col_idx_list.append(mapped[good]); vals_list.append(np.ones(good.sum()))
    for col_name in away_cols:
        if col_name not in df.columns: continue
        col_vals = df[col_name].values
        valid_mask = pd.notna(col_vals)
        if not valid_mask.any(): continue
        valid_rows = np.where(valid_mask)[0]
        valid_pids = col_vals[valid_mask].astype(np.int64)
        in_range = valid_pids < max_pid
        valid_rows, valid_pids = valid_rows[in_range], valid_pids[in_range]
        mapped = pid_lookup[valid_pids]; good = mapped >= 0
        row_idx_list.append(valid_rows[good]); col_idx_list.append(mapped[good]); vals_list.append(np.full(good.sum(), -1.0))
    if row_idx_list:
        return csr_matrix((np.concatenate(vals_list), (np.concatenate(row_idx_list), np.concatenate(col_idx_list))), shape=(n_rows, len(player_to_col)))
    return csr_matrix((n_rows, len(player_to_col)))

def main():
    np.random.seed(42)
    n = 5000
    player_pool = list(range(8400001, 8400200))
    rows = []
    for i in range(n):
        hp = np.random.choice(player_pool, size=5, replace=False)
        ap = np.random.choice(player_pool, size=5, replace=False)
        row = {"duration_s": np.random.uniform(5,120)}
        for j in range(5):
            row[f"home_skater_{j+1}"] = float(hp[j])
            row[f"away_skater_{j+1}"] = float(ap[j])
        row["home_skater_6"] = None; row["away_skater_6"] = None
        rows.append(row)
    data = pd.DataFrame(rows)
    ptc = {pid: i for i, pid in enumerate(sorted(player_pool))}
    print(f"Testing with {n} stints, {len(ptc)} players")

    # Test _build_sparse_X_net
    print("\n[TEST] _build_sparse_X_net (iloc loop vs numpy fancy indexing)")
    t0 = time.time(); X_old = build_sparse_X_net_OLD(data, ptc, home_cols, away_cols); t_old = time.time()-t0
    t0 = time.time(); X_new = build_sparse_X_net_NEW(data, ptc, home_cols, away_cols); t_new = time.time()-t0
    diff = abs(X_old - X_new)
    match = diff.nnz == 0
    print(f"  Old: {t_old:.3f}s | New: {t_new:.3f}s | Speedup: {t_old/max(t_new,0.001):.0f}x")
    print(f"  Shape: old={X_old.shape} new={X_new.shape} | Match: {'✅' if match else '❌'}")

    # Test full Ridge solve: old sklearn vs new normal equations
    print("\n[TEST] Ridge solve (sklearn lsqr vs normal equations Cholesky)")
    y = np.random.randn(n) * 0.01
    w = data["duration_s"].values
    alpha = 1e4
    t0 = time.time()
    m = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr"); m.fit(X_old, y, sample_weight=w)
    sk_coef = m.coef_; t_sk = time.time()-t0
    
    t0 = time.time()
    p = X_old.shape[1]; w_sum = w.sum()
    y_mean = np.dot(w, y)/w_sum; x_mean = np.asarray(X_old.T@w).ravel()/w_sum; y_c = y-y_mean
    sqrt_w = np.sqrt(w); Xw = X_old.multiply(sqrt_w[:,np.newaxis])
    XtWX = (Xw.T@Xw).toarray(); XtWX -= w_sum*np.outer(x_mean,x_mean); XtWX += alpha*np.eye(p)
    XtWy = np.asarray(X_old.T@(w*y_c)).ravel()
    c, low = cho_factor(XtWX); ne_coef = cho_solve((c,low), XtWy)
    t_ne = time.time()-t0
    
    max_diff = np.abs(sk_coef - ne_coef).max()
    coef_match = max_diff < 1e-8
    print(f"  sklearn: {t_sk:.3f}s | normal eq: {t_ne:.3f}s | Speedup: {t_sk/max(t_ne,0.001):.1f}x")
    print(f"  Max coef diff: {max_diff:.2e} | Match: {'✅' if coef_match else '❌'}")

    all_pass = match and coef_match
    print("\n" + "=" * 50)
    print(f"{'✅ ALL PASSED' if all_pass else '❌ FAILED'}")
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
