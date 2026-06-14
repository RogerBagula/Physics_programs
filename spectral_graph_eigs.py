#!/usr/bin/env python3
"""
spectral_graph_eigs.py

Discrete graph-Laplacian approximation of the Laplace-Beltrami spectrum
for three 4-generator models across scales.

Outputs:
 - spectral_eigenvalues_table.csv  (sc0, model, eig1..eig8)
 - spectral_eigenvalues_plot.png   (first 8 eigenvalues vs sc0)
"""

import numpy as np
import math
import cmath
import csv
import time
import os
from typing import Tuple, Dict, List
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# ---------------------------
# User parameters (tune for speed/accuracy)
# ---------------------------
SC0_LIST = [0.25, 1/3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
W_A = (2, 3, 5, -30)            # Dodecahedral Poincare (det=+1)
W_B = (2, 3, 11, 66.0/5.0)      # D6 / Weeks-related (det=+1)
W_C = W_A                       # Seifert-Weber uses same angles but det=-1

MODEL_NAMES = ['A_dodeca_poincare', 'B_D6_weeks', 'C_seifert_weber']

# Sampling and graph parameters
N_SAMPLE_POINTS = 2500          # number of sample points in fundamental region (increase for accuracy)
K_NEIGHBORS = 12                # k for k-NN graph
SIGMA_SCALE = 0.5               # weight sigma = SIGMA_SCALE * median(kNN distances)
RNG_SEED = 20240614

CSV_OUT = "spectral_eigenvalues_table.csv"
PNG_OUT = "spectral_eigenvalues_plot.png"

# ---------------------------
# Linear algebra and Möbius helpers
# ---------------------------
def normalize_to_det(M: np.ndarray, target_det: complex = 1+0j) -> np.ndarray:
    M = np.array(M, dtype=complex)
    d = np.linalg.det(M)
    if abs(d) == 0:
        raise ValueError("Matrix determinant is zero")
    alpha = cmath.sqrt(target_det / d)
    return M * alpha

def mobius_apply(M: np.ndarray, z: complex) -> complex:
    a, b = M[0,0], M[0,1]
    c, d = M[1,0], M[1,1]
    denom = c*z + d
    if abs(denom) == 0:
        return None
    return (a*z + b) / denom

def safe_cos_term(val):
    if val == float('inf') or val == math.inf:
        return 1.0
    return math.cos(2.0 * math.pi / val)

def real_exp_over(val):
    if val == float('inf') or val == math.inf:
        return math.exp(0.0)
    return math.exp(2.0 * math.pi / val)

def complex_exp_over(val):
    if val == float('inf') or val == math.inf:
        return cmath.exp(0j)
    return cmath.exp(2j * math.pi / val)

def build_generators(w: Tuple[float,float,float,float], sc0: float, target_det: complex = 1+0j) -> Dict[str, np.ndarray]:
    p, q, r, s = w
    sc_p = sc0 * safe_cos_term(p)
    sc_q = sc0 * safe_cos_term(q)
    sc_r = sc0 * safe_cos_term(r)
    sc_s = sc0 * safe_cos_term(s)

    s1 = np.array([
        [ real_exp_over(p), sc_p ],
        [ 0.0, 1.0/real_exp_over(p) ]
    ], dtype=complex)

    s2 = np.array([
        [ sc_q, complex_exp_over(q) ],
        [ -np.conj(complex_exp_over(q)), 0.0 ]
    ], dtype=complex)

    s3 = np.array([
        [ 1j * complex_exp_over(r), 0.0 ],
        [ sc_r, -1j * np.conj(complex_exp_over(r)) ]
    ], dtype=complex)

    s4 = np.array([
        [ 0.0, 1j * complex_exp_over(s) ],
        [ 1j * np.conj(complex_exp_over(s)), sc_s ]
    ], dtype=complex)

    return {
        's1': normalize_to_det(s1, target_det),
        's2': normalize_to_det(s2, target_det),
        's3': normalize_to_det(s3, target_det),
        's4': normalize_to_det(s4, target_det)
    }

def hemisphere_from_matrix(M: np.ndarray, z0: complex = 0+0j):
    w = mobius_apply(M, z0)
    if w is None:
        return None
    c = 0.5 * (z0 + w)
    r = abs(w) / 2.0
    return (c.real, c.imag, r)

# ---------------------------
# Sampling fundamental region
# ---------------------------
def bounding_box_from_hemispheres(hemispheres: Dict[str, Dict], margin: float = 0.25):
    xs = []
    ys = []
    rs = []
    for h in hemispheres.values():
        cx, cy = h['center']
        r = h['radius']
        xs.extend([cx - r, cx + r])
        ys.extend([cy - r, cy + r])
        rs.append(r)
    if not xs:
        return (-1,1,-1,1, 0.0, 1.0)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    tmax = max(rs) if rs else 1.0
    xspan = xmax - xmin if xmax > xmin else 1.0
    yspan = ymax - ymin if ymax > ymin else 1.0
    xmin -= margin * max(1.0, xspan)
    xmax += margin * max(1.0, xspan)
    ymin -= margin * max(1.0, yspan)
    ymax += margin * max(1.0, yspan)
    tmin = 0.0
    tmax = tmax * (1.0 + margin)
    if tmax <= 0:
        tmax = 1.0
    return (xmin, xmax, ymin, ymax, tmin, tmax)

def sample_fundamental_region(hemispheres: Dict[str, Dict], n_points: int, rng: np.random.Generator):
    xmin, xmax, ymin, ymax, tmin, tmax = bounding_box_from_hemispheres(hemispheres)
    pts = []
    # rejection sampling: sample in box and keep points outside all hemispheres
    batch = max(1000, n_points // 5)
    attempts = 0
    while len(pts) < n_points and attempts < 2000:
        xs = rng.uniform(xmin, xmax, size=batch)
        ys = rng.uniform(ymin, ymax, size=batch)
        ts = rng.uniform(tmin, tmax, size=batch)
        keep_mask = np.ones(batch, dtype=bool)
        for h in hemispheres.values():
            cx, cy = h['center']
            r = h['radius']
            dx = xs - cx
            dy = ys - cy
            sq = dx*dx + dy*dy + ts*ts
            keep_mask &= (sq >= r*r)
        for x,y,t,keep in zip(xs,ys,ts,keep_mask):
            if keep:
                pts.append((x,y,t))
                if len(pts) >= n_points:
                    break
        attempts += 1
    if len(pts) < n_points:
        # fallback: pad with random points in box (may include inside hemispheres)
        extra = n_points - len(pts)
        xs = rng.uniform(xmin, xmax, size=extra)
        ys = rng.uniform(ymin, ymax, size=extra)
        ts = rng.uniform(tmin, tmax, size=extra)
        pts.extend(list(zip(xs,ys,ts)))
    return np.array(pts)

# ---------------------------
# Build sparse k-NN Gaussian-weight graph
# ---------------------------
def build_graph(points: np.ndarray, k: int = 12, sigma_scale: float = 0.5):
    n = points.shape[0]
    tree = cKDTree(points)
    dists, idxs = tree.query(points, k=k+1)  # includes self at index 0
    # drop self
    dists = dists[:,1:]
    idxs = idxs[:,1:]
    # choose sigma as scaled median of nonzero distances
    median_dist = np.median(dists)
    sigma = max(1e-6, sigma_scale * median_dist)
    rows = []
    cols = []
    vals = []
    for i in range(n):
        for j_idx, dist in zip(idxs[i], dists[i]):
            w = math.exp(- (dist*dist) / (2.0 * sigma * sigma))
            rows.append(i)
            cols.append(j_idx)
            vals.append(w)
            # symmetric entry will be added; we'll symmetrize later
    W = coo_matrix((vals, (rows, cols)), shape=(n,n))
    # symmetrize: W = (W + W^T)/2
    W = (W + W.T) * 0.5
    # degree and Laplacian
    degs = np.array(W.sum(axis=1)).flatten()
    D = diags(degs)
    L = D - W
    return L.tocsr(), D.tocsr(), sigma

# ---------------------------
# Compute first m eigenvalues of generalized problem L x = lambda D x
# ---------------------------
def compute_first_eigenvalues(L: csr_matrix, D: csr_matrix, m: int = 8):
    # convert to symmetric generalized eigenproblem: use eigsh with M = D
    # shift small positive regularization to D diagonal to avoid singularity
    # but eigsh accepts M as positive definite; ensure D has positive entries
    diagD = D.diagonal()
    # if any zeros, add tiny epsilon to those entries
    eps = 1e-12
    diagD_safe = np.where(diagD <= 0, eps, diagD)
    M_safe = diags(diagD_safe)
    try:
        # compute smallest nonzero eigenvalues; use sigma=0 with which='SM' may be unstable
        # instead compute smallest algebraic eigenvalues with which='SM'
        vals, vecs = eigsh(L, k=m+1, M=M_safe, sigma=0.0, which='LM', tol=1e-6, maxiter=5000)
        # eigsh with sigma=0 returns eigenvalues near sigma; we asked for m+1 to drop the zero mode
        vals = np.real(vals)
        vals_sorted = np.sort(vals)
    except Exception:
        # fallback: compute smallest algebraic eigenvalues without shift
        vals, vecs = eigsh(L, k=m+1, M=M_safe, which='SM', tol=1e-6, maxiter=5000)
        vals = np.real(vals)
        vals_sorted = np.sort(vals)
    # drop the smallest eigenvalue (near zero) if present
    # keep first m positive eigenvalues
    positive_vals = [v for v in vals_sorted if v > 1e-10]
    if len(positive_vals) < m:
        # pad with large numbers
        positive_vals.extend([float('inf')] * (m - len(positive_vals)))
    return positive_vals[:m]

# ---------------------------
# Main sweep
# ---------------------------
def run_sweep():
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    start = time.time()
    for i, sc0 in enumerate(SC0_LIST):
        print(f"\n=== sc0 = {sc0}  ({i+1}/{len(SC0_LIST)}) ===")
        # build hemispheres for each model
        gensA = build_generators(W_A, sc0, target_det=1+0j)
        hemisA = {}
        for name, M in gensA.items():
            hemi = hemisphere_from_matrix(M)
            if hemi is None:
                continue
            cx, cy, r = hemi
            hemisA[name] = {'center':[cx, cy], 'radius': r}

        gensB = build_generators(W_B, sc0, target_det=1+0j)
        hemisB = {}
        for name, M in gensB.items():
            hemi = hemisphere_from_matrix(M)
            if hemi is None:
                continue
            cx, cy, r = hemi
            hemisB[name] = {'center':[cx, cy], 'radius': r}

        gensC = build_generators(W_C, sc0, target_det=-1+0j)
        hemisC = {}
        for name, M in gensC.items():
            hemi = hemisphere_from_matrix(M)
            if hemi is None:
                continue
            cx, cy, r = hemi
            hemisC[name] = {'center':[cx, cy], 'radius': r}

        model_hemis = [hemisA, hemisB, hemisC]
        for model_idx, hemis in enumerate(model_hemis):
            model_name = MODEL_NAMES[model_idx]
            # sample points in fundamental region
            pts = sample_fundamental_region(hemis, N_SAMPLE_POINTS, rng)
            # build graph
            L, D, sigma = build_graph(pts, k=K_NEIGHBORS, sigma_scale=SIGMA_SCALE)
            # compute eigenvalues
            try:
                eigs = compute_first_eigenvalues(L, D, m=8)
            except Exception as e:
                print(f"  model {model_name} sc0={sc0}: eigen computation failed: {e}")
                eigs = [float('nan')] * 8
            print(f"  {model_name}: sigma={sigma:.4g}  eigs[1..8]={['{:.6g}'.format(x) for x in eigs]}")
            row = {
                'sc0': sc0,
                'model': model_name,
                'eigs': eigs
            }
            rows.append(row)
    elapsed = time.time() - start
    print(f"\nSweep complete in {elapsed:.1f}s")
    return rows

# ---------------------------
# Save CSV and plot
# ---------------------------
def save_csv(rows, filename=CSV_OUT):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['sc0', 'model'] + [f'eig{i+1}' for i in range(8)]
        writer.writerow(header)
        for r in rows:
            writer.writerow([r['sc0'], r['model']] + r['eigs'])
    print(f"CSV written: {filename}")

def plot_rows(rows, filename=PNG_OUT):
    # organize by model
    models = {}
    for r in rows:
        models.setdefault(r['model'], []).append(r)
    plt.figure(figsize=(10,7))
    markers = {'A_dodeca_poincare':'o', 'B_D6_weeks':'s', 'C_seifert_weber':'^'}
    colors = {'A_dodeca_poincare':'C0', 'B_D6_weeks':'C1', 'C_seifert_weber':'C2'}
    for model, recs in models.items():
        recs_sorted = sorted(recs, key=lambda x: x['sc0'])
        scs = [r['sc0'] for r in recs_sorted]
        eigs_matrix = np.array([r['eigs'] for r in recs_sorted])  # shape (len(scs), 8)
        # plot each of the 8 eigenvalues as a faint line, and highlight the first 3
        for j in range(8):
            plt.plot(scs, eigs_matrix[:,j], marker=markers.get(model,'o'), color=colors.get(model,'k'),
                     linestyle='-' if j<3 else '--', alpha=0.9 if j<3 else 0.45,
                     label=f"{model} eig{j+1}" if j==0 else None)
    plt.xlabel('sc0')
    plt.ylabel('Approx eigenvalues (graph Laplacian)')
    plt.title('First 8 approximate eigenvalues vs scale (models A,B,C)')
    plt.grid(True)
    # create legend entries for models
    handles = []
    for model in MODEL_NAMES:
        handles.append(plt.Line2D([0],[0], marker=markers.get(model,'o'), color=colors.get(model,'k'), label=model, linestyle=''))
    plt.legend(handles=handles, loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved: {filename}")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    print("Starting spectral graph eigenvalue sweep...")
    rows = run_sweep()
    save_csv(rows, CSV_OUT)
    plot_rows(rows, PNG_OUT)
    print("\nDone. Files produced:")
    print(" -", CSV_OUT)
    print(" -", PNG_OUT)
