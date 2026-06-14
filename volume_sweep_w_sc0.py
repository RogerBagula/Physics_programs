#!/usr/bin/env python3
"""
volume_sweep_w_sc0.py

Sweep volumes over many angular tuples w and sc0 values.
Vectorized Monte-Carlo sampling; outputs CSV "volume_table.csv".
"""

import numpy as np
import math
import cmath
import csv
import time
from typing import List, Tuple

# ---------------------------
# Parameters (edit if desired)
# ---------------------------
DEFAULT_SC0_LIST = [0.25, 1/3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
DEFAULT_W_LIST = [
    (2, 3, 3, -6),
    (2, 3, 4, -12),
    (2, 3, 5, -30),
    (2, 3, 6, float('inf')),
    (2, 3, 7, 42),
    (2, 3, 11, 66.0/5.0)
]
N_SAMPLES = 200_000     # vectorized Monte-Carlo samples per (w,sc0)
MARGIN = 0.25           # bounding box margin fraction
RNG_SEED = 123456       # reproducible results
OUT_CSV = "volume_table.csv"

# ---------------------------
# Helpers
# ---------------------------
def normalize_matrix(M: np.ndarray) -> np.ndarray:
    M = np.array(M, dtype=complex)
    d = np.linalg.det(M)
    if abs(d) == 0:
        raise ValueError("Matrix determinant is zero")
    return M / (d ** 0.5)

def mobius_apply(M: np.ndarray, z: complex) -> complex:
    a, b = M[0,0], M[0,1]
    c, d = M[1,0], M[1,1]
    denom = c*z + d
    if abs(denom) == 0:
        return None
    return (a*z + b) / denom

def safe_cos_term(val):
    if val is None:
        return 1.0
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

# ---------------------------
# Build generators for given w tuple and sc0
# (follows the Mathematica-style definitions you provided)
# ---------------------------
def build_generators_from_w(w_tuple: Tuple[float,float,float,float], sc0: float):
    p, q, r, s = w_tuple

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
        's1': normalize_matrix(s1),
        's2': normalize_matrix(s2),
        's3': normalize_matrix(s3),
        's4': normalize_matrix(s4)
    }

# ---------------------------
# Hemisphere from Möbius image of z0=0
# center c = (0 + w)/2, radius r = |w|/2
# ---------------------------
def hemisphere_from_matrix(M, z0=0+0j):
    w = mobius_apply(M, z0)
    if w is None:
        return None
    c = 0.5 * (z0 + w)
    r = abs(w) / 2.0
    return (c.real, c.imag, r)

# ---------------------------
# Vectorized Monte-Carlo estimator
# ---------------------------
def estimate_volume_vectorized(hemispheres: dict, n_samples: int = N_SAMPLES, margin: float = MARGIN, rng_seed: int = None) -> float:
    if not hemispheres:
        return 0.0

    xs = []
    ys = []
    rs = []
    for h in hemispheres.values():
        cx, cy = h['center']
        r = h['radius']
        xs.extend([cx - r, cx + r])
        ys.extend([cy - r, cy + r])
        rs.append(r)

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

    vol_box = (xmax - xmin) * (ymax - ymin) * (tmax - tmin)

    rng = np.random.default_rng(rng_seed)

    xsamp = rng.uniform(xmin, xmax, size=n_samples)
    ysamp = rng.uniform(ymin, ymax, size=n_samples)
    tsamp = rng.uniform(tmin, tmax, size=n_samples)

    inside_mask = np.ones(n_samples, dtype=bool)

    for h in hemispheres.values():
        cx, cy = h['center']
        r = h['radius']
        dx = xsamp - cx
        dy = ysamp - cy
        sq = dx*dx + dy*dy + tsamp*tsamp
        inside_mask &= (sq >= r*r)

    count_inside = np.count_nonzero(inside_mask)
    return vol_box * (count_inside / float(n_samples))

# ---------------------------
# Sweep and write CSV
# ---------------------------
def sweep_and_write(w_list: List[Tuple[float,float,float,float]], sc0_list: List[float], out_csv: str = OUT_CSV, samples: int = N_SAMPLES):
    rows = []
    total = len(w_list) * len(sc0_list)
    idx = 0
    start_time = time.time()

    for w in w_list:
        for sc0 in sc0_list:
            idx += 1
            t0 = time.time()
            gens = build_generators_from_w(w, sc0)
            hemisphere_data = {}
            for name, M in gens.items():
                hemi = hemisphere_from_matrix(M)
                if hemi is None:
                    continue
                cx, cy, r = hemi
                hemisphere_data[name] = {'center':[cx, cy], 'radius': r}

            vol = estimate_volume_vectorized(hemisphere_data, n_samples=samples, margin=MARGIN, rng_seed=(RNG_SEED + idx))
            rows.append((w, sc0, vol))
            dt = time.time() - t0
            elapsed = time.time() - start_time
            print(f"[{idx}/{total}] w={w}, sc0={sc0:.3f}  →  vol={vol:.6f}  (t={dt:.2f}s, elapsed={elapsed:.1f}s)")

    # write CSV
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['p','q','r','s','sc0','volume'])
        for w, sc0, vol in rows:
            writer.writerow([w[0], w[1], w[2], w[3], sc0, vol])

    print(f"\nDone. Results written to {out_csv}")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("Starting sweep over W list and SC0 list...")
    sweep_and_write(DEFAULT_W_LIST, DEFAULT_SC0_LIST, out_csv=OUT_CSV, samples=N_SAMPLES)
