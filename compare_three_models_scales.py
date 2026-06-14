#!/usr/bin/env python3
"""
compare_three_models_scales.py

Compare three 4-generator Fuchsian/quaternion models across scales:
 - Group A: Dodecahedral Poincare {2,3,5,-30} (det = +1)
 - Group B: D6 / Weeks-related {2,3,11,66/5} (det = +1)
 - Group C: Seifert-Weber hyperbolic dodecahedral {2,3,5,-30} (det = -1)

Outputs:
 - CSV: model_comparison_table.csv
 - PNG:  model_comparison_scales.png

Vectorized Monte-Carlo sampling (upper half-space) is used for volume estimates.
"""

import numpy as np
import math
import cmath
import csv
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# ---------------------------
# Parameters
# ---------------------------
SC0_LIST = [0.25, 1/3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
W_A = (2, 3, 5, -30)            # Dodecahedral Poincare
W_B = (2, 3, 11, 66.0/5.0)      # D6 / Weeks-related
W_C = W_A                       # Seifert-Weber uses same angles but det = -1
N_SAMPLES = 200_000
MARGIN = 0.25
RNG_SEED = 20240614

CSV_OUT = "model_comparison_table.csv"
PNG_OUT = "model_comparison_scales.png"

# ---------------------------
# Linear algebra helpers
# ---------------------------
def normalize_to_det(M: np.ndarray, target_det: complex = 1+0j) -> np.ndarray:
    """
    Scale matrix M by scalar alpha so that det(alpha * M) = target_det.
    alpha^2 * det(M) = target_det  => alpha = sqrt(target_det / det(M))
    """
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

# ---------------------------
# Build generators from user's Mathematica-style definitions
# ---------------------------
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

    # Normalize each generator to the requested determinant (target_det)
    return {
        's1': normalize_to_det(s1, target_det),
        's2': normalize_to_det(s2, target_det),
        's3': normalize_to_det(s3, target_det),
        's4': normalize_to_det(s4, target_det)
    }

# ---------------------------
# Hemisphere geometry
# ---------------------------
def hemisphere_from_matrix(M: np.ndarray, z0: complex = 0+0j):
    w = mobius_apply(M, z0)
    if w is None:
        return None
    c = 0.5 * (z0 + w)
    r = abs(w) / 2.0
    return (c.real, c.imag, r)

# ---------------------------
# Vectorized Monte-Carlo estimator
# ---------------------------
def estimate_volume_vectorized(hemispheres: Dict[str, Dict], n_samples: int = N_SAMPLES, margin: float = MARGIN, rng_seed: int = None) -> float:
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
# Run sweep for three models
# ---------------------------
def run_comparison():
    results = []
    rng_base = RNG_SEED

    for i, sc0 in enumerate(SC0_LIST):
        # Group A: det = +1
        gens_A = build_generators(W_A, sc0, target_det=1+0j)
        hemis_A = {}
        for name, M in gens_A.items():
            hemi = hemisphere_from_matrix(M)
            if hemi is None:
                continue
            cx, cy, r = hemi
            hemis_A[name] = {'center':[cx, cy], 'radius': r}

        # Group B: det = +1
        gens_B = build_generators(W_B, sc0, target_det=1+0j)
        hemis_B = {}
        for name, M in gens_B.items():
            hemi = hemisphere_from_matrix(M)
            if hemi is None:
                continue
            cx, cy, r = hemi
            hemis_B[name] = {'center':[cx, cy], 'radius': r}

        # Group C: det = -1 (Seifert-Weber style)
        gens_C = build_generators(W_C, sc0, target_det=-1+0j)
        hemis_C = {}
        for name, M in gens_C.items():
            hemi = hemisphere_from_matrix(M)
            if hemi is None:
                continue
            cx, cy, r = hemi
            hemis_C[name] = {'center':[cx, cy], 'radius': r}

        # Use distinct but reproducible seeds per run so A/B/C share same base randomness for comparability
        seedA = int(rng_base + 10*i + 1)
        seedB = int(rng_base + 10*i + 2)
        seedC = int(rng_base + 10*i + 3)

        volA = estimate_volume_vectorized(hemis_A, n_samples=N_SAMPLES, margin=MARGIN, rng_seed=seedA)
        volB = estimate_volume_vectorized(hemis_B, n_samples=N_SAMPLES, margin=MARGIN, rng_seed=seedB)
        volC = estimate_volume_vectorized(hemis_C, n_samples=N_SAMPLES, margin=MARGIN, rng_seed=seedC)

        ratio_AB = volA / volB if volB != 0 else float('inf')
        ratio_AC = volA / volC if volC != 0 else float('inf')
        ratio_BC = volB / volC if volC != 0 else float('inf')

        results.append({
            'sc0': sc0,
            'volA': volA,
            'volB': volB,
            'volC': volC,
            'ratio_AB': ratio_AB,
            'ratio_AC': ratio_AC,
            'ratio_BC': ratio_BC
        })

        print(f"sc0={sc0:.6g}  volA={volA:.6f}  volB={volB:.6f}  volC={volC:.6f}  AB={ratio_AB:.6f}  AC={ratio_AC:.6f}  BC={ratio_BC:.6f}")

    return results

# ---------------------------
# Save CSV and plot
# ---------------------------
def save_csv(results, filename=CSV_OUT):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sc0','volA','volB','volC','ratio_A_over_B','ratio_A_over_C','ratio_B_over_C'])
        for r in results:
            writer.writerow([r['sc0'], r['volA'], r['volB'], r['volC'], r['ratio_AB'], r['ratio_AC'], r['ratio_BC']])
    print(f"\nCSV written: {filename}")

def plot_results(results, filename=PNG_OUT):
    scs = [r['sc0'] for r in results]
    volA = [r['volA'] for r in results]
    volB = [r['volB'] for r in results]
    volC = [r['volC'] for r in results]
    ratio_AB = [r['ratio_AB'] for r in results]
    ratio_AC = [r['ratio_AC'] for r in results]
    ratio_BC = [r['ratio_BC'] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

    ax1.plot(scs, volA, marker='o', label='Dodecahedral (Poincare) A')
    ax1.plot(scs, volB, marker='s', label='D6 / Weeks B')
    ax1.plot(scs, volC, marker='^', label='Seifert-Weber (det=-1) C')
    ax1.set_ylabel('Estimated Volume')
    ax1.set_title('Model Volumes vs scale sc0')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(scs, ratio_AB, marker='o', label='A / B')
    ax2.plot(scs, ratio_AC, marker='s', label='A / C')
    ax2.plot(scs, ratio_BC, marker='^', label='B / C')
    ax2.set_xlabel('sc0')
    ax2.set_ylabel('Volume Ratio')
    ax2.set_title('Pairwise Volume Ratios vs scale sc0')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Plot saved: {filename}")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("Running comparison sweep for three models...")
    results = run_comparison()
    save_csv(results, CSV_OUT)
    plot_results(results, PNG_OUT)
    print("\nDone. Files produced:")
    print(" -", CSV_OUT)
    print(" -", PNG_OUT)
