#!/usr/bin/env python3
"""
dodecahedral_scale_volume_table.py

Estimate hyperbolic volumes for the Dodecahedral Fuchsian quaternion 4-group
(w = {2,3,5,-30}) while varying sc0 over a symmetric range of scales.
Vectorized Monte-Carlo sampling is used for speed.
"""

import numpy as np
import math
import cmath
import random

# ---------------------------
# Parameters (edit if desired)
# ---------------------------
W = (2, 3, 5, -30)                     # the tuple (p,q,r,s)
SC0_LIST = [0.25, 1/3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
N_SAMPLES = 200_000                    # increase for more accuracy
MARGIN = 0.25                          # bounding box margin fraction

# ---------------------------
# Helpers
# ---------------------------
def normalize_matrix(M):
    M = np.array(M, dtype=complex)
    d = np.linalg.det(M)
    if abs(d) == 0:
        raise ValueError("Matrix determinant is zero")
    return M / (d ** 0.5)

def mobius_apply(M, z):
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
# Build generators for given sc0 and fixed W=(2,3,5,-30)
# Following the Mathematica-style definitions you provided:
# s1 = {{Exp[2*Pi/p], sc0*Cos[2*Pi/p]}, {0, Exp[-2*Pi/p]}};
# s2 = {{sc0*Cos[2*Pi/q], Exp[2*Pi*I/q]}, {-Exp[-2*Pi*I/q], 0}};
# s3 = {{I*Exp[2*Pi*I/r], 0}, {sc0*Cos[2*Pi/r], -I*Exp[-2*Pi*I/r]}};
# s4 = {{0, I*Exp[2*Pi*I/s]}, {I*Exp[-2*Pi*I/s], sc0*Cos[2*Pi/s]}};
# ---------------------------
def build_generators_for_sc0(sc0, w_tuple=W):
    p, q, r, s = w_tuple

    sc_p = sc0 * safe_cos_term(p)
    sc_q = sc0 * safe_cos_term(q)
    sc_r = sc0 * safe_cos_term(r)
    sc_s = sc0 * safe_cos_term(s)

    # s1: uses real exponential for p
    s1 = np.array([
        [ real_exp_over(p), sc_p ],
        [ 0.0, 1.0/real_exp_over(p) ]
    ], dtype=complex)

    # s2: complex exponential for q
    s2 = np.array([
        [ sc_q, complex_exp_over(q) ],
        [ -np.conj(complex_exp_over(q)), 0.0 ]
    ], dtype=complex)

    # s3: complex exponential for r, multiplied by I
    s3 = np.array([
        [ 1j * complex_exp_over(r), 0.0 ],
        [ sc_r, -1j * np.conj(complex_exp_over(r)) ]
    ], dtype=complex)

    # s4: complex exponential for s, multiplied by I
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
def estimate_volume_vectorized(hemispheres, n_samples=N_SAMPLES, margin=MARGIN):
    if not hemispheres:
        return 0.0

    # bounding box from hemisphere centers and radii
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

    # vectorized sampling
    xsamp = np.random.uniform(xmin, xmax, size=n_samples)
    ysamp = np.random.uniform(ymin, ymax, size=n_samples)
    tsamp = np.random.uniform(tmin, tmax, size=n_samples)

    inside_mask = np.ones(n_samples, dtype=bool)

    # for each hemisphere, mark points that fall inside it (these are excluded)
    for h in hemispheres.values():
        cx, cy = h['center']
        r = h['radius']
        dx = xsamp - cx
        dy = ysamp - cy
        # compute squared distance in (x,y,t)
        sq = dx*dx + dy*dy + tsamp*tsamp
        inside_mask &= (sq >= r*r)   # keep points outside hemisphere

    count_inside = np.count_nonzero(inside_mask)
    return vol_box * (count_inside / float(n_samples))

# ---------------------------
# Main: run table for SC0_LIST
# ---------------------------
if __name__ == "__main__":
    print("\nDodecahedral Fuchsian 4-group (w = {2,3,5,-30})")
    print("Vectorized Monte-Carlo volume estimates")
    print(f"N_SAMPLES = {N_SAMPLES}, margin = {MARGIN}\n")
    print(f"{'sc0':>6s}   {'Volume (estimate)':>20s}")

    for sc0 in SC0_LIST:
        gens = build_generators_for_sc0(sc0, W)
        hemisphere_data = {}
        for name, M in gens.items():
            hemi = hemisphere_from_matrix(M)
            if hemi is None:
                continue
            cx, cy, r = hemi
            hemisphere_data[name] = {'center':[cx, cy], 'radius': r}

        vol = estimate_volume_vectorized(hemisphere_data, n_samples=N_SAMPLES, margin=MARGIN)
        print(f"{sc0:6.3f}   {vol:20.6f}")

    print("\nDone.\n")
