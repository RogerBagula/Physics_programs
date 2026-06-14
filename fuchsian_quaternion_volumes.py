#!/usr/bin/env python3
"""
fuchsian_quaternion_volumes.py

Estimate hyperbolic volumes for Platonic Fuchsian quaternion 4-groups
as defined by Roger's Mathematica-style matrices.
"""

import numpy as np
import math
import random
import cmath

# ---------------------------
# Utility: normalize to SL(2,C)
# ---------------------------
def normalize_matrix(M):
    M = np.array(M, dtype=complex)
    d = np.linalg.det(M)
    if abs(d) == 0:
        raise ValueError("Matrix determinant is zero")
    return M / (d ** 0.5)

# ---------------------------
# Möbius action
# ---------------------------
def mobius_apply(M, z):
    a, b = M[0,0], M[0,1]
    c, d = M[1,0], M[1,1]
    denom = c*z + d
    if abs(denom) == 0:
        return None
    return (a*z + b) / denom

# ---------------------------
# Build generators from (p,q,r,s)
# ---------------------------
def build_generators_from_w(w_tuple, sc0=2.0):
    # w_tuple may contain 'math.inf' for Infinity
    p, q, r, s = w_tuple

    def safe_cos_term(val):
        if val is None:
            return 1.0
        if val == float('inf') or val == math.inf:
            return 1.0
        return math.cos(2.0 * math.pi / val)

    # scales for each generator (following your note sc = sc0 * Cos[2*Pi/w])
    sc_p = sc0 * safe_cos_term(p)
    sc_q = sc0 * safe_cos_term(q)
    sc_r = sc0 * safe_cos_term(r)
    sc_s = sc0 * safe_cos_term(s)

    # build matrices exactly as in your Mathematica lines
    # s1 = {{Exp[2*Pi/p], sc0*Cos[2*Pi/p]}, {0, Exp[-2*Pi/p]}};
    # s2 = {{sc0*Cos[2*Pi/q], Exp[2*Pi*I/q]}, {-Exp[-2*Pi*I/q], 0}};
    # s3 = {{I*Exp[2*Pi*I/r], 0}, {sc0*Cos[2*Pi/r], -I*Exp[-2*Pi*I/r]}};
    # s4 = {{0, I*Exp[2*Pi*I/s]}, {I*Exp[-2*Pi*I/s], sc0*Cos[2*Pi/s]}};

    # For real exponentials use math.exp(2*pi/p). For complex exponentials use cmath.exp(2j*pi/val).
    def real_exp_over(val):
        if val == float('inf') or val == math.inf:
            return math.exp(0.0)
        return math.exp(2.0 * math.pi / val)

    def complex_exp_over(val):
        if val == float('inf') or val == math.inf:
            return cmath.exp(0j)
        return cmath.exp(2j * math.pi / val)

    # Build matrices (dtype complex)
    s1 = np.array([
        [ real_exp_over(p), sc_p ],
        [ 0.0, 1.0/real_exp_over(p) ]  # Exp[-2*Pi/p] = 1/Exp[2*Pi/p]
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

    # Normalize to SL(2,C)
    return {
        's1': normalize_matrix(s1),
        's2': normalize_matrix(s2),
        's3': normalize_matrix(s3),
        's4': normalize_matrix(s4)
    }

# ---------------------------
# Hemisphere from Möbius image of z0=0
# ---------------------------
def isometric_hemisphere_from_matrix(M, z0=0+0j):
    w = mobius_apply(M, z0)
    if w is None:
        return None
    c = 0.5 * (z0 + w)
    r = abs(w) / 2.0
    return (c.real, c.imag, r)

# ---------------------------
# Monte-Carlo volume estimator (upper half-space)
# ---------------------------
def estimate_volume_from_hemispheres(hemispheres, n_samples=100000, margin=0.2):
    # hemispheres: dict name -> {'center':[cx,cy], 'radius':r}
    if not hemispheres:
        return 0.0

    # compute bounding box from hemisphere centers and radii
    xs = []
    ys = []
    rs = []
    for h in hemispheres.values():
        cx, cy = h['center']
        r = h['radius']
        xs.append(cx - r)
        xs.append(cx + r)
        ys.append(cy - r)
        ys.append(cy + r)
        rs.append(r)

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    tmax = max(rs) if rs else 1.0

    # expand by margin fraction
    xspan = xmax - xmin
    yspan = ymax - ymin
    xmin -= margin * max(1.0, xspan)
    xmax += margin * max(1.0, xspan)
    ymin -= margin * max(1.0, yspan)
    ymax += margin * max(1.0, yspan)
    tmin = 0.0
    tmax = tmax * (1.0 + margin)
    if tmax <= 0:
        tmax = 1.0

    vol_box = (xmax - xmin) * (ymax - ymin) * (tmax - tmin)
    count = 0

    # Monte Carlo sampling (simple loop)
    for _ in range(n_samples):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        t = random.uniform(tmin, tmax)
        inside = True
        for h in hemispheres.values():
            cx, cy = h['center']
            r = h['radius']
            if (x - cx)**2 + (y - cy)**2 + t**2 < r**2:
                inside = False
                break
        if inside:
            count += 1

    return vol_box * (count / n_samples)

# ---------------------------
# Main: run over your w list
# ---------------------------
if __name__ == "__main__":
    # Your list of Platonic-like tuples (Mathematica list)
    w_list = [
        (2, 3, 3, -6),
        (2, 3, 4, -12),
        (2, 3, 5, -30),
        (2, 3, 6, float('inf')),   # Infinity -> treated as large
        (2, 3, 7, 42),
        (2, 3, 11, 66.0/5.0)
    ]

    sc0 = 2.0
    samples = 120000

    print("\nFuchsian Quaternion 4-groups: volume estimates")
    print("sc0 =", sc0, ", Monte-Carlo samples =", samples)
    print("-------------------------------------------------\n")

    for w in w_list:
        try:
            gens = build_generators_from_w(w, sc0=sc0)
        except Exception as e:
            print(f"w={w}  -> error building generators: {e}")
            continue

        # compute hemispheres
        hemisphere_data = {}
        for name, M in gens.items():
            hemi = isometric_hemisphere_from_matrix(M)
            if hemi is None:
                continue
            cx, cy, r = hemi
            hemisphere_data[name] = {'center':[cx, cy], 'radius': r}

        # estimate volume
        vol = estimate_volume_from_hemispheres(hemisphere_data, n_samples=samples)
        print(f"w={w}  -> Volume ≈ {vol:.6f}")

    print("\nDone.\n")
