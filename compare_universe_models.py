#!/usr/bin/env python3
"""
compare_universe_models.py

Compare three 4-generator SL(2,C) "universe" models:

  A: Elliptic dodecahedral (Poincaré)   wA = (2,3,5,-30)
  B: Hyperbolic D6-type                 wB = (2,3,11,66/5)
  C: Seifert–Weber hyperbolic dodecahedral (complex-rotated wA)

For each scale sc0 in SC0_LIST, estimate hyperbolic volume (upper half-space)
via vectorized Monte-Carlo, then compute pairwise ratios.

Outputs:
  - CSV: volume_models_table.csv
  - PNG: volume_models_plot.png
"""

import numpy as np
import math
import cmath
import csv
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

SC0_LIST = [0.25, 1/3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# A: Poincaré dodecahedral (elliptic)
W_A = (2, 3, 5, -30)

# B: D6 / Weeks-type hyperbolic
W_B = (2, 3, 11, 66.0/5.0)

# C: Seifert–Weber hyperbolic dodecahedral (same tuple as A, but complex rotations)
W_C = (2, 3, 5, -30)

N_SAMPLES = 200_000
MARGIN = 0.25
RNG_SEED = 20240614

OUT_CSV = "volume_models_table.csv"
OUT_PNG = "volume_models_plot.png"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Build generators for a given w and sc0
# mode:
#   "mixed"   -> real Exp[2π/p] for p, complex for q,r,s (your original)
#   "complex" -> complex Exp[2πi/w] for all four (Seifert–Weber style)
# ------------------------------------------------------------

def build_generators(w_tuple, sc0: float, mode: str):
    p, q, r, s = w_tuple

    sc_p = sc0 * safe_cos_term(p)
    sc_q = sc0 * safe_cos_term(q)
    sc_r = sc0 * safe_cos_term(r)
    sc_s = sc0 * safe_cos_term(s)

    if mode == "mixed":
        # Poincaré / D6 style: real for p, complex for q,r,s
        e_p = real_exp_over(p)
        e_q = complex_exp_over(q)
        e_r = complex_exp_over(r)
        e_s = complex_exp_over(s)
    elif mode == "complex":
        # Seifert–Weber style: complex rotations for all
        e_p = complex_exp_over(p)
        e_q = complex_exp_over(q)
        e_r = complex_exp_over(r)
        e_s = complex_exp_over(s)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # s1 = {{Exp[2*Pi/p], sc0*Cos[2*Pi/p]}, {0, Exp[-2*Pi/p]}};
    s1 = np.array([
        [ e_p, sc_p ],
        [ 0.0, 1.0/e_p ]
    ], dtype=complex)

    # s2 = {{sc0*Cos[2*Pi/q], Exp[2*Pi*I/q]}, {-Exp[-2*Pi*I/q], 0}};
    s2 = np.array([
        [ sc_q, e_q ],
        [ -np.conj(e_q), 0.0 ]
    ], dtype=complex)

    # s3 = {{I*Exp[2*Pi*I/r], 0}, {sc0*Cos[2*Pi/r], -I*Exp[-2*Pi*I/r]}};
    s3 = np.array([
        [ 1j*e_r, 0.0 ],
        [ sc_r, -1j*np.conj(e_r) ]
    ], dtype=complex)

    # s4 = {{0, I*Exp[2*Pi*I/s]}, {I*Exp[-2*Pi*I/s], sc0*Cos[2*Pi/s]}};
    s4 = np.array([
        [ 0.0, 1j*e_s ],
        [ 1j*np.conj(e_s), sc_s ]
    ], dtype=complex)

    return {
        "s1": normalize_matrix(s1),
        "s2": normalize_matrix(s2),
        "s3": normalize_matrix(s3),
        "s4": normalize_matrix(s4),
    }

# ------------------------------------------------------------
# Hemisphere from Möbius image of z0=0
# ------------------------------------------------------------

def hemisphere_from_matrix(M, z0=0+0j):
    w = mobius_apply(M, z0)
    if w is None:
        return None
    c = 0.