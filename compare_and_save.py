#!/usr/bin/env python3
# compare_and_save.py
# Computes scalar vs quadratic K variants, prints table, and writes CSV.

import math
import csv

# --- Constants (cgs) ---
e = 4.8032068e-10
c = 2.99792458e10
hbar = 1.05457266e-27
alpha = 1.0/137.035999206
thetaW0 = 0.231208
thetaWrad = math.asin(math.sqrt(thetaW0))

# --- base g definitions ---
g1 = alpha
g2 = math.cos(thetaWrad)**2 / (e**2 * c**2)
g3 = math.sin(thetaWrad)**2 / (e**2 * c**2)
g4 = alpha**4 / (g1 * g2 * g3)
g5_product = 24.0 * g1 * g2 * g3 * g4

# --- K variants ---
def K_scalar():
    Kn = (g1/3.0) + (2.0/3.0)*g4 + (1.0/3.0)*g2 + (2.0/3.0)*g3 - g5_product
    return Kn / hbar

def K_quadratic():
    Kn = (g1**2)/3.0 + (2.0/3.0)*(g4**2) + (1.0/3.0)*(g2**2) + (2.0/3.0)*(g3**2) - (g5_product**2)
    return Kn / hbar

K_s = K_scalar()
K_q = K_quadratic()

# --- helpers ---
def rx_from_tau(tau, K):
    return 1.0 / (c * tau * K)

def mx_from_rx(rx):
    return 4.0 * e**2 / (rx * c**2)

def t0_of_x(x, K):
    den = c * x * K
    return float('inf') if den == 0 else 1.0/den

def gamma_from_tau(tau):
    return hbar / tau

# --- particle list (name, symbol, mass MeV, lifetime s) ---
particles = [
    ("Muon","μ",105.7,2.1969811e-6),
    ("Tau","τ",1777.0,2.903e-13),
    ("Pi0","π0",135.0,8.52e-17),
    ("PiPlusMinus","π±",139.6,2.6033e-8),
    ("KPlusMinus","K±",493.677,1.2380e-8),
    ("K0S","K0S",497.611,0.8953e-10),
    ("K0L","K0L",497.611,5.116e-8),
    ("Neutron","n",939.6,880.2),
    ("W","W±",80400.0,3.0e-25),
    ("Z","Z0",91000.0,3.0e-25),
    ("Higgs","H",125090.0,1.56e-22),
    ("Top","t",172760.0,5.0e-25),
    ("D0","D0",1864.83,4.101e-13),
    ("DPlus","D+",1869.65,1.040e-12),
    ("Ds","Ds",1968.34,5.00e-13),
    ("Bplus","B+",5279.34,1.638e-12),
    ("B0","B0",5279.65,1.517e-12),
    ("Bs","Bs",5366.88,1.520e-12),
    ("Bc","Bc",6274.9,0.510e-12),
    ("Lambda_b","Λb",5619.6,1.470e-12),
    ("Proton","p",938.272,1.0e32)
]

# --- compute rows and write CSV ---
out_filename = "particle_lifetimes_compare.csv"
header = [
    "Particle","Symbol","Mass_MeV","Lifetime_s",
    "rx_scalar_cm","mx_scalar_g","t0_scalar_s",
    "rx_quad_cm","mx_quad_g","t0_quad_s",
    "ratio_rx_q_over_s","ratio_mx_q_over_s","Gamma_erg"
]

with open(out_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for name, sym, mass, tau in particles:
        try:
            rx_s = rx_from_tau(tau, K_s)
            mx_s = mx_from_rx(rx_s)
            t0_s = t0_of_x(rx_s, K_s)
        except Exception:
            rx_s = mx_s = t0_s = float('nan')
        try:
            rx_q = rx_from_tau(tau, K_q)
            mx_q = mx_from_rx(rx_q)
            t0_q = t0_of_x(rx_q, K_q)
        except Exception:
            rx_q = mx_q = t0_q = float('nan')
        ratio_rx = rx_q / rx_s if (rx_s and rx_s != 0) else float('nan')
        ratio_mx = mx_q / mx_s if (mx_s and mx_s != 0) else float('nan')
        gamma = gamma_from_tau(tau)
        writer.writerow([
            name, sym, f"{mass:.6g}", f"{tau:.6e}",
            f"{rx_s:.6e}", f"{mx_s:.6e}", f"{t0_s:.6e}",
            f"{rx_q:.6e}", f"{mx_q:.6e}", f"{t0_q:.6e}",
            f"{ratio_rx:.6e}", f"{ratio_mx:.6e}", f"{gamma:.6e}"
        ])

print(f"Saved comparison CSV to: {out_filename}")
print(f"K_scalar = {K_s:.6e}, K_quadratic = {K_q:.6e}, K_scalar/K_quadratic = {K_s/K_q:.6e}")
