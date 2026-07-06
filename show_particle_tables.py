#!/usr/bin/env python3
# show_particle_tables.py
# Prints scalar, quadratic, and ratio tables to the console.
# No external libraries required; will use pandas/tabulate if available for nicer output.

import math
from collections import OrderedDict

# --- Constants (cgs) ---
e = 4.8032068e-10
c = 2.99792458e10
hbar = 1.05457266e-27
alpha = 1.0/137.035999206
thetaW0 = 0.231208
thetaWrad = math.asin(math.sqrt(thetaW0))

# --- g definitions ---
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

# --- compute tables ---
scalar_rows = []
quad_rows = []
ratio_rows = []

for name, sym, mass, tau in particles:
    # scalar
    rx_s = rx_from_tau(tau, K_s)
    mx_s = mx_from_rx(rx_s)
    t0_s = t0_of_x(rx_s, K_s)
    # quadratic
    rx_q = rx_from_tau(tau, K_q)
    mx_q = mx_from_rx(rx_q)
    t0_q = t0_of_x(rx_q, K_q)
    # ratios
    ratio_rx = rx_q / rx_s if (rx_s and rx_s != 0) else float('nan')
    ratio_mx = mx_q / mx_s if (mx_s and mx_s != 0) else float('nan')

    scalar_rows.append(OrderedDict([
        ("Particle", name), ("Symbol", sym), ("Mass_MeV", mass),
        ("Lifetime_s", tau), ("rx_cm", rx_s), ("mx_g", mx_s), ("t0_s", t0_s)
    ]))
    quad_rows.append(OrderedDict([
        ("Particle", name), ("Symbol", sym), ("Mass_MeV", mass),
        ("Lifetime_s", tau), ("rx_cm", rx_q), ("mx_g", mx_q), ("t0_s", t0_q)
    ]))
    ratio_rows.append(OrderedDict([
        ("Particle", name), ("rx_ratio_q_over_s", ratio_rx), ("mx_ratio_q_over_s", ratio_mx)
    ]))

# --- pretty print: try pandas, then tabulate, else fallback to manual formatting ---
def try_print_table(rows, title):
    try:
        import pandas as pd
        print("\n" + title)
        df = pd.DataFrame(rows)
        with pd.option_context('display.float_format', '{:.6e}'.format):
            print(df.to_string(index=False))
        return
    except Exception:
        pass
    try:
        from tabulate import tabulate
        print("\n" + title)
        print(tabulate(rows, headers="keys", floatfmt=".6e"))
        return
    except Exception:
        pass
    # fallback
    print("\n" + title)
    keys = list(rows[0].keys())
    # header
    header = " | ".join(k.ljust(16) for k in keys)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = " | ".join(f"{str(r[k]):16}" if not isinstance(r[k], float) else f"{r[k]:16.6e}" for k in keys)
        print(line)

# --- show K values and tables ---
print(f"K_scalar = {K_s:.6e}, K_quadratic = {K_q:.6e}, ratio K_s/K_q = {K_s/K_q:.6e}")
try_print_table(scalar_rows, "Scalar model table (linear g contributions)")
try_print_table(quad_rows, "Quadratic model table (squared g contributions)")
try_print_table(ratio_rows, "Ratio table (quadratic / scalar) for rx and mx")

# --- optional: interactive lookup ---
def lookup(name):
    for r in scalar_rows:
        if r["Particle"].lower() == name.lower() or r["Symbol"].lower() == name.lower():
            return r
    return None

# Example: print Higgs row from both tables
print("\nExample: Higgs (scalar vs quadratic)")
h_s = next(r for r in scalar_rows if r["Particle"]=="Higgs")
h_q = next(r for r in quad_rows if r["Particle"]=="Higgs")
print("Scalar:", h_s)
print("Quadratic:", h_q)
