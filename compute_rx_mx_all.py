# compute_rx_mx_all.py
# Python 3.8+. No external libraries required.

import math

# --- Physical constants (cgs where appropriate) ---
e = 4.8032068e-10       # statcoulomb (esu)
c = 2.99792458e10       # cm/s
hbar = 1.05457266e-27   # erg*s (cgs)
G = 6.67259e-8          # cgs
alpha = 1.0/137.035999206

# --- Electroweak parameters (as used in your notebook) ---
thetaW0 = 0.231208                      # sin^2(theta_W) as in your notebook
thetaWrad = math.asin(math.sqrt(thetaW0))
ga = alpha
g2 = math.cos(thetaWrad)**2 / (e**2 * c**2)
g3 = math.sin(thetaWrad)**2 / (e**2 * c**2)
g4 = alpha**4 / (ga * g2 * g3)          # from ga*g2*g3*g4 == alpha^4
g5 = G / 24.0

# K factor in t0(x) = 1/(c*x*K)
K_numer = (ga/3.0) + (2.0/3.0)*g4 + (1.0/3.0)*g2 + (2.0/3.0)*g3 - g5
K = K_numer / hbar

def rx_from_tau(tau):
    """Analytic inversion x = 1/(c * tau * K)."""
    if tau <= 0 or not math.isfinite(tau):
        raise ValueError("tau must be positive finite")
    return 1.0 / (c * tau * K)

def mx_from_rx(rx):
    """mx from 4*e^2/(mx*c^2) == rx  => mx = 4*e^2/(rx*c^2)."""
    if rx <= 0 or not math.isfinite(rx):
        raise ValueError("rx must be positive finite")
    return 4.0 * e**2 / (rx * c**2)

def t0_of_x(x):
    denom = c * x * K
    return 1.0/denom if denom != 0 else float('inf')

# --- Lifetimes to process (omit electron and proton) ---
# Values are best-mean approximations (seconds). Add or edit as you wish.
lifetimes = [
    ("Muon", 2.1969811e-6),
    ("Tau", 2.903e-13),
    ("Pi0", 8.52e-17),
    ("PiPlusMinus", 2.6033e-8),
    ("KPlusMinus", 1.2380e-8),
    ("K0S", 0.8953e-10),
    ("K0L", 5.116e-8),
    ("Neutron", 880.2),
    ("W", 3.0e-25),
    ("Z", 3.0e-25),
    ("Higgs", 1.56e-22),
    ("Top", 5.0e-25),
    ("D0", 4.101e-13),
    ("DPlus", 1.040e-12),
    ("Ds", 5.00e-13),
    ("Bplus", 1.638e-12),
    ("B0", 1.517e-12),
    ("Bs", 1.520e-12),
    ("Bc", 0.510e-12),
    ("Lambda_b", 1.470e-12)
]

# --- Compute and print results ---
print("{:<12s} {:>14s} {:>18s} {:>18s} {:>18s}".format(
    "Particle", "Lifetime(s)", "rx (cm)", "mx (g)", "t0_check(s)"))
print("-"*92)
for name, tau in lifetimes:
    try:
        rx = rx_from_tau(tau)
        mx = mx_from_rx(rx)
        t0check = t0_of_x(rx)
        print("{:<12s} {:14.6e} {:18.6e} {:18.6e} {:18.6e}".format(
            name, tau, rx, mx, t0check))
    except Exception as exc:
        print("{:<12s} {:14} {:>18s} {:>18s} {:>18s}".format(
            name, str(tau), "FAILED", "FAILED", "FAILED"))
