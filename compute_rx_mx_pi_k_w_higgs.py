# compute_rx_mx_pi_k_w_higgs.py
# Python 3.8+. No scipy required.
import math

# --- Constants (cgs where appropriate) ---
e = 4.8032068e-10       # statcoulomb (esu)
c = 2.99792458e10       # cm/s
hbar = 1.05457266e-27   # erg*s (cgs)
G = 6.67259e-8          # cgs
alpha = 1.0/137.035999206

# thetaW0 as you used (sin^2 theta_W)
thetaW0 = 0.231208
# convert to angle as in your Mathematica: thetaWrad = ArcSin[Sqrt[thetaW0]]
thetaWrad = math.asin(math.sqrt(thetaW0))

# g2, g3 consistent with your notebook
g2 = math.cos(thetaWrad)**2 / (e**2 * c**2)
g3 = math.sin(thetaWrad)**2 / (e**2 * c**2)
ga = alpha
# g4 from ga*g2*g3*g4 == alpha^4
g4 = alpha**4 / (ga * g2 * g3)
g5 = G / 24.0

# K numerator and K (matching your t0 expression)
K_numer = (ga/3.0) + (2.0/3.0)*g4 + (1.0/3.0)*g2 + (2.0/3.0)*g3 - g5
K = K_numer / hbar

# Analytical inversion: x = 1/(c * tau * K)
def rx_from_tau(tau):
    if tau <= 0 or not math.isfinite(tau):
        raise ValueError("tau must be positive finite")
    return 1.0 / (c * tau * K)

# mx from 4*e^2/(mx*c^2) == rx  => mx = 4*e^2/(rx*c^2)
def mx_from_rx(rx):
    if rx <= 0 or not math.isfinite(rx):
        raise ValueError("rx must be positive finite")
    return 4.0 * e**2 / (rx * c**2)

# t0 for verification: t0(x) = 1/(c*x*K)
def t0_of_x(x):
    denom = c * x * K
    return 1.0/denom if denom != 0 else float('inf')

# --- Lifetimes you provided (omit electron and proton) ---
lifetimes = [
    ("Pi", 2.6e-8),        # pi meson
    ("K", 1.2380e-8),      # K meson
    ("W", 3.0e-25),        # W boson
    ("Higgs", 1.56e-22)    # Higgs boson
]

# --- Compute and print results ---
print("{:<8s} {:>14s} {:>18s} {:>18s} {:>18s}".format(
    "Particle", "Lifetime(s)", "rx (cm)", "mx (g)", "t0_check(s)"))
print("-"*90)
for name, tau in lifetimes:
    rx = rx_from_tau(tau)
    mx = mx_from_rx(rx)
    t0check = t0_of_x(rx)
    print("{:<8s} {:14.6e} {:18.6e} {:18.6e} {:18.6e}".format(
        name, tau, rx, mx, t0check))
