#!/usr/bin/env python3
"""
plot_eigen_sums.py

Load spectral_eigenvalues_table.csv, sum the first 8 eigenvalues per model/scale,
save eigen_sums.csv, and plot sums vs sc0 (eigen_sum_plot.png).
"""

import csv
import os
import math
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = "spectral_eigenvalues_table.csv"   # produced by spectral_graph_eigs.py
OUTPUT_CSV = "eigen_sums.csv"
OUTPUT_PNG = "eigen_sum_plot.png"

# ---------------------------
# Load CSV
# ---------------------------
if not os.path.exists(INPUT_CSV):
    raise SystemExit(f"Input file not found: {INPUT_CSV}\nRun the spectral eigenvalue sweep first or place the CSV in this directory.")

rows = []
with open(INPUT_CSV, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    # Expect header like: sc0,model,eig1,...,eig8
    # find indices
    try:
        sc0_idx = header.index('sc0')
        model_idx = header.index('model')
    except ValueError:
        # fallback to first two columns
        sc0_idx = 0
        model_idx = 1
    eig_indices = []
    for i, h in enumerate(header):
        if h.lower().startswith('eig'):
            eig_indices.append(i)
    if len(eig_indices) < 8:
        raise SystemExit("Input CSV does not contain 8 eigenvalue columns named eig1..eig8.")

    for r in reader:
        sc0 = float(r[sc0_idx])
        model = r[model_idx]
        eigs = []
        for idx in eig_indices[:8]:
            val = r[idx]
            try:
                eigs.append(float(val))
            except:
                eigs.append(float('nan'))
        rows.append({'sc0': sc0, 'model': model, 'eigs': eigs})

# ---------------------------
# Aggregate sums
# ---------------------------
# Organize by model, then sort by sc0
models = {}
for r in rows:
    models.setdefault(r['model'], []).append(r)

for model in models:
    models[model] = sorted(models[model], key=lambda x: x['sc0'])

# Compute sums and write CSV
out_rows = []
for model, recs in models.items():
    for rec in recs:
        sc0 = rec['sc0']
        eigs = np.array(rec['eigs'], dtype=float)
        # treat NaN as large or skip; here we sum finite values only
        finite = eigs[np.isfinite(eigs)]
        sum8 = float(np.sum(finite)) if finite.size > 0 else float('nan')
        out_rows.append({'sc0': sc0, 'model': model, 'sum8': sum8})

# Save eigen_sums.csv
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['sc0', 'model', 'sum8'])
    for r in out_rows:
        writer.writerow([r['sc0'], r['model'], r['sum8']])

print(f"Saved summed eigenvalues to {OUTPUT_CSV}")

# ---------------------------
# Print table to stdout
# ---------------------------
print("\nsc0\tmodel\tsum8")
for r in out_rows:
    print(f"{r['sc0']}\t{r['model']}\t{r['sum8']:.6g}")

# ---------------------------
# Plot sums vs sc0
# ---------------------------
plt.figure(figsize=(9,6))
markers = {'A_dodeca_poincare':'o', 'B_D6_weeks':'s', 'C_seifert_weber':'^'}
colors = {'A_dodeca_poincare':'C0', 'B_D6_weeks':'C1', 'C_seifert_weber':'C2'}

global_peak = {'model': None, 'sc0': None, 'sum8': -math.inf}

for model, recs in models.items():
    scs = [r['sc0'] for r in recs]
    sums = []
    for r in recs:
        eigs = np.array(r['eigs'], dtype=float)
        finite = eigs[np.isfinite(eigs)]
        sums.append(float(np.sum(finite)) if finite.size>0 else float('nan'))
    sums = np.array(sums)
    plt.plot(scs, sums, marker=markers.get(model,'o'), color=colors.get(model,'k'),
             linestyle='-', label=model)
    # model peak
    if np.any(np.isfinite(sums)):
        idx = int(np.nanargmax(sums))
        peak_sc0 = scs[idx]
        peak_sum = float(sums[idx])
        plt.scatter([peak_sc0], [peak_sum], s=120, facecolors='none', edgecolors=colors.get(model,'k'), linewidths=2)
        plt.text(peak_sc0, peak_sum, f"  peak {model}\n  sc0={peak_sc0}, sum={peak_sum:.3g}", fontsize=8, va='bottom')
        if peak_sum > global_peak['sum8']:
            global_peak = {'model': model, 'sc0': peak_sc0, 'sum8': peak_sum}

plt.xlabel('sc0')
plt.ylabel('Sum of first 8 eigenvalues')
plt.title('Sum of first 8 approximate eigenvalues vs scale')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()
print(f"Plot saved: {OUTPUT_PNG}")

# ---------------------------
# Print peak summary
# ---------------------------
if global_peak['model'] is not None:
    print("\nGlobal peak across all models:")
    print(f" Model: {global_peak['model']}")
    print(f" sc0:  {global_peak['sc0']}")
    print(f" sum8: {global_peak['sum8']:.6g}")
else:
    print("\nNo finite sums found to determine a peak.")
