# examples/example_power_map_and_placement.py

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------
# Make sure we can import the local "harvester" package
# -------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harvester.models import PiezoParams, simulate_piezo
from harvester.vibration import read_vibration_csv, preprocess_vibration
from harvester.placement import compute_scaling_field
from harvester.optimize import greedy_placement


def main():
    # ---------------------------------------------------------------
    # 1) Load global base vibration (same CSV you used before)
    # ---------------------------------------------------------------
    vib_path = os.path.join(ROOT, "data", "sample_vibration.csv")
    t, a_global = read_vibration_csv(vib_path)

    # Optional filtering: keep 1–50 Hz band
    t, a_global = preprocess_vibration(t, a_global, f_low=1.0, f_high=50.0)

    # ---------------------------------------------------------------
    # 2) Define panel / hull section
    # ---------------------------------------------------------------
    # Example: 10 m by 5 m plate, coarse grid
    Lx, Ly = 10.0, 5.0      # meters
    nx, ny = 20, 10         # grid resolution

    # Mode-shape-like scaling field (phi normalized to [0,1])
    X, Y, S = compute_scaling_field(Lx, Ly, nx, ny, mx=1, my=1)

    # ---------------------------------------------------------------
    # 3) Piezo harvester parameters (same for every location)
    # ---------------------------------------------------------------
    # Tuned to be physically reasonable for ~5 Hz excitation
    params = PiezoParams(
        m=0.02,      # 20 g
        c=0.063,     # ≈ 5% damping
        k=20.0,      # N/m, ~5 Hz natural frequency
        Cp=1e-7,     # 100 nF
        theta=1e-4,  # weaker coupling
        R=1e5,       # 100 kΩ
    )

    # ---------------------------------------------------------------
    # 4) Compute average power at each grid location
    # ---------------------------------------------------------------
    P_avg = np.zeros_like(S)

    ny_, nx_ = S.shape
    for j in range(ny_):
        for i in range(nx_):
            scale = S[j, i]
            if scale <= 0.0:
                P_avg[j, i] = 0.0
                continue

            # Local acceleration = scaled global acceleration
            a_loc = scale * a_global

            res = simulate_piezo(t, a_loc, params)
            P_avg[j, i] = res["P_avg"]

    # ---------------------------------------------------------------
    # 5) Greedy placement of N patches
    # ---------------------------------------------------------------
    N_patches = 5
    chosen = greedy_placement(P_avg, N_patches, min_dist_cells=1)
    print("Chosen patch indices (i, j):", chosen)

    # ---------------------------------------------------------------
    # 6) Plot heatmap + chosen patch positions
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))

    mesh = ax.pcolormesh(X, Y, P_avg, shading="auto")
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("Average Power [W]")

    # Overlay chosen patch centers
    for (i, j) in chosen:
        ax.plot(X[j, i], Y[j, i], "ko", markersize=6)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Harvested Power Map & Greedy Patch Placement")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
