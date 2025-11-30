# examples/example_kcs_nominal_panel_map.py

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harvester.marine_waves import build_wave_spectrum
from harvester.raos import (
    load_kcs_raos,
    interpolate_raos_to_grid,
    motion_psds_from_raos,
    base_accel_psd,
)
from harvester.panel import local_accel_psd
from harvester.synthesis import synthesize_from_psd
from harvester.models import PiezoParams, simulate_piezo
from harvester.placement import compute_scaling_field
from harvester.optimize import greedy_placement


def main():
    # ----------------- 1) Frequency grid -----------------
    f = np.arange(0.03, 1.0 + 1e-9, 0.01)  # 0.03:0.01:1.0 Hz

    # ----------------- 2) JONSWAP (Nominal) ---------------
    S_eta = build_wave_spectrum("nominal", f)

    # ----------------- 3) KCS RAOs ------------------------
    raos_csv = os.path.join(ROOT, "data", "kcs_raos_heave_pitch_roll.csv")
    f_raos, H3, H4, H5 = load_kcs_raos(raos_csv)
    H3_i, H4_i, H5_i = interpolate_raos_to_grid(f_raos, H3, H4, H5, f)

    # Motion PSDs
    S_z3, S_z4, S_z5 = motion_psds_from_raos(S_eta, H3_i, H4_i, H5_i)

    # ----------------- 4) Base accel PSD ------------------
    Sa_hull, a_rms = base_accel_psd(f, S_z3, S_z4, S_z5, x=0.0, y=16.1)
    print(f"[Nominal] vertical a_rms at side shell ≈ {a_rms:.5f} m/s^2")

    # ----------------- 5) Panel filtering -----------------
    Sa_local, f11, Hpan = local_accel_psd(Sa_hull, f)
    print(f"Panel first-mode frequency f11 ≈ {f11:.2f} Hz")

    # ----------------- 6) Time-domain synthesis -----------
    duration = 60.0  # s
    fs = 200.0       # Hz
    t_sig, a_sig = synthesize_from_psd(f, Sa_local, duration, fs)

    # ----------------- 7) Power map on panel --------------
    Lx, Ly = 10.0, 5.0
    nx, ny = 20, 10
    X, Y, S_shape = compute_scaling_field(Lx, Ly, nx, ny, mx=1, my=1)

    params = PiezoParams(
        m=0.02,
        c=0.063,
        k=20.0,
        Cp=1e-7,
        theta=1e-4,
        R=1e5,
    )

    P_avg = np.zeros_like(S_shape)
    ny_, nx_ = S_shape.shape

    for j in range(ny_):
        for i in range(nx_):
            scale = S_shape[j, i]
            if scale <= 0.0:
                P_avg[j, i] = 0.0
                continue

            a_loc = scale * a_sig
            res = simulate_piezo(t_sig, a_loc, params)
            P_avg[j, i] = res["P_avg"]

    N_patches = 5
    chosen = greedy_placement(P_avg, N_patches, min_dist_cells=1)
    print("Chosen patch cells (i, j):", chosen)

    # ----------------- 8) Plots ---------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax0, ax1, ax2 = axes

    # Base acceleration PSD
    ax0.plot(f, Sa_hull)
    ax0.set_xlabel("f [Hz]")
    ax0.set_ylabel("S_a,hull [(m/s^2)^2 / Hz]")
    ax0.set_title("Base vertical acceleration PSD (Nominal)")
    ax0.grid(True, alpha=0.3)

    # Power map + patches
    mesh = ax1.pcolormesh(X, Y, P_avg, shading="auto")
    cbar = plt.colorbar(mesh, ax=ax1)
    cbar.set_label("Average Power [W]")
    for (i, j) in chosen:
        ax1.plot(X[j, i], Y[j, i], "ko", markersize=5)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Power Map & Greedy Patch Placement")
    ax1.grid(True, alpha=0.3)

    # Local time signal
    ax2.plot(t_sig, a_sig)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("a_local [m/s^2]")
    ax2.set_title("Synthesized local acceleration")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
