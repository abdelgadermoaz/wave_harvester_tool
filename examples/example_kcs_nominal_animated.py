# examples/example_kcs_nominal_animated.py

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    # ----------------- 1) Frequency grid + waves -----------------
    f = np.arange(0.03, 1.0 + 1e-9, 0.01)  # 0.03:0.01:1.0 Hz
    S_eta = build_wave_spectrum("nominal", f)

    # ----------------- 2) KCS RAOs & motions ---------------------
    raos_csv = os.path.join(ROOT, "data", "kcs_raos_heave_pitch_roll.csv")
    f_raos, H3, H4, H5 = load_kcs_raos(raos_csv)
    H3_i, H4_i, H5_i = interpolate_raos_to_grid(f_raos, H3, H4, H5, f)

    S_z3, S_z4, S_z5 = motion_psds_from_raos(S_eta, H3_i, H4_i, H5_i)

    # ----------------- 3) Base accel PSD -------------------------
    Sa_hull, a_rms = base_accel_psd(f, S_z3, S_z4, S_z5, x=0.0, y=16.1)
    print(f"[Nominal] vertical a_rms at side shell ≈ {a_rms:.5f} m/s^2")

    # ----------------- 4) Panel filtering ------------------------
    Sa_local, f11, Hpan = local_accel_psd(Sa_hull, f)
    print(f"Panel first-mode frequency f11 ≈ {f11:.2f} Hz")

    # ----------------- 5) Time-domain synthesis ------------------
    duration = 60.0  # s
    fs = 200.0       # Hz
    t_sig, a_sig = synthesize_from_psd(f, Sa_local, duration, fs)

    # ----------------- 6) Power map on panel ---------------------
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

    # Pick N patches by greedy algorithm (for plotting)
    N_patches = 5
    chosen = greedy_placement(P_avg, N_patches, min_dist_cells=1)
    print("Chosen patch cells (i, j):", chosen)

    # Also pick the single BEST patch for time animation
    j_best, i_best = np.unravel_index(np.argmax(P_avg), P_avg.shape)
    print(f"Best patch at grid cell (i={i_best}, j={j_best})")

    # Re-simulate piezo at best patch and keep full time history
    scale_best = S_shape[j_best, i_best]
    a_best = scale_best * a_sig
    res_best = simulate_piezo(t_sig, a_best, params)
    v_best = res_best["v"]
    P_best = res_best["P"]

    # ----------------- 7) Build animated figure ------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax_psd, ax_map, ax_acc, ax_volt = axes.ravel()

    # (a) Base acceleration PSD (static)
    ax_psd.plot(f, Sa_hull)
    ax_psd.set_xlabel("f [Hz]")
    ax_psd.set_ylabel("S_a,hull [(m/s²)² / Hz]")
    ax_psd.set_title("Base vertical acceleration PSD (Nominal)")
    ax_psd.grid(True, alpha=0.3)

    # (b) Power map + chosen patches (static)
    mesh = ax_map.pcolormesh(X, Y, P_avg, shading="auto")
    cbar = plt.colorbar(mesh, ax=ax_map)
    cbar.set_label("Average Power [W]")
    for (i, j) in chosen:
        ax_map.plot(X[j, i], Y[j, i], "ko", markersize=5)
    # highlight the best patch with a red marker
    ax_map.plot(X[j_best, i_best], Y[j_best, i_best], "ro", markersize=7)
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.set_title("Power Map & Patch Placement")
    ax_map.grid(True, alpha=0.3)

    # (c) Local acceleration over time (animated)
    ax_acc.set_xlim(0, duration)
    ax_acc.set_ylim(1.1 * np.min(a_sig), 1.1 * np.max(a_sig))
    ax_acc.set_xlabel("t [s]")
    ax_acc.set_ylabel("a_local [m/s²]")
    ax_acc.set_title("Local acceleration at best patch")
    ax_acc.grid(True, alpha=0.3)
    line_acc, = ax_acc.plot([], [], lw=1.5)

    # (d) Voltage over time (animated)
    ax_volt.set_xlim(0, duration)
    ax_volt.set_ylim(1.1 * np.min(v_best), 1.1 * np.max(v_best))
    ax_volt.set_xlabel("t [s]")
    ax_volt.set_ylabel("v [V]")
    ax_volt.set_title("Piezo voltage at best patch")
    ax_volt.grid(True, alpha=0.3)
    line_volt, = ax_volt.plot([], [], lw=1.5)

    plt.tight_layout()

    # ----------------- 8) Animation setup ------------------------
    # To avoid 12k frames, downsample frames for animation
    step = 10  # use every 10th sample
    frame_indices = np.arange(1, len(t_sig), step)

    def init():
        line_acc.set_data([], [])
        line_volt.set_data([], [])
        return line_acc, line_volt

    def update(k):
        i = frame_indices[k]
        t_slice = t_sig[:i]
        a_slice = a_sig[:i]
        v_slice = v_best[:i]

        line_acc.set_data(t_slice, a_slice)
        line_volt.set_data(t_slice, v_slice)
        return line_acc, line_volt

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=40,  # ms between frames (~25 fps)
        blit=True,
    )

    # If you want to save as a video, uncomment this (needs ffmpeg installed):
    # ani.save("kcs_nominal_animation.mp4", fps=25, dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
