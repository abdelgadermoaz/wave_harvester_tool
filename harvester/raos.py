# harvester/raos.py

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def load_kcs_raos(csv_path: str):
    """
    Load KCS RAOs from CSV.

    Expected CSV columns (header row):
        f_Hz,H_heave_m_per_m,H_pitch_rad_per_m,H_roll_rad_per_m

    - f_Hz:                encounter frequency [Hz]
    - H_heave_m_per_m:     heave RAO [m/m]
    - H_pitch_rad_per_m:   pitch RAO [rad/m]
    - H_roll_rad_per_m:    roll RAO [rad/m]

    Column names are stripped of whitespace, so
    ' H_heave_m_per_m' is also accepted.
    """
    df = pd.read_csv(csv_path)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    required = [
        "f_Hz",
        "H_heave_m_per_m",
        "H_pitch_rad_per_m",
        "H_roll_rad_per_m",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in RAO CSV: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    f = df["f_Hz"].to_numpy(dtype=float)
    H3 = df["H_heave_m_per_m"].to_numpy(dtype=float)   # ζ3  (heave)
    H5 = df["H_pitch_rad_per_m"].to_numpy(dtype=float) # ζ5  (pitch)
    H4 = df["H_roll_rad_per_m"].to_numpy(dtype=float)  # ζ4  (roll)

    return f, H3, H4, H5


def interpolate_raos_to_grid(
    f_src: np.ndarray,
    H3: np.ndarray,
    H4: np.ndarray,
    H5: np.ndarray,
    f_grid: np.ndarray,
):
    """
    Interpolate RAOs from their original frequency points onto
    the analysis grid f_grid.
    """
    kind = "linear"

    f_src = np.asarray(f_src, dtype=float)
    f_grid = np.asarray(f_grid, dtype=float)

    H3_i = interp1d(
        f_src, H3, kind=kind, bounds_error=False, fill_value=0.0
    )(f_grid)
    H4_i = interp1d(
        f_src, H4, kind=kind, bounds_error=False, fill_value=0.0
    )(f_grid)
    H5_i = interp1d(
        f_src, H5, kind=kind, bounds_error=False, fill_value=0.0
    )(f_grid)

    return H3_i, H4_i, H5_i


def motion_psds_from_raos(S_eta: np.ndarray, H3: np.ndarray, H4: np.ndarray, H5: np.ndarray):
    """
    From wave elevation spectrum S_eta(f) to motion PSDs:

        S_xi(f) = |H_xi(f)|^2 * S_eta(f)

    where xi in {3, 4, 5} corresponds to heave, roll, pitch.
    """
    S_eta = np.asarray(S_eta, dtype=float)

    S_z3 = np.abs(H3) ** 2 * S_eta  # heave PSD [m^2/Hz]
    S_z4 = np.abs(H4) ** 2 * S_eta  # roll PSD [rad^2/Hz]
    S_z5 = np.abs(H5) ** 2 * S_eta  # pitch PSD [rad^2/Hz]

    return S_z3, S_z4, S_z5


def base_accel_psd(
    f: np.ndarray,
    S_z3: np.ndarray,
    S_z4: np.ndarray,
    S_z5: np.ndarray,
    x: float = 0.0,
    y: float = 16.1,
):
    """
    Compute vertical base acceleration PSD at a point relative to
    the ship's center of gravity, using:

        S_a,hull(f) = (2πf)^2 [S_ξ3 + x^2 S_ξ5 + y^2 S_ξ4]

    where:
      - S_ξ3 : heave PSD [m^2/Hz]
      - S_ξ4 : roll PSD  [rad^2/Hz]
      - S_ξ5 : pitch PSD [rad^2/Hz]
      - x, y : longitudinal and transverse lever arms [m]
    """
    f = np.asarray(f, dtype=float)
    S_z3 = np.asarray(S_z3, dtype=float)
    S_z4 = np.asarray(S_z4, dtype=float)
    S_z5 = np.asarray(S_z5, dtype=float)

    omega2 = (2 * np.pi * f) ** 2
    Sa = omega2 * (S_z3 + (x**2) * S_z5 + (y**2) * S_z4)

    # RMS over the main wave band 0.05–1.0 Hz
    mask = (f >= 0.05) & (f <= 1.0)
    if np.any(mask):
        Sa_sub = Sa[mask]
        f_sub = f[mask]
        a_rms = float(np.sqrt(np.trapz(Sa_sub, f_sub)))
    else:
        a_rms = 0.0

    return Sa, a_rms
