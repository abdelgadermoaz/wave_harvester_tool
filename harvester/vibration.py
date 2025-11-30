# harvester/vibration.py

from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import detrend, butter, filtfilt


def read_vibration_csv(path: str,
                       t_col: str = "t",
                       a_col: str = "a_hull") -> Tuple[np.ndarray, np.ndarray]:
    """
    Read vibration time series from CSV.

    CSV format example:
        t, a_hull
        0.00, 0.12
        0.01, 0.15
        ...

    Parameters
    ----------
    path : str
        Path to CSV file.
    t_col : str
        Column name for time.
    a_col : str
        Column name for acceleration.

    Returns
    -------
    (t, a) : (np.ndarray, np.ndarray)
    """
    df = pd.read_csv(path)
    if t_col not in df.columns or a_col not in df.columns:
        raise ValueError(f"CSV must contain '{t_col}' and '{a_col}' columns")

    t = df[t_col].to_numpy(dtype=float)
    a = df[a_col].to_numpy(dtype=float)
    return t, a


def _design_band_filter(fs: float,
                        f_low: Optional[float],
                        f_high: Optional[float],
                        order: int = 4):
    """
    Design a Butterworth filter for optional bandpass/low/high pass.
    """
    nyq = 0.5 * fs
    if f_low is not None and f_high is not None:
        low = f_low / nyq
        high = f_high / nyq
        b, a = butter(order, [low, high], btype="band")
    elif f_low is not None:
        low = f_low / nyq
        b, a = butter(order, low, btype="high")
    elif f_high is not None:
        high = f_high / nyq
        b, a = butter(order, high, btype="low")
    else:
        b, a = None, None
    return b, a


def preprocess_vibration(
    t: np.ndarray,
    a: np.ndarray,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
    detrend_flag: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basic preprocessing for vibration data.

    Steps:
    - ensure uniform sampling (assumes nearly uniform t)
    - optional detrend
    - optional bandpass / high / low pass

    Parameters
    ----------
    t : np.ndarray
        Time array.
    a : np.ndarray
        Acceleration array.
    f_low : float, optional
        Lower cutoff frequency (Hz).
    f_high : float, optional
        Upper cutoff frequency (Hz).
    detrend_flag : bool
        If True, remove linear trend.

    Returns
    -------
    (t_out, a_out) : (np.ndarray, np.ndarray)
    """
    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=float)
    if t.shape != a.shape:
        raise ValueError("t and a must have the same shape")

    # Detrend
    if detrend_flag:
        a = detrend(a, type="linear")

    dt = np.mean(np.diff(t))
    fs = 1.0 / dt

    # Optional filtering
    b, filt_a = _design_band_filter(fs, f_low, f_high)
    if b is not None:
        a = filtfilt(b, filt_a, a)

    return t, a
