# harvester/synthesis.py

import numpy as np

def synthesize_from_psd(f, S_psd, duration, fs, random_state=None):
    """
    Generate a real-valued time series x(t) whose one-sided PSD
    approximately matches S_psd(f) using random phases + IFFT.

    f: frequency array [Hz] (one-sided, starting > 0)
    S_psd: same length, one-sided PSD
    duration: seconds
    fs: sampling frequency [Hz]
    """
    rng = np.random.default_rng(random_state)

    n = int(duration * fs)
    if n % 2 != 0:
        n += 1

    df = fs / n
    # frequency bins for FFT: 0..fs/2 (one-sided)
    f_fft = np.fft.rfftfreq(n, d=1/fs)

    # interpolate PSD onto FFT grid
    S_interp = np.interp(f_fft, f, S_psd, left=0.0, right=0.0)

    # amplitude from PSD: S(f) ≈ (2 / (fs * n)) |X(f)|^2
    # => |X(f)| = sqrt(0.5 * S(f) * fs * n)
    A = np.sqrt(0.5 * S_interp * fs * n)

    phases = rng.uniform(0, 2 * np.pi, size=len(f_fft))
    X = A * np.exp(1j * phases)
    X[0] = 0.0  # remove DC

    # IFFT to time series
    x = np.fft.irfft(X, n=n)
    t = np.arange(n) / fs
    return t, x
