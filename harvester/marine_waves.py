# harvester/marine_waves.py

import numpy as np

JONSWAP_GAMMA = 3.3

SCENARIOS = {
    "calm": {
        "Hs": 0.414,
        "Tp": 5.841,
    },
    "nominal": {
        "Hs": 0.729,
        "Tp": 5.395,
    },
    "rough": {
        "Hs": 1.849,
        "Tp": 6.561,
    },
}

def jonswap_spectrum(f, Hs, Tp, gamma=JONSWAP_GAMMA):
    """
    JONSWAP spectrum S_eta(f) using the classic parametrization,
    then rescaled so that m0 = Hs^2 / 16 as in your methodology.
    f: array of frequencies [Hz]
    returns S_eta [m^2 / Hz]
    """
    f = np.asarray(f)
    fp = 1.0 / Tp

    sigma = np.where(f <= fp, 0.07, 0.09)

    alpha = 0.076 * (Hs**2 * fp**4) ** 0.22  # initial guess

    r = f / fp
    S = (
        alpha
        * Hs**2
        * Tp
        * r ** -5
        * np.exp(-1.25 * r ** -4)
        * gamma ** np.exp(-0.5 * ((f - fp) ** 2) / (sigma**2 * fp**2))
    )

    # enforce m0 = Hs^2 / 16 numerically
    df = f[1] - f[0]
    m0 = np.trapz(S, f)
    target_m0 = Hs**2 / 16.0
    if m0 > 0:
        S *= target_m0 / m0

    return S


def build_wave_spectrum(scenario, f_grid):
    """
    scenario: "calm", "nominal", or "rough"
    f_grid: frequency array (e.g. 0.03:0.01:1.0 Hz, as in thesis)
    returns S_eta(f) for that scenario
    """
    sc = SCENARIOS[scenario.lower()]
    return jonswap_spectrum(f_grid, sc["Hs"], sc["Tp"])
