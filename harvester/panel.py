# harvester/panel.py

import numpy as np

def panel_first_mode_frequency(a=0.8, b=0.9, t=0.016,
                               E=210e9, nu=0.30, rho=7850.0):
    """
    Compute f11 for a simply-supported rectangular bay (1,1 mode).
    """
    D = E * t**3 / (12 * (1 - nu**2))
    m_areal = rho * t

    term = (1.0 / a) ** 2 + (1.0 / b) ** 2
    omega11 = (np.pi**2) * np.sqrt(D / m_areal) * term
    f11 = omega11 / (2 * np.pi)
    return f11


def panel_transmissibility(f, f11, zeta_p=0.03):
    """
    |H_panel(f)| displacement transmissibility under base excitation
    as in your Eq. for SDOF panel. :contentReference[oaicite:8]{index=8}
    """
    r = f / f11
    num = 1 + (2 * zeta_p * r) ** 2
    den = (1 - r**2) ** 2 + (2 * zeta_p * r) ** 2
    return np.sqrt(num / den)


def local_accel_psd(Sa_hull, f, a=0.8, b=0.9, t=0.016,
                    E=210e9, nu=0.30, rho=7850.0, zeta_p=0.03):
    """
    Sa_local(f) = |H_panel(f)|^2 * Sa_hull(f)
    """
    f11 = panel_first_mode_frequency(a, b, t, E, nu, rho)
    Hpan = panel_transmissibility(f, f11, zeta_p)
    Sa_local = (np.abs(Hpan) ** 2) * Sa_hull
    return Sa_local, f11, Hpan
