# harvester/models.py

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class PiezoParams:
    """
    Parameters for a single piezoelectric harvester modeled
    as a base-excited SDOF with electrical coupling.
    """
    m: float      # kg, effective mass
    c: float      # N·s/m, mechanical damping
    k: float      # N/m, stiffness
    Cp: float     # F, piezo capacitance
    theta: float  # N/V or C/m, electromechanical coupling
    R: float      # Ohms, load resistance


def simulate_piezo(
    t: np.ndarray,
    a_base: np.ndarray,
    params: PiezoParams,
    x0: Optional[list] = None
) -> Dict[str, Any]:
    """
    Simulate a piezoelectric energy harvester subjected to base acceleration.

    State vector: x = [z, zdot, v]
    where:
        z    = relative displacement (mass w.r.t base)
        zdot = relative velocity
        v    = output voltage across R

    Equations:
        m * zdd + c * zdot + k * z + theta * v = -m * ydd
        Cp * vdot + v / R + theta * zdot = 0

    Parameters
    ----------
    t : np.ndarray
        Time array (monotonic, 1D).
    a_base : np.ndarray
        Base acceleration array ydd(t), same length as t [m/s^2].
    params : PiezoParams
        Harvester parameters.
    x0 : list, optional
        Initial state [z0, zdot0, v0]. Defaults to zeros.

    Returns
    -------
    dict
        Keys: "t", "z", "zdot", "v", "P", "P_avg"
    """
    t = np.asarray(t)
    a_base = np.asarray(a_base)

    if t.shape != a_base.shape:
        raise ValueError("t and a_base must have the same shape")

    m = params.m
    c = params.c
    k = params.k
    Cp = params.Cp
    th = params.theta
    R = params.R

    # Interpolated base acceleration
    a_func = lambda tau: np.interp(tau, t, a_base)

    def ode(tau, x):
        z, zdot, v = x
        ydd = a_func(tau)
        zdd = (-c * zdot - k * z - th * v - m * ydd) / m
        vdot = (-v / R - th * zdot) / Cp
        return [zdot, zdd, vdot]

    if x0 is None:
        x0 = [0.0, 0.0, 0.0]

    sol = solve_ivp(
        ode,
        (float(t[0]), float(t[-1])),
        x0,
        t_eval=t,
        method="RK45",
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    z = sol.y[0, :]
    zdot = sol.y[1, :]
    v = sol.y[2, :]

    # Instantaneous power
    P = (v ** 2) / R

    # Average power
    duration = float(t[-1] - t[0])
    if duration <= 0:
        raise ValueError("Time array duration must be > 0")
    P_avg = float(np.trapz(P, t) / duration)

    return {
        "t": t,
        "z": z,
        "zdot": zdot,
        "v": v,
        "P": P,
        "P_avg": P_avg,
    }
