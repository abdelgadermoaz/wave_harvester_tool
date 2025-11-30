# harvester/placement.py

from typing import Tuple

import numpy as np


def generate_grid(Lx: float, Ly: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D grid over a rectangular panel.

    Parameters
    ----------
    Lx : float
        Panel length in x-direction (m).
    Ly : float
        Panel length in y-direction (m).
    nx : int
        Number of grid points along x.
    ny : int
        Number of grid points along y.

    Returns
    -------
    (X, Y) : (np.ndarray, np.ndarray)
        Meshgrid arrays of shape (ny, nx).
    """
    xs = np.linspace(0.0, Lx, nx)
    ys = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return X, Y


def mode_shape_phi(
    X: np.ndarray,
    Y: np.ndarray,
    Lx: float,
    Ly: float,
    mx: int = 1,
    my: int = 1,
) -> np.ndarray:
    """
    Simple sinusoidal plate mode shape approximation:

        phi(x, y) = sin(mx * pi * x / Lx) * sin(my * pi * y / Ly)

    Parameters
    ----------
    X, Y : np.ndarray
        Meshgrid coordinates.
    Lx, Ly : float
        Plate dimensions.
    mx, my : int
        Mode indices.

    Returns
    -------
    phi : np.ndarray
        Mode shape values (same shape as X).
    """
    return np.sin(mx * np.pi * X / Lx) * np.sin(my * np.pi * Y / Ly)


def compute_scaling_field(
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    mx: int = 1,
    my: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a normalized scaling field over the panel, representing
    relative vibration intensity (e.g. from mode shape magnitude).

    Parameters
    ----------
    Lx, Ly : float
        Panel dimensions.
    nx, ny : int
        Grid resolution.
    mx, my : int
        Mode indices.

    Returns
    -------
    (X, Y, S) : (np.ndarray, np.ndarray, np.ndarray)
        X, Y grid and normalized scaling S in [0, 1].
    """
    X, Y = generate_grid(Lx, Ly, nx, ny)
    Phi = np.abs(mode_shape_phi(X, Y, Lx, Ly, mx=mx, my=my))
    max_val = Phi.max()
    if max_val <= 0:
        S = Phi
    else:
        S = Phi / max_val
    return X, Y, S
