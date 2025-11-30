# harvester/optimize.py

from typing import List, Tuple

import numpy as np


def greedy_placement(
    P_avg: np.ndarray,
    N_patches: int,
    min_dist_cells: int = 1,
) -> List[Tuple[int, int]]:
    """
    Greedy patch placement: choose grid cells with highest average power,
    enforcing a minimum spacing in grid cells.

    Parameters
    ----------
    P_avg : np.ndarray
        2D array (ny, nx) of average power for a patch centered at each grid node.
    N_patches : int
        Number of patches to place.
    min_dist_cells : int
        Minimum Manhattan-like spacing in grid index units.

    Returns
    -------
    chosen : list of (i, j)
        List of chosen indices: (i, j) where i is x-index, j is y-index.
    """
    ny, nx = P_avg.shape
    chosen: List[Tuple[int, int]] = []

    # Flatten indices sorted by descending power
    flat_indices = np.argsort(P_avg.ravel())[::-1]

    def is_far_enough(i: int, j: int) -> bool:
        for (pi, pj) in chosen:
            if abs(pi - i) <= min_dist_cells and abs(pj - j) <= min_dist_cells:
                return False
        return True

    for idx in flat_indices:
        if len(chosen) >= N_patches:
            break
        j = idx // nx
        i = idx % nx
        if is_far_enough(i, j):
            chosen.append((i, j))

    return chosen
