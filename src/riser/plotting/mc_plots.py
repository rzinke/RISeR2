# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

"""
Monte Carlo sampling plots.
"""


# Public API
__all__ = [
    "plot_mc_picks",
]


# Import modules
import numpy as np


#################### SAMPLE PLOTTING ####################
def plot_mc_picks(
    ax,
    age_picks: np.ndarray,
    disp_picks: np.ndarray,
    max_picks: int = 500,
) -> None:
    """Plot valid displacement-age picks.
    """
    # Plot lines connecting points
    ax.plot(
        age_picks[:, :max_picks],
        disp_picks[:, :max_picks],
        color="k",
        alpha=0.1,
        zorder=1,
    )

    # Plot pick values
    ax.scatter(
        age_picks[:, :max_picks],
        disp_picks[:, :max_picks],
        s=2**2,
        color="b",
        alpha=0.1,
        zorder=2,
    )


# end of file
