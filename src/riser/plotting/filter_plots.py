# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
Filter kernel plots.
"""


# Public API
__all__ = [
    "plot_filter_kernel",
]


# Import modules
from matplotlib.axes import Axes

from ..sampling import filtering


#################### FILTER KERNEL PLOTTING ####################
def plot_filter_kernel(
    ax: Axes, filt: filtering.FIRFilter
) -> None:
    """Plot a filter kernel.

    Parameters
    ----------
    ax : Axes
        Axis on which to plot the filter kernel.
    filt : FIRfilter
        Filter to plot.
    """
    # Plot kernel values
    ax.plot(filt.h, color="k", linewidth=2)


# end of file
