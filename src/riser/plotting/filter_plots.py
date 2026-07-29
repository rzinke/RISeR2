# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

"""
Filter kernel plots.
"""


# Public API
__all__ = [
    "plot_filter_kernel",
]


# Import modules
from ..sampling import filtering


#################### FILTER KERNEL PLOTTING ####################
def plot_filter_kernel(ax, filt: filtering.FIRFilter) -> None:
    """Plot a filter kernel.

    Parameters
    ----------
    ax
        Axis on which to plot the filter kernel.
    filt : FIRfilter
        Filter to plot.
    """
    # Plot kernel values
    ax.plot(filt.h, color="k", linewidth=2)


# end of file
