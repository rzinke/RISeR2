# src/riser/plotting/__init__.py
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
Plotting functions.
"""

# Import modules
from . import (  # noqa: I001
    cdf_plots,
    filter_plots,
    mc_plots,
    pdf_plots,
    variable_pair_plots,
)

from .cdf_plots import *
from .filter_plots import *
from .mc_plots import *
from .pdf_plots import *
from .variable_pair_plots import *


# Public API
__all__ = (
    # Submodules
    "cdf_plots",
    "filter_plots",
    "mc_plots",
    "pdf_plots",
    "variable_pair_plots",
    # PDF plots
    "axis_label_from_pdf",
    "axis_label_from_pdfs",
    "plot_pdf_line",
    "plot_pdf_filled",
    "plot_pdf_labeled",
    "plot_pdf_confidence_range",
    "plot_pdf_stack",
    # CDF plots
    "plot_cdf_line",
    "plot_cdf_filled",
    "plot_cdf_labeled",
    # Variable pair plots
    "plot_variable_pair_whisker",
    "plot_variable_pairs_whisker",
    "plot_variable_pair_rectangle",
    "plot_variable_pairs_rectangle",
    "plot_variable_pairs_joint_pdf",
    "get_markers_plot",
    "plot_variable_pairs",
    # Monte Carlo plots
    "plot_mc_picks",
    # Filter plots
    "plot_filter_kernel",
)


# end of file
