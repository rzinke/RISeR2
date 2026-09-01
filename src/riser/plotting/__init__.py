# src/riser/plotting/__init__.py
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
Plotting functions.
"""

# Import modules
from . import pdf_plots
from . import cdf_plots
from . import variable_pair_plots
from . import filter_plots
from . import mc_plots

# Public API
__all__ = (
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
