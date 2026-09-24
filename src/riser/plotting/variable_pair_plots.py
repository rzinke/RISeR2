# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
Functions for plotting variable pairs.
"""


# Public API
__all__ = [
    "plot_variable_pair_whisker",
    "plot_variable_pairs_whisker",
    "plot_variable_pair_rectangle",
    "plot_variable_pairs_rectangle",
    "plot_variable_pairs_joint_pdf",
    "get_markers_plot",
    "plot_variable_pairs",
]


# Import modules
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from .. import (
    constants,
    probability_functions as PDFs,
    variable_pairs,
)
from .pdf_plots import axis_label_from_pdf, axis_label_from_pdfs


#################### WARNINGS ####################
nameless_label_warning = (
    "`label` flag was passed, but no name was given to variable pair"
)


#################### VARIABLE PAIR PLOTTING ####################
def set_origin_zero(ax: Axes) -> None:
    """Set the plot origin at zero.

    Parameters
    ----------
    ax
        Axes to set at zero.
    """
    ax.set_xlim((0, ax.get_xlim()[1]))
    ax.set_ylim((0, ax.get_ylim()[1]))


def format_marker_plot(
    ax: Axes,
    markers: (
        variable_pairs.VariablePair | Mapping[str, variable_pairs.VariablePair]
    ),
) -> None:
    """Add axis labels, formulated in the standardized manner.

    Parameters
    ----------
    ax
        Axis on which to plot the variable pair.
    markers : VariablePair or Mapping[str, VariablePair]
        Variable pair to plot.
    """
    if isinstance(markers, variable_pairs.VariablePair):
        # Axis labels based on single marker
        pdf1_label = axis_label_from_pdf(markers.pdf1)
        pdf2_label = axis_label_from_pdf(markers.pdf2)

    elif isinstance(markers, Mapping):
        # Axis labels based on multiple markers
        pdf1_label = axis_label_from_pdfs(
            [marker.pdf1 for marker in markers.values()]
        )
        pdf2_label = axis_label_from_pdfs(
            [marker.pdf2 for marker in markers.values()]
        )

    else:
        raise TypeError(
            f"Markers must be passed as a single VariablePair "
            f"or dictionary of VariablePairs, got {type(markers).__name__}"
        )

    # Label axes
    ax.set_xlabel(pdf1_label)
    ax.set_ylabel(pdf2_label)


def plot_variable_pair_whisker(
    ax: Axes,
    marker: variable_pairs.VariablePair,
    confidence: float = constants.Psigma["2"],
    *,
    # Style args
    color: str = "royalblue",
    zorder: int = 1,
    label: bool = False,
) -> None:
    """Plot a variable pair as a cross.

    Parameters
    ----------
    ax
        Axis on which to plot the variable pair.
    marker : VariablePair
        Variable pair to plot.
    confidence : float
        Confidence range to plot.
    color : str
        Marker color.
    zorder : int
        Order in which to plot the variable pairs relative to other items.
    label : bool
        Label the variable pairs.
    """
    # Define function that determines the central locaction of a data point
    pdf_center = PDFs.analytics.pdf_mean

    # Compute confidence limits for x-variable
    pdf1_center = pdf_center(marker.pdf1)
    pdf1_range = PDFs.analytics.compute_interquantile_range(marker.pdf1, confidence)

    # Plot pdf1 values (first and only cluster range)
    pdf1_vals = pdf1_range.range_values[0]
    pdf1_err = [[pdf1_center - pdf1_vals[0]], [pdf1_vals[1] - pdf1_center]]

    # Compute pdf2 confidence limits
    pdf2_center = pdf_center(marker.pdf2)
    pdf2_range = PDFs.analytics.compute_interquantile_range(marker.pdf2, confidence)

    # Plot y values (first and only cluster range)
    pdf2_vals = pdf2_range.range_values[0]
    pdf2_err = [[pdf2_center - pdf2_vals[0]], [pdf2_vals[1] - pdf2_center]]

    # Plot marker
    ax.errorbar(
        pdf1_center,
        pdf2_center,
        xerr=pdf1_err,
        yerr=pdf2_err,
        color=color,
        zorder=zorder,
    )

    # Label if requested
    if label:
        if marker.name is not None:
            ax.text(
                1.01 * pdf1_center, 1.01 * pdf2_center, marker.name, color=color
            )
        else:
            warnings.warn(nameless_label_warning)


def plot_variable_pairs_whisker(
    ax: Axes,
    markers: Mapping[str, variable_pairs.VariablePair],
    confidence: float = constants.Psigma["2"],
    *,
    # Style args
    color: str = "royalblue",
    zorder: int = 1,
    label: bool = False,
) -> None:
    """Plot a variable pair as a cross.

    Parameters
    ----------
    ax
        Axis on which to plot the variable pair.
    markers : dict[str, VariablePair]
        Variable pairs to plot.
    confidence : float
        Confidence range to plot.
    color : str
        Marker color.
    zorder : int
        Order in which to plot the variable pairs relative to other items.
    label : bool
        Label the variable pairs.
    """
    for marker in markers.values():
        plot_variable_pair_whisker(
            ax=ax,
            marker=marker,
            color=color,
            zorder=zorder,
            label=label,
        )


def plot_variable_pair_rectangle(
    ax: Axes,
    marker: variable_pairs.VariablePair,
    confidence: float = constants.Psigma["2"],
    *,
    # Style args
    color: str = "royalblue",
    zorder: int = 1,
    label: bool = False,
) -> None:
    """Plot a variable pair as a rectangle.

    Parameters
    ----------
    ax
        Axis on which to plot the variable pair.
    marker : VariablePair
        Variable pair to plot.
    confidence : float
        Confidence range to plot.
    color : str
        Marker color.
    zorder : int
        Order in which to plot the variable pairs relative to other items.
    label : bool
        Label the variable pairs.
    """
    # Compute x confidence limits
    pdf1_range = PDFs.analytics.compute_interquantile_range(marker.pdf1, confidence)

    # Plot x values (first and only cluster range)
    pdf1_vals = pdf1_range.range_values[0]
    box_pdf1 = pdf1_vals[0]
    box_width = pdf1_vals[1] - box_pdf1

    # Compute y confidence limits
    pdf2_range = PDFs.analytics.compute_interquantile_range(marker.pdf2, confidence)

    # Plot pdf2 values (first and only cluster range)
    pdf2_vals = pdf2_range.range_values[0]
    box_pdf2 = pdf2_vals[0]
    box_height = pdf2_vals[1] - box_pdf2

    # Plot rectangle
    ax.add_patch(
        Rectangle(
            (box_pdf1, box_pdf2),
            box_width,
            box_height,
            edgecolor=color,
            fill=False,
            zorder=zorder,
        )
    )

    # Label if requested
    if label:
        if marker.name is not None:
            ax.text(pdf1_vals[1], pdf2_vals[1], marker.name, color=color)
        else:
            warnings.warn(nameless_label_warning)

    # Adjust axis limits
    ax.set_xlim((0, 1.1 * pdf1_vals[1]))
    ax.set_ylim((0, 1.1 * pdf2_vals[1]))


def plot_variable_pairs_rectangle(
    ax: Axes,
    markers: Mapping[str, variable_pairs.VariablePair],
    confidence: float = constants.Psigma["2"],
    *,
    # Style args
    color: str = "royalblue",
    zorder: int = 1,
    label: bool = False,
) -> None:
    """Plot variable pairs as rectangles.

    Parameters
    ----------
    ax : Axes
        Axis on which to plot the variable pair.
    markers : dict[str, VariablePair]
        Variable pairs to plot.
    confidence : float
        Confidence range to plot.
    color : str
        Marker color.
    zorder : int
        Order in which to plot the variable pairs relative to other items.
    label : bool
        Label the variable pairs.
    """
    for marker in markers.values():
        plot_variable_pair_rectangle(
            ax=ax,
            marker=marker,
            confidence=confidence,
            color=color,
            zorder=zorder,
            label=label,
        )


def plot_variable_pairs_joint_pdf(
    ax: Axes,
    markers: Mapping[str, variable_pairs.VariablePair],
    *,
    n: int = 1_000,
    pdf1_min: float = 0.0,
    pdf2_min: float = 0.0,
    pdf1_max: float = 0.0,
    pdf2_max: float = 0.0,
    # Style args
    cmap: str = "Greys",
    label: bool = False,
) -> None:
    """Plot markers as joint PDFs.

    Parameters
    ----------
    ax : Axes
        Axis on which to plot the variable pair.
    markers : dict[str, VariablePair]
        Variable pairs to plot.
    n : int
        Number of grid points to use in x and y.
    pdf1_min : float
        Minimum pdf1-axis value.
    pdf2_min : float
        Minimum pdf2-axis value.
    pdf1_max : float
        Maximum pdf1-axis value.
    pdf2_max : float
        Maximum pdf2-axis value.
    cmap : str
        Density colormap.
    label : bool
        Label the variable pairs.
    """
    # Determine plot limits based on markers if necessary
    if pdf1_max is None or pdf1_max == 0:
        pdf1_max = max(marker.pdf1.x.max() for marker in markers.values())

    if pdf2_max is None or pdf2_max == 0:
        pdf2_max = max(marker.pdf2.x.max() for marker in markers.values())

    # Establish a coarse grid on which to sample
    pdf1 = np.linspace(pdf1_min, pdf1_max, n)
    pdf2 = np.linspace(pdf2_min, pdf2_max, n)
    X1, X2 = np.meshgrid(pdf1, pdf2)

    # Initialize total joing probability
    Pjoint = np.zeros(X1.shape)

    # Loop through markers
    for marker_name, marker in markers.items():
        # Interpolate PDFs on coarse grid
        ppdf1 = marker.pdf1.pdf_at_value(pdf1)
        ppdf2 = marker.pdf2.pdf_at_value(pdf2)

        # Compute joint probability
        Pjoint += np.outer(ppdf1, ppdf2)

        # Label if requested
        if label:
            if not marker_name is None:
                pdf1_mode = PDFs.analytics.pdf_mode(marker.pdf1)
                pdf2_mode = PDFs.analytics.pdf_mode(marker.pdf2)
                ax.text(pdf1_mode, pdf2_mode, marker_name, color="royalblue")
            else:
                warnings.warn(nameless_label_warning)

    # Plot joint probability
    ax.pcolormesh(X1, X2, Pjoint.T, cmap=cmap)


VARIABLE_PAIR_PLOT_TYPES: dict[str, Callable[..., Any]] = {
    "whisker": plot_variable_pairs_whisker,
    "rectangle": plot_variable_pairs_rectangle,
    "pdf": plot_variable_pairs_joint_pdf,
}


def get_markers_plot(
    marker_plot_type: str, verbose: bool = False
) -> Callable[..., Any]:
    """Retrieve a variable pairs plot by type.

    Parameters
    ----------
    marker_plot_type : str
        Marker plot type.

    Returns
    -------
    Callable
        Marker plot function.
    """
    if marker_plot_type not in VARIABLE_PAIR_PLOT_TYPES:
        raise ValueError(
            f"Variable pairs plot type '{marker_plot_type}' not supported. "
            f"Used one of {', '.join(VARIABLE_PAIR_PLOT_TYPES)}"
        )

    if verbose:
        print(f"Retrieving '{marker_plot_type}'-type variable pairs plot")

    return VARIABLE_PAIR_PLOT_TYPES[marker_plot_type]


def plot_variable_pairs(
    ax: Axes,
    markers: Mapping[str, variable_pairs.VariablePair],
    marker_plot_type = "whisker",
    *,
    confidence: float = constants.Psigma["2"],
    pdf1_min: float = 0.0,
    pdf2_min: float = 0.0,
    pdf1_max: float = 0.0,
    pdf2_max: float = 0.0,
    label: bool = False,
) -> None:
    """Plot multiple variable pairs.

    Parameters
    ----------
    ax : Axes
        Axis on which to plot variable pairs.
    markers : dict[str, VariablePair]
        Variable pairs to plot.
    marker_plot_type : str
        Marker plot type.
    confidence : float
        Confidence range to plot.
    pdf1_min : float
        Minimum pdf1-axis value.
    pdf2_min : float
        Minimum pdf2-axis value.
    pdf1_max : float
        Maximum pdf1-axis value.
    pdf2_max : float
        Maximum pdf2-axis value.
    label : bool
        Label the variable pairs.
    """
    # Arguments common to any plot
    plt_args = {
        "ax": ax,
        "markers": markers,
        "label": label,
    }

    # Update plot arguments based on marker plot type
    if marker_plot_type in ["whisker", "rectangle"]:
        # Update plot args
        plt_args["confidence"] = confidence

    elif marker_plot_type == "pdf":
        # Update plot args
        plt_args |= {
            "pdf1_min": pdf1_min,
            "pdf2_min": pdf2_min,
            "pdf1_max": pdf1_max,
            "pdf2_max": pdf2_max,
        }

    # Loop through markers
    get_markers_plot(marker_plot_type)(**plt_args)

    # Ensure origin set at zero
    set_origin_zero(ax)

    # Label axes
    format_marker_plot(ax, markers)


# end of file
