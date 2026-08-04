# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
Functions for plotting variable pairs.
"""


# Public API
__all__ = [
    "plot_markers",
]


# Import modules
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from .. import (
    constants,
    probability_functions as PDFs,
    variable_pairs,
)
from .pdf_plots import axis_label_from_pdf, axis_label_from_pdfs


#################### VARIABLE PAIR PLOTTING ####################
def set_origin_zero(ax: Axes) -> None:
    """Set the plot origin at zero.

    Parameters
    ----------
    ax
        Axes to set at zero.
    """
    ax.set_xlim([0, ax.get_xlim()[1]])
    ax.set_ylim([0, ax.get_ylim()[1]])


def format_marker_plot(
    ax: Axes,
    markers: variable_pair.VariablePair | dict[str, variable_pair.VariablePair],
) -> None:
    """Add axis labels, formulated in the standardized manner.

    Parameters
    ----------
    ax
        Axis on which to plot the variable pair.
    markers : VariablePair or dict[str, VariablePair]
        Variable pair to plot.
    """
    if isinstance(markers, variable_pairs.VariablePair):
        # Axis labels based on single marker
        x1_label = axis_label_from_pdf(markers.x1)
        x2_label = axis_label_from_pdf(markers.x2)

    elif isinstance(markers, dict):
        # Axis labels based on multiple markers
        x1_label = axis_label_from_pdfs(
            [marker.x1 for marker in markers.values()]
        )
        x2_label = axis_label_from_pdfs(
            [marker.x2 for marker in markers.values()]
        )

    else:
        raise TypeError(
            f"Markers must be passed as a single VariablePair "
            f"or dictionary of VariablePairs, got {type(markers).__name__}"
        )

    # Label axes
    ax.set_xlabel(x1_label)
    ax.set_ylabel(x2_label)


def plot_marker_whisker(
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
    x1_center = pdf_center(marker.x1)
    x1_range = PDFs.analytics.compute_interquantile_range(marker.x1, confidence)

    # Plot x1 values (first and only cluster range)
    x1_vals = x1_range.range_values[0]
    x1_err = [[x1_center - x1_vals[0]], [x1_vals[1] - x1_center]]

    # Compute x2 confidence limits
    x2_center = pdf_center(marker.x2)
    x2_range = PDFs.analytics.compute_interquantile_range(marker.x2, confidence)

    # Plot y values (first and only cluster range)
    x2_vals = x2_range.range_values[0]
    x2_err = [[x2_center - x2_vals[0]], [x2_vals[1] - x2_center]]

    # Plot marker
    ax.errorbar(
        x1_center,
        x2_center,
        xerr=x1_err,
        yerr=x2_err,
        color=color,
        zorder=zorder,
    )

    # Label if requested
    if label:
        ax.text(1.01 * x1_center, 1.01 * x2_center, marker.name, color=color)


def plot_markers_whisker(
    ax: Axes,
    markers: dict[str, variable_pairs.VariablePair],
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
        plot_marker_whisker(
            ax=ax,
            marker=marker,
            color=color,
            zorder=zorder,
            label=label,
        )


def plot_marker_rectangle(
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
    x1_range = PDFs.analytics.compute_interquantile_range(marker.x1, confidence)

    # Plot x values (first and only cluster range)
    x1_vals = x1_range.range_values[0]
    box_x1 = x1_vals[0]
    box_width = x1_vals[1] - box_x1

    # Compute y confidence limits
    x2_range = PDFs.analytics.compute_interquantile_range(marker.x2, confidence)

    # Plot x2 values (first and only cluster range)
    x2_vals = x2_range.range_values[0]
    box_x2 = x2_vals[0]
    box_height = x2_vals[1] - box_x2

    # Plot rectangle
    ax.add_patch(
        Rectangle(
            (box_x1, box_x2),
            box_width,
            box_height,
            edgecolor=color,
            fill=False,
            zorder=zorder,
        )
    )

    # Label if requested
    if label:
        ax.text(x1_vals[1], x2_vals[1], marker.name, color=color)

    # Adjust axis limits
    ax.set_xlim([0, 1.1 * x1_vals[1]])
    ax.set_ylim([0, 1.1 * x2_vals[1]])


def plot_markers_rectangle(
    ax: Axes,
    markers: dict[str, variable_pairs.VariablePair],
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
        plot_marker_rectangle(
            ax=ax,
            marker=marker,
            confidence=confidence,
            color=color,
            zorder=zorder,
            label=label,
        )


def plot_markers_joint_pdf(
    ax: Axes,
    markers: dict[str, variable_pairs.VariablePair],
    *,
    n: int = 1_000,
    x1min: float = 0.0,
    x2min: float = 0.0,
    x1max: float = 0.0,
    x2max: float = 0.0,
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
    x1min : float
        Minimum x1-axis value.
    x2min : float
        Minimum x2-axis value.
    x1max : float
        Maximum x1-axis value.
    x2max : float
        Maximum x2-axis value.
    cmap : str
        Density colormap.
    label : bool
        Label the variable pairs.
    """
    # Determine plot limits based on markers if necessary
    if x1max is None or x1max == 0:
        x1max = max(marker.x1.x.max() for marker in markers.values())

    if x2max is None or x2max == 0:
        x2max = max(marker.x2.x.max() for marker in markers.values())

    # Establish a coarse grid on which to sample
    x1 = np.linspace(x1min, x1max, n)
    x2 = np.linspace(x2min, x2max, n)
    X1, X2 = np.meshgrid(x1, x2)

    # Initialize total joing probability
    Pjoint = np.zeros(X1.shape)

    # Loop through markers
    for marker_name, marker in markers.items():
        # Interpolate PDFs on coarse grid
        px1 = marker.x1.pdf_at_value(x1)
        px2 = marker.x2.pdf_at_value(x2)

        # Compute joint probability
        Pjoint += np.outer(px1, px2)

        # Label if requested
        if label:
            x1_mode = PDFs.analytics.pdf_mode(marker.x1)
            x2_mode = PDFs.analytics.pdf_mode(marker.x2)
            ax.text(x1_mode, x2_mode, marker_name, color="royalblue")

    # Plot joint probability
    ax.pcolormesh(X1, X2, Pjoint.T, cmap=cmap)


VARIABLE_PAIR_PLOT_TYPES = {
    "whisker": plot_markers_whisker,
    "rectangle": plot_markers_rectangle,
    "pdf": plot_markers_joint_pdf,
}


def get_markers_plot(
    marker_plot_type: str, verbose: bool = False
) -> "Callable":
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
        print(f"Retrieving '{marker_type}'-type variable pairs plot")

    return VARIABLE_PAIR_PLOT_TYPES.get(marker_plot_type)


def plot_markers(
    ax: Axes,
    markers: dict[str, variable_pairs.VariablePair],
    marker_plot_type = "whisker",
    *,
    confidence: float = constants.Psigma["2"],
    x1min: float = 0.0,
    x2min: float = 0.0,
    x1max: float = 0.0,
    x2max: float = 0.0,
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
    x1min : float
        Minimum x1-axis value.
    x2min : float
        Minimum x2-axis value.
    x1max : float
        Maximum x1-axis value.
    x2max : float
        Maximum x2-axis value.
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
    if marker_plot_type == "whisker":
        # Update plot args
        plt_args["confidence"] = confidence

    elif marker_plot_type == "rectangle":
        # Update plot args
        plt_args["confidence"] = confidence

    elif marker_plot_type == "pdf":
        # Update plot args
        plt_args |= {
            "x1min": x1min,
            "x2min": x2min,
            "x1max": x1max,
            "x2max": x2max,
        }

    # Loop through markers
    get_markers_plot(marker_plot_type)(**plt_args)

    # Ensure origin set at zero
    set_origin_zero(ax)

    # Label axes
    format_marker_plot(ax, markers)


# end of file
