# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
Functions for plotting dated markers.
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
    dated_markers,
)
from .pdf_plots import axis_label_from_pdf, axis_label_from_pdfs


#################### DATED MARKER PLOTTING ####################
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
    markers: dated_markers.DatedMarker | dict[str, dated_markers.DatedMarker],
) -> None:
    """Add axis labels, formulated in the standardized manner.

    Parameters
    ----------
    ax
        Axis on which to plot the dated marker.
    markers : DatedMarker or dict[str, DatedMarker]
        Dated markers to plot.
    """
    if type(markers) == dated_markers.DatedMarker:
        # Axis labels based on single marker
        xlabel = axis_label_from_pdf(markers.age)
        ylabel = axis_label_from_pdf(markers.displacement)

    elif type(markers) == dict:
        # Axis labels based on multiple markers
        xlabel = axis_label_from_pdfs(
            [marker.age for marker in markers.values()]
        )
        ylabel = axis_label_from_pdfs(
            [marker.displacement for marker in markers.values()]
        )

    else:
        raise TypeError(
            f"Markers must be passed as a single DatedMarker "
            f"or dictionary of DatedMarkers, got {type(markers).__name__}"
        )

    # Label axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_marker_whisker(
    ax: Axes,
    marker: dated_markers.DatedMarker,
    confidence: float = constants.Psigma["2"],
    *,
    # Style args
    color: str = "royalblue",
    zorder: int = 1,
    label: bool = False,
) -> None:
    """Plot a dated marker as a cross.

    Parameters
    ----------
    ax
        Axis on which to plot the dated marker.
    marker : DatedMarker
        Dated marker to plot.
    confidence : float
        Confidence range to plot.
    color : str
        Marker color.
    zorder : int
        Order in which to plot the dated markers relative to other items.
    label : bool
        Label the dated markers.
    """
    # Define function that determines the central locaction of a data point
    pdf_center = PDFs.analytics.pdf_mean

    # Compute age confidence limits
    age_center = pdf_center(marker.age)
    age_range = PDFs.analytics.compute_interquantile_range(
        marker.age, confidence
    )

    # Plot age values (first and only cluster range)
    age_vals = age_range.range_values[0]
    age_err = [[age_center - age_vals[0]], [age_vals[1] - age_center]]

    # Compute displacement confidence limits
    disp_center = pdf_center(marker.displacement)
    disp_range = PDFs.analytics.compute_interquantile_range(
        marker.displacement, confidence
    )

    # Plot displacement values (first and only cluster range)
    disp_vals = disp_range.range_values[0]
    disp_err = [[disp_center - disp_vals[0]], [disp_vals[1] - disp_center]]

    # Plot marker
    ax.errorbar(
        age_center,
        disp_center,
        xerr=age_err,
        yerr=disp_err,
        color=color,
        zorder=zorder,
    )

    # Label if requested
    if label:
        ax.text(
            1.01 * age_center, 1.01 * disp_center, marker.name, color=color
        )


def plot_markers_whisker(
    ax: Axes,
    markers: dict[str, dated_markers.DatedMarker],
    confidence: float = constants.Psigma["2"],
    *,
    # Style args
    color: str = "royalblue",
    zorder: int = 1,
    label: bool = False,
) -> None:
    """Plot a dated marker as a cross.

    Parameters
    ----------
    ax
        Axis on which to plot the dated marker.
    markers : dict[str, DatedMarker]
        Dated markers to plot.
    confidence : float
        Confidence range to plot.
    color : str
        Marker color.
    zorder : int
        Order in which to plot the dated markers relative to other items.
    label : bool
        Label the dated markers.
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
    marker: dated_markers.DatedMarker,
    confidence: float = constants.Psigma["2"],
    *,
    # Style args
    color: str = "royalblue",
    zorder: int = 1,
    label: bool = False,
) -> None:
    """Plot a dated marker as a rectangle.

    Parameters
    ----------
    ax
        Axis on which to plot the dated marker.
    marker : DatedMarker
        Dated marker to plot.
    confidence : float
        Confidence range to plot.
    color : str
        Marker color.
    zorder : int
        Order in which to plot the dated markers relative to other items.
    label : bool
        Label the dated markers.
    """
    # Compute age confidence limits
    age_range = PDFs.analytics.compute_interquantile_range(
        marker.age, confidence
    )

    # Plot age values (first and only cluster range)
    age_vals = age_range.range_values[0]
    box_x = age_vals[0]
    box_width = age_vals[1] - box_x

    # Compute displacement confidence limits
    disp_range = PDFs.analytics.compute_interquantile_range(
        marker.displacement, confidence
    )

    # Plot displacement values (first and only cluster range)
    disp_vals = disp_range.range_values[0]
    box_y = disp_vals[0]
    box_height = disp_vals[1] - box_y

    # Plot rectangle
    ax.add_patch(
        Rectangle(
            (box_x, box_y),
            box_width,
            box_height,
            edgecolor=color,
            fill=False,
            zorder=zorder,
        )
    )

    # Label if requested
    if label:
        ax.text(age_vals[1], disp_vals[1], marker.name, color=color)

    # Adjust axis limits
    ax.set_xlim([0, 1.1 * age_vals[1]])
    ax.set_ylim([0, 1.1 * disp_vals[1]])


def plot_markers_rectangle(
    ax: Axes,
    markers: dict[str, dated_markers.DatedMarker],
    confidence: float = constants.Psigma["2"],
    *,
    # Style args
    color: str = "royalblue",
    zorder: int = 1,
    label: bool = False,
) -> None:
    """Plot dated markers as rectangles.

    Parameters
    ----------
    ax : Axes
        Axis on which to plot the dated marker.
    markers : dict[str, DatedMarker]
        Dated markers to plot.
    confidence : float
        Confidence range to plot.
    color : str
        Marker color.
    zorder : int
        Order in which to plot the dated markers relative to other items.
    label : bool
        Label the dated markers.
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
    markers: dict[str, dated_markers.DatedMarker],
    *,
    n: int = 1_000,
    xmin: float = 0.0,
    ymin: float = 0.0,
    xmax: float = 0.0,
    ymax: float = 0.0,
    # Style args
    cmap: str = "Greys",
    label: bool = False,
) -> None:
    """Plot markers as joint PDFs.

    Parameters
    ----------
    ax : Axes
        Axis on which to plot the dated marker.
    markers : dict[str, DatedMarker]
        Dated markers to plot.
    n : int
        Number of grid points to use in x and y.
    xmin : float
        Minimum x-axis value.
    ymin : float
        Minimum y-axis value.
    xmax : float
        Maximum x-axis value.
    ymax : float
        Maximum y-axis value.
    cmap : str
        Density colormap.
    label : bool
        Label the dated markers.
    """
    # Determine plot limits based on markers if necessary
    if xmax == 0:
        for marker in markers.values():
            # Max age
            age_max = marker.age.x.max()
            xmax = age_max if age_max > xmax else xmax

    if ymax == 0:
        for marker in markers.values():
            # Max displacement
            disp_max = marker.displacement.x.max()
            ymax = disp_max if disp_max > ymax else ymax

    # Establish grid boundaries
    xmax = marker.age.x.max() if xmax is None else xmax
    ymax = marker.displacement.x.max() if ymax is None else ymax

    # Establish a coarse grid on which to sample
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(x, y)

    # Initialize total joing probability
    Pjoint = np.zeros(X.shape)

    # Loop through markers
    for marker_name, marker in markers.items():
        # Interpolate PDFs on coarse grid
        px = marker.age.pdf_at_value(x)
        py = marker.displacement.pdf_at_value(y)

        # Compute joint probability
        Pjoint += np.outer(px, py)

        # Label if requested
        if label:
            age_mode = PDFs.analytics.pdf_mode(marker.age)
            disp_mode = PDFs.analytics.pdf_mode(marker.displacement)
            ax.text(age_mode, disp_mode, marker_name, color="royalblue")

    # Plot joint probability
    ax.pcolormesh(X, Y, Pjoint.T, cmap=cmap)


DATED_MARKER_PLOT_TYPES = {
    "whisker": plot_markers_whisker,
    "rectangle": plot_markers_rectangle,
    "pdf": plot_markers_joint_pdf,
}


def get_markers_plot(
    marker_plot_type: str, verbose: bool = False
) -> "Callable":
    """Retrieve a dated markers plot by type.

    Parameters
    ----------
    marker_plot_type : str
        Marker plot type.

    Returns
    -------
    Callable
        Marker plot function.
    """
    if marker_plot_type not in DATED_MARKER_PLOT_TYPES:
        raise ValueError(
            f"Dated markers plot type '{marker_plot_type}' not supported. "
            f"Used one of {', '.join(DATED_MARKER_PLOT_TYPES)}"
        )

    if verbose:
        print(f"Retrieving '{marker_type}'-type dated markers plot")

    return DATED_MARKER_PLOT_TYPES.get(marker_plot_type)


def plot_markers(
    ax: Axes,
    markers: dict[str, dated_markers.DatedMarker],
    marker_plot_type = "whisker",
    *,
    confidence: float = constants.Psigma["2"],
    xmin: float = 0.0,
    ymin: float = 0.0,
    xmax: float = 0.0,
    ymax: float = 0.0,
    label: bool = False,
) -> None:
    """Plot multiple dated markers.

    Parameters
    ----------
    ax : Axes
        Axis on which to plot dated markers.
    markers : dict[str, DatedMarker]
        Dated markers to plot.
    marker_plot_type : str
        Marker plot type.
    confidence : float
        Confidence range to plot.
    xmin : float
        Minimum x-axis value.
    ymin : float
        Minimum y-axis value.
    xmax : float
        Maximum x-axis value.
    ymax : float
        Maximum y-axis value.
    label : bool
        Label the dated markers.
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
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }

    # Loop through markers
    get_markers_plot(marker_plot_type)(**plt_args)

    # Ensure origin set at zero
    set_origin_zero(ax)

    # Label axes
    format_marker_plot(ax, markers)

    # Set title
    ax.set_title("Displacement-Age History")


# end of file
