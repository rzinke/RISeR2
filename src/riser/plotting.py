# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants
from riser.constants import Psigma


# Import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from riser import variable_types, units
from riser.probability_functions import PDF, analytics
from riser.sampling import filtering
from riser.markers import DatedMarker


#################### GENERAL LABELING ####################
def formulate_axis_label(variable_type:str, unit:str) -> str:
    """Formulate an axis label in a standardized manner.

    Args    pdf - PDF from which to draw the metadata
    Returns ax_label - str, standardized axis label
    """
    # Set variable type if available
    ax_label = f"{variable_type.capitalize()} " \
            if variable_type is not None else ""

    # Add unit if available
    ax_label += f"({unit})" if unit is not None else ""

    return ax_label


def axis_label_from_pdf(pdf:PDF) -> str:
    """Formulate an axis label from PDF metadata in a standardized manner.

    Args    pdf - PDF from which to draw the metadata
    Returns ax_label - str, standardized axis label
    """
    # Set variable type
    variable_type = pdf.variable_type

    # Set unit
    unit = pdf.unit

    return formulate_axis_label(variable_type, unit)


def axis_label_from_pdfs(pdfs:list[PDF]) -> str:
    """Formulate an axis label from PDF metadata in a standardized manner.

    Args    pdfs - list[PDF], PDFs from which to draw the metadata
    Returns ax_label - str, standardized axis label
    """
    # Set variable type
    variable_type = variable_types.check_same_pdf_variable_types(pdfs)

    # Set unit
    unit = units.check_same_pdf_units(pdfs)

    return formulate_axis_label(variable_type, unit)


#################### PDF PLOTTING ####################
def plot_pdf_line(fig, ax, pdf:PDF,
                  color:str="black", linewidth:int=2,
                  offset:float=0, scale:float=1, zorder=1):
    """Basic plot of a probability density function (PDF).
    """
    # Plot PDF
    ax.plot(pdf.x, scale * pdf.px + offset,
            color=color, linewidth=linewidth, label=pdf.name, zorder=zorder)


def plot_pdf_filled(fig, ax, pdf:PDF,
                    color:str="black", linewidth:int=2,
                    offset:float=0, scale:float=1, alpha=0.3, zorder=1):
    """Filled plot of a probability density function (PDF).
    """
    # Plot filled PDF
    ax.fill_between(pdf.x, scale * pdf.px + offset, y2=offset,
                    color=color, alpha=alpha, zorder=zorder)

    # Plot PDF outline
    plot_pdf_line(fig, ax, pdf,
                  color=color, linewidth=linewidth,
                  offset=offset, scale=scale, zorder=zorder)


def plot_pdf_labeled(fig, ax, pdf:PDF,
                     color:str="black", linewidth:int=2,
                     offset:float=0, scale:float=1, alpha:float=0.3,
                     zorder=1):
    """Labeled plot of a PDF.
    """
    # Plot filled PDF
    plot_pdf_filled(fig, ax, pdf, color, linewidth, offset, scale, alpha,
                    zorder)

    # Set title
    title = pdf.name if pdf.name is not None else "PDF"
    ax.set_title(title)

    # Set value label
    ax.set_xlabel(axis_label_from_pdf(pdf))

    # Set probability density label
    if all([offset == 0, scale == 0]):
        # y-values are probability density
        ax.set_ylabel("Probability density")
    else:
        # y-values are scaled and/or shifted: the exact value is meaningless
        ax.set_yticks([])
        ax.set_ylabel("Rel probability density")


# PDF Confidence
def plot_pdf_confidence_range(
        fig, ax, pdf:PDF, conf_range:analytics.ConfidenceRange,
        color:str="royalblue", offset:float=0, scale:float=1, alpha:float=0.3,
        incl_label:bool=False, zorder=1):
    """Plot confidence ranges as fields overlying a PDF.
    """
    # Formulate label
    label = f"{100 * conf_range.confidence:.2f} %" if incl_label == True \
            else None

    # Plot confidence ranges
    for rng in conf_range:
        # Indices within range
        rng_ndx = (pdf.x >= rng[0]) & (pdf.x <= rng[1])

        # Plot range
        plot_args = {
            'x': pdf.x[rng_ndx],
            'y1': scale * pdf.px[rng_ndx] + offset,
            'y2': offset,
            'color': color,
            'alpha': alpha,
            'label': label,
            'zorder': zorder,
        }
        ax.fill_between(**plot_args)


# Multi-PDF
def plot_pdf_stack(fig, ax, pdfs:dict, height:float=0.9, colors:dict={},
                   conf_ranges:dict={}, priors:dict={}, same_height:bool=False):
    """Plot multiple PDFs as rows on the same figure.
    Check all PDFs for the maximum px value, scale the largest max to 1.0,
    and scale the other PDF maxima accordingly.

    Args    fig, ax - figure and axis on which to plot
            pdfs - dict, PDFs stored by PDF name
            conf_ranges - dict, ConfidenceRanges stored by PDF name
            height - float, height of hightest PDF peak relative to line
                spacing
    """
    # Determine highest peak
    max_peak = 0
    for pdf in pdfs.values():
        px_max = pdf.px.max()
        max_peak = px_max if px_max > max_peak else max_peak

    # Loop through PDFs
    for i, (name, pdf) in enumerate(pdfs.items()):
        # Determine scale
        if same_height == True:
            scale = height / pdf.px.max()
        else:
            scale = height / max_peak

        # Plot prior if available
        if priors.get(name) is not None:
            plot_pdf_line(fig, ax, priors.get(name), offset=i, scale=scale,
                          color="darkgrey", zorder=1)

        # Plot PDF
        plot_pdf_filled(fig, ax, pdf, offset=i, scale=scale,
                        color=colors.get(name, "black"), zorder=2)

        # Plot confidence range if available
        if conf_ranges.get(name) is not None:
            plot_pdf_confidence_range(fig, ax, pdf, conf_ranges.get(name),
                                      offset=i, scale=scale, incl_label=False,
                                      zorder=3)

    # Format plot
    ax.set_xlabel(axis_label_from_pdfs([*priors.values()] + [*pdfs.values()]))
    ax.set_yticks(range(len(pdfs)))
    ax.set_yticklabels([*pdfs.keys()])
    ax.set_ylabel("Rel probability density")


#################### CDF PLOTTING ####################
def plot_cdf_line(fig, ax, pdf:PDF,
                  color="black", linewidth=2):
    """Basic plot of a cumulative distribution function (CDF).
    """
    # Plot PDF
    ax.plot(pdf.x, pdf.Px, color=color, linewidth=linewidth, label=pdf.name)


def plot_cdf_filled(fig, ax, pdf:PDF,
                    color="black", linewidth=2, alpha:float=0.3):
    """Filled plot of a cumulative distribution function (CDF).
    """
    # Plot filled PDF
    ax.fill_between(pdf.x, pdf.Px, color=color, alpha=alpha)

    # Plot PDF outline
    plot_cdf_line(fig, ax, pdf, color=color, linewidth=linewidth)


def plot_cdf_labeled(fig, ax, pdf:PDF,
                     color="black", linewidth=2, alpha:float=0.3):
    """Labeled plot of a CDF.
    """
    # Plot filled CDF
    plot_cdf_filled(fig, ax, pdf,
                    color=color, linewidth=linewidth, alpha=alpha)

    # Set title
    title = pdf.name if pdf.name is not None else "CDF"
    ax.set_title(title)

    # Set value label
    xlabel = f"{pdf.variable_type.capitalize()} " \
            if pdf.variable_type is not None else ""
    xlabel += f"({pdf.unit})" if pdf.unit is not None else ""
    ax.set_xlabel(xlabel)

    # Set probability density label
    ax.set_ylabel("P(X <= x)")


#################### FILTER KERNEL PLOTTING ####################
def plot_filter_kernel(fig, ax, filt:filtering.FIRFilter):
    """Plot a filter kernel.
    """
    # Plot kernel values
    ax.plot(filt.h, color="k", linewidth=2)


#################### DATED MARKER PLOTTING ####################
def set_origin_zero(ax):
    """Set the plot origin at zero.
    """
    ax.set_xlim([0, ax.get_xlim()[1]])
    ax.set_ylim([0, ax.get_ylim()[1]])


def format_marker_plot(fig, ax, markers:DatedMarker|dict):
    """Add axis labels, formulated in the standardized manner.
    """
    if type(markers) == DatedMarker:
        # Axis labels based on single marker
        xlabel = axis_label_from_pdf(markers.age)
        ylabel = axis_label_from_pdf(markers.displacement)

    elif type(markers) == dict:
        # Axis labels based on multiple markers
        xlabel = axis_label_from_pdfs(
                [marker.age for marker in markers.values()])
        ylabel = axis_label_from_pdfs(
                [marker.displacement for marker in markers.values()])

    else:
        raise Exception("Markers must be passed as a single DatedMarker "
                        "or dictionary of DatedMarkers")

    # Label axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_marker_whisker(
        fig, ax, marker:DatedMarker, confidence:float=Psigma['2'],
        color="royalblue", zorder=1, label=False):
    """Plot a dated marker as a cross.
    """
    # Compute confidence limits
    age_mode = analytics.pdf_mode(marker.age)
    age_range = analytics.compute_interquantile_range(
            marker.age, confidence)
    age_vals = age_range.range_values[0]  # first and only cluster of ranges
    age_err = [[age_mode - age_vals[0]], [age_vals[1] - age_mode]]

    disp_mode = analytics.pdf_mode(marker.displacement)
    disp_range = analytics.compute_interquantile_range(
            marker.displacement, confidence)
    disp_vals = disp_range.range_values[0]  # first and only cluster of ranges
    disp_err = [[disp_mode - disp_vals[0]], [disp_vals[1] - disp_mode]]

    # Plot marker
    ax.errorbar(age_mode, disp_mode, xerr=age_err, yerr=disp_err,
                color=color, zorder=zorder)

    # Label if requested
    if label == True:
        ax.text(1.01 * age_mode, 1.01 * disp_mode, marker.name)


def plot_marker_rectangle(
        fig, ax, marker:DatedMarker, confidence:float=Psigma['2'],
        color="royalblue", zorder=1, label=False):
    """Plot a dated marker as a rectangle.
    """
    # Compute confidence limits
    age_range = analytics.compute_interquantile_range(
            marker.age, confidence)
    age_vals = age_range.range_values[0]  # first and only cluster of ranges
    box_x = age_vals[0]
    box_width = age_vals[1] - box_x

    disp_range = analytics.compute_interquantile_range(
            marker.displacement, confidence)
    disp_vals = disp_range.range_values[0]  # first and only cluster of ranges
    box_y = disp_vals[0]
    box_height = disp_vals[1] - box_y

    # Plot rectangle
    ax.add_patch(Rectangle((box_x, box_y), box_width, box_height,
                 edgecolor=color, fill=False, zorder=zorder))

    # Label if requested
    if label == True:
        ax.text(age_vals[1], disp_vals[1], marker.name)

    # Adjust axis limits
    ax.set_xlim([0, 1.1 * age_vals[1]])
    ax.set_ylim([0, 1.1 * disp_vals[1]])


def plot_markers_joint_pdf(fig, ax, markers:dict, **kwargs):
    """Plot markers as joint PDFs.
    """
    # Initialize plot limits
    xmin = kwargs.get('xmin', 0)
    ymin = kwargs.get('ymin', 0)
    xmax = kwargs.get('xmax', 0)
    ymax = kwargs.get('xmax', 0)

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
    n = kwargs.get('n', 1000)
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
        if kwargs.get('label') == True:
            age_mode = analytics.pdf_mode(marker.age)
            disp_mode = analytics.pdf_mode(marker.displacement)
            ax.text(age_mode, disp_mode, marker_name, color="royalblue")

    # Plot joint probability
    ax.pcolormesh(X, Y, Pjoint.T, cmap=kwargs.get('cmap', 'Greys'))


def plot_markers(fig, ax, markers:dict, marker_plot_type="whisker", **kwargs):
    """Plot multiple dated markers.
    """
    # Determine action based on marker plot type
    if marker_plot_type == "pdf":
        # Plot joint PDFs
        plot_markers_joint_pdf(fig, ax, markers, **kwargs)

    else:
        # Retrieve marker plot
        if marker_plot_type == "whisker":
            # Retrieve whisker plot
            plotter = plot_marker_whisker

        elif marker_plot_type == "rectangle":
            # Retrieve rectangle plot
            plotter = plot_marker_rectangle

        elif marker_plot_type == "pdf":
            # Retrieve joint PDF plot
            plotter = plot_markers_joint_pdf

        else:
            raise ValueError(f"Marker plot type {marker_plot_type} not "
                             f"recognized")

        # Loop through markers
        for marker_name, marker in markers.items():
            plotter(fig, ax, marker, **kwargs)

    # Ensure origin set at zero
    set_origin_zero(ax)

    # Label axes
    format_marker_plot(fig, ax, markers)

    # Set title
    ax.set_title("Displacement-Age History")


#################### SAMPLE PLOTTING ####################
def plot_mc_picks(fig, ax, age_picks:np.ndarray, disp_picks:np.ndarray,
                  max_picks:int=500):
    """Plot valid displacement-age picks.
    """
    # Plot lines connecting points
    ax.plot(age_picks[:,:max_picks], disp_picks[:,:max_picks],
            color="k", alpha=0.1, zorder=1)

    # Plot pick values
    ax.scatter(age_picks[:,:max_picks], disp_picks[:,:max_picks], s=2**2,
               color="b", alpha=0.1, zorder=2)


# end of file