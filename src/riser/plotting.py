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

from riser.probability_functions import PDF, analytics
from riser.sampling import filtering
from riser.markers import DatedMarker


#################### PDF PLOTTING ####################
def plot_pdf_line(fig, ax, pdf:PDF,
                  color="black", linewidth=2):
    """Basic plot of a probability density function (PDF).
    """
    # Plot PDF
    ax.plot(pdf.x, pdf.px, color=color, linewidth=linewidth, label=pdf.name)


def plot_pdf_filled(fig, ax, pdf:PDF, alpha=0.3, **kwargs):
    """Filled plot of a probability density function (PDF).
    """
    # Parse args
    color = kwargs.get('color', "black")
    linewidth = kwargs.get('linewidth', 2)

    # Plot filled PDF
    ax.fill_between(pdf.x, pdf.px, color=color, alpha=alpha)

    # Plot PDF outline
    plot_pdf_line(fig, ax, pdf, color=color, linewidth=linewidth)


def plot_pdf_labeled(fig, ax, pdf:PDF, **kwargs):
    """Labeled plot of a PDF.
    """
    # Plot filled PDF
    plot_args = {
        'fig': fig,
        'ax': ax,
        'pdf': pdf,
        'color': kwargs.get('color', "black"),
        'linewidth': kwargs.get('linewidth', 2),
        'alpha': kwargs.get('alpha', 0.3)
    }
    plot_pdf_filled(**plot_args)

    # Set title
    title = pdf.name if pdf.name is not None else "PDF"
    ax.set_title(title)

    # Set value label
    xlabel = f"{pdf.variable_type.capitalize()} " \
            if pdf.variable_type is not None else ""
    xlabel += f"({pdf.unit})" if pdf.unit is not None else ""
    ax.set_xlabel(xlabel)

    # Set probability density label
    ax.set_ylabel("Probability density")


# PDF Confidence
def plot_pdf_confidence_range(
        fig, ax, pdf:PDF, conf_range:analytics.ConfidenceRange,
        color="royalblue", alpha=0.3):
    """Plot confidence ranges as fields overlying a PDF.
    """
    # Plot confidence ranges
    for rng in conf_range:
        rng_ndx = (pdf.x >= rng[0]) & (pdf.x <= rng[1])
        ax.fill_between(pdf.x[rng_ndx], pdf.px[rng_ndx],
                        color=color, alpha=alpha,
                        label=f"{100 * conf_range.confidence:.2f} %")

    # Format plot
    ax.legend()


#################### CDF PLOTTING ####################
def plot_cdf_line(fig, ax, pdf:PDF,
                  color="black", linewidth=2):
    """Basic plot of a cumulative distribution function (CDF).
    """
    # Plot PDF
    ax.plot(pdf.x, pdf.Px, color=color, linewidth=linewidth, label=pdf.name)


def plot_cdf_filled(fig, ax, pdf:PDF, alpha=0.3, **kwargs):
    """Filled plot of a cumulative distribution function (CDF).
    """
    # Parse args
    color = kwargs.get('color', "black")
    linewidth = kwargs.get('linewidth', 2)

    # Plot filled PDF
    ax.fill_between(pdf.x, pdf.Px, color=color, alpha=alpha)

    # Plot PDF outline
    plot_cdf_line(fig, ax, pdf, color=color, linewidth=linewidth)


def plot_cdf_labeled(fig, ax, pdf:PDF, **kwargs):
    """Labeled plot of a CDF.
    """
    # Plot filled CDF
    plot_args = {
        'fig': fig,
        'ax': ax,
        'pdf': pdf,
        'color': kwargs.get('color', "black"),
        'linewidth': kwargs.get('linewidth', 2),
        'alpha': kwargs.get('alpha', 0.3)
    }
    plot_cdf_filled(**plot_args)

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


def format_marker_plot(fig, ax, marker:DatedMarker):
    """Add axis labels.
    """
    # Label axes
    ax.set_xlabel(marker.age.variable_type)
    ax.set_ylabel(marker.displacement.variable_type)

    # Format figure
    fig.tight_layout()


def plot_marker_whisker(
        fig, ax, marker:DatedMarker, confidence:float=Psigma['2'],
        color="royalblue", label=False):
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
    ax.errorbar(age_mode, disp_mode, xerr=age_err, yerr=disp_err, color=color)

    # Label if requested
    if label == True:
        ax.text(1.01 * age_mode, 1.01 * disp_mode, marker.name)


def plot_marker_rectangle(
        fig, ax, marker:DatedMarker, confidence:float=Psigma['2'],
        color="royalblue", label=False):
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
                 edgecolor=color, fill=False))

    # Label if requested
    if label == True:
        ax.text(age_vals[1], disp_vals[1], marker.name)

    # Adjust axis limits
    ax.set_xlim([0, 1.1 * age_vals[1]])
    ax.set_ylim([0, 1.1 * disp_vals[1]])


def plot_markers(fig, ax, markers:dict, marker_plot_type="whisker", **kwargs):
    """Plot multiple dated markers.
    """
    # Retrieve marker plot
    if marker_plot_type == "whisker":
        plotter = plot_marker_whisker
    elif marker_plot_type == "rectangle":
        plotter = plot_marker_rectangle
    else:
        raise ValueError(f"Marker plot type {marker_plot_type} not recognized")

    # Loop through markers
    for marker_name, marker in markers.items():
        # Plot marker
        plot_args = {
            'fig': fig,
            'ax': ax,
            'marker': marker,
            'confidence': kwargs.get('confidence', Psigma['2']),
            'color': kwargs.get('color', "royalblue"),
            'label': kwargs.get('label', False),
        }
        plotter(**plot_args)

    # Set origin at zero
    set_origin_zero(ax)

    # Label axes
    format_marker_plot(fig, ax, marker)



# end of file