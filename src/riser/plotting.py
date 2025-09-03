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
                  color="k", linewidth=2):
    """Basic plot of a probability density function (PDF).
    """
    # Plot PDF
    ax.plot(pdf.x, pdf.px, color=color, linewidth=linewidth, label=pdf.name)


def plot_pdf_filled(fig, ax, pdf:PDF,
                    color="k", linewidth=2, alpha=0.3):
    """Filled plot of a probability density function (PDF).
    """
    # Plot filled PDF
    ax.fill_between(pdf.x, pdf.px, color=color, alpha=alpha)

    # Plot PDF outline
    plot_pdf_line(fig, ax, pdf, color=color, linewidth=linewidth)


def plot_pdf_labeled(fig, ax, pdf:PDF,
                     color="k", linewidth=2, alpha=0.3):
    """Labeled plot of a PDF.
    """
    # Plot filled PDF
    plot_pdf_filled(fig, ax, pdf,
                    color=color, linewidth=linewidth, alpha=alpha)

    # Set title
    title = pdf.name if pdf.name is not None else "PDF"
    ax.set_title(title)

    # Set value label
    xlabel = f"{pdf.variable_type} " if pdf.variable_type is not None else ""
    xlabel += f"({pdf.unit})" if pdf.unit is not None else ""
    ax.set_xlabel(xlabel)

def plot_pdf_labeled(fig, ax, pdf:PDF,
                     color="k", linewidth=2, alpha=0.3):
    """Labeled plot of a PDF.
    """
    # Plot filled PDF
    plot_pdf_filled(fig, ax, pdf,
                    color=color, linewidth=linewidth, alpha=alpha)

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
                  color="k", linewidth=2):
    """Basic plot of a cumulative distribution function (CDF).
    """
    # Plot PDF
    ax.plot(pdf.x, pdf.Px, color=color, linewidth=linewidth, label=pdf.name)


def plot_cdf_filled(fig, ax, pdf:PDF,
                    color="k", linewidth=2, alpha=0.3):
    """Filled plot of a cumulative distribution function (CDF).
    """
    # Plot filled PDF
    ax.fill_between(pdf.x, pdf.Px, color=color, alpha=alpha)

    # Plot PDF outline
    plot_cdf_line(fig, ax, pdf, color=color, linewidth=linewidth)


def plot_cdf_labeled(fig, ax, pdf:PDF,
                     color="k", linewidth=2, alpha=0.3):
    """Labeled plot of a CDF.
    """
    # Plot filled PDF
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
        ax.text(1.01 * age_mode, 1.01 * disp_mode, marker.displacement.name)


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
        ax.text(box_x + box_width, box_y + box_height, marker.displacement.name)


# end of file