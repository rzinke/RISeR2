# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import matplotlib.pyplot as plt

from riser.probability_functions import PDF, analytics


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


def plot_confidence_range(
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


# end of file