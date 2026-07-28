# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

"""
Functions for plotting cumulative distribution functions (CDFs).
"""


# Public API
__all__ = [
    "plot_cdf_line",
    "plot_cdf_filled",
    "plot_cdf_labeled",
]


# Import modules
from .. import (
    units,
    probability_functions as PDFs,
)


#################### CDF PLOTTING ####################
def plot_cdf_line(
    ax,
    pdf: PDFs.PDF,
    *,
    # Style args
    color: str = "black",
    linewidth: float = 2,
) -> None:
    """Basic plot of a cumulative distribution function (CDF).
    """
    # Plot CDF
    ax.plot(
        pdf.x,
        pdf.Px,
        color=color,
        linewidth=linewidth,
        label=pdf.name,
    )


def plot_cdf_filled(
    ax,
    pdf: PDFs.PDF,
    *,
    # Style args
    color: str = "black",
    linewidth: float = 2.0,
    alpha: float = 0.3,
) -> None:
    """Filled plot of a cumulative distribution function (CDF).
    """
    # Plot filled PDF
    ax.fill_between(
        pdf.x,
        pdf.Px,
        color=color,
        alpha=alpha,
    )

    # Plot PDF outline
    plot_cdf_line(
        ax,
        pdf,
        color=color,
        linewidth=linewidth,
    )


def plot_cdf_labeled(
    ax,
    pdf: PDFs.PDF,
    *,
    # Style args
    color: str = "black",
    linewidth: float = 2.0,
    alpha: float = 0.3,
) -> None:
    """Labeled plot of a CDF.
    """
    # Plot filled CDF
    plot_cdf_filled(
        ax,
        pdf,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )

    # Set title
    title = pdf.name if pdf.name is not None else "CDF"
    ax.set_title(title)

    # Set value label
    xlabel = (
        f"{pdf.variable_type.capitalize()} " if pdf.variable_type is not None
        else ""
    )
    xlabel += f"({pdf.unit})" if pdf.unit is not None else ""
    ax.set_xlabel(xlabel)

    # Set probability density label
    ax.set_ylabel("P(X <= x)")


# end of file
