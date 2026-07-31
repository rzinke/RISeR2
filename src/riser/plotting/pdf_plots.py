# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
Functions for plotting probability density functions (PDFs).
"""


# Public API
__all__ = [
    "axis_label_from_pdf",
    "axis_label_from_pdfs",
    "plot_pdf_line",
    "plot_pdf_filled",
    "plot_pdf_labeled",
    "plot_pdf_confidence_range",
    "plot_pdf_stack",
]


# Import modules
from matplotlib.axes import Axes

from .. import (
    units,
    probability_functions as PDFs,
    variable_types,
)


#################### GENERAL LABELING ####################
def formulate_axis_label(variable_type: str, unit: str) -> str:
    """Formulate an axis label in a standardized manner.

    Parameters
    ----------
    variable_type : str
        Variable type from which to draw the metadata.
    unit : str
        Unit.

    Returns
    -------
    ax_label : str
        Standardized axis label.
    """
    # Set variable type if available
    ax_label = (
        f"{variable_type.capitalize()} " if variable_type is not None else ""
    )

    # Add unit if available
    ax_label += f"({unit})" if unit is not None else ""

    return ax_label


def axis_label_from_pdf(pdf: PDFs.PDF) -> str:
    """Formulate an axis label from PDF metadata in a standardized manner.

    Parameters
    ----------
    pdf : PDF
        PDF from which to draw the metadata.

    Returns
    -------
    ax_label : str
        Standardized axis label.
    """
    # Set variable type
    variable_type = pdf.variable_type

    # Set unit
    unit = pdf.unit

    return formulate_axis_label(variable_type, unit)


def axis_label_from_pdfs(pdfs: list[PDFs.PDF]) -> str:
    """Formulate an axis label from PDF metadata in a standardized manner.

    Parameters
    ----------
    pdfs : list[PDF]
        PDFs from which to draw the metadata.

    Returns
    -------
    ax_label : str
        Standardized axis label.
    """
    # Find common metadata values
    common_metadata = PDFs.metadata.get_common_metadata(pdfs)

    return formulate_axis_label(
        variable_type=common_metadata.variable_type,
        unit=common_metadata.unit,
    )


#################### PDF PLOTTING ####################
def plot_pdf_line(
    ax: Axes,
    pdf: PDFs.PDF,
    *,
    # Style args
    color: str = "black",
    linewidth: float = 2.0,
    zorder: int = 1,
    # Scaling args
    offset: float = 0.0,
    scale: float = 1.0
) -> None:
    """Basic line plot of a probability density function (PDF).

    Parameters
    ----------
    ax : Axes
        Pyplot axis.
    pdf : PDF
        Probability density function to plot.
    color : str
        PDF color.
    linewidth : float
        PDF linewidth.
    zorder : int
        Position on plot.
    offset : float
        y-axis offset.
    scale : float
        y-axis scale.
    """
    # Plot PDF
    ax.plot(
        pdf.x,
        scale * pdf.px + offset,
        color=color,
        linewidth=linewidth,
        zorder=zorder,
        label=pdf.name,
    )


def plot_pdf_filled(
    ax: Axes,
    pdf: PDFs.PDF,
    *,
    # Style args
    color: str = "black",
    linewidth: float = 2.0,
    zorder: int = 1,
    alpha: float = 0.3,
    # Scaling args
    offset: float = 0.0,
    scale: float = 1.0,
) -> None:
    """Filled plot of a probability density function (PDF).

    Parameters
    ----------
    ax : Axes
        Pyplot axis.
    pdf : PDF
        Probability density function to plot.
    color : str
        PDF color.
    linewidth : float
        PDF linewidth.
    zorder : int
        Position on plot.
    alpha : float
        PDF fill opacity.
    offset : float
        y-axis offset.
    scale : float
        y-axis scale.
    """
    # Plot filled PDF
    ax.fill_between(
        pdf.x,
        scale * pdf.px + offset,
        y2=offset,
        color=color,
        zorder=zorder,
        alpha=alpha,
    )

    # Plot PDF outline
    plot_pdf_line(
        ax,
        pdf,
        color=color,
        linewidth=linewidth,
        zorder=zorder,
        offset=offset,
        scale=scale,
    )


def plot_pdf_labeled(
    ax: Axes,
    pdf: PDFs.PDF,
    *,
    # Style args
    color: str = "black",
    linewidth: float = 2.0,
    zorder: int = 1,
    alpha: float = 0.3,
    # Scaling args
    offset: float = 0.0,
    scale: float = 1.0,
) -> None:
    """Labeled plot of a PDF.

    Parameters
    ----------
    ax : Axes
        Pyplot axis.
    pdf : PDF
        Probability density function to plot.
    color : str
        PDF color.
    linewidth : float
        PDF linewidth.
    zorder : int
        Position on plot.
    alpha : float
        Opacity.
    offset : float
        y-axis offset.
    scale : float
        y-axis scale.
    """
    # Plot filled PDF
    plot_pdf_filled(
        ax,
        pdf,
        color=color,
        linewidth=linewidth,
        zorder=zorder,
        alpha=alpha,
        offset=offset, 
        scale=scale,
    )

    # Set title
    title = pdf.name if pdf.name is not None else "PDF"
    ax.set_title(title)

    # Set value label
    ax.set_xlabel(axis_label_from_pdf(pdf))

    # Set probability density label
    if offset == 0 and scale == 1:
        # y-values are probability density
        ax.set_ylabel("Probability density")
    else:
        # y-values are scaled and/or shifted: the exact value is meaningless
        ax.set_yticks([])
        ax.set_ylabel("Rel probability density")


# PDF Confidence
def plot_pdf_confidence_range(
    ax: Axes,
    pdf: PDFs.PDF,
    conf_range: PDFs.analytics.ConfidenceRange,
    *,
    # Style args
    color: str = "royalblue",
    zorder: int = 1,
    alpha: float = 0.3,
    incl_label: bool = False,
    # Scaling args
    offset: float = 0.0,
    scale: float = 1.0,
) -> None:
    """Plot confidence ranges as fields overlying a PDF.

    Parameters
    ----------
    ax : Axes
        Pyplot axis.
    pdf : PDF
        Probability density function to plot.
    conf_range : ConfidenceRange
        PDF confidence range.
    color : str
        PDF color.
    linewidth : float
        PDF linewidth.
    zorder : int
        Position on plot.
    alpha : float
        Opacity.
    incl_label : bool
        Include label in PDF plot.
    offset : float
        y-axis offset.
    scale : float
        y-axis scale.
    """
    # Formulate label
    label = (
        f"{100 * conf_range.confidence:.2f} %" if incl_label
        else None
    )

    # Plot confidence ranges
    for rng in conf_range:
        # Indices within range
        rng_ndx = (pdf.x >= rng[0]) & (pdf.x <= rng[1])

        # Plot range
        ax.fill_between(
            pdf.x[rng_ndx],
            y1=scale * pdf.px[rng_ndx] + offset,
            y2=offset,
            color=color,
            zorder=zorder,
            alpha=0.5,
            label=label
        )


# Multi-PDF
def plot_pdf_stack(
    ax: Axes,
    pdfs: dict[str, PDFs.PDF],
    height: float = 0.9,
    colors: dict[str, str] | None = None,
    conf_ranges: dict[str, PDFs.analytics.ConfidenceRange] | None = None,
    priors: dict[str, PDFs.PDF] | None = None,
    same_height: bool | None = False,
) -> None:
    """Plot multiple PDFs as rows on the same figure.

    Check all PDFs for the maximum px value, scale the largest max to 1.0,
    and scale the other PDF maxima accordingly.

    Parameters
    ----------
    ax : Axes
        Axis on which to plot PDF stack.
    pdfs : dict[str, PDF]
        PDFs stored by PDF name.
    height : float
        Height of hightest PDF peak relative to line spacing.
    colors : dict[str, str]
        PDF colors.
    conf_ranges : dict[str, ConfidenceRange]
        Confidence ranges stored by PDF name.
    priors : dict[str, PDF]
        PDF prior distributions.
    same_height : bool
        Scale PDF modes to the same height.
    """
    # Set defaults
    if colors is None:
        colors = {}

    if conf_ranges is None:
        conf_ranges = {}

    if priors is None:
        priors = {}

    # Determine highest peak
    max_peak = 0
    for pdf in pdfs.values():
        px_max = pdf.px.max()
        max_peak = px_max if px_max > max_peak else max_peak

    # Loop through PDFs
    for i, (name, pdf) in enumerate(pdfs.items()):
        # Determine scale
        if same_height:
            scale = height / pdf.px.max()
        else:
            scale = height / max_peak

        # Plot prior if available
        if priors.get(name) is not None:
            plot_pdf_line(
                ax,
                priors.get(name),
                color="darkgrey",
                zorder=3,
                offset=i,
                scale=scale,
            )

        # Plot PDF
        plot_pdf_filled(
            ax,
            pdf,
            color=colors.get(name, "black"),
            zorder=2,
            offset=i,
            scale=scale,
        )

        # Plot confidence range if available
        if conf_ranges.get(name) is not None:
            plot_pdf_confidence_range(
                ax,
                pdf,
                conf_range=conf_ranges.get(name),
                zorder=1,
                offset=i,
                scale=scale,
            )

    # Format plot
    ax.set_xlabel(axis_label_from_pdfs([*priors.values()] + [*pdfs.values()]))
    ax.set_yticks(range(len(pdfs)))
    ax.set_yticklabels([*pdfs.keys()])
    ax.set_ylabel("Rel probability density")


# end of file
