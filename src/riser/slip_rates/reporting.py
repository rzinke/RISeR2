# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Public API
__all__ = [
    "establish_output_dir",
    "save_marker_fig",
    "save_slip_rate_fig",
    "write_picks_to_file",
    "write_slip_rates_report",
]


# Import modules
import os
from datetime import datetime

import numpy as np
from matplotlib.figure import Figure

from .. import probability_functions as PDFs
from .. import sampling


#################### FILENAME FORMATTING ####################
def establish_output_dir(output_prefix: str, verbose: bool = False) -> None:
    """Create a slip rate output directory name.

    Determine the output directory based on the current directory and the
    output_prefix.

    Create it if it does not already exist.

    Parameters
    ----------
    output_prefix : str
        Output <prefix> or <folder>/<prefix>.
    """
    # Check for folder
    outfldr = os.path.dirname(output_prefix)

    # Establish absolute filepath
    outdir = os.path.abspath(outfldr)

    # Report if requested
    if verbose:
        print(f"Output directory: {outdir}")

        # Check if it already exists
        if os.path.exists(outdir):
            print(" ... already exists")
        else:
            print(" ... creating")

    # Create output folder if it does not alrady exist
    os.makedirs(outdir, exist_ok=True)


#################### FIGURE SAVING ####################
def save_marker_fig(
    output_prefix: str,
    marker_fig: Figure,
    verbose: bool = False,
) -> None:
    """Save figure showing the dated displacement history to a file.

    Parameters
    ----------
    output_prefix : str
        Prefix for output file name.
    marker_fig : Figure
        Dated marker figure to save.
    """
    # Formulate outname
    outname = f"{output_prefix}_markers.pdf"

    # Format figure
    marker_fig.tight_layout()

    # Save figure to file
    marker_fig.savefig(outname)

    # Report if requested
    if verbose:
        print(f"Saved dated displacement history (markers) to: "
              f"{os.path.abspath(outname)}")


def save_slip_rate_fig(
    output_prefix: str,
    rate_fig: Figure,
    verbose: bool = False,
) -> None:
    """Save figure showing slip rate measurements to a file.

    Parameters
    ----------
    output_prefix : str
        Prefix for output file name.
    rate_fig : Figure
        Slip rate figure to save.
    """
    # Formulate outname
    outname = f"{output_prefix}_slip_rates.pdf"

    # Format figure
    rate_fig.tight_layout()

    # Save figure to file
    rate_fig.savefig(outname)

    # Report if requested
    if verbose:
        print(f"Saved slip rate figure to: {os.path.abspath(outname)}")


#################### MC PICKS SAVING ####################
def write_picks_to_file(
    output_prefix: str,
    age_picks: np.ndarray,
    disp_picks: np.ndarray,
    rate_picks: np.ndarray,
    verbose: bool = False,
) -> None:
    """Save all valid samples (picks) to a numpy binary file (npz).

    Parameters
    ----------
    output_prefix : str
        Prefix for output file name.
    age_picks : np.ndarray
        Age samples that meet the sample criterion.
    disp_picks : np.ndarray
        Displacement samples that meet the sample criterion.
    rate_picks : np.ndarray
        Slip rate samples that meet the sample criterion.
    """
    # Formulate outname
    outname = f"{output_prefix}_picks"

    # Save to file
    np.savez(outname,
        age_picks=age_picks,
        disp_picks=disp_picks,
        rate_picks=rate_picks,
    )

    # Report if requested
    if verbose:
        print(f"Saved MC picks to: {os.path.abspath(outname)}.npz")


#################### SLIP RATE SUMMARY REPORTS ####################
def write_slip_rates_report(
    output_prefix: str,
    formulation: str,
    slip_rates: dict[str, PDFs.PDF],
    *,
    pdf_statistics: (
        dict[str, PDFs.analytics.PDFstatistics] | None
    ) = None,
    confidence_ranges: (
        dict[str, PDFs.analytics.ConfidenceRange] | None
    ) = None,
    sample_statistics: (
        dict[str, sampling.sample_statistics.SampleStatistics] | None
    ) = None,
    verbose: bool = False,
) -> None:
    """Write slip rate statistics to a file.

    Include:
    Description, date, time
    Slip rate name
        slip rate stats
        slip rate confidence intervals

    Parameters
    ----------
    output_prefix : str
        Prefix for output file name.
    formulation : str
        Slip rates computation description.
    slip_rates : dict[str, PDF]
        Incremental slip rate PDFs.
    pdf_statistics : dict[str, PDFstatistics], optional
        Slip rate PDF statistics.
    confidence_ranges : dict[str, ConfidenceRange], optional
        Slip rate PDF confidence ranges.
    sample_statistics : dict[str, SampleStatistics], optional
        Slip rate sample statistics.
    """
    # Check that slip rate statistical products pertain to same pairs
    if sample_statistics is not None:
        if sample_statistics.keys() != slip_rates.keys():
            raise ValueError(
                "One SampleStatistics object must be provided for each "
                "slip rate"
            )

    if pdf_statistics is not None:
        if pdf_statistics.keys() != slip_rates.keys():
            raise ValueError(
                "One PDFstatistics object must be provided for each slip rate"
            )

    if confidence_ranges is not None:
        if confidence_ranges.keys() != slip_rates.keys():
            raise ValueError(
                "One ConfidenceRange object must be provided for each slip rate"
            )

    # Formulate outname
    outname = f"{output_prefix}_slip_rate_report.txt"

    # Write file contents
    with open(outname, 'w') as outfile:
        # Overall header
        outfile.write(
            f"Incremental slip rates from {formulation} formulation "
            f"({datetime.now().strftime('%Y %m %d:%H %M %S')})"
        )

        # Loop through incremental slip rates based on marker pairs
        for marker_pair in slip_rates.keys():
            # Breathe
            outfile.write("\n\n")

            # Write sample statistics
            if sample_statistics is not None:
                outfile.write(str(sample_statistics[marker_pair]) + "\n")

            # Write PDF statistics
            if pdf_statistics is not None:
                outfile.write(str(pdf_statistics[marker_pair]) + "\n")

            # Write confidence ranges
            if confidence_ranges is not None:
                outfile.write(str(confidence_ranges[marker_pair]) + "\n")

    # Report if requested
    if verbose:
        print(f"Saved slip rate report to: {os.path.abspath(outname)}")


# end of file
