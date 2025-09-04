# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import os
from datetime import datetime

from riser.probability_functions import analytics


#################### FILENAME FORMATTING ####################
def establish_output_dir(output_prefix:str, verbose=False):
    """Determine the output directory based on the current directory and the
    output_prefix.
    Create it if it does not already exist.

    Args    output_prefix - str, output <prefix> or <folder>/<prefix>
    Returns outdir
    """
    # Check for folder
    outfldr = os.path.dirname(output_prefix)

    # Establish absolute filepath
    outdir = os.path.abspath(outfldr)

    # Report if requested
    if verbose == True:
        print(f"Output directory: {outdir}")

        # Check if it already exists
        if os.path.exists(outdir):
            print(" ... already exists")
        else:
            print(" ... creating")

    # Create output folder if it does not alrady exist
    os.makedirs(outdir, exist_ok=True)


#################### FIGURE SAVING ####################
def save_marker_fig(output_prefix:str, marker_fig, verbose=False):
    """Save figure showing the dated displacement history to a file.
    """
    # Formulate outname
    outname = f"{output_prefix}_markers.pdf"

    # Format figure
    marker_fig.tight_layout()

    # Save figure to file
    marker_fig.savefig(outname)

    # Report if requested
    if verbose == True:
        print(f"Saved dated displacement history (markers) to: "
              f"{os.path.abspath(outname)}")


def save_slip_rate_fig(output_prefix:str, rate_fig, verbose=False):
    """
    """
    # Formulate outname
    outname = f"{output_prefix}_slip_rates.pdf"

    # Format figure
    rate_fig.tight_layout()

    # Save figure to file
    rate_fig.savefig(outname)

    # Report if requested
    if verbose == True:
        print(f"Saved slip rate figure to: {os.path.abspath(outname)}")


#################### SLIP RATE SUMMARY REPORTS ####################
def write_slip_rates_report(output_prefix:str,
                            formulation:str,
                            slip_rates:dict,
                            pdf_statistics:dict=None,
                            confidence_ranges:dict=None,
                            verbose=False):
    """Write slip rate statistics to a file.
    Include:
    Description, date, time
    Slip rate name
        slip rate stats
        slip rate confidence intervals
    """
    # Check that slip rate statistical products pertain to same pairs
    if pdf_statistics is not None:
        if pdf_statistics.keys() != slip_rates.keys():
            raise ValueError("One PDFstatistics object must be provided for "
                             "each slip rate")

    if confidence_ranges is not None:
        if confidence_ranges.keys() != slip_rates.keys():
            raise ValueError("One ConfidenceRange object must be provided "
                             "for each slip rate")

    # Formulate outname
    outname = f"{output_prefix}_slip_rate_report.txt"

    # Write file contents
    with open(outname, 'w') as outfile:
        # Overall header
        outfile.write(f"Incremental slip rates from {formulation} formulation "
                      f"({datetime.now().strftime('%Y %m %d:%H %M %S')})")

        # Loop through incremental slip rates based on marker pairs
        for marker_pair in slip_rates.keys():
            # Breathe
            outfile.write("\n\n")

            # Write PDF statistics
            if pdf_statistics is not None:
                outfile.write(str(pdf_statistics[marker_pair]))

            # Write confidence ranges
            if confidence_ranges is not None:
                outfile.write(str(confidence_ranges[marker_pair]))

    # Report if requested
    if verbose == True:
        print(f"Saved slip rate report to: {os.path.abspath(outname)}")


# end of file