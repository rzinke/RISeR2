#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants
from riser.sampling.filtering import FILTER_TYPES


# Import modules
import argparse

import matplotlib.pyplot as plt

from riser.probability_functions import readers, PDF, analytics
from riser.sampling import filtering
from riser import units, plotting


#################### ARGUMENT PARSER ####################
Description = """Convert an age PDF from calendar years to (kilo)years before present."""

Examples = """Examples:
"""

def create_parser():
    parser = argparse.ArgumentParser(description=Description,
            formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    return parser

def cmd_parser(iargs=None):
    parser = create_parser()

    input_args = parser.add_argument_group("Inputs")
    input_args.add_argument(dest='fname',
        type=str,
        help="Calendar years file name.")

    # Metadata
    metadata_args = parser.add_argument_group("Metadata")
    metadata_args.add_argument('--name', dest='name',
        type=str,
        help="Name.")
    metadata_args.add_argument('--variable-type', dest='variable_type',
        type=str, default="age",
        help="Variable type. [age]")

    # Referencing
    referencing_args = parser.add_argument_group("Referencing")
    referencing_args.add_argument('--reference-date', dest='reference_date',
        type=float, default=1950,
        help="Reference date, e.g., 1950 for radiocarbon, 2000 for OSL, etc. "
             "[1950]")
    referencing_args.add_argument('--limit-zero', dest='limit_zero',
        action='store_true',
        help="Limit youngest value to zero.")

    # Units
    unit_args = parser.add_argument_group("Units")
    unit_args.add_argument('--input-unit', dest='input_unit',
        type=str,
        help="Unit of input data. [y]")
    unit_args.add_argument('--output-unit', dest='output_unit',
        type=str, default="ky",
        help="Unit of output data. [ky]")

    # Smoothing
    smoothing_args = parser.add_argument_group("Smoothing")
    smoothing_args.add_argument('--smoothing-type', dest='smoothing_type',
        type=str, choices=FILTER_TYPES, default="gauss",
        help="Smoothing filter type. [gauss]")
    smoothing_args.add_argument('--smoothing-width', dest='smoothing_width',
        type=int, default=0,
        help="Smoothing kernel width. [0]")

    # Outputs
    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument('-o', '--outname', dest='outname',
        type=str, required=True,
        help="Output file.")
    output_args.add_argument('-v', '--verbose', dest='verbose',
        action='store_true',
        help="Verbose mode.")
    output_args.add_argument('-p', '--plot', dest='plot',
        action='store_true',
        help="Plot.")

    return parser.parse_args(args=iargs)


#################### MAIN ####################
def main():
    # Parse arguments
    inps = cmd_parser()

    # Read calendar years file
    calyr, calpx, metadata = readers.read_calendar_file(inps.fname,
                                                     verbose=inps.verbose)

    # Convert calendar year to years before present (ypb)
    if inps.verbose == True:
        print(f"Shifting dates relative to reference: {inps.reference_date}")
    ybp = inps.reference_date - calyr

    # Check input units
    input_unit = units.get_priority_unit(metadata.get('unit'), inps.input_unit)

    # Scale from input units to output units
    x = units.scale_by_units(ybp, input_unit, inps.output_unit,
                          verbose=inps.verbose)

    # Flip left for right, so age is increasing
    x = x[::-1]
    px = calpx[::-1]

    # Limit minimum age to zero
    if inps.limit_zero == True:
        # Non-negative indices
        non_neg_ndx = (x >= 0)

        # Crop arrays
        x = x[non_neg_ndx]
        px = px[non_neg_ndx]

    # Form data into PDF
    pdf_args = {
        'x': x,
        'px': px,
        'name': inps.name,
        'variable_type': inps.variable_type,
        'unit': inps.output_unit,
    }
    pdf = PDF(**pdf_args, normalize_area=True)

    # Smooth data
    if inps.smoothing_width > 0:
        filter_args = {
            'pdf': pdf,
            'filter_type': inps.smoothing_type,
            'filter_width': inps.smoothing_width,
            'edge_padding': "zeros",
            'verbose': inps.verbose,
        }
        pdf = filtering.filter_pdf(**filter_args)

    # Report final stats if requested
    if inps.verbose == True:
        stats = analytics.PDFstatistics(pdf)
        print(stats)

    # Save to file
    readers.save_pdf(inps.outname, pdf, verbose=inps.verbose)

    # Plot function if requested
    if inps.plot == True:
        # Initialize figure and axis
        fig, axes = plt.subplots(nrows=2)

        # Plot calendar years
        axes[0].plot(calyr, calpx, color="k")

        # Plot PDF
        plotting.plot_pdf_labeled(fig, axes[1], pdf)

        # Format plot
        axes[0].invert_xaxis()
        axes[0].set_xlabel("Calendar year (CE / BCE)")
        axes[0].set_ylabel("Probability density")

        fig.tight_layout()

        plt.show()


if __name__ == '__main__':
    main()


# end of file