#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import argparse

import matplotlib.pyplot as plt

from riser.probability_functions import readers, interpolation
from riser.variable_operations import subtract_variables
from riser import plotting


#################### ARGUMENT PARSER ####################
Description = """Subtract two random variables expressed as PDFs."""

Examples = """Examples:
subtract_variables.py pdf1.txt pdf2.txt -o pdf12.txt
"""

def create_parser():
    parser = argparse.ArgumentParser(description=Description,
            formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    return parser

def cmd_parser(iargs=None):
    parser = create_parser()

    input_args = parser.add_argument_group("Inputs")
    input_args.add_argument(dest='fname1',
        type=str,
        help="File name of first PDF.")
    input_args.add_argument(dest='fname2',
        type=str,
        help="File name of second PDF.")

    input_args.add_argument('--limit-positive', dest='limit_positive',
        action='store_true',
        help="Enforce the condition that values are greater than or equal to "
             "0.")

    input_args.add_argument('--name', dest='name',
        type=str,
        help="Name of differenced PDF. [None]")

    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument('-o', '--outname', dest='outname',
        type=str, required=True,
        help="Output file.")
    output_args.add_argument('-v', '--verbose', dest='verbose',
        action='store_true',
        help="Verbose mode.")
    output_args.add_argument('-p', '--plot', dest='plot',
        action='store_true',
        help="Plot distribution.")

    return parser.parse_args(args=iargs)


#################### MAIN ####################
def main():
    # Parse arguments
    inps = cmd_parser()

    # Read PDFs from files
    pdf1 = readers.read_pdf(inps.fname1, verbose=inps.verbose)
    pdf2 = readers.read_pdf(inps.fname2, verbose=inps.verbose)

    # Sample PDFs on same axis
    pdf1, pdf2 = interpolation.interpolate_pdfs([pdf1, pdf2],
                                                verbose=inps.verbose)

    # Compute summed PDF
    diff_pdf = subtract_variables(
            pdf1, pdf2, limit_positive=inps.limit_positive, name=inps.name,
            verbose=inps.verbose)

    # Save to file
    readers.save_pdf(inps.outname, diff_pdf, verbose=inps.verbose)

    # Plot function if requested
    if inps.plot == True:
        # Initialize figure and axis
        fig, axes = plt.subplots(nrows=2)

        # Plot input PDFs
        plotting.plot_pdf_filled(fig, axes[0], pdf1)
        plotting.plot_pdf_filled(fig, axes[0], pdf2)

        # Plot PDF
        plotting.plot_pdf_labeled(fig, axes[1], diff_pdf)

        # Format figure
        axes[0].legend()
        axes[0].set_title("Inputs")
        axes[1].set_title("Differenced PDF")
        fig.tight_layout()

        plt.show()


if __name__ == '__main__':
    main()


# end of file