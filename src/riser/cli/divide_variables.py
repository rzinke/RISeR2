#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import argparse

import matplotlib.pyplot as plt

from riser.probability_functions import readers, interpolation
from riser.variable_operations import divide_variables
from riser import plotting


#################### ARGUMENT PARSER ####################
Description = """Divide two random variables expressed as PDFs."""

Examples = """Examples:
divide_variables.py displacement.txt age.txt -o sliprate.txt
"""

def create_parser():
    parser = argparse.ArgumentParser(description=Description,
            formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    return parser

def cmd_parser(iargs=None):
    parser = create_parser()

    input_args = parser.add_argument_group("Inputs")
    input_args.add_argument(dest='numer_fname',
        type=str,
        help="File name of the numerator PDF.")
    input_args.add_argument(dest='denom_fname',
        type=str,
        help="File name of the denominator PDF.")

    input_args.add_argument('--name', dest='name',
        type=str,
        help="Name of summed PDF. [None]")

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
    numer_pdf = readers.read_pdf(inps.numer_fname, verbose=inps.verbose)
    denom_pdf = readers.read_pdf(inps.denom_fname, verbose=inps.verbose)

    # Sample PDFs on same axis
    (numer_pdf,
     denom_pdf) = interpolation.interpolate_pdfs([numer_pdf, denom_pdf],
                                                 verbose=inps.verbose)

    # Compute summed PDF
    quot_pdf = divide_variables(numer_pdf, denom_pdf, name=inps.name,
                                verbose=inps.verbose)

    # Save to file
    readers.save_pdf(inps.outname, quot_pdf, verbose=inps.verbose)

    # Plot function if requested
    if inps.plot == True:
        # Initialize figure and axis
        fig, axes = plt.subplots(nrows=2)

        # Plot input PDFs
        plotting.plot_pdf_filled(fig, axes[0], pdf1)
        plotting.plot_pdf_filled(fig, axes[0], pdf2)

        # Plot PDF
        plotting.plot_pdf_labeled(fig, axes[1], quot_pdf)

        # Format figure
        axes[0].legend()
        axes[0].set_title("Inputs")
        axes[1].set_title("PDF Quotient")
        fig.tight_layout()

        plt.show()


if __name__ == '__main__':
    main()


# end of file