#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants


# Import modules
import argparse

import matplotlib.pyplot as plt

from riser.probability_functions import readers, interpolation
from riser.variable_operations import combine_variables
from riser import plotting


#################### ARGUMENT PARSER ####################
Description = """Compute the joint PDF of two or more functions."""

Examples = """Examples:
combine_variables.py pdf1.txt pdf2.txt -o joint_pdf.txt
combine_variables.py pdf1.txt pdf2.txt pdf3.txt -o joint_pdf.txt
"""

def create_parser():
    parser = argparse.ArgumentParser(description=Description,
            formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    return parser

def cmd_parser(iargs=None):
    parser = create_parser()

    input_args = parser.add_argument_group("Inputs")
    input_args.add_argument(dest='fnames',
        type=str, nargs='+',
        help="PDF file names.")

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
    pdfs = readers.read_pdfs(inps.fnames)

    # Sample PDFs on same axis
    pdfs = interpolation.interpolate_pdfs(pdfs, verbose=inps.verbose)

    # Compute joint PDF
    joint_pdf = combine_variables(pdfs, verbose=inps.verbose)

    # Save to file
    readers.save_pdf(inps.outname, joint_pdf, verbose=inps.verbose)

    # Plot function if requested
    if inps.plot == True:
        # Initialize figure and axis
        fig, axes = plt.subplots(nrows=2)

        # Plot input PDFs
        for pdf in pdfs:
            plotting.plot_pdf_filled(fig, axes[0], pdf)

        # Plot PDF
        plotting.plot_pdf_labeled(fig, axes[1], joint_pdf)

        # Format figure
        axes[0].legend()
        axes[0].set_title("Inputs")
        axes[1].set_title("Joint PDF")
        fig.tight_layout()

        plt.show()


if __name__ == '__main__':
    main()


# end of file