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
from riser.variable_operations import compute_overlap_index
from riser import plotting


#################### ARGUMENT PARSER ####################
Description = """Compute the overlap statistic of two or more PDFs."""

Examples = """Examples:
compute_overlap_index.py pdf1.txt pdf2.txt
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
    pdfs = interpolation.interpolate_pdfs(pdfs)

    # Compute overlap index
    px_min, eta = compute_overlap_index(pdfs, verbose=True)

    # Plot function if requested
    if inps.plot == True:
        # Initialize figure and axis
        fig, ax = plt.subplots()

        # Plot input PDFs
        for pdf in pdfs:
            plotting.plot_pdf_filled(fig, ax, pdf)

        # Plot minimum
        ax.fill_between(pdfs[0].x, px_min, color="blue")

        # Format figure
        ax.legend()
        ax.set_title("Overlap")

        plt.show()


if __name__ == '__main__':
    main()


# end of file