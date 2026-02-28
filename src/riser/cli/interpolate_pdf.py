#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import argparse

import numpy as np
import matplotlib.pyplot as plt

import riser.probability_functions as PDFs
from riser import plotting


#################### ARGUMENT PARSER ####################
description = (
    "Interpolate (and extrapolate) the PDF of a random variable over a new "
    "range of values."
)

examples = """Examples:
interpolate_pdf.py pdf_file.txt --dx 0.01
interpolate_pdf.py pdf_file.txt --xmin 0 --xmax 100 -o interp_pdf_file.txt
"""

def create_parser():
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=examples
    )

    return parser

def cmd_parser(iargs=None):
    parser = create_parser()

    input_args = parser.add_argument_group("Inputs")
    input_args.add_argument(dest="pdf_name",
        type=str,
        help="PDF file name.")

    input_args.add_argument("--xmin", dest="xmin",
        type=float,
        help="Minimum x-value. [None]")
    input_args.add_argument("--xmax", dest="xmax",
        type=float,
        help="Maximum x-value. [None]")
    input_args.add_argument("--dx", dest="dx",
        type=float,
        help="x-step. [None]")

    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument("-o", "--outname", dest="outname",
        type=str, required=True,
        help="Output file.")
    output_args.add_argument("-v", "--verbose", dest="verbose",
        action="store_true",
        help="Verbose mode.")
    output_args.add_argument("-p", "--plot", dest="plot",
        action="store_true",
        help="Plot distribution.")


    return parser.parse_args(args=iargs)


#################### MAIN ####################
def main():
    # Parse arguments
    inps = cmd_parser()

    # Read PDF from file
    pdf = PDFs.readers.read_pdf(inps.pdf_name, verbose=inps.verbose)

    # Determine bounds of interpolation and sampling rate
    dx = inps.dx if inps.dx is not None else np.median(np.diff(pdf.x))
    xmin = inps.xmin if inps.xmin is not None else pdf.x.min()
    xmax = inps.xmax if inps.xmax is not None else pdf.x.max()

    # Report sampling parameters if requested
    if inps.verbose == True:
        print(f"xmin {xmin}")
        print(f"xmax {xmax}")

    # Formulate value array
    x = PDFs.value_arrays.create_precise_value_array(
        xmin, xmax, dx, verbose=inps.verbose
    )

    # Interpolate PDF
    pdf_resamp = PDFs.interpolation.interpolate_pdf(pdf, x)

    # Plot if requested
    if inps.plot == True:
        # Initialize figure and axis
        fig, ax = plt.subplots()

        # Plot PDF
        plotting.plot_pdf_labeled(ax, pdf_resamp)

    plt.show()


if __name__ == "__main__":
    main()


# end of file