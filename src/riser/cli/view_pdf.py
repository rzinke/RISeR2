#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants
from riser import constants


# Import modules
import argparse

import matplotlib.pyplot as plt

import riser.probability_functions as PDFs
from riser import plotting


#################### ARGUMENT PARSER ####################
description = "View the PDF of a random variable and its properties."

examples = """Examples:
view_pdf.py pdf_file.txt
view_pdf.py pdf_file.txt --show-confidence --confidence-limits 0.9545 --confidence-methods IQR
view_pdf.py pdf_file.txt -o pdf_fig.png --no-show
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
    input_args.add_argument(dest="fname",
        type=str,
        help="PDF file name.")

    output_args = parser.add_argument_group("Outputs")

    output_args.add_argument("--show-confidence", dest="show_confidence",
        action="store_true",
        help="Show confidence range on PDF and print as text.")
    output_args.add_argument("--confidence-limits", dest="confidence_limits",
        type=float, default=constants.Psigma["1"],
        help="Confidence limits. [0.682...]")
    output_args.add_argument("--confidence-method", dest="confidence_method",
        type=str, default="HPD",
        help="Method for determining confidence limits. [HPD]")

    output_args.add_argument("--show-cdf", dest="show_cdf",
        action="store_true",
        help="Show the cumulative distribution function.")

    output_args.add_argument("-v", "--verbose", dest="verbose",
        action="store_true",
        help="Verbose mode.")
    output_args.add_argument("-o", "--outname", dest="outname",
        type=str,
        help="Output file.")
    output_args.add_argument("--no-show", dest="no_show",
        action="store_true",
        help="Forego showing plot.")

    return parser.parse_args(args=iargs)


#################### MAIN ####################
def main():
    # Parse arguments
    inps = cmd_parser()

    # Read PDF from file
    pdf = PDFs.readers.read_pdf(inps.fname, verbose=inps.verbose)

    # Report stats if requested
    if inps.verbose == True:
        print(PDFs.analytics.PDFstatistics(pdf))

    # Initialize figure and axis
    pdf_fig, pdf_ax = plt.subplots()

    # Plot PDF
    plotting.plot_pdf_labeled(pdf_ax, pdf)

    # Compute and plot confidence range
    if inps.show_confidence == True:
        # Retrieve confidence range function
        conf_fcn = PDFs.analytics.get_pdf_confidence_function(
                inps.confidence_method, verbose=inps.verbose
        )

        # Compute confidence range
        conf_range = conf_fcn(pdf, inps.confidence_limits)
        if inps.verbose == True:
            print(conf_range)

        # Plot confidence range(s)
        plotting.plot_pdf_confidence_range(pdf_ax, pdf, conf_range)

    # Show CDF
    if inps.show_cdf == True:
        # Initialize figure and axis
        cdf_fig, cdf_ax = plt.subplots()

        # Plot CDF
        plotting.plot_cdf_labeled(cdf_ax, pdf)

    # Save figure to file
    if inps.outname is not None:
        # Check output filename
        if inps.outname[-4:] != ".png":
            raise ValueError("File output name must end in .png")

        # Save figure
        pdf_fig.savefig(inps.outname)

    # Show unless specified not to
    if inps.no_show == False:
        plt.show()


if __name__ == "__main__":
    main()


# end of file