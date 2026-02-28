#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import argparse

import matplotlib.pyplot as plt

import riser.probability_functions as PDFs
import riser.variable_operations as var_ops
from riser import plotting


#################### ARGUMENT PARSER ####################
description = "Subtract two random variables expressed as PDFs."

examples = """Examples:
subtract_variables.py pdf1.txt pdf2.txt -o pdf12.txt
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
    input_args.add_argument(dest="fname1",
        type=str,
        help="File name of first PDF.")
    input_args.add_argument(dest="fname2",
        type=str,
        help="File name of second PDF.")

    input_args.add_argument("--limit-positive", dest="limit_positive",
        action="store_true",
        help="Enforce the condition that values are >= to 0.")

    input_args.add_argument("--name", dest="name",
        type=str,
        help="Name of differenced PDF. [None]")

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

    # Read PDFs from files
    pdf1 = PDFs.readers.read_pdf(inps.fname1, verbose=inps.verbose)
    pdf2 = PDFs.readers.read_pdf(inps.fname2, verbose=inps.verbose)

    # Sample PDFs on same axis
    pdf1, pdf2 = PDFs.interpolation.interpolate_pdfs(
        [pdf1, pdf2], verbose=inps.verbose
    )

    # Compute summed PDF
    diff_pdf = var_ops.subtract_variables(
        pdf1,
        pdf2,
        limit_positive=inps.limit_positive,
        name=inps.name,
        verbose=inps.verbose
    )

    # Save to file
    PDFs.readers.save_pdf(inps.outname, diff_pdf, verbose=inps.verbose)

    # Plot function if requested
    if inps.plot == True:
        # Initialize figure and axis
        fig, (inpt_ax, diff_ax) = plt.subplots(nrows=2)

        # Plot input PDFs
        plotting.plot_pdf_filled(inpt_ax, pdf1)
        plotting.plot_pdf_filled(inpt_ax, pdf2)

        # Plot difference PDF
        plotting.plot_pdf_labeled(diff_ax, diff_pdf)

        # Format figure
        inpt_ax.legend()
        inpt_ax.set_title("Inputs")
        diff_ax.set_title("Difference PDF")
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()


# end of file