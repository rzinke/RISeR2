#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.


# Import modules
import argparse

import matplotlib.pyplot as plt

from riser import (
    plotting,
    probability_functions as PDFs,
    variable_functions as var_fcns,
)


#################### ARGUMENT PARSER ####################
description = "Compute the probabilties of values between two bracketing PDFs."

examples = """Examples:
create_bracketed_pdf.py smaller_pdf.txt larger_pdf.txt -o bracketed.txt
"""

def create_parser():
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=examples,
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

    output_args = parser.add_argument_group("Outputs")

    output_args.add_argument("-o", "--outname", dest="outname",
        type=str, required=True,
        help="Output file.")
    output_args.add_argument("--name", dest="name",
        type=str,
        help="Name of bracketed PDF.")
    output_args.add_argument("--variable-type", dest="variable_type",
        type=str,
        help="Variable type of bracketed PDF.")
    output_args.add_argument("--unit", dest="unit",
        type=str,
        help="Unit of bracketed PDF.")

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
    bracketed_pdf, _ = var_fcns.condition.bracketing.infer_bracketed(
        pdf1=pdf1,
        pdf2=pdf2,
        name=inps.name,
        variable_type=inps.variable_type,
        unit=inps.unit,
        verbose=inps.verbose,
    )

    # Save to file
    PDFs.readers.save_pdf(inps.outname, bracketed_pdf, verbose=inps.verbose)

    # Plot bracketed PDF
    if inps.plot:
        # Initialize figure and axis
        fig, ax = plt.subplots()

        # Plot PDF
        plotting.plot_pdf_line(ax, pdf1)
        plotting.plot_pdf_line(ax, pdf2)
        plotting.plot_pdf_labeled(ax, bracketed_pdf)

        # Format figure
        ax.legend()
        ax.set_title("Bracketed PDF")
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()


# end of file
