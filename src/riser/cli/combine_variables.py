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
description = "Compute the intersection of two or more PDFs."

examples = """Examples:
riser-combine-variables pdf1.txt pdf2.txt -o joint_pdf.txt
riser-combine-variables pdf1.txt pdf2.txt pdf3.txt -o joint_pdf.txt
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
    input_args.add_argument(dest="fnames",
        type=str, nargs="+",
        help="PDF file names.")

    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument("-o", "--outname", dest="outname",
        type=str, required=True,
        help="Output file.")

    metadata_args = parser.add_argument_group("Metadata")
    metadata_args.add_argument("--name", dest="name",
        type=str,
        help="Name of combined PDF.")
    metadata_args.add_argument("--variable-type", dest="variable_type",
        type=str,
        help="Variable type of combined PDF.")
    metadata_args.add_argument("--unit", dest="unit",
        type=str,
        help="Unit of combined PDF.")

    diagnostic_args = parser.add_argument_group("Diagnostics")
    diagnostic_args.add_argument("-v", "--verbose", dest="verbose",
        action="store_true",
        help="Verbose mode.")
    diagnostic_args.add_argument("-p", "--plot", dest="plot",
        action="store_true",
        help="Plot distribution.")

    return parser.parse_args(args=iargs)


#################### MAIN ####################
def main():
    # Parse arguments
    inps = cmd_parser()

    # Read PDFs from files
    pdfs = PDFs.readers.read_pdfs(inps.fnames)

    # Sample PDFs on same axis
    pdfs = PDFs.interpolation.interpolate_pdfs(pdfs, verbose=inps.verbose)

    # Compute joint PDF
    joint_pdf, _ = var_fcns.condition.combination.combine_variables(
        pdfs,
        name=inps.name,
        variable_type=inps.variable_type,
        unit=inps.unit,
        verbose=inps.verbose,
    )

    # Save to file
    PDFs.readers.save_pdf(inps.outname, joint_pdf, verbose=inps.verbose)

    # Plot combined PDF
    if inps.plot:
        # Initialize figure and axis
        fig, (inpt_ax, comb_ax) = plt.subplots(nrows=2)

        # Plot input PDFs
        for pdf in pdfs:
            plotting.plot_pdf_filled(inpt_ax, pdf)

        # Plot PDF
        plotting.plot_pdf_labeled(comb_ax, joint_pdf)

        # Format figure
        inpt_ax.legend()
        inpt_ax.set_title("Inputs")
        comb_ax.set_title("Joint PDF")
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()


# end of file
