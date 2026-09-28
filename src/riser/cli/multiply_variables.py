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
description = "Multiply two random variables expressed as PDFs."

examples = """Examples:
riser-multiply-variables sliprate.txt age.txt -o displacement.txt
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
    input_args.add_argument(dest='pdf1_fname',
        type=str,
        help="File name of the first PDF.")
    input_args.add_argument(dest='pdf2_fname',
        type=str,
        help="File name of the second PDF.")

    domain_args = parser.add_argument_group("Domain")
    domain_args.add_argument("--dz", dest="dz",
        type=float, default=0.01,
        help="Product sample spacing.")
    domain_args.add_argument("--min-product", dest="min_product",
        type=float,
        help="Minimum-allowable product to consider.")
    domain_args.add_argument("--max-product", dest="max_product",
        type=float,
        help="Maximum-allowable product to consider.")

    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument("-o", "--outname", dest="outname",
        type=str, required=True,
        help="Output file.")

    metadata_args = parser.add_argument_group("Metadata")
    metadata_args.add_argument("--name", dest="name",
        type=str,
        help="Name of product PDF.")
    metadata_args.add_argument("--variable-type", dest="variable_type",
        type=str,
        help="Variable type of product PDF.")
    metadata_args.add_argument("--unit", dest="unit",
        type=str,
        help="Unit of product PDF.")

    diagnostic_args = parser.add_argument_group("Diagnostics")
    diagnostic_args.add_argument("-v", "--verbose", dest="verbose",
        action="store_true",
        help="Verbose mode.")
    diagnostic_args.add_argument("-p", "--plot", dest="plot",
        action="store_true",
        help="Plot distribution.")

    return parser.parse_args(args=iargs)


#################### MAIN ####################
def main() -> None:
    # Parse arguments
    inps = cmd_parser()

    # Read PDFs from files
    pdf1 = PDFs.readers.read_pdf(inps.pdf1_fname, verbose=inps.verbose)
    pdf2 = PDFs.readers.read_pdf(inps.pdf2_fname, verbose=inps.verbose)

    # Compute product of PDFs
    prod_pdf, _ = var_fcns.transform.arithmetic.multiply_variables(
        pdf1=pdf1,
        pdf2=pdf2,
        dz=inps.dz,
        min_product=inps.min_product,
        max_product=inps.max_product,
        name=inps.name,
        variable_type=inps.variable_type,
        unit=inps.unit,
        verbose=inps.verbose,
    )

    # Save to file
    PDFs.readers.save_pdf(inps.outname, prod_pdf, verbose=inps.verbose)

    # Plot function if requested
    if inps.plot:
        # Initialize figure and axis
        fig, (inpt_ax, prod_ax) = plt.subplots(nrows=2)

        # Plot input PDFs
        plotting.plot_pdf_filled(inpt_ax, pdf1)
        plotting.plot_pdf_filled(inpt_ax, pdf2)

        # Plot PDF
        plotting.plot_pdf_labeled(prod_ax, prod_pdf)

        # Format figure
        inpt_ax.legend()
        inpt_ax.set_title("Inputs")
        prod_ax.set_title("PDF Product")
        fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()


# end of file
