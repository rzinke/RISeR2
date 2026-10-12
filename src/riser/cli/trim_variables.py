#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.


# Import modules
import argparse

import matplotlib.pyplot as plt

from riser import (
    plotting,
    probability_functions as PDFs,
    variable_functions as var_fcns,
)


#################### ARGUMENT PARSER ####################
description = (
    "Trim two or more PDFs based on their relative ordering. "
    "PDFs must be ordered from smallest to largest (X_1 < ... < X_n)."
)

examples = """Examples:
riser-trim-variables younger.txt older.txt -o trimmed
riser-trim-variables pdf1.txt pdf2.txt pdf3.txt -o tr
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
    input_args.add_argument(
        dest="fnames", type=str, nargs="+", help="PDF file names."
    )

    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument(
        "-o",
        "--output-suffix",
        dest="output_suffix",
        type=str,
        required=True,
        help="Output suffix.",
    )

    metadata_args = parser.add_argument_group("Metadata")
    metadata_args.add_argument(
        "--variable-type",
        dest="variable_type",
        type=str,
        help="Override variable type of trimmed PDFs.",
    )
    metadata_args.add_argument(
        "--unit", dest="unit", type=str, help="Override unit of trimmed PDFs."
    )

    diagnostic_args = parser.add_argument_group("Diagnostics")
    diagnostic_args.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose mode.",
    )
    diagnostic_args.add_argument(
        "-p",
        "--plot",
        dest="plot",
        action="store_true",
        help="Plot distribution.",
    )

    return parser.parse_args(args=iargs)


#################### MAIN ####################
def main():
    # Parse arguments
    inps = cmd_parser()

    # Read PDFs from files
    pdfs = PDFs.readers.read_pdfs(inps.fnames)

    # Sample PDFs on same axis
    pdfs = PDFs.interpolation.interpolate_pdfs(pdfs, verbose=inps.verbose)

    # Compute trimmed PDFs
    trimmed_pdfs = var_fcns.condition.trimming.trim_variable_sequence(
        pdfs,
        variable_type=inps.variable_type,
        unit=inps.unit,
        verbose=inps.verbose,
    )

    # Save PDFs to file
    for fname, trimmed_pdf in zip(inps.fnames, trimmed_pdfs):
        trimmed_outname = fname.replace(".txt", f"{inps.output_suffix}.txt")
        PDFs.readers.save_pdf(
            trimmed_outname, trimmed_pdf, verbose=inps.verbose
        )

    # Plot trimmed PDFs
    if inps.plot:
        # Initialize figure and axis
        fig, (inpt_ax, comb_ax) = plt.subplots(nrows=2)

        # Plot input PDFs
        for pdf in pdfs:
            plotting.plot_pdf_filled(inpt_ax, pdf)

        # Plot PDF
        for trimmed_pdf in trimmed_pdfs:
            plotting.plot_pdf_labeled(comb_ax, trimmed_pdf)

        # Format figure
        inpt_ax.legend()
        inpt_ax.set_title("Inputs")
        comb_ax.set_title("Trimmed PDF")
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()


# end of file
