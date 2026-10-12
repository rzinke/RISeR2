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
description = "Constrain a variable to be above or below the specified value."

examples = """Examples:
riser-constrain-variable pdf.txt --value 0.0 --above -o constrained
riser-constrain-variable pdf.txt --value 10.0 --below -o constr
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
    input_args.add_argument(dest="fname", type=str, help="PDF to constrain.")

    constraint_args = parser.add_argument_group("Constraint")
    constraint_args.add_argument(
        "--value",
        dest="value",
        type=float,
        required=True,
        help="Value above/below which to constrain the PDF.",
    )
    constraint_direction_args = constraint_args.add_mutually_exclusive_group(
        required=True
    )
    constraint_direction_args.add_argument(
        "--above",
        dest="above",
        action="store_true",
        help="Allow non-zero probability density only above the specified "
        "value.",
    )
    constraint_direction_args.add_argument(
        "--below",
        dest="below",
        action="store_true",
        help="Allow non-zero probability density only below the specified "
        "value.",
    )
    constraint_args.add_argument(
        "--crop",
        dest="crop",
        action="store_true",
        help="Crop to non-zero domain.",
    )

    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument(
        "-o",
        "--outname",
        dest="outname",
        type=str,
        required=True,
        help="Output file name.",
    )

    metadata_args = parser.add_argument_group("Metadata")
    metadata_args.add_argument(
        "--name",
        dest="name",
        type=str,
        help="Override descriptive name of constrained PDF.",
    )
    metadata_args.add_argument(
        "--variable-type",
        dest="variable_type",
        type=str,
        help="Override variable type of constrained PDF.",
    )
    metadata_args.add_argument(
        "--unit",
        dest="unit",
        type=str,
        help="Override unit of constrained PDF.",
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
def main() -> None:
    # Parse arguments
    inps = cmd_parser()

    # Read PDF from file
    pdf = PDFs.readers.read_pdf(inps.fname, verbose=inps.verbose)

    # Retrieve variable constraint function
    if inps.above:
        constraint_fcn = var_fcns.condition.self_constraint.constrain_above
    else:
        constraint_fcn = var_fcns.condition.self_constraint.constrain_below

    # Constrain variable
    constrained_pdf, _ = constraint_fcn(
        pdf=pdf,
        value=inps.value,
        crop=inps.crop,
        name=inps.name,
        variable_type=inps.variable_type,
        unit=inps.unit,
        verbose=inps.verbose,
    )

    # Save to file
    PDFs.readers.save_pdf(inps.outname, constrained_pdf, verbose=inps.verbose)

    # Plot function if requested
    if inps.plot:
        # Initialize figure and axis
        _, ax = plt.subplots()

        # Plot PDF
        plotting.plot_pdf_labeled(ax, constrained_pdf)

    plt.show()


if __name__ == "__main__":
    main()


# end of file
