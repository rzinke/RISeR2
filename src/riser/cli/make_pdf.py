#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.


# Import modules
import argparse
import inspect
import warnings

import matplotlib.pyplot as plt

from riser import (
    plotting,
    precision,
    probability_functions as PDFs,
    units,
    variable_types,
)


#################### ARGUMENT PARSER ####################
description = (
    "Make a PDF for a random variable based on a parametric function."
)

examples = """Examples:
riser-make-pdf triangular --a 9.0 --c 11.0 --b 12.5 -o T1.txt
riser-make-pdf trapezoidal --a 3.5 --b 4.0 --c 5.0 --d 6.0 --dx 0.01 -o T2.txt
riser-make-pdf gaussian --mu 11.3 --sigma 1.2 --dx 0.1 --name T3 --variable-type age --unit ky -o T3.txt

Use `riser-make-pdf <function> -h` to list the parameters of a function.
Options common to all functions must follow the function name.
"""


def create_common_parser():
    """Create a parent parser holding the options common to all functions."""
    common = argparse.ArgumentParser(add_help=False)

    domain_args = common.add_argument_group("Domain")
    domain_args.add_argument(
        "-dx",
        "--dx",
        dest="dx",
        type=float,
        help="Function value array x-step.",
    )
    domain_args.add_argument(
        "--xmin", dest="xmin", type=float, help="Minimum function value."
    )
    domain_args.add_argument(
        "--xmax", dest="xmax", type=float, help="Maximum function value."
    )
    domain_args.add_argument(
        "--limit-positive",
        dest="limit_positive",
        action="store_true",
        help="Enforce the condition that values are > 0.",
    )

    output_args = common.add_argument_group("Outputs")
    output_args.add_argument(
        "-o",
        "--outname",
        dest="outname",
        type=str,
        required=True,
        help="Output file name.",
    )

    metadata_args = common.add_argument_group("Metadata")
    metadata_args.add_argument(
        "--name", dest="name", type=str, help="PDF descriptive name. [None]"
    )
    metadata_args.add_argument(
        "--variable-type",
        dest="variable_type",
        type=str,
        choices=variable_types.SUPPORTED_VARIABLE_TYPES,
        help="PDF variable type, e.g., 'age', 'displacement', 'slip rate'. "
        "[None]",
    )
    metadata_args.add_argument(
        "--unit",
        dest="unit",
        type=str,
        help="Value unit (e.g., 'y'; 'm'. [None]",
    )

    diagnostic_args = common.add_argument_group("Diagnostics")
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

    return common


def create_parser():
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=examples,
    )

    common = create_common_parser()
    common_dests = {action.dest for action in common._actions}

    # One subcommand per parametric function
    subparsers = parser.add_subparsers(
        title="Functions",
        dest="function_name",
        metavar="FUNCTION",
        required=True,
    )
    for (
        function_name,
        function,
    ) in PDFs.parametric_functions.PARAMETRIC_FUNCTIONS.items():
        subparser = subparsers.add_parser(
            function_name,
            parents=[common],
            formatter_class=argparse.RawTextHelpFormatter,
            help=f"{function_name} function",
        )

        # Function parameters, in the order of the function signature
        param_names = [
            name
            for name in inspect.signature(function).parameters
            if name != "x"
        ]
        assert not common_dests.intersection(param_names), function_name

        param_args = subparser.add_argument_group("Function parameters")
        for param_name in param_names:
            param_args.add_argument(
                f"--{param_name}",
                dest=param_name,
                type=float,
                required=True,
                help=f"{function_name} parameter '{param_name}'.",
            )
        subparser.set_defaults(param_names=param_names)

    return parser


def cmd_parser(iargs=None):
    inps = create_parser().parse_args(args=iargs)

    # Collect parameter values in the order of the function signature
    inps.values = [getattr(inps, name) for name in inps.param_names]

    return inps


#################### MAIN ####################
def main() -> None:
    # Parse arguments
    inps = cmd_parser()

    # Check required inputs
    PDFs.parametric_functions.check_number_inputs(
        inps.function_name, inps.values
    )

    # Check optional variable type input
    if inps.variable_type is None:
        warnings.warn(
            "It is strongly suggested to specify variable-type "
            "(e.g., 'age'; 'displacement')",
            stacklevel=2,
        )
    else:
        # Check whether variable type is supported - warn only, no error
        variable_types.check_variable_type_supported(inps.variable_type)

    # Check optional unit input
    if inps.unit is None:
        warnings.warn(
            "It is strongly suggested to specify unit (e.g., 'y'; 'm')",
            stacklevel=2,
        )
    else:
        # Check unit specification
        units.parse_unit(inps.unit)

    # Determine suggested min/max values based on function shape
    (suggested_xmin, suggested_xmax) = (
        PDFs.parametric_functions.determine_min_max_limits(
            function=inps.function_name,
            values=inps.values,
            limit_positive=inps.limit_positive,
            verbose=inps.verbose,
        )
    )

    # Determine min/max values to use
    xmin = suggested_xmin if inps.xmin is None else inps.xmin
    xmax = suggested_xmax if inps.xmax is None else inps.xmax

    # Determine increment
    dx = (xmax - xmin) / 1000 if inps.dx is None else inps.dx

    # Check increment
    precision.check_precision(dx)

    # Create x-array
    x = PDFs.value_arrays.precise_array(xmin, xmax, dx, verbose=inps.verbose)

    # Ensure x-values are greater than zero
    if inps.limit_positive:
        x = x[x > 0.0]

    # Retrieve parameteric function
    para_fcn = PDFs.parametric_functions.get_function_by_name(
        inps.function_name
    )

    # Create PDF
    if inps.verbose:
        print(f"Creating {inps.function_name} function")

    px = para_fcn(x, *inps.values)

    # Instantiate PDF
    pdf = PDFs.PDF(
        x=x,
        px=px,
        name=inps.name,
        variable_type=inps.variable_type,
        unit=inps.unit,
    )

    # Save to file
    PDFs.readers.save_pdf(inps.outname, pdf, verbose=inps.verbose)

    # Plot function if requested
    if inps.plot:
        # Initialize figure and axis
        _, ax = plt.subplots()

        # Plot PDF
        plotting.plot_pdf_labeled(ax, pdf)

    plt.show()


if __name__ == "__main__":
    main()


# end of file
