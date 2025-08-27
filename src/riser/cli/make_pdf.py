#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Constants
from riser.probability_functions.parametric_functions import \
    PARAMETRIC_FUNCTIONS


# Import modules
import argparse

import numpy as np
import matplotlib.pyplot as plt

from riser import precision
from riser.probability_functions import PDF, readers, value_arrays, \
    parametric_functions
from riser import plotting


#################### ARGUMENT PARSER ####################
Description = """Make a PDF for a random variable based on a parametric distribution."""

Examples = """Examples:
make_pdf.py -d triangular -s 9.0 11.0 12.5 -dx 0.1 -o T1.txt
make_pdf.py -d trapezoidal -s 3.5 4.0 5.0 6.0 -dx 0.01 -o T2.txt
make_pdf.py -d gaussian -s 11.3 1.2 -dx 0.1 --name T3 --variable age --unit ky -o T3.txt
"""

def create_parser():
    parser = argparse.ArgumentParser(description=Description,
            formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    return parser

def cmd_parser(iargs=None):
    parser = create_parser()

    input_args = parser.add_argument_group("Inputs")
    input_args.add_argument('-d', '--distribution', dest='distribution',
        type=str, choices=PARAMETRIC_FUNCTIONS, required=True,
        help="Parametric function.")
    input_args.add_argument('-s', '--values', dest='values',
        type=float, nargs='+', required=True,
        help="Parameter values.")
    input_args.add_argument('-dx', '--dx', dest='dx',
        type=float, required=True,
        help="x-step.")

    input_args.add_argument('--name', dest='name',
        type=str,
        help="PDF name. [None]")
    input_args.add_argument('--variable-type', dest='variable_type',
        type=str,
        help="PDF variable type, e.g., age, displacement, slip rate. [None]")
    input_args.add_argument('--unit', dest='unit',
        type=str,
        help="Value unit. [None]")

    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument('-o', '--outname', dest='outname',
        type=str, required=True,
        help="Output file.")
    output_args.add_argument('-v', '--verbose', dest='verbose',
        action='store_true',
        help="Verbose mode.")
    output_args.add_argument('-p', '--plot', dest='plot',
        action='store_true',
        help="Plot distribution.")

    return parser.parse_args(args=iargs)


#################### SUPPORT FUNCTIONS ####################
def check_number_inputs(distribution:str, values:list[float]):
    """Check that the appropriate number of inputs are provided for the given
    distribution.

    Args    distribution - str, parametric function
            values - list[float], parameter values
    """
    # Number of values
    n_vals = len(values)

    # Check against distribution
    if distribution == "boxcar":
        if n_vals != 2:
            raise ValueError("2 parameters must be specified for a boxcar "
                             "distribution")

    elif distribution == "triangular":
        if n_vals != 3:
            raise ValueError("3 parameters must be specified for a triangular "
                             "distribution")

    if distribution == "trapezoidal":
        if n_vals != 4:
            raise ValueError("4 parameters must be specified for a trapezoidal "
                             "distribution")

    if distribution == "gaussian":
        if n_vals != 2:
            raise ValueError("2 parameters must be specified for a gaussian "
                             "distribution")


def determine_min_max_limits(distribution:str, values:list[float],
                             verbose=False) -> (float, float):
    """Determine the minimum and maximum values of the PDF domain.

    Args    distribution - str, parametric function
            values - list[float], parameter values
    Returns xmin, xmax - float, min/max values
    """
    # Behave based on function type
    if distribution in ["boxcar", "triangular", "trapezoidal"]:
        # Use first and last values
        xmin = values[0]
        xmax = values[-1]

    elif distribution in ["gaussian"]:
        # Parse values
        mu, sigma = values

        # Use 4-sigma limit
        sigma_lim = 4 * sigma

        # Limit at zero
        xmin = np.max([mu - sigma_lim, 0])

        # Max value
        xmax = mu + sigma_lim

    # Report if requested
    if verbose == True:
        print(f"Minimum value {xmin}\nMaximum value {xmax}")

    return xmin, xmax


#################### MAIN ####################
def main():
    # Parse arguments
    inps = cmd_parser()

    # Check inputs
    check_number_inputs(inps.distribution, inps.values)
    precision.check_precision(inps.dx)

    # Determine min/max values
    xmin, xmax = determine_min_max_limits(inps.distribution, inps.values,
                                          verbose=inps.verbose)

    # Create x-array
    x = value_arrays.create_precise_value_array(xmin, xmax, inps.dx,
                                                verbose=inps.verbose)

    # Retrieve parameteric function
    para_fcn = parametric_functions.get_function_by_name(inps.distribution)

    # Create PDF
    if inps.verbose == True:
        print(f"Creating {inps.distribution} distribution")

    px = para_fcn(x, *inps.values)

    # Instantiate PDF
    pdf = PDF(x, px, normalize_area=True,
              name=inps.name, variable_type=inps.variable_type, unit=inps.unit)

    # Save to file
    readers.save_pdf(inps.outname, pdf, verbose=inps.verbose)

    # Plot function if requested
    if inps.plot == True:
        # Initialize figure and axis
        fig, ax = plt.subplots()

        # Plot PDF
        plotting.plot_pdf_labeled(fig, ax, pdf)

        plt.show()


if __name__ == '__main__':
    main()


# end of file