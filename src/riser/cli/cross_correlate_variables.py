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
import riser.variable_operations as var_ops
from riser import units, plotting


#################### ARGUMENT PARSER ####################
description = "Cross-correlate two random variables expressed as PDFs."

examples = """Examples:
cross_correlate_variables.py ref_pdf.txt sec_pdf.txt
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
    input_args.add_argument(dest="ref_fname",
        type=str,
        help="File name of the reference PDF.")
    input_args.add_argument(dest="sec_fname",
        type=str,
        help="File name of the secondary PDF.")

    output_args = parser.add_argument_group("Outputs")
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
    ref_pdf = PDFs.readers.read_pdf(inps.ref_fname, verbose=inps.verbose)
    sec_pdf = PDFs.readers.read_pdf(inps.sec_fname, verbose=inps.verbose)

    # Sample PDFs on same axis
    (ref_pdf,
     sec_pdf) = PDFs.interpolation.interpolate_pdfs(
        [ref_pdf, sec_pdf], verbose=inps.verbose
    )

    # Cross-correlate PDFs
    lags, corr_vals = var_ops.cross_correlate_variables(
        ref_pdf, sec_pdf, verbose=inps.verbose
    )

    # Convert integer lags to lag values with units of the variables
    lag_vals = lags * PDFs.value_arrays.sample_spacing_from_pdf(ref_pdf)

    # Optimal lag index
    opt_ndx = np.argmax(corr_vals)

    # Optimal lag
    opt_lag = lags[opt_ndx]
    opt_lag_val = lag_vals[opt_ndx]

    # Optimal correlation
    opt_corr = corr_vals[opt_ndx]

    # PDF unit
    unit = units.check_same_pdf_units([ref_pdf, sec_pdf])

    # Shift secondary correlation to optimal lag
    px_shifted = np.roll(sec_pdf.px, opt_lag)
    pdf_shifted = PDFs.PDF(sec_pdf.x, px_shifted, name="shifted")

    # Report optimal lag and correlation value if requested
    if inps.verbose == True:
        # Formulate lag string
        lag_str = f"Optimal lag value {opt_lag_val}"
        if unit is not None:
            lag_str += f" ({unit})"

        # Print correlation value
        print(f"Optimal correlation value {opt_corr:.2f}")

    # Plot function if requested
    if inps.plot == True:
        # Initialize figure and axis
        fig, (inpt_ax, corr_ax) = plt.subplots(nrows=2)

        # Plot input PDFs
        plotting.plot_pdf_labeled(inpt_ax, ref_pdf)
        plotting.plot_pdf_labeled(inpt_ax, sec_pdf)

        # Plot correlation
        corr_ax.plot(lag_vals, corr_vals, "k", linewidth=2)

        # Plot shifted PDF
        plotting.plot_pdf_line(inpt_ax, pdf_shifted, color="g")

        # Format figure
        inpt_ax.legend()
        inpt_ax.set_title("Inputs")

        corr_ax.set_title("Cross Correlation")
        corr_ax.set_xlabel(f"Lag ({unit})")

        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()


# end of file