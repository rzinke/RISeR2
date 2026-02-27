#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import argparse

import numpy as np
import matplotlib.pyplot as plt

from riser.probability_functions import(
    readers, interpolation, value_arrays, PDF
)
from riser.variable_operations import compute_ks_statistic
from riser import plotting, units


#################### ARGUMENT PARSER ####################
Description = "Compute the K-S statistic for two PDFs."

Examples = """Examples:
compute_ks_statistic.py pdf1.txt pdf2.txt
"""

def create_parser():
    parser = argparse.ArgumentParser(description=Description,
            formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    return parser

def cmd_parser(iargs=None):
    parser = create_parser()

    input_args = parser.add_argument_group("Inputs")
    input_args.add_argument(dest='fname1',
        type=str,
        help="File name of first PDF.")
    input_args.add_argument(dest='fname2',
        type=str,
        help="File name of second PDF.")

    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument('-v', '--verbose', dest='verbose',
        action='store_true',
        help="Verbose mode.")
    output_args.add_argument('-p', '--plot', dest='plot',
        action='store_true',
        help="Plot distribution.")

    return parser.parse_args(args=iargs)


#################### MAIN ####################
def main():
    # Parse arguments
    inps = cmd_parser()

    # Read PDFs from files
    pdf1 = readers.read_pdf(inps.fname1, verbose=inps.verbose)
    pdf2 = readers.read_pdf(inps.fname2, verbose=inps.verbose)

    # Sample PDFs on same axis
    pdf1, pdf2 = interpolation.interpolate_pdfs(
        [pdf1, pdf2], verbose=inps.verbose
    )

    # Compute K-S statistic
    ks_stat, ks_ndx = compute_ks_statistic(pdf1, pdf2, verbose=True)

    # Plot functions if requested
    if inps.plot == True:
        # Initialize figure and axis
        fig, (inpt_ax, cdf_ax) = plt.subplots(nrows=2)

        # Plot input PDFs
        plotting.plot_pdf_filled(inpt_ax, pdf1, color="r")
        plotting.plot_pdf_filled(inpt_ax, pdf2, color="b")

        # Plot CDFs
        plotting.plot_cdf_line(cdf_ax, pdf1, color="r")
        plotting.plot_cdf_line(cdf_ax, pdf2, color="b")

        # Plot maximum distance
        cdf_ax.plot(
            [pdf1.x[ks_ndx], pdf2.x[ks_ndx]],
            [pdf1.Px[ks_ndx], pdf2.Px[ks_ndx]],
            color="k",
            linewidth=2,
            label="KS stat"
        )

        # Format figure
        inpt_ax.legend()
        inpt_ax.set_title("Inputs")

        cdf_ax.legend()
        cdf_ax.set_title("K-S Analysis")
        fig.tight_layout()

        plt.show()


if __name__ == '__main__':
    main()


# end of file