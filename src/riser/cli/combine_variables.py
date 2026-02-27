#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants


# Import modules
import argparse

import matplotlib.pyplot as plt

import riser.probability_functions.readers as readers
import riser.probability_functions.interpolation as interpolation
import riser.variable_operations as var_ops
import riser.plotting as plotting


#################### ARGUMENT PARSER ####################
Description = "Compute the joint PDF of two or more functions."

Examples = """Examples:
combine_variables.py pdf1.txt pdf2.txt -o joint_pdf.txt
combine_variables.py pdf1.txt pdf2.txt pdf3.txt -o joint_pdf.txt
"""

def create_parser():
    parser = argparse.ArgumentParser(description=Description,
            formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    return parser

def cmd_parser(iargs=None):
    parser = create_parser()

    input_args = parser.add_argument_group("Inputs")
    input_args.add_argument(dest='fnames',
        type=str, nargs='+',
        help="PDF file names.")

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


#################### MAIN ####################
def main():
    # Parse arguments
    inps = cmd_parser()

    # Read PDFs from files
    pdfs = readers.read_pdfs(inps.fnames)

    # Sample PDFs on same axis
    pdfs = interpolation.interpolate_pdfs(pdfs, verbose=inps.verbose)

    # Compute joint PDF
    joint_pdf = var_ops.combine_variables(pdfs, verbose=inps.verbose)

    # Save to file
    readers.save_pdf(inps.outname, joint_pdf, verbose=inps.verbose)

    # Plot function if requested
    if inps.plot == True:
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


if __name__ == '__main__':
    main()


# end of file