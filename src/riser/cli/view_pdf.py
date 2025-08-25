#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import argparse

import matplotlib.pyplot as plt

from riser.probability_functions.readers import read_pdf


#################### ARGUMENT PARSER ####################
Description = """."""

Examples = """Examples:
"""

def create_parser():
    parser = argparse.ArgumentParser(description=Description,
            formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    return parser

def cmd_parser(iargs=None):
    parser = create_parser()

    input_args = parser.add_argument_group("Inputs")
    input_args.add_argument(dest='fname',
        type=str,
        help="PDF file name.")

    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument('-v', '--verbose', dest='verbose',
        action='store_true',
        help="Verbose mode.")

    return parser.parse_args(args=iargs)


#################### MAIN ####################
def main():
    # Parse arguments
    inps = cmd_parser()


    # Read PDF from file
    pdf = read_pdf(inps.fname, verbose=inps.verbose)


    # Initialize figure and axis
    fig, ax = plt.subplots()

    # Plot PDF
    ax.plot(pdf.x, pdf.px, color="k", linewidth=3)


    plt.show()


if __name__ == '__main__':
    main()


# end of file