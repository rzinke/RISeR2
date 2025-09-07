#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Import modules
import argparse

import toml
import matplotlib.pyplot as plt

from riser.probability_functions import readers
from riser import units, plotting


#################### ARGUMENT PARSER ####################
Description = ("Plot multiple PDFs in a stack.")

Examples = """Examples:
view_pdf_stack.py pdfs_config.toml
"""

def create_parser():
    parser = argparse.ArgumentParser(description=Description,
            formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    return parser

def cmd_parser(iargs=None):
    parser = create_parser()

    # Inputs
    input_args = parser.add_argument_group("Inputs")
    input_args.add_argument(dest='pdf_config',
        type=str,
        help="PDFs configuration file.")

    # Units
    unit_args = parser.add_argument_group("Units")
    unit_args.add_argument('--unit-out', dest='unit_out',
        type=str,
        help="Output units.")

    # Outputs
    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument('-v', '--verbose', dest='verbose',
        action='store_true',
        help="Verbose mode.")
    output_args.add_argument('-o', '--outname', dest='outname',
        type=str,
        help="Output file.")
    output_args.add_argument('--no-show', dest='no_show',
        action='store_true',
        help="Forego showing plot.")

    return parser.parse_args(args=iargs)


#################### MAIN ####################
def main():
    # Parse arguments
    inps = cmd_parser()

    # Read config file contents
    with open(inps.pdf_config, 'r') as config_file:
        pdf_specs = toml.load(config_file)

    # Empty dictionaries to store PDFs and metadata
    pdfs = {}
    colors = {}
    priors  = {}

    # Loop through PDFs in config file
    for pdf_name, pdf_spec in pdf_specs.items():
        print(pdf_name)

        # Check that a PDF file is specified
        if "pdf file" not in pdf_spec.keys():
            raise ValueError("A file name must be associated with each PDF")

        # Read PDF from file
        pdf = readers.read_pdf(pdf_spec.get('pdf file'),
                               verbose=inps.verbose)

        # Scale PDF units
        pdfs[pdf_name] = units.scale_pdf_by_units(pdf, inps.unit_out,
                                                  verbose=inps.verbose)

        # Read color if specified
        colors[pdf_name] = pdf_spec.get('color', "black")

        # Read prior if specified
        if pdf_spec.get('prior') is not None:
            prior = readers.read_pdf(pdf_spec.get('prior'),
                                     verbose=inps.verbose)

            # Scale prior units
            priors[pdf_name] = units.scale_pdf_by_units(prior, inps.unit_out,
                                                        verbose=inps.verbose)

    # Initialize figure and axis for input markers
    fig, ax = plt.subplots()

    # Plot PDFs
    plotting.plot_pdf_stack(fig, ax, pdfs, colors=colors, priors=priors)

    # Save figure to file
    if inps.outname is not None:
        # Save figure
        fig.savefig(inps.outname)

    # Show unless specified not to
    if inps.no_show == False:
        plt.show()


if __name__ == '__main__':
    main()


# end of file