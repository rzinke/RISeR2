#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import os
import argparse

import matplotlib.pyplot as plt

from riser.markers import readers as marker_readers
from riser.slip_rate_computation import compute_slip_rate
from riser.probability_functions import readers as pdf_readers
from riser import units, plotting


#################### ARGUMENT PARSER ####################
Description = """Compute the slip rate for a single marker by dividing feature displacement by age, using the analytical formulation."""

Examples = """Examples:
compute_slip_rate.py marker_config.toml -o slip_rate
"""

def create_parser():
    parser = argparse.ArgumentParser(description=Description,
            formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    return parser

def cmd_parser(iargs=None):
    parser = create_parser()

    # Inputs
    input_args = parser.add_argument_group("Inputs")
    input_args.add_argument(dest='marker_config',
        type=str,
        help="Dated displacement marker configuration file.")

    # Units
    unit_args = parser.add_argument_group("Units")
    unit_args.add_argument('--age-unit-out', dest='age_unit_out',
        type=str,
        help="Output age units.")
    unit_args.add_argument(
        '--displacement-unit-out', dest='displacement_unit_out',
        type=str,
        help="Output displacement units.")

    # Slip rate
    rate_args = parser.add_argument_group("Slip rates")
    rate_args.add_argument('--max-rate', dest='max_rate',
        type=float, default=100,
        help="Maximum slip rate to consider. [100]")
    rate_args.add_argument('--dv', dest='dv',
        type=float, default=0.01,
        help="Slip rate step. [0.01]")

    # Outputs
    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument('-o', '--outfldr', dest='outfldr',
        type=str, required=True,
        help="Output folder.")
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

    # Read markers
    markers = marker_readers.read_markers_from_config(inps.marker_config,
                                                      verbose=inps.verbose)

    # Check that only one marker is specified
    if len(markers) > 1:
        raise Exception("Only one marker can be specified")

    # Use only first marker
    marker = [*markers.values()][0]

    # Scale input units to output units
    marker.age = units.scale_pdf_by_units(marker.age, inps.age_unit_out)

    marker.displacement = units.scale_pdf_by_units(
            marker.displacement, inps.displacement_unit_out)

    # Compute slip rate
    slip_rate_args = {
        'marker': marker,
        'max_quotient': inps.max_rate,
        'dq': inps.dv,
    }
    slip_rate = compute_slip_rate(**slip_rate_args, verbose=inps.verbose)

    # Formulate slip rate output file name
    slip_rate_outname = os.path.join(inps.outfldr,
                                     f"{inps.outfldr}_slip_rate.txt")

    # Save to file
    pdf_readers.save_pdf(slip_rate_outname, slip_rate, verbose=inps.verbose)

    # Plot if requested
    if inps.plot == True:
        # Initialize figure and axis for input marker
        marker_fig, marker_ax = plt.subplots()

        # Plot marker
        plotting.plot_marker_whisker(marker_fig, marker_ax, marker, label=True)

        # Format marker fig
        plotting.set_origin_zero(marker_ax)
        plotting.format_marker_plot(marker_fig, marker_ax, marker)

        # Save marker fig
        marker_fig_outname = os.path.join(inps.outfldr,
                                          f"{inps.outfldr}_marker.png")
        marker_fig.savefig(marker_fig_outname)

        # Initialize figure and axis for slip rate PDF
        rate_fig, rate_ax = plt.subplots()

        # Plot slip rate PDF
        plotting.plot_pdf_labeled(rate_fig, rate_ax, slip_rate)

        # Format slip rate figure
        rate_fig.tight_layout()

        # Save slip rate figure
        rate_fig_outname = os.path.join(inps.outfldr,
                                        f"{inps.outfldr}_slip_rate.png")
        rate_fig.savefig(rate_fig_outname)

        plt.show()


if __name__ == '__main__':
    main()


# end of file