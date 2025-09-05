#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants
from riser.constants import Psigma
from riser.probability_functions.analytics import CONFIDENCE_FUNCTIONS


# Import modules
import argparse

import numpy as np
import matplotlib.pyplot as plt

from riser.probability_functions import readers as pdf_readers, analytics
from riser.markers import readers as marker_readers
from riser.slip_rates import reporting
from riser import units, plotting


#################### ARGUMENT PARSER ####################
Description = ("Plot the displacement-time history constrained by a set of "
               "markers.")

Examples = """Examples:
view_displacement_time_history.py marker_config.toml
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
    input_args.add_argument('--marker-type', dest='marker_plot_type',
        type=str, default="whisker",
        help="Marker type. [whisker]")
    input_args.add_argument('--show-marginals', dest='show_marginals',
        action='store_true',
        help="Plot displacement and age marginal distributions.")

    # Units
    unit_args = parser.add_argument_group("Units")
    unit_args.add_argument('--age-unit-out', dest='age_unit_out',
        type=str,
        help="Output age units.")
    unit_args.add_argument(
        '--displacement-unit-out', dest='displacement_unit_out',
        type=str,
        help="Output displacement units.")

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

    # Read markers
    markers = marker_readers.read_markers_from_config(inps.marker_config,
                                                      verbose=inps.verbose)

    # Scale input units to output units before further action
    for marker in markers.values():
        marker.age = units.scale_pdf_by_units(marker.age, inps.age_unit_out)
        marker.displacement = units.scale_pdf_by_units(
                marker.displacement, inps.displacement_unit_out)

    # Initialize figure and axis for input markers
    if inps.show_marginals == False:
        # Standard single-axis figure
        fig, marker_ax = plt.subplots()

    else:
        # Three-part figure with marginal distributions
        fig = plt.figure()
        marker_ax = fig.add_subplot(position=(0.32, 0.32, 0.60, 0.60))
        age_ax = fig.add_subplot(position=(0.32, 0.05, 0.60, 0.15))
        disp_ax = fig.add_subplot(position=(0.05, 0.32, 0.15, 0.60))

    # Plot markers
    plot_args = {
        'fig': fig,
        'ax': marker_ax,
        'markers': markers,
        'marker_plot_type': inps.marker_plot_type,
        'label': True,
    }
    plotting.plot_markers(**plot_args)

    # Plot marginal distributions
    for name, marker in markers.items():
        # Plot age
        age_ax.fill_between(marker.age.x, marker.age.px,
                            color="dimgrey", alpha=0.3)

        # Plot displacement
        disp_ax.fill(marker.displacement.px, marker.displacement.x,
                             color="dimgrey", alpha=0.3)

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