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

import matplotlib.pyplot as plt

from riser.probability_functions import readers as pdf_readers, analytics
from riser.markers import readers as marker_readers
from riser.slip_rates import rate_computation, reporting
from riser import units, plotting


#################### ARGUMENT PARSER ####################
Description = ("Compute the incremental slip rates for a series of dated "
        "markers by computing the deltas between adjacent pairs of "
        "displacements and ages, and dividing displacement by age, using the "
        "analytical formulation.")

Examples = """Examples:
compute_slip_rates.py marker_config.toml -o incr_slip_rates
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
    rate_args.add_argument('--limit-positive', dest='limit_positive',
        action='store_true',
        help="Enforce the condition that values are >= to 0.")
    rate_args.add_argument('--max-rate', dest='max_rate',
        type=float, default=100,
        help="Maximum slip rate to consider. [100]")
    rate_args.add_argument('--dv', dest='dv',
        type=float, default=0.01,
        help="Slip rate step. [0.01]")

    # Reporting
    reporting_args = parser.add_argument_group("Reporting")
    reporting_args.add_argument(
        '--confidence-method', dest='confidence_method',
        type=str, choices=CONFIDENCE_FUNCTIONS, default="HPD",
        help="Function for computing function confidence. [HPD]")
    reporting_args.add_argument(
        '--confidence-limits', dest='confidence_limits',
        type=float, default=Psigma['1'],
        help="Confidence level. [0.682]")

    # Outputs
    output_args = parser.add_argument_group("Outputs")
    output_args.add_argument('-o', '--output-prefix', dest='output_prefix',
        type=str, required=True,
        help="Output prefix as <prefix> or <folder>/<prefix>.")
    output_args.add_argument('-v', '--verbose', dest='verbose',
        action='store_true',
        help="Verbose mode.")
    output_args.add_argument('-p', '--plot', dest='plot',
        action='store_true',
        help="Show results plots. Figures will be generated and saved "
             "whether plot flag is raised.")

    return parser.parse_args(args=iargs)


#################### MAIN ####################
def main():
    # Parse arguments
    inps = cmd_parser()

    # Establish output directory
    reporting.establish_output_dir(inps.output_prefix, verbose=inps.verbose)

    # Read markers
    markers = marker_readers.read_markers_from_config(inps.marker_config,
                                                      verbose=inps.verbose)

    # Check that multiple markers are specified
    if len(markers) < 2:
        raise Exception("Multiple markers must be specified")

    # Initialize figure and axis for input markers
    marker_fig, marker_ax = plt.subplots()

    # Plot markers
    plot_args = {
        'fig': marker_fig,
        'ax': marker_ax,
        'markers': markers,
        'marker_plot_type': "whisker",
        'label': True,
    }
    plotting.plot_markers(**plot_args)

    # Scale input units to output units
    for marker in markers.values():
        marker.age = units.scale_pdf_by_units(marker.age, inps.age_unit_out)
        marker.displacement = units.scale_pdf_by_units(
                marker.displacement, inps.displacement_unit_out)

    # Save marker fig
    reporting.save_marker_fig(inps.output_prefix, marker_fig,
                              verbose=inps.verbose)

    # Compute slip rates
    slip_rate_args = {
        'markers': markers,
        'limit_positive': inps.limit_positive,
        'max_quotient': inps.max_rate,
        'dq': inps.dv,
    }
    slip_rates = rate_computation.compute_slip_rates_analytical(
            **slip_rate_args, verbose=inps.verbose)

    # Save PDFs to file
    for marker_pair, slip_rate in slip_rates.items():
        rate_outname = f"{inps.output_prefix}_{marker_pair}_slip_rate.txt"
        pdf_readers.save_pdf(rate_outname, slip_rate,
                             verbose=inps.verbose)

    # Compute PDF statistics
    pdf_stats = {}
    for marker_pair, slip_rate in slip_rates.items():
        pdf_stats[marker_pair] = analytics.PDFstatistics(slip_rate)
        if inps.verbose == True:
            print(pdf_stats[marker_pair])

    # Retrieve confidence range function
    conf_fcn = analytics.get_confidence_function(inps.confidence_method,
                                                 verbose=inps.verbose)

    # Compute confidence ranges
    conf_ranges = {}
    for marker_pair, slip_rate in slip_rates.items():
        conf_ranges[marker_pair] = conf_fcn(slip_rate, inps.confidence_limits)
        if inps.verbose == True:
            print(conf_ranges[marker_pair])

    # Initialize figure and axis for slip rate PDF
    rate_fig, rate_ax = plt.subplots()

    # Plot slip rate PDFs
    plotting.plot_pdf_stack(rate_fig, rate_ax, slip_rates,
                            conf_ranges=conf_ranges)

    # Save slip rate figure
    reporting.save_slip_rate_fig(inps.output_prefix, rate_fig,
                                 verbose=inps.verbose)

    # Save slip rate report to file
    report_args = {
        'output_prefix': inps.output_prefix,
        'formulation': "analytical",
        'slip_rates': slip_rates,
        'pdf_statistics': pdf_stats,
        'confidence_ranges': conf_ranges,
    }
    reporting.write_slip_rates_report(**report_args, verbose=inps.verbose)

    # Plot if requested
    if inps.plot == True:
        plt.show()


if __name__ == '__main__':
    main()


# end of file