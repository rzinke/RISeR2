#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants
from riser.constants import Psigma
from riser.sampling.filtering import FILTER_TYPES
from riser.probability_functions.analytics import PDF_CONFIDENCE_METRICS


# Import modules
import argparse

import matplotlib.pyplot as plt

from riser.probability_functions import readers as pdf_readers, analytics
from riser.markers import readers as marker_readers
from riser.sampling import sample_statistics
from riser.slip_rates import rate_computation, reporting
from riser import units, plotting


#################### ARGUMENT PARSER ####################
Description = ("Compute the incremental slip rates for a series of dated "
        "markers using Monte Carlo sampling, and applying a criterion.")

Examples = """Examples:
compute_slip_rates_mc.py marker_config.toml -o incr_slip_rates
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

    # Sampling
    sampling_args = parser.add_argument_group("Sampling")
    sampling_args.add_argument('--n-samples', dest='n_samples',
        type=int, default=10000,
        help="Desired number of successful sample combinations. [1 000 000]")

    # Slip rate
    rate_args = parser.add_argument_group("Slip rates")
    rate_args.add_argument('--min-rate', dest='min_rate',
        type=float, default=0,
        help="Minimum slip rate to consider. [0]")
    rate_args.add_argument('--max-rate', dest='max_rate',
        type=float, default=100,
        help="Maximum slip rate to consider. [100]")
    rate_args.add_argument('--dv', dest='dv',
        type=float, default=0.01,
        help="Slip rate step. [0.01]")

    # Smoothing
    smoothing_args = parser.add_argument_group("Smoothing")
    smoothing_args.add_argument('--smoothing-type', dest='smoothing_type',
        type=str, choices=FILTER_TYPES,
        help="Smoothing filter type. [None]")
    smoothing_args.add_argument('--smoothing-width', dest='smoothing_width',
        type=int, default=0,
        help="Smoothing kernel width. [0]")

    # Reporting
    reporting_args = parser.add_argument_group("Reporting")
    reporting_args.add_argument(
        '--confidence-metric', dest='confidence_metric',
        type=str, choices=PDF_CONFIDENCE_METRICS, default="IQR",
        help="Function for computing function confidence. [IQR]")
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
        'marker_plot_type': "rectangle",
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
        'condition': "pass-non-negative-bounded",
        'n_samples': inps.n_samples,
        'pdf_min': inps.min_rate,
        'pdf_max': inps.max_rate,
        'pdf_dx': inps.dv,
        'smoothing_type': inps.smoothing_type,
        'smoothing_width': inps.smoothing_width,
        'max_rate': inps.max_rate,
    }
    (slip_rates,
     age_picks,
     disp_picks,
     rate_picks) = rate_computation.compute_slip_rates_mc(
            **slip_rate_args, verbose=inps.verbose)

    # Save PDFs to file
    for marker_pair, slip_rate in slip_rates.items():
        rate_outname = f"{inps.output_prefix}_{marker_pair}_slip_rate.txt"
        pdf_readers.save_pdf(rate_outname, slip_rate,
                             verbose=inps.verbose)

    # Save picks to file
    reporting.write_picks_to_file(inps.output_prefix, age_picks, disp_picks,
                                  rate_picks, verbose=inps.verbose)

    # Retrieve confidence range function
    conf_fcn = analytics.get_pdf_confidence_function(inps.confidence_metric,
                                                     verbose=inps.verbose)

    # Empty dictionaries for summary statistics
    sample_stats = {}
    pdf_stats = {}
    conf_ranges = {}

    # Loop through marker pairs to compute summary statistics
    for i, (marker_pair, slip_rate) in enumerate(slip_rates.items()):
        # Compute sample statistics
        sample_stats[marker_pair] = \
                sample_statistics.compute_sample_confidence(
                    rate_picks[i,:], inps.confidence_limits,
                    name=marker_pair, unit=slip_rates[marker_pair].unit)
        if inps.verbose == True:
            print(sample_stats[marker_pair])

        # Compute PDF statistics
        pdf_stats[marker_pair] = analytics.PDFstatistics(slip_rate)
        if inps.verbose == True:
            print(pdf_stats[marker_pair])

        # Compute confidence ranges
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
        'sample_statistics': sample_stats,
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