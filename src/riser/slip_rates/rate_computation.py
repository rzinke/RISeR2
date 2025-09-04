# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
from datetime import datetime

from riser.probability_functions import PDF, interpolation, analytics
from riser import variable_operations
from riser.markers import DatedMarker


#################### ANALYTIC COMPUTATION ####################
def compute_slip_rate(marker:DatedMarker, verbose=False, **kwargs) -> PDF:
    """Compute a single slip rate based on a dated displacement marker.

    Args    marker - DatedMarker, displacement-age pair used to calculate
                slip rate
            kwargs - dict, keyword agruments for divide_variables
    Returns slip_rate - PDF
    """
    if verbose == True:
        print("Computing slip rate")

    # Divide displacement by age
    division_args = {
        'numerator': marker.displacement,
        'denominator': marker.age,
        'max_quotient': kwargs.get('max_quotient', 100.),
        'dq': kwargs.get('dq', 0.01),
        'name': marker.name,
        'variable_type': "slip rate",
    }
    slip_rate = variable_operations.divide_variables(**division_args)

    return slip_rate


def compute_slip_rates_analytical(markers:dict, verbose=False, **kwargs) -> \
        dict:
    """Compute the incremental slip rates between multiple dated displacement
    markers using analytical functions.
    First, compute the difference between each pair of adjacent displacements
    and corresponding pairs of ages to get DeltaD's and DeltaT's.
    Then, compute the slip rate over each increment by dividing the DeltaD by
    the corresponding DeltaT.

    Args    markers - dict, DatedMarkers bounding each interval
            kwargs - dict, keyword arguments for:
                subtract_variables
                divide_variables
    Retrusn slip_rates - dict, incremental slip rates
    """
    # Marker parameters
    n_markers = len(markers)
    marker_names = [*markers.keys()]

    # Number of slip rates
    n_rates = n_markers - 1

    if verbose == True:
        print(f"Computing {n_rates} incremental slip rates")

    # Empty dictionary to store slip rates
    slip_rates = {}

    # Loop through marker pairs
    for i in range(n_rates):
        # Younger marker
        younger_name = marker_names[i]
        younger_marker = markers[younger_name]

        # Older marker
        older_name = marker_names[i+1]
        older_marker = markers[older_name]

        # Interpolate ages on same axis
        (younger_age,
         older_age) = interpolation.interpolate_pdfs(
                [younger_marker.age, older_marker.age])

        # Compute age difference
        age_args = {
            'pdf1': older_age,
            'pdf2': younger_age,
            'limit_positive': kwargs.get('limit_positive', False)
        }
        DeltaT = variable_operations.subtract_variables(**age_args)

        # Interpolate displacements on same axis
        (younger_displacement,
         older_displacement) = interpolation.interpolate_pdfs(
                [younger_marker.displacement, older_marker.displacement])

        # Compute displacement difference
        disp_args = {
            'pdf1': older_displacement,
            'pdf2': younger_displacement,
            'limit_positive': kwargs.get('limit_positive', False)
        }
        DeltaU = variable_operations.subtract_variables(**disp_args)

        # Formulate incremental slip rate name
        rate_name = f"{older_marker.name}-{younger_marker.name}"

        # Divide displacement by age
        rate_args = {
            'numerator': DeltaU,
            'denominator': DeltaT,
            'max_quotient': kwargs.get('max_quotient', 100.),
            'dq': kwargs.get('dq', 0.01),
            'name': rate_name,
            'variable_type': "slip rate",
        }
        slip_rate = variable_operations.divide_variables(**rate_args)

        # Report if requested
        if verbose == True:
            print(slip_rate)
            print(f"Mean: {analytics.pdf_mean(slip_rate):.3f} "
                  f"+- {analytics.pdf_std(slip_rate):.3f}")

        # Record to slip rate dictionary
        slip_rates[rate_name] = slip_rate

    return slip_rates


# end of file