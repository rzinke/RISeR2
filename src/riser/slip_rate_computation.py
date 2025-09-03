# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
from riser.probability_functions import PDF
from riser import variable_operations
from riser.markers import DatedMarker


#################### ANALYTIC COMPUTATION ####################
def compute_slip_rate(marker:DatedMarker, verbose=False, **kwargs) -> PDF:
    """Compute a single slip rate based on a dated displacement marker.

    Args    marker - DatedMarker, displacement-age pair used to calculate
                slip rate
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
    }
    slip_rate = variable_operations.divide_variables(**division_args)

    return slip_rate


def compute_slip_rates_analytical(markers:dict, verbose=False, **kwargs) -> dict:
    """Compute the incremental slip rates between multiple dated displacement
    markers using analytical functions.
    First, compute the difference between each pair of adjacent displacements
    and corresponding pairs of ages to get DeltaD's and DeltaT's.
    Then, compute the slip rate over each increment by dividing the DeltaD by
    the corresponding DeltaT.

    Args    markers - dict, DatedMarkers bounding each interval
    Retrusn slip_rates - dict, incremental slip rates
    """
    return


# end of file