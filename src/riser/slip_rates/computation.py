# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
from riser.probability_functions import PDF
from riser import variable_operations
from riser.slip_rates import DatedMarker


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
        'name': marker.displacement.name,
    }
    slip_rate = variable_operations.divide_variables(**division_args)

    return slip_rate


# end of file