# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants
SAMPLING_CRITERIA = [
    'non-negative',
]

# Import modules
import numpy as np
from tqdm import tqdm


#################### SAMPLING CRITERIA ####################
def pass_all(ages, displacements, **kwargs) -> bool:
    """Trivial condition for which all random samples will be passed.
    (Allows negative slip rates).

    Args    ages - np.ndarray, sampled age values
            displacements - np.ndarray, sampled displacement values
            kwargs are ignored
    Returns bool, pass (True) or fail (False)
    """
    return True


def pass_nonnegative(ages, displacements, **kwargs) -> bool:
    """Pass sample combinations that result in non-negative slip rates.
    Args    kwargs are ignored
    """
    # Compute age, displacement deltas
    age_diffs = np.diff(ages)
    disp_diffs = np.diff(displacements)

    # Check condition
    if all([age_diffs.min() > 0, disp_diffs.min() >= 0]):
        return True
    else:
        return False


def pass_nonnegative_bounded(ages, displacements, **kwargs) -> bool:
    """Pass sample combinations that result in slip rates that are non-negative
    and less than the specified maximum value.
    Args    kwargs must contain the maximum allowable slip rate as max_rate
    """
    # Compute age, displacement deltas
    age_diffs = np.diff(ages)
    disp_diffs = np.diff(displacements)

    # Compute incremental slip rates
    slip_rates = disp_diffs / age_diffs

    # Check condition
    if all([age_diffs.min() > 0, disp_diffs.min() >= 0,
            slip_rates.max() <= kwargs.get('max_rate', np.inf)]):
        return True
    else:
        return False


def get_sample_criterion(criterion_name:str, verbose=False):
    """Retrieve a sample criterion by name.
    """
    # Format criterion name
    formatted_name = criterion_name.lower()
    formatted_name = formatted_name.replace("-", "")
    formatted_name = formatted_name.replace("_", "")
    formatted_name = formatted_name.lstrip("pass")

    # Return criterion object
    if formatted_name in ["all"]:
        return pass_all
    elif formatted_name in ["nonnegative"]:
        return pass_nonnegative
    elif formatted_name in ["nonnegativebounded"]:
        return pass_nonnegative_bounded
    else:
        raise ValueError(f"Criterion name {criterion_name} not recognized.")


#################### MONTE CARLO SAMPLING ####################
def sample_monte_carlo(markers:dict, criterion:"function",
                       n_samples:int=10000, seed_val:int=0,
                       hard_stop:int=1000000000, verbose=False, **kwargs) -> \
                       (np.ndarray, np.ndarray):
    """This method uses the inverse transform sampling method to randomly
    sample the displacement and age PDFs constraining a DatedMarker.
    The random samples are checked against a criterion, e.g., "no negative
    slip rates".

    Args    markers - dict, DatedMarkers between which to compute incremental
                slip rates
            criterion - function to evaluate validity of samples
            n_samples - int, number of valid samples to achieve
            max_rate - float, maximum slip rate to consider
            seed_val - int, random number generator seed value
    """
    if verbose == True:
        print(f"Initializing MC sampling for {n_samples} samples")

    # Number of markers
    m_markers = len(markers)

    # Initialize sample arrays
    age_samps = np.empty(m_markers)
    disp_samps = np.empty(m_markers)

    # Initialize output arrays
    age_picks = np.empty((m_markers, n_samples))
    disp_picks = np.empty((m_markers, n_samples))

    # Seed random number generator
    np.random.seed(seed_val)

    # Initialize counter
    successes = 0
    tossed = 0

    # Initialize progress bar
    pbar = tqdm(total=n_samples)

    # Loop through samples until enough successess collected
    for i in range(hard_stop):
        # Generate random numbers over interval [0.0, 1.0)
        r_ages = np.random.rand(m_markers)
        r_disps = np.random.rand(m_markers)

        # Transform random numbers to age, displacement values
        for j, name in enumerate(markers.keys()):
            age_samps[j] = markers[name].age.pit(r_ages[j])
            disp_samps[j] = markers[name].displacement.pit(r_disps[j])

        # Check samples against condition
        if criterion(age_samps, disp_samps, **kwargs) == True:
            # Condition met, record valid samples
            age_picks[:,successes] = age_samps
            disp_picks[:,successes] = disp_samps

            # Update counter
            successes += 1

            # Update progress bar
            pbar.update()

            # Break loop if desired successes achieved
            if successes == n_samples:
                break
        else:
            # Condition not met
            tossed += 1

    # Close progress bar
    pbar.close()

    # Report if requested
    if verbose == True:
        print(f"MC sampling finished with:"
              f"\n\t{successes} successes"
              f"\n\t{tossed} tossed")

    # Report if hard stop met
    if i == (hard_stop - 1):
        print(f"WARNING: Only {successes} valid samples found before reaching "
              f"trial limit")

    return age_picks, disp_picks


# end of file