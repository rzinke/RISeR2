# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.

# Public API
__all__ = [
    "interpolate_variable_pairs",
]


# Import modules
import copy
from collections.abc import Mapping

from .. import probability_functions as PDFs
from .variable_pair import VariablePair


#################### RESAMPLING/INTERPOLATION ####################
def interpolate_variable_pairs(
    pairs: Mapping[str, VariablePair],
    verbose: bool = False,
) -> Mapping[str, VariablePair]:
    """Resample the PDFs of multiple VariablePairs along common value arrays.

    Common value arrays are found for `pdf1` in all pairs, and `pdf2` in all
    pairs.

    This function is analogous to
    `probability_functions/interpolation.interpolate_pdfs`.

    Parameters
    ----------
    pairs : dict[str, VariablePair]
        Variable pairs for which to interpolate the PDFs.

    Returns
    -------
    interp_pairs : dict[str, VariablePair]
        Variable pairs with interpolated PDFs.
    """
    if verbose:
        print(f"Interpolating {len(pairs)} variable pair PDFs onto common axes")

    # Pair names
    pair_names = [*pairs.keys()]

    # Copy dict to avoid overwriting original
    interp_pairs = copy.deepcopy(pairs)

    # Interpolate pdf1's onto common value array
    interp_pdf1s = PDFs.interpolation.interpolate_pdfs(
        [pairs[pair_name].pdf1 for pair_name in pair_names]
    )

    # Interpolate pdf2's onto common value array
    interp_pdf2s = PDFs.interpolation.interpolate_pdfs(
        [pairs[pair_name].pdf2 for pair_name in pair_names]
    )

    # Write interpolated PDFs to pairs
    for i, pair_name in enumerate(pair_names):
        interp_pairs[pair_name].pdf1 = interp_pdf1s[i]
        interp_pairs[pair_name].pdf2 = interp_pdf2s[i]

    return interp_pairs


# end of file
