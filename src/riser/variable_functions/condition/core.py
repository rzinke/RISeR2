# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

"""
These functions condition a random variable by multiplying by a weight and
then renormalizing.
"""


# Public API
__all__ = [
    "weigh",
    "condition",
]


# Import modules
import numpy as np

from riser import (
    integration,
    probability_functions as PDFs,
)


#################### CONDITIONING FUNCTIONS ####################
def weigh(
    prior: PDFs.weight_functions.WeightFunction,
    weight: PDFs.weight_functions.WeightFunction,
) -> PDFs.weight_functions.WeightFunction:
    """Primitive to reshape a prior by a weight function.
    """
    return PDFs.weight_functions.WeightFunction(
        x=prior.x, wx=prior.wx * weight.wx
    )

def condition(
    prior: PDFs.PDF | PDFs.weight_functions.WeightFunction,
    weight: PDFs.PDF | PDFs.weight_functions.WeightFunction,
    **metadata,
) -> tuple[PDFs.PDF, float]:
    """Weight a prior distribution, according to

        posterior(x) ~ prior(x) . weighting(x)

    The resulting posterior distribution is then normalized to unit area and
    when it is formed into a PDF.

    The unnormalized area of the resulting distribution represents the fraction
    of the prior distribution that remains after weighting is applied. 
    The specific interpretation depends on the weight chosen by the calling 
    function.

    Metadata are preserved unless explicitly overridden with the `name`
    parameter.

    Parameters
    ----------
    prior : PDF
        Prior to weight.
    weight : np.ndarray
        Weighting array.

    Returns
    -------
    posterior : PDF
        Weighted prior.
    area : float
        Area of the conditioned distribution before normalization.
    """
    # Check the size of the weighting array equals the size of the prior
    if len(weight) != len(prior):
        raise ValueError(
            f"The size of the weighting array ({len(weight)}) must equal the "
            f"size of the prior domain ({len(prior)})"
        )

    # Extract weights from prior
    prior_weight = (
        PDFs.weight_functions.WeightFunction.from_pdf(prior)
        if isinstance(prior, PDFs.PDF) else prior
    )

    weight_function = (
        PDFs.weight_functions.WeightFunction.from_pdf(weight)
        if isinstance(weight, PDFs.PDF) else weight
    )

    # Weight the probability densities of the prior
    post_weight = weigh(prior_weight, weight_function)

    # Format weighted prior as PDF (scaling carried out by PDF.__init__)
    posterior = post_weight.normalize(**metadata)

    return posterior, post_weight.area()

# end of file
