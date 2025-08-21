# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import numpy as np

from riser.probability_functions import ProbabilityDensityFunctions as PDFs


####################  ####################
class ConfidenceValues:
    """Class to conveniently hold statistics.
    """
    def __init__(self, confidences:list[(float, float)],
                 pdf_name:str=None, method:str=None):
        """
        """
        # Record confidence level-value pairs
        self.confidences = confidences

        # Record descriptive parameters
        self.pdf_name = pdf_name
        self.method = method

    def __str__(self):
        print_str = "Confidence values:"

        if self.pdf_name is not None:
            print_str += f"\nPDF: {self.pdf_name}"

        if self.method is not None:
            print_str += f" ({self.method})"

        for level, value in self.confidences:
            print_str += f"\n\t{level:.3f}: {value:.3f}"

        return print_str


#################### FUNCTIONS ####################
def compute_interquantile_range(pdf:PDFs.ProbabilityDensityFunction,
                                confidence_levels:list[float],
                                pdf_name:str=None,
                                verbose=False):
    """Compute the interquantile range (IQR) values of a PDF based on the CDF.

    Args    pdf - PDF to analyse
            confidence - list[float], confidence levels
    """
    # Compute the CDF value for each confidence level
    values = [pdf.pit(conf) for conf in confidence_levels]

    # Format in ConfidenceValues object
    confidences = ConfidenceValues(zip(confidence_levels, values),
                                   pdf_name=pdf_name,
                                   method="IQR")

    # Report if requested
    if verbose == True:
        print(confidences)

    return confidences


# end of file