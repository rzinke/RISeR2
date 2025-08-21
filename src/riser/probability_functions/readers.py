# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
from riser.probability_functions import ProbabilityDensityFunctions as PDFs


#################### PDF READERS ####################
def read_pdf(fname:str, pdf_type:str=None,
        verbose=False) -> PDFs.ProbabilityDensityFunction:
    """Read a PDF from a file.

    Args    fname - str, file name
            pdf_type - str, type

    Returns PDF - ProbabilityDensityFunction
    """
    # Open file and read contents
    with open(fname, 'r') as raw_file:
        lines = raw_file.readlines()

    # Parse header and data lines
    header_lines = [line for line in lines if line[0] == "#"]
    data_lines = [line for line in lines if line[0] != "#"]

    if verbose == True:
        print(f"{len(header_lines)} header lines")

    return


#################### PDF SAVERS ####################


# end of file