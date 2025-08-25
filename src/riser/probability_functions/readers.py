# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Constants
from riser.probability_functions.PDFs import PDF_METADATA_ITEMS, PDF_TYPES


# Import modules
import numpy as np

from riser.probability_functions import PDFs


#################### PDF READERS ####################
def parse_metadata_from_header(header_lines:list[str], verbose=False) -> dict:
    """Parse the header of a PDF file.
    Retrieve the metadata pertinent to the PDF. Metadata items correspond to
    those listed in PDFs.PDF_METADATA_ITEMS, and are demarkated by the item
    name, a colon, and a space.

    Args    header_lines - list[str], file header lines
    Returns metadata - dict, metadata dictionary
    """
    if verbose == True:
        print("Reading metadata from header")

    # Empty metadata dictionary
    metadata = {}

    # Loop through header lines
    for line in header_lines:
        # Loop through header items
        for meta_item in PDF_METADATA_ITEMS:
            # Determine if header line contains a metadata item
            if line.startswith(f"# {meta_item}"):
                # Strip newline character
                line = line.strip("\n")

                # Split metadata value from key
                meta_value = line.split(": ")[1]

                # Record to metadata dictionary
                metadata[meta_item] = meta_value

                # Report if requested
                if verbose == True:
                    print(f"{meta_item}: {meta_value}")

    return metadata


def format_data_line(data_line:str) -> str:
    """Remove leading and trailing spaces and newline characters.
    Ensure the delimiter is a comma.

    Args    data_line - str, line to format
    Returns fmtd_line - str, formatted data line
    """
    # Strip newline character
    data_line = data_line.strip("\n")

    # Remove leading and trailing spaces
    data_line = data_line.lstrip()
    data_line = data_line.rstrip()

    # Change multiple spaces to single space
    while "  " in " ":
        line = line.replace("  ", " ")

    # Ensure delimiter is comma
    data_line = data_line.replace(" ", ",")
    data_line = data_line.replace("\t", ",")
    data_line = data_line.replace(", ", ",")

    return data_line


def parse_data_lines(data_lines:list[str], verbose=False) -> \
        (np.ndarray, np.ndarray):
    """Parse the value-probability density pairs of a PDF file.
    Values x and probability densities px should be recorded as floats.
    One x-px pair per line.
    x's and px's should be delimited by a comma, comma space, space,
    multiple space, or tab.

    Args    data_lines - list[str], lines containing x-px pairs
    Returns x - np.ndarray, PDF values
            px - np.ndarray, PDF probability densities
    """
    if verbose == True:
        print("Reading data from file")

    # Empty lists for x, px
    x = []
    px = []

    # Loop through lines
    for line in data_lines:
        # Format data line
        line = format_data_line(line)

        # Parse x, px from line
        line_x, line_px = line.split(",")

        # Record to list
        x.append(float(line_x))
        px.append(float(line_px))

    # Convert lists to numpy arrays
    x = np.array(x)
    px = np.array(px)

    return x, px


def read_pdf(fname:str, pdf_type:str=None, normalize_area:bool=True,
        verbose=False) -> PDFs.ProbabilityDensityFunction:
    """Read a PDF from a file.

    Args    fname - str, file name
            pdf_type - str, type
            normalize_area - bool, scale px value to so the area = 1.0

    Returns PDF - ProbabilityDensityFunction
    """
    # Check if pdf_type if valid
    pdf_types = [pdftype.lower() for pdftype in PDF_TYPES]

    if pdf_type is not None and pdf_type.lower() not in pdf_types:
        raise ValueError(f"PDF type {pdf_type} not valid")

    # Open file and read contents
    with open(fname, 'r') as raw_file:
        lines = raw_file.readlines()

    # Remove blank or malformed lines
    lines = [line for line in lines if len(line) > 3]

    # Parse header lines
    header_lines = [line for line in lines if line[0] == "#"]
    if verbose == True:
        print(f"{len(header_lines)} header lines")

    # Retrieve metadata
    metadata = parse_metadata_from_header(header_lines, verbose=verbose)

    # Parse data lines
    data_lines = [line for line in lines if line[0] != "#"]

    # Retrieve data
    x, px = parse_data_lines(data_lines, verbose=verbose)

    if verbose == True:
        print(f"{len(data_lines)} data lines")

    # Determine appropriate PDF class to use
    if pdf_type is None:
        # Generic
        PDF = PDFs.ProbabilityDensityFunction

    elif pdf_type.lower() == "age":
        PDF = PDFs.Age

    elif pdf_type.lower() == "displacement":
        PDF = PDFs.Displacement

    elif pdf_type.lower() == "sliprate":
        PDF = PDFs.SlipRate

    # Instatiate PDF object
    pdf = PDF(x, px, normalize_area=normalize_area, **metadata)

    return pdf


#################### PDF SAVERS ####################


# end of file