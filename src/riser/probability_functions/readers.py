# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Constants
from riser.probability_functions.ProbabilityDensityFunction import \
    PDF_METADATA_ITEMS


# Import modules
import numpy as np

from riser.probability_functions import PDF


#################### CHECKS ####################
def check_extension(fname:str, ext:str):
    """Check that the filename has the appropriate extension.

    Args    fname - str, filename
            ext - str, required extension
    """
    # Get filename extension
    fname_ext = fname.split(".")[-1]

    # Check filename has required extension
    if fname_ext != ext:
        raise ValueError(f"Filename must have extension: {ext}")


#################### PDF READERS ####################
def parse_metadata_from_header(header_lines:list[str], verbose=False) -> dict:
    """Parse the header of a PDF file.
    Retrieve the metadata pertinent to the PDF. Metadata items correspond to
    those listed in PDF_METADATA_ITEMS, and are demarkated by the item
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
            if line.startswith(f"# {meta_item.capitalize()}"):
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
    while "  " in data_line:
        data_line = data_line.replace("  ", " ")

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


def read_pdf(fname:str, normalize_area:bool=True, verbose=False) -> PDF:
    """Read a PDF from a file.

    Args    fname - str, file name
            normalize_area - bool, scale px value to so the area = 1.0
    Returns PDF - ProbabilityDensityFunction
    """
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

    # Instatiate PDF object
    pdf = PDF(x, px, normalize_area=normalize_area, **metadata)

    return pdf


def read_pdfs(fnames:list[str], normalize_area:bool=True,
              verbose=False) -> list[PDF]:
    """Read multiple PDFs from files.

    Args    fnames - list[str], file names
            normalize_area - bool, scale px value to so the area = 1.0
    Returns pdfs - list[PDF], list of PDFs
    """
    if verbose == True:
        print(f"Reading {len(fnames)} PDF names")

    # Read PDFs
    pdfs = [read_pdf(fname, verbose=verbose) for fname in fnames]

    return pdfs


def read_calendar_file(fname:str, verbose=False):
    """Read a file (e.g., OxCal output) in which probability densities are
    recorded as a function of calendar year, as opposed to years before
    present (or some reference date).

    Args    fname - str, name of calendar year file
    Returns calyr - np.ndarray, calendar years
            calpx - np.ndarray, probability density of each year increment
            metdata - dict, metadata retrieved from file
    """
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
    data_lines = [line for line in lines if line[0] != "#" and len(line) > 1]

    # Empty lists for x, px
    calyr = []
    calpx = []

    # Loop through lines
    for line in data_lines:
        # Format data line
        line = format_data_line(line)

        # Parse x, px from line
        line_calyr, line_px = line.split(",")

        # Record to list
        calyr.append(float(line_calyr))
        calpx.append(float(line_px))

    # Convert lists to numpy arrays
    calyr = np.array(calyr)
    calpx = np.array(calpx)

    return calyr, calpx, metadata


#################### PDF SAVERS ####################
def create_header_from_pdf(pdf:PDF) -> str:
    """Create the header of a PDF file.

    Args    pdf - PDF
    Returns header - str, block of header text
    """
    # Empty header lines
    header_lines = []

    # Loop through header items
    for meta_item in PDF_METADATA_ITEMS:
        # Determine if PDF contains a metadata item
        if hasattr(pdf, meta_item) and getattr(pdf, meta_item) is not None:
            # Get metadata item
            meta_value = getattr(pdf, meta_item)

            # Format metadata item in header string
            header_str = f"# {meta_item.capitalize()}: {meta_value}\n"

            # Append to list
            header_lines.append(header_str)

    # Format list into text block
    header = "".join(header_lines)

    return header


def pdf_data_to_str(pdf:PDF) -> str:
    """Format the data of a PDF into string format.

    Args    pdf - PDF
    Returns block of PDF data
    """
    return [f"{x},{px}\n" for x, px in zip(pdf.x, pdf.px)]


def save_pdf(outname:str, pdf:PDF, verbose=False):
    """Save a PDF to a file.

    Args    outname - str, output file name
            pdf - PDF to save
    """
    # Check that outname is a text file
    check_extension(outname, "txt")

    # Create header
    header = create_header_from_pdf(pdf)

    # Format data to string
    data = pdf_data_to_str(pdf)

    # Write to file
    with open(outname, 'w') as outfile:
        # Write header
        for header_line in header:
            outfile.write(header_line)

        # Write data
        for datum in data:
            outfile.write(datum)

    # Report if requested
    if verbose == True:
        print(f"Wrote PDF to file: {outname}")


# end of file