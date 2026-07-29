# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Public API
__all__ = [
    "check_extension",
    "parse_metadata_from_header",
    "read_pdf",
    "read_pdfs",
    "read_calendar_file",
    "save_pdf",
]


# Import modules
import copy
import warnings

import numpy as np

from .. import (
    variable_types,
    units,
)

from .metadata import METADATA_ITEMS
from .probability_density_function import ProbabilityDensityFunction as PDF


#################### CHECKS ####################
def check_extension(fname: str, ext: str):
    """Check that the filename has the appropriate extension.

    Parameters
    ----------
    fname : str
        Filename.
    ext : str
        Filename extension.
    """
    # Get filename extension
    fname_ext = fname.split(".")[-1]

    # Check filename has required extension
    if fname_ext != ext:
        raise ValueError(f"Filename must have extension: '{ext}'")


#################### PDF READERS ####################
def parse_metadata_from_header(
    header_lines: list[str], verbose: bool = False
) -> dict[str, str]:
    """Parse the header of a PDF file.

    Retrieve the metadata pertinent to the PDF. Metadata items correspond to
    those listed in PDF.metadata_items, and are demarkated by the item
    name, a colon, and a space.

    Parameters
    ----------
    header_lines : list[str]
        File header lines.

    Return
    ------
    metadata : dict[str, str]
        Metadata dictionary.
    """
    if verbose:
        print("Reading metadata from header")

    # Empty metadata dictionary
    metadata = {}

    # Loop through header lines
    for line in header_lines:
        # Loop through header items
        for meta_item in METADATA_ITEMS:
            # Determine if header line contains a metadata item
            if line.startswith(f"# {meta_item.capitalize()}"):
                # Strip newline character
                line = line.strip("\n")

                # Split metadata value from key
                meta_value = line.split(": ")[1]

                # Record to metadata dictionary
                metadata[meta_item] = meta_value

                # Report if requested
                if verbose:
                    print(f"{meta_item}: {meta_value}")

    return metadata


def reconcile_metadata(
    user_metadata: dict[str, str],
    file_metadata: dict[str, str],
    verbose: bool = False,
) -> dict[str, str]:
    """Determine metadata value if user and file specifications collide.

    Parameters
    ----------
    user_metadata : dict[str, str]
        User-specified metadata entries.
    file_metadata : dict[str, str]
        File-recorded metadata entries.

    Returns
    -------
    metadata : dict[str, str]
        Reconciled metadata.
    """
    metadata = copy.copy(file_metadata)

    for meta_item in METADATA_ITEMS:
        user_item = user_metadata.get(meta_item)
        file_item = file_metadata.get(meta_item)

        if user_item is not None:
            if file_item is not None:
                warnings.warn(
                    f"User-specified metadata for {meta_item} ({user_item}) "
                    f"differs from metadata in file ({file_item}). "
                    f"Using user-specified value.",
                    stacklevel=2,
                )

            # Overwrite metadata from file
            metadata[meta_item] = user_item

    return metadata


def format_data_line(data_line: str) -> str:
    """Format a line in a PDF text file to be properly parsed.

    Remove leading and trailing spaces and newline characters.
    Ensure the delimiter is a comma.

    Parameters
    ----------
    data_line : str
        Line to format.

    Returns
    -------
    fmtd_line : str
        Formatted data line.
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


def parse_data_lines(
    data_lines: list[str], verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Parse the value-probability density pairs of a PDF file.

    Values x and probability densities px should be recorded as floats.
    One x-px pair per line.

    x's and px's should be delimited by a comma, comma space, space,
    multiple space, or tab.

    Parameters
    ----------
    data_lines : list[str]
        Lines containing x-px pairs.

    Returns
    -------
    x : np.ndarray,
        PDF value array.
    px : np.ndarray
        PDF probability densities.
    """
    if verbose:
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


def read_pdf(
    fname: str,
    normalize_area: bool = True,
    *,
    name: str | None = None,
    variable_type: str | None = None,
    unit: str | None = None,
    verbose: bool = False,
) -> PDF:
    """Read a PDF from a file.

    Parameters
    ----------
    fname : str
        File name.
    normalize_area : bool
        Scale px value to so the area = 1.0.

    Returns
    -------
    PDF: PDF
        Probability density function read from file.
    """
    # Open file and read contents
    with open(fname, "r") as raw_file:
        lines = raw_file.readlines()

    # Remove blank or malformed lines
    lines = [line for line in lines if len(line) > 3]

    # Parse header lines
    header_lines = [line for line in lines if line[0] == "#"]
    if verbose:
        print(f"{len(header_lines)} header lines")

    # Retrieve metadata
    file_metadata = parse_metadata_from_header(header_lines, verbose=verbose)

    # Check metadata from file against user-specified metadata
    user_metadata = {
        "name": name,
        "variable_type": variable_type,
        "unit": unit,
    }
    metadata = reconcile_metadata(
        user_metadata, file_metadata, verbose=verbose
    )

    # Parse data lines
    data_lines = [line for line in lines if line[0] != "#"]

    # Retrieve data
    x, px = parse_data_lines(data_lines, verbose=verbose)

    if verbose:
        print(f"{len(data_lines)} data lines")

    # Instatiate PDF object
    pdf = PDF(x, px, normalize_area=normalize_area, **metadata)

    return pdf


def read_pdfs(
    fnames: list[str], normalize_area: bool = True, verbose: bool = False
) -> list[PDF]:
    """Read multiple PDFs from files.

    Parameters
    ----------
    fnames : list[str]
        File names.
    normalize_area : bool
        Scale px value to so the area = 1.0.

    Returns
    -------
    pdfs : list[PDF]
        List of PDFs.
    """
    if verbose:
        print(f"Reading {len(fnames)} PDF names")

    # Read PDFs
    pdfs = [read_pdf(fname, verbose=verbose) for fname in fnames]

    return pdfs


def read_calendar_file(
    fname: str, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Convert a PDF in calendar years to one in age.

    Read a file (e.g., OxCal output) in which probability densities are
    recorded as a function of calendar year, as opposed to years before
    present (or some reference date).

    Parameters
    ----------
    fname : str
        Name of calendar year file.

    Returns
    -------
    calyr : np.ndarray
        Calendar years.
    calpx : np.ndarray
        Probability density of each year increment.
    metdata : dict[str, str]
        Metadata retrieved from file.
    """
    # Open file and read contents
    with open(fname, 'r') as raw_file:
        lines = raw_file.readlines()

    # Remove blank or malformed lines
    lines = [line for line in lines if len(line) > 3]

    # Parse header lines
    header_lines = [line for line in lines if line[0] == "#"]
    if verbose:
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
def create_header_from_pdf(pdf: PDF) -> str:
    """Create the header of a PDF file.

    Parameters
    ----------
    pdf : PDF
        PDF for which to create a text file header.

    Returns
    -------
    header : str
        Block of header text.
    """
    # Empty header lines
    header_lines = []

    # Loop through header items
    for meta_item in PDF.metadata_items:
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


def pdf_data_to_str(pdf: PDF) -> str:
    """Format the data of a PDF into string format.

    Parameters
    ----------
    pdf : PDF
        PDF to reformat as a string.

    Returns
    -------
    str
        Block of PDF data.
    """
    return [f"{x},{px}\n" for x, px in zip(pdf.x, pdf.px)]


def save_pdf(outname: str, pdf: PDF, verbose: bool = False) -> None:
    """Save a PDF to a file.

    Parameters
    ----------
    outname : str
        Output file name.
    pdf : PDF
        PDF to save.
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
    if verbose:
        print(f"Wrote PDF to file: {outname}")


# end of file