# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Public API
__all__ = [
    "VariablePair",
]


# Import modules
import copy

from .. import probability_functions as PDFs


#################### DATED MARKER ####################
class VariablePair:
    """Store a pair of related random variables, each expressed as a PDF.

    A VariablePair associates two PDFs that describe a paired observation,
    without requiring the two variables to share any particular type or unit.
    """

    def __init__(
        self,
        pdf1: PDFs.PDF,
        pdf2: PDFs.PDF,
        name: str | None = None,
    ) -> None:
        """Initialize a VariablePair.

        Parameters
        ----------
        pdf1 : PDF
            One of the two paired random variables.
        pdf2 : PDF
            The other of the two paired random variables.
        name : str, optional
            Brief descriptive identifier of the marker.
        """
        # Set variables
        self.pdf1 = pdf1
        self.pdf2 = pdf2

        # Record metadata
        self.name = name

    @property
    def pdf1(self) -> PDFs.PDF:
        return self._pdf1

    @pdf1.setter
    def pdf1(self, value: PDFs.PDF) -> None:
        if not isinstance(value, PDFs.PDF):
            raise TypeError(
                f"Variable `pdf1` must be provided as a PDF, "
                f"got {type(value).__name__}"
            )

        # Set pdf1 value - deep copy just in case
        self._pdf1 = copy.deepcopy(value)

    @property
    def pdf2(self) -> PDFs.PDF:
        return self._pdf2

    @pdf2.setter
    def pdf2(self, value: PDFs.PDF) -> None:
        if not isinstance(value, PDFs.PDF):
            raise TypeError(
                f"Variable `pdf2` must be provided as a PDF, "
                f"got {type(value).__name__}"
            )

        # Set pdf2 value - deep copy just in case
        self._pdf2 = copy.deepcopy(value)

    def __str__(self) -> str:
        print_str = "VariablePair "

        # Report marker name
        if self.name is not None:
            print_str += f"{self.name}, "

        # Report x
        print_str += (
            f"comprising: "
            f"\n\tpdf1: {self.pdf1.name} "
            f"{PDFs.analytics.pdf_mean(self.pdf1):.2f} "
            f"+- {PDFs.analytics.pdf_std(self.pdf1):.2f} "
        )
        if self.pdf1.unit is not None:
            print_str += f"{self.pdf1.unit}"

        # Report pdf2
        print_str += (
            f"\n\tpdf2: {self.pdf2.name} "
            f"{PDFs.analytics.pdf_mean(self.pdf2):.2f} "
            f"+- {PDFs.analytics.pdf_std(self.pdf2):.2f} "
        )
        if self.pdf2.unit is not None:
            print_str += f"{self.pdf2.unit}"

        return print_str


# end of file
