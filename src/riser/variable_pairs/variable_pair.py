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
        x1: PDFs.PDF,
        x2: PDFs.PDF,
        name: str | None = None,
    ):
        """Initialize a VariablePair.

        Parameters
        ----------
        x1 : PDF
            One of the two paired random variables.
        x2 : PDF
            The other of the two paired random variables.
        name : str
            Brief descriptive identifier of the marker.
        """
        # Set variables
        self.x1 = x1
        self.x2 = x2

        # Record metadata
        self.name = name

    @property
    def x1(self) -> PDFs.PDF:
        return self._x1

    @x1.setter
    def x1(self, value):
        if not isinstance(value, PDFs.PDF):
            raise TypeError(
                f"Variable `x1` must be provided as a PDF, "
                f"got {type(x1).__name__}"
            )

        # Set x1 value - deep copy just in case
        self._x1 = copy.deepcopy(value)

    @property
    def x2(self) -> PDFs.PDF:
        return self._x2

    @x2.setter
    def x2(self, value):
        if not isinstance(value, PDFs.PDF):
            raise TypeError(
                f"Variable `x2` must be provided as a PDF, "
                f"got {type(x2).__name__}"
            )

        # Set x2 value - deep copy just in case
        self._x2 = copy.deepcopy(value)

    def __str__(self):
        print_str = f"VariablePair "

        # Report marker name
        if self.name is not None:
            print_str += f"{self.name}, "

        # Report x
        print_str += (
            f"comprising: "
            f"\n\tx1: {self.x1.name} "
            f"{PDFs.analytics.pdf_mean(self.x1):.2f} "
            f"+- {PDFs.analytics.pdf_std(self.x1):.2f} "
        )
        if self.x1.unit is not None:
            print_str += f"{self.x1.unit}"

        # Report x2
        print_str += (
            f"\n\tx2: {self.x2.name} "
            f"{PDFs.analytics.pdf_mean(self.x2):.2f} "
            f"+- {PDFs.analytics.pdf_std(self.x2):.2f} "
        )
        if self.x2.unit is not None:
            print_str += f"{self.x2.unit}"

        return print_str


# end of file
