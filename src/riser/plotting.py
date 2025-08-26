# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import matplotlib.pyplot as plt

from riser.probability_functions import PDF


#################### PDF PLOTTING ####################
def plot_pdf(fig, ax, pdf:PDF,
             color="k", linewidth=3):
    """Basic plot of a probability density function (PDF).
    """
    # Plot PDF
    ax.plot(pdf.x, pdf.px, color=color, linewidth=linewidth)


# end of file