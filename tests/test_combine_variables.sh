#!/bin/bash
# Find the joint probability of two PDFs.

# Create marginal distributions
X1name="tmp/boxcarX.txt"
make_pdf.py -d boxcar -s 4.0 5.0 -dx 0.05 --name "boxcar" -o $X1name

X2name="tmp/trapezX.txt"
make_pdf.py -d trapezoidal -s 4.0 4.5 6.0 7.0 -dx 0.1 --name "trapezoidal" -o $X2name


# Compute joint probability
combine_variables.py $X1name $X2name -o tmp/X12joint.txt -v -p
