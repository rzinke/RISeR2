#!/bin/bash
# Find the sum of two random variables.

# Create marginal distributions
X1name="tmp/pdf1.txt"
make_pdf.py -d triangular -s 3 4 5 -dx 0.2 -o $X1name

X2name="tmp/pdf2.txt"
make_pdf.py -d triangular -s 5 6 7 -dx 0.2 -o $X2name


# Compute joint probability
X12name="tmp/gap_pdf.txt"
compute_gap_probabilities.py $X1name $X2name --name "gap" -o $X12name -v -p

# # Examine output
# view_pdf.py $X1_2name -v
