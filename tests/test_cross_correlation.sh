#!/bin/bash
# Find the sum of two random variables.

# Create marginal distributions
X1name="tmp/ref.txt"
make_pdf.py -d gaussian -s 6.0 1.0 -dx 0.1 --name "ref" --unit "ky" -o $X1name

X2name="tmp/sec.txt"
make_pdf.py -d triangular -s 3.5 5.0 6.5 -dx 0.1 --name "sec" --unit "ky" -o $X2name


# Compute joint probability
cross_correlate_variables.py $X1name $X2name -v -p

