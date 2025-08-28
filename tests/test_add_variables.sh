#!/bin/bash
# Find the sum of two random variables.

# Create marginal distributions
X1name="tmp/pdf1.txt"
make_pdf.py -d gaussian -s 6.0 1.0 -dx 0.01 --name "pdf1" -o $X1name

X2name="tmp/pdf2.txt"
make_pdf.py -d gaussian -s 4.0 1.0 -dx 0.01 --name "pdf2" -o $X2name


# Compute joint probability
X12name="tmp/pdf12.txt"
add_variables.py $X1name $X2name -o $X12name -v -p

# # Examine output
# view_pdf.py $X12name -v
