#!/bin/bash
# Find the quotient of two random variables.

# Create marginal distributions
X1name="tmp/displacement.txt"
make_pdf.py -d gaussian -s 30.0 1.0 -dx 0.01 --name "displacement" -o $X1name

X2name="tmp/age.txt"
make_pdf.py -d gaussian -s 10.0 1.0 -dx 0.01 --name "age" -o $X2name


# Compute joint probability
X12name="tmp/pdf12.txt"
divide_variables.py $X1name $X2name -o $X12name -v -p

# # Examine output
# view_pdf.py $X12name -v
