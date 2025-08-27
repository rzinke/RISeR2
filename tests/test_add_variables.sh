#!/bin/bash
# Find the joint probability of two PDFs.

# Create marginal distributions
X1name="tmp/pdfX1.txt"
make_pdf.py -d gaussian -s 4.0 1.0 -dx 0.05 \
    --name "X" --unit "y" -o $X1name

X2name="tmp/pdfX2.txt"
make_pdf.py -d gaussian -s 6.0 1.0 -dx 0.05 \
    --name "Y" --unit "y" -o $X2name


# Compute joint probability
add_variables.py $X1name $X2name -o sumXY.txt -v -p
