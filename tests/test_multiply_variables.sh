#!/bin/bash

# Create marginal distributions
echo "Creating PDFs"

X1name="tmp/sliprate.txt"
make_pdf.py -d gaussian -s 2.0 0.2 -dx 0.001 \
    --name "slip rate" --variable-type "slip rate" -o $X1name

X2name="tmp/age.txt"
make_pdf.py -d gaussian -s 8.0 0.5 -dx 0.001 \
    --name "age" --variable-type "age" --unit "ky" -o $X2name


# Compute joint probability
echo ""
echo "Computing PDF product"

X12name="tmp/pdf12.txt"
multiply_variables.py $X1name $X2name -o $X12name -v -p