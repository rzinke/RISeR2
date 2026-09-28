#!/bin/bash

# Create marginal distributions
echo "Creating PDFs"

X1name="tmp/displacement.txt"
riser-make-pdf -d gaussian -s 30.0 1.0 -dx 0.01 \
    --name "displacement" --variable-type "displacement" --unit "m" -o $X1name

X2name="tmp/age.txt"
riser-make-pdf -d gaussian -s 10.0 1.0 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" -o $X2name


# Compute joint probability
echo ""
echo "Computing PDF ratio"

X12name="tmp/pdf12.txt"
riser-divide-variables $X1name $X2name -o $X12name -v -p
