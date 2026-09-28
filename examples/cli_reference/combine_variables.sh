#!/bin/bash

# Create marginal distributions
echo "Creating PDFs"
X1name="tmp/uniform.txt"
riser-make-pdf -d uniform -s 4.0 5.0 \
    --name "uniform" --variable-type "displacement" --unit "m" -o $X1name

X2name="tmp/trapezoidal.txt"
riser-make-pdf -d trapezoidal -s 4.0 4.5 6.0 7.0 \
    --name "trapezoidal" --variable-type "displacement" --unit "m" -o $X2name


# Compute joint probability
echo ""
echo "Combining PDFs"
X12name="tmp/combined.txt"
riser-combine-variables $X1name $X2name -o $X12name -v -p
