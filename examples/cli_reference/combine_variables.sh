#!/bin/bash

# Create marginal distributions
echo "Creating PDFs"
X1name="tmp/uniform.txt"
riser-make-pdf uniform --a 4.0 --b 5.0 \
    --name "uniform" --variable-type "displacement" --unit "m" -o $X1name

X2name="tmp/trapezoidal.txt"
riser-make-pdf trapezoidal --a 4.0 --b 4.5 --c 6.0 --d 7.0 \
    --name "trapezoidal" --variable-type "displacement" --unit "m" -o $X2name


# Compute joint probability
echo ""
echo "Combining PDFs"
X12name="tmp/combined.txt"
riser-combine-variables $X1name $X2name -o $X12name -v -p
