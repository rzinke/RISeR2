#!/bin/bash

# Create marginal distributions
echo "Creating PDFs"
X1name="tmp/uniform.txt"
riser-make-pdf uniform --a 4.0 --b 5.0 --dx 0.05 \
    --name "uniform" --variable-type "displacement" --unit "m" -o $X1name

X2name="tmp/trapezoid.txt"
riser-make-pdf trapezoidal --a 4.0 --c 4.5 --b 6.0 7.0 --dx 0.1 \
    --name "trapezoidal" --variable-type "displacement" --unit "m" -o $X2name


# Merge PDFs
echo ""
echo "Pooling PDFs"
X12name="tmp/pooled.txt"
riser-pool-variables $X1name $X2name -o $X12name -v -p
