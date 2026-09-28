#!/bin/bash

# Create marginal distributions
echo "Creating PDFs"
X1name="tmp/uniform.txt"
riser-make-pdf -d uniform -s 4.0 5.0 -dx 0.05 \
    --name "uniform" --variable-type "displacement" --unit "m" -o $X1name

X2name="tmp/trapezoid.txt"
riser-make-pdf -d trapezoidal -s 4.0 4.5 6.0 7.0 -dx 0.1 \
    --name "trapezoidal" --variable-type "displacement" --unit "m" -o $X2name


# Merge PDFs
echo ""
echo "Pooling PDFs"
X12name="tmp/pooled.txt"
riser-pool-variables $X1name $X2name -o $X12name -v -p
