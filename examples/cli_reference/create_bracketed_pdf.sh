#!/bin/bash

mkdir -p tmp

# Create marginal distributions
echo "Creating PDFs"
X1name="tmp/smaller.txt"
riser-make-pdf -d triangular -s 3 4 5 -dx 0.1 \
    --variable-type "displacement" --unit "m" -o $X1name

X2name="tmp/larger.txt"
riser-make-pdf -d triangular -s 7 8 11 -dx 0.1 \
    --variable-type "displacement" --unit "m" -o $X2name


# Compute joint probability
echo ""
echo "Computing bracketed PDF"
X12name="tmp/bracketed.txt"
riser-create-bracketed-pdf $X1name $X2name --name "bracketed" -o $X12name -v -p
