#!/bin/bash

mkdir -p tmp

# Create marginal distributions
echo "Creating PDFs"
X1name="tmp/smaller.txt"
riser-make-pdf triangular --a 3 --c 4 --b 5 --dx 0.1 \
    --variable-type "displacement" --unit "m" -o $X1name

X2name="tmp/larger.txt"
riser-make-pdf triangular --a 7 --c 8 --b 11 --dx 0.1 \
    --variable-type "displacement" --unit "m" -o $X2name


# Compute joint probability
echo ""
echo "Computing bracketed PDF"
X12name="tmp/bracketed.txt"
riser-create-bracketed-pdf $X1name $X2name --name "bracketed" -o $X12name -v -p
