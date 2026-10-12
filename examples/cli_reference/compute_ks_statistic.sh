#!/bin/bash

mkdir -p tmp

# Create marginal distributions
echo "Creating PDFs"

X1name="tmp/pdf1.txt"
riser-make-pdf gaussian --mu 6.0 --sigma 1.0 --dx 0.1 \
    --name "pdf1" --variable-type "age" --unit "ky" -o $X1name

X2name="tmp/pdf2.txt"
riser-make-pdf triangular --a 3.5 --c 5.0 --b 6.5 --dx 0.1 \
    --name "pdf2" --variable-type "age" --unit "ky" -o $X2name


# Compute KS statistic
echo ""
echo "Computing KS statistic"

riser-compute-ks-statistic $X1name $X2name -v -p
