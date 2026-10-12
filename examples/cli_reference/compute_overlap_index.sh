#!/bin/bash

mkdir -p tmp

# Create marginal distributions
echo "Creating PDFs"

A1name="tmp/age1.txt"
riser-make-pdf gaussian --mu 7.0 --sigma 1.0 --dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A1name

A2name="tmp/age2.txt"
riser-make-pdf gaussian --mu 10.0 --sigma 1.0 --dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A2name


# Compute slip rate
echo ""
echo "Computing overlap index"
riser-compute-overlap-index $A1name $A2name -v -p
