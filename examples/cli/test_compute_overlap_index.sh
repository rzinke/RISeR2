#!/bin/bash

# Create marginal distributions
echo "Creating PDFs"

A1name="tmp/age1.txt"
riser-make-pdf -d gaussian -s 7.0 1.0 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A1name

A2name="tmp/age2.txt"
riser-make-pdf -d gaussian -s 10.0 1.0 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A2name


# Compute slip rate
echo ""
echo "Computing overlap index"
riser-compute-overlap-index $A1name $A2name -v -p
