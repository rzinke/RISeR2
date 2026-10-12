#!/bin/bash

mkdir -p tmp

# Create marginal distributions
X1name="tmp/pdf1.txt"
riser-make-pdf gaussian --mu 6.0 --sigma 1.0 --dx 0.01 \
    --name "pdf1" --variable-type "age" --unit "ky" -o $X1name

X2name="tmp/pdf2.txt"
riser-make-pdf gaussian --mu 4.0 --sigma 1.0 --dx 0.01 \
    --name "pdf2" --variable-type "age" --unit "ky" -o $X2name


# Compute joint probability
X12name="tmp/differenced.txt"
riser-subtract-variables $X1name $X2name -o $X12name -v -p
