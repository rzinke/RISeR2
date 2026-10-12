#!/bin/bash

mkdir -p tmp

# Create marginal distributions
echo "Creating PDFs"
X1name="tmp/ref.txt"
riser-make-pdf gaussian --mu 6.0 --sigma 1.0 --dx 0.1 \
    --name "ref" --variable-type age --unit "ky" -o $X1name

X2name="tmp/sec.txt"
riser-make-pdf triangular --a 3.5 --c 5.0 --b 6.5 --dx 0.1 \
    --name "sec" --variable-type age --unit "ky" -o $X2name


# Cross correlate
echo ""
echo "Computing cross correlation"
riser-cross-correlate-variables $X1name $X2name -v -p
