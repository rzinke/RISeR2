#!/bin/bash

# Create marginal distributions
echo "Creating PDFs"
X1name="tmp/ref.txt"
riser-make-pdf -d gaussian -s 6.0 1.0 -dx 0.1 \
    --name "ref" --unit "ky" -o $X1name

X2name="tmp/sec.txt"
riser-make-pdf -d triangular -s 3.5 5.0 6.5 -dx 0.1 \
    --name "sec" --unit "ky" -o $X2name


# Cross correlate
echo ""
echo "Computing cross correlation"
riser-cross-correlate-variables $X1name $X2name -v -p
