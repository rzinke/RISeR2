#!/bin/bash

mkdir -p tmp

# Create PDF
echo "Creating PDFs"
X1name="tmp/pdf1.txt"
riser-make-pdf -d gaussian -s 6.0 1.0 -dx 0.1 \
    --name "pdf1" --variable-type "age" --unit "y" -o $X1name

# Interpolate PDF
echo ""
echo "Interpolating PDF"
outname="interpolated.txt"
riser-interpolate-pdf $X1name -o $outname \
    --xmin 0 --dx 0.01 \
    -v -p
