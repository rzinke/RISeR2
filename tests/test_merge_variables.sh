#!/bin/bash

# Create marginal distributions
echo "Creating PDFs"
X1name="tmp/boxcarX.txt"
make_pdf.py -d boxcar -s 4.0 5.0 -dx 0.05 \
    --name "boxcar" --unit "m" -o $X1name

X2name="tmp/trapezX.txt"
make_pdf.py -d trapezoidal -s 4.0 4.5 6.0 7.0 -dx 0.1 \
    --name "trapezoidal" --unit "m" -o $X2name


# Merge PDFs
echo ""
echo "Merging PDFs"
merge_variables.py $X1name $X2name -o tmp/X12merged.txt -v -p