#!/bin/bash

mkdir -p tmp

# Create marginal distributions
echo "Creating PDFs"
X1name="tmp/pdf1.txt"
riser-make-pdf gaussian --mu 6.0 --sigma 1.0 --dx 0.01 \
    --name "pdf1" --variable-type "age" --unit "y" -o $X1name

X2name="tmp/pdf2.txt"
riser-make-pdf gaussian --mu 4.0 --sigma 1.0 --dx 0.01 \
    --name "pdf2" --variable-type "age" --unit "y" -o $X2name


# Add variables
echo ""
echo "Adding PDFs"
X12name="tmp/summed.txt"
riser-add-variables $X1name $X2name -o $X12name -v -p
