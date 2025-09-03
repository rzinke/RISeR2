#!/bin/bash

# Create marginal distributions
echo "Creating PDFs"
X1name="tmp/pdf1.txt"
make_pdf.py -d gaussian -s 6.0 1.0 -dx 0.01 \
    --name "pdf1" --variable-type "age" --unit "y" -o $X1name

X2name="tmp/pdf2.txt"
make_pdf.py -d gaussian -s 4.0 1.0 -dx 0.01 \
    --name "pdf2" --variable-type "age" --unit "y" -o $X2name


# Add variables
echo ""
echo "Adding PDFs"
X12name="tmp/pdf12.txt"
add_variables.py $X1name $X2name -o $X12name -v -p