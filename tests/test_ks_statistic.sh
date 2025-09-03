#!/bin/bash

# Create marginal distributions
echo "Creating PDFs"

X1name="tmp/pdf1.txt"
make_pdf.py -d gaussian -s 6.0 1.0 -dx 0.1 \
    --name "pdf1" --variable-type "age" --unit "ky" -o $X1name

X2name="tmp/pdf2.txt"
make_pdf.py -d triangular -s 3.5 5.0 6.5 -dx 0.1 \
    --name "pdf2" --variable-type "age" --unit "ky" -o $X2name


# Compute KS statistic
echo ""
echo "Computing KS statistic"

compute_ks_statistic.py $X1name $X2name -v -p