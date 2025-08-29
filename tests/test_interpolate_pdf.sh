#!/bin/bash

# Create PDF
echo "Creating PDF"
bash test_make_pdf.sh

# View PDF
echo ""
echo "Interpolating PDF"
interpolate_pdf.py "tmp/PDF_X.txt" -o tmp/PDF_X_interp.txt \
    --xmin 0 --dx 0.01 \
    -v -p