#!/bin/bash

# Create PDF
echo "Creating PDFs"
X1name="tmp/pdf1.txt"
riser-make-pdf -d gaussian -s 6.0 1.0 -dx 0.1 \
    --name "pdf1" --variable-type "age" --unit "y" -o $X1name

# View PDF
echo ""
echo "Viewing PDF"
riser-view-pdf $X1name -v --show-confidence
