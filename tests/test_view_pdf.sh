#!/bin/bash


# Create PDF
echo "Creating PDF"
bash test_make_pdf.sh

# View PDF
echo ""
echo "Viewing PDF"
view_pdf.py "tmp/PDF_X.txt" -v --show-confidence


# end of file