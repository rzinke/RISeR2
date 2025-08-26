#!/bin/bash


make_pdf.py -d triangular -s 9.0 11.0 12.5 -dx 0.1 \
    --name "PDF X" --variable-type "displacement" --unit "m" \
    -o tmp/PDF_X.txt -v


# end of file