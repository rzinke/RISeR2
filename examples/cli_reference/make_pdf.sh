#!/bin/bash

mkdir -p tmp

riser-make-pdf -d triangular -s 9.0 11.0 12.5 -dx 0.1 \
    --name "AsymmTri" --variable-type "displacement" --unit "m" \
    -o tmp/pdf.txt -v -p
