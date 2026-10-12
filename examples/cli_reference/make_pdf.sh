#!/bin/bash

mkdir -p tmp

riser-make-pdf triangular --a 9.0 --c 11.0 --b 12.5 --dx 0.1 \
    --name "AsymmTri" --variable-type "displacement" --unit "m" \
    -o tmp/pdf.txt -v -p
