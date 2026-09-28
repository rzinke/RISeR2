#!/bin/bash

mkdir -p tmp

# Create age and displacement PDFs
Aname="tmp/age.txt"
riser-make-pdf -d gaussian -s 10.0 1.0 -dx 0.01 \
    --name "C14-01" --variable-type "age" --unit "ky" \
    -o $Aname

echo "Creating PDFs"
Uname="tmp/displacement.txt"
riser-make-pdf -d gaussian -s 30.0 1.0 -dx 0.01 \
    --name "marker 1" --variable-type "displacement" --unit "m" \
    -o $Uname

# Compute slip rate
echo ""
echo "Computing slip rate"
outdir="tmp/single_slip_rate"
riser-compute-slip-rate \
    --age $Aname --displacement $Uname \
    --age-unit-out "y" --displacement-unit-out "mm" \
    -o $outdir -v -p
