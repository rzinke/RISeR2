#!/bin/bash

# Create marginal distributions
echo "Creating PDFs"
X1name="tmp/displacement.txt"
make_pdf.py -d gaussian -s 30.0 1.0 -dx 0.01 \
    --name "marker 1" --variable-type "displacement" --unit "m" \
    -o $X1name

X2name="tmp/age.txt"
make_pdf.py -d gaussian -s 10.0 1.0 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $X2name


echo ""
echo "Writing config file"
config_file="tmp/slip_rate_config.toml"
echo "[\"slip rate X\"]" > $config_file
echo "\"age file\" = \"${X2name}\"" >> $config_file
echo "\"displacement file\" = \"${X1name}\"" >> $config_file


# Compute slip rate
echo ""
echo "Computing slip rate"
compute_slip_rate.py $config_file -o "tmp" -v -p