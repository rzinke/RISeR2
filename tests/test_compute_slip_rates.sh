#!/bin/bash


# compute_slip_rates.py -h


# Create marginal distributions
echo "Creating PDFs"

U1name="tmp/displacement1.txt"
make_pdf.py -d gaussian -s 21.0 1.0 -dx 0.01 \
    --name "marker 1" --variable-type "displacement" --unit "m" \
    -o $U1name

A1name="tmp/age1.txt"
make_pdf.py -d gaussian -s 7.0 1.0 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A1name


U2name="tmp/displacement2.txt"
make_pdf.py -d gaussian -s 30.0 1.0 -dx 0.01 \
    --name "marker 2" --variable-type "displacement" --unit "m" \
    -o $U2name

A2name="tmp/age2.txt"
make_pdf.py -d gaussian -s 10.0 1.0 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A2name


U3name="tmp/displacement3.txt"
make_pdf.py -d gaussian -s 45.0 1.0 -dx 0.01 \
    --name "marker 3" --variable-type "displacement" --unit "m" \
    -o $U3name

A3name="tmp/age3.txt"
make_pdf.py -d gaussian -s 15.0 1.0 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A3name


echo ""
echo "Writing config file"

config_file="tmp/slip_rates_config.toml"
echo "[\"Feat 1\"]" > $config_file
echo "\"age file\" = \"${A1name}\"" >> $config_file
echo "\"displacement file\" = \"${U1name}\"" >> $config_file

echo "[\"Feat 2\"]" >> $config_file
echo "\"age file\" = \"${A2name}\"" >> $config_file
echo "\"displacement file\" = \"${U2name}\"" >> $config_file

echo "[\"Feat 3\"]" >> $config_file
echo "\"age file\" = \"${A3name}\"" >> $config_file
echo "\"displacement file\" = \"${U3name}\"" >> $config_file


# Compute slip rate
echo ""
echo "Computing slip rate"
compute_slip_rates.py $config_file \
    --age-unit-out "y" --displacement-unit-out "mm" \
    --max-rate 10 \
    -o "tmp/incr_analyt/v1ia" -v -p