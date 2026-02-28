#!/bin/bash


config_file="tmp/slip_rates_mc_config.toml"


# Create marginal distributions
echo "Creating PDFs"

U1name="tmp/displacement1.txt"
make_pdf.py -d triangular -s 7.5 9.5 11.0 -dx 0.01 \
    --name "disp 1" --variable-type "displacement" --unit "m" \
    -o $U1name

A1name="tmp/age1.txt"
make_pdf.py -d gaussian -s 4.0 1.0 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A1name


U2name="tmp/displacement2.txt"
make_pdf.py -d trapezoidal -s 13.0 14.0 15.0 17.0 -dx 0.01 \
    --name "disp 2" --variable-type "displacement" --unit "m" \
    -o $U2name

A2name="tmp/age2.txt"
make_pdf.py -d boxcar -s 5.2 5.9 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A2name


U3name="tmp/displacement3.txt"
make_pdf.py -d triangular -s 17.0 18.5 20.0 -dx 0.01 \
    --name "disp 3" --variable-type "displacement" --unit "m" \
    -o $U3name

A3name="tmp/age3.txt"
make_pdf.py -d gaussian -s 6.3 1.2 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A3name


U4name="tmp/displacement4.txt"
make_pdf.py -d trapezoidal -s 39.5 41.5 44.0 45.5 -dx 0.01 \
    --name "disp 4" --variable-type "displacement" --unit "m" \
    -o $U4name

A4name="tmp/age4.txt"
make_pdf.py -d gaussian -s 7.7 0.5 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A4name


U5name="tmp/displacement5.txt"
make_pdf.py -d triangular -s 45.0 50.0 55.0 -dx 0.01 \
    --name "disp 5" --variable-type "displacement" --unit "m" \
    -o $U5name

A5name="tmp/age5.txt"
make_pdf.py -d triangular -s 7.0 9.0 11.0 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A5name


echo ""
echo "Writing config file"

echo "[\"Feat 1\"]" > $config_file
echo "\"age file\" = \"${A1name}\"" >> $config_file
echo "\"displacement file\" = \"${U1name}\"" >> $config_file

echo "[\"Feat 2\"]" >> $config_file
echo "\"age file\" = \"${A2name}\"" >> $config_file
echo "\"displacement file\" = \"${U2name}\"" >> $config_file

echo "[\"Feat 3\"]" >> $config_file
echo "\"age file\" = \"${A3name}\"" >> $config_file
echo "\"displacement file\" = \"${U3name}\"" >> $config_file

echo "[\"Feat 4\"]" >> $config_file
echo "\"age file\" = \"${A4name}\"" >> $config_file
echo "\"displacement file\" = \"${U4name}\"" >> $config_file

echo "[\"Feat 5\"]" >> $config_file
echo "\"age file\" = \"${A5name}\"" >> $config_file
echo "\"displacement file\" = \"${U5name}\"" >> $config_file


# Compute slip rate
echo ""
echo "Computing slip rates (Monte Carlo)"
compute_slip_rates_mc.py $config_file \
    --n-samples 100000 --max-rate 60 \
    --smoothing-type gaussian --smoothing-width 9 \
    --confidence-metric HPD \
    -o "tmp/mcx" -v -p