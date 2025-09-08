#!/bin/bash


config_file="tmp/ages_config.toml"


# Create marginal distributions
echo "Creating PDFs"

A1name="tmp/age1.txt"
make_pdf.py -d gaussian -s 4.0 0.4 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A1name

A1prior_name="tmp/age1_prior.txt"
make_pdf.py -d gaussian -s 4.1 0.7 -dx 0.01 \
    --name "age prior" --variable-type "age" --unit "ky" \
    -o $A1prior_name

A2name="tmp/age2.txt"
make_pdf.py -d gaussian -s 6.3 0.6 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A2name

A2prior_name="tmp/age2_prior.txt"
make_pdf.py -d gaussian -s 6.1 0.8 -dx 0.01 \
    --name "age prior" --variable-type "age" --unit "ky" \
    -o $A2prior_name

A3name="tmp/age3.txt"
make_pdf.py -d gaussian -s 7.7 0.5 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A3name

A4name="tmp/age4.txt"
make_pdf.py -d gaussian -s 9.0 0.7 -dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A4name


echo ""
echo "Writing config file"

echo "[\"Age 1\"]" > $config_file
echo "\"pdf file\" = \"${A1name}\"" >> $config_file
echo "\"color\" = \"blue\"" >> $config_file
echo "\"prior\" = \"${A1prior_name}\"" >> $config_file

echo "[\"Age 2\"]" >> $config_file
echo "\"pdf file\" = \"${A2name}\"" >> $config_file
echo "\"color\" = \"blue\"" >> $config_file
echo "\"prior\" = \"${A2prior_name}\"" >> $config_file

echo "[\"Age 3\"]" >> $config_file
echo "\"pdf file\" = \"${A3name}\"" >> $config_file

echo "[\"Age 4\"]" >> $config_file
echo "\"pdf file\" = \"${A4name}\"" >> $config_file


# Compute slip rate
echo ""
echo "Viewing history"
view_pdf_stack.py $config_file