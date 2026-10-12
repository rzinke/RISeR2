#!/bin/bash

mkdir -p tmp

# Create marginal distributions
echo "Creating PDFs"

A1name="tmp/age1.txt"
riser-make-pdf gaussian --mu 4.0 --sigma 0.4 --dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A1name

A1prior_name="tmp/age1_prior.txt"
riser-make-pdf gaussian --mu 4.1 --sigma 0.7 --dx 0.01 \
    --name "age prior" --variable-type "age" --unit "ky" \
    -o $A1prior_name

A2name="tmp/age2.txt"
riser-make-pdf gaussian --mu 6.3 --sigma 0.6 --dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A2name

A2prior_name="tmp/age2_prior.txt"
riser-make-pdf gaussian --mu 6.1 --sigma 0.8 --dx 0.01 \
    --name "age prior" --variable-type "age" --unit "ky" \
    -o $A2prior_name

A3name="tmp/age3.txt"
riser-make-pdf gaussian --mu 7.7 --sigma 0.5 --dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A3name

A4name="tmp/age4.txt"
riser-make-pdf gaussian --mu 9.0 --sigma 0.7 --dx 0.01 \
    --name "age" --variable-type "age" --unit "ky" \
    -o $A4name


echo ""
echo "Writing config file"

config_file="tmp/ages_config.toml"


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
riser-view-pdf-stack $config_file
