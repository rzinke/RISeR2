#!/bin/bash
# Compute incremental fault slip rates using Monte Carlo sampling based on a
# series of dated displacement markers.

mkdir -p tmp

# Convert ages from dates to years before present
for date_name in $(ls *_date.txt); do
    # Sample and file names
    sample_name=$(echo $date_name | cut -d "_" -f 1)
    age_name="tmp/${date_name//date/age}"

    # Convert date to age
    riser-calyr-to-age $date_name --name $sample_name --variable-type age \
        --reference-date 1950 --output-unit ky \
        -v -o $age_name
done


# View ages
riser-view-pdf-stack age_list.toml --same-height -v


# Check displacement-age history
riser-view-displacement-age-history marker_config.toml \
    --age-unit-out ky --marker-type rectangle --show-marginals --show-labels \
    -v


# Compute incremental slip rates
riser-compute-slip-rates-mc marker_config.toml \
    --age-unit-out y --displacement-unit-out mm \
    --n-samples 1000000 --max-rate 100 --dv 0.2 \
    --smoothing-type mean --smoothing-width 3 \
    --confidence-metric HPD \
    -v -p -o tmp/complex_incremental_rates
