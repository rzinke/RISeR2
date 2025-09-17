#!/bin/bash
# Compute incremental fault slip rates using Monte Carlo sampling based on a
# series of dated displacement markers.

# Convert ages from dates to years before present
for date_name in $(ls *_date.txt); do
    # Sample and file names
    sample_name=$(echo $date_name | cut -d "_" -f 1)
    age_name=${date_name//date/age}

    # Convert date to age
    calyr_to_age.py $date_name --name $sample_name --variable-type age \
        --reference-date 1950 --output-unit ky \
        -v -o $age_name
done


# View ages
view_pdf_stack.py age_list.toml --same-height -v


# # Check displacement-age history
# view_displacement_age_history.py marker_config.toml \
#     --age-unit-out ky --marker-type rectangle --show-marginals --show-labels \
#     -v


# # Compute incremental slip rates
# compute_slip_rates_mc.py marker_config.toml \
#     --age-unit-out y --displacement-unit-out mm \
#     --n-samples 1000000 --max-rate 100 --dv 0.2 \
#     --confidence-metric HPD \
#     -v -p -o mc_slip_rate