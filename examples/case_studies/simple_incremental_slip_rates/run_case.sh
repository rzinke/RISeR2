#!/bin/bash
# Compute an incremental fault slip rate based on a series of dated
# displacement markers.


view_displacement_age_history.py marker_config.toml \
    --age-unit-out ky --show-labels \
    -v

riser-compute-slip-rates marker_config.toml \
    --age-unit-out y --displacement-unit-out mm \
    --limit-positive --max-rate 10 \
    -v -p -o tmp/simple_incremental_rates
