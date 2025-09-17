#!/bin/bash
# Compute an incremental fault slip rate based on a series of dated
# displacement markers.


view_displacement_age_history.py marker_config.toml \
    --age-unit-out ky --show-labels \
    -v

compute_slip_rates.py marker_config.toml \
    --age-unit-out y --displacement-unit-out mm \
    --limit-positive --max-rate 12 \
    -v -p -o slip_rate