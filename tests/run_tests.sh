#!/bin/bash

# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.

for test in $(ls test*.py); do
    pytest $test
done
