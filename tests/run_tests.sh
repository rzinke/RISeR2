#!/bin/bash

for test in $(ls test*.py); do
    pytest $test
done
