#!/bin/sh

echo 'Executing par2_generator.py:'
python3 par2_generator.py

echo 'Executing random_sampling_plots.py:'
python3 random_sampling_plots.py

echo 'Executing family_and_SAT_random_sampling_plots.py:'
python3 family_and_SAT_random_sampling_plots.py
