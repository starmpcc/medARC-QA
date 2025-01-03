#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name o1-preview\
                 --output_dir eval_results \
                 --assigned_subjects all
