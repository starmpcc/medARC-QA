#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name o1-mini\
                 --output_dir eval_results \
                 --assigned_subjects all
