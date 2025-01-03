#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name claude-3-sonnet-20240229 \
                 --output_dir eval_results \
                 --assigned_subjects all
