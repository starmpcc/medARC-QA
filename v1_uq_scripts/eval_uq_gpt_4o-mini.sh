#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name gpt-4o-mini \
                 --output_dir eval_uq_results \
                 --assigned_subjects all \
                 --uncertainty_quantification 1
