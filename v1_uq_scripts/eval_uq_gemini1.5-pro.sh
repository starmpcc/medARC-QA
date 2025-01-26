#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name gemini-1.5-pro-latest \
                 --output_dir eval_uq_results \
                 --assigned_subjects all \
                 --uncertainty_quantification 1

