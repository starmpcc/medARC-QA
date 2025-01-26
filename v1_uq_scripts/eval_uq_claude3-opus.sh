#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name claude-3-opus-20240229 \
                 --output_dir eval_uq_results \
                 --assigned_subjects all \
                 --uncertainty_quantification 1
