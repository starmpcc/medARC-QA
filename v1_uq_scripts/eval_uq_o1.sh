#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name o1-2024-12-17\
                 --num_tokens 100000\
                 --output_dir eval_uq_results \
                 --assigned_subjects all \
                 --uncertainty_quantification 1
