#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name llama3.1-70b-instruct-fp8\
                 --output_dir eval_uq_results \
                 --assigned_subjects all \
                 --uncertainty_quantification 1
