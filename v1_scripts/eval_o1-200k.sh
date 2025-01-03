#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name o1-2024-12-17\
                 --num_tokens 195000\
                 --output_dir eval_results \
                 --assigned_subjects all
