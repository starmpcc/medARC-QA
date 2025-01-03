#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name Llama-3.1-70B-Instruct\
                 --output_dir eval_results \
                 --assigned_subjects all
