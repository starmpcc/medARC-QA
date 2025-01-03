#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name Mistral-7B-Instruct-v0.3\
                 --output_dir eval_results \
                 --assigned_subjects all
