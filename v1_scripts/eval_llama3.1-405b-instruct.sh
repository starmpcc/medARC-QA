#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name llama3.1-405b-instruct-fp8\
                 --output_dir eval_results \
                 --assigned_subjects all
