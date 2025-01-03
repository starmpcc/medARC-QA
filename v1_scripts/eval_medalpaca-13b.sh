#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name medalpaca-13b\
                 --output_dir eval_results \
                 --assigned_subjects all
