#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name Meditron3-70B \
                 --output_dir eval_results \
                 --assigned_subjects all
