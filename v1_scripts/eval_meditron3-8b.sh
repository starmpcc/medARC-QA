#!/bin/bash

cd ../

python evaluate_from_api.py \
                 --model_name Meditron3-8B \
                 --output_dir eval_results \
                 --assigned_subjects all
