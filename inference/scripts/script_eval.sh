#! /bin/bash

# export the api key
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY ## Modify this

model_path="path/to/model" ## Modify this
test_gt_path="../annotations/OpenView_bench.json"
output_path="../results/$(basename $model_path)_output.json"

python eval.py --output_path $output_path --model gpt-5-mini