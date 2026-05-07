#! /bin/bash
test_gt_path="../annotations/OpenView_bench.json"

model_path="path/to/qwen_vl_model" ## Modify this
python inference_qwen_vl.py \
    --model_path $model_path \
    --test_gt_path $test_gt_path

model_path="path/to/internvl_model" ## Modify this
python inference_internvl.py \
    --model_path $model_path \
    --test_gt_path $test_gt_path

model_path="path/to/llavanext_model" ## Modify this
python inference_llavanext.py \
    --model_path $model_path \
    --test_gt_path $test_gt_path