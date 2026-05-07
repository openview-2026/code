#! /bin/bash
#SBATCH --job-name=inference
#SBATCH --gres=gpu:A100:2
#SBATCH --ntasks=2
#SBATCH --time=24:00:00
#SBATCH --mem=128GB

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NUM_THREADS=1

mamba activate openview_inf ## Modify your own env name
cd path/to/OpenView/inference
test_gt_path="../annotations/OpenView_bench.json"

model_path="path/to/qwen_vl_model" ## Modify this
srun --cpu-bind=cores bash -lc "
python inference_qwen_vl.py \
    --model_path $model_path \
    --test_gt_path $test_gt_path
"

model_path="path/to/internvl_model" ## Modify this
srun --cpu-bind=cores bash -lc "
python inference_internvl.py \
    --model_path $model_path \
    --test_gt_path $test_gt_path
"

model_path="path/to/llavanext_model" ## Modify this
srun --cpu-bind=cores bash -lc "
python inference_llavanext.py \
    --model_path $model_path \
    --test_gt_path $test_gt_path
"