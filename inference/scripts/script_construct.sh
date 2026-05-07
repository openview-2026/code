#! /bin/bash

python batch_stage1.py
python batch_stage2.py --config_path configs/batch_config.yaml
python batch_stage3.py --config_path configs/batch_config.yaml
python batch_stage4.py --config_path configs/batch_config.yaml