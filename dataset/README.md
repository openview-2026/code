# 🪛 Data Preparation Guide

This directory contains utilities for preparing panorama images and extracting question views for the OpenView-Dataset.

## Panorama Preparation

To begin, you’ll need to collect the required panorama images or videos from five datasets. 

Please download the dataset and benchmark, following the instructions in the [Hugging Face page](https://huggingface.co/datasets/openview2026/OpenView_data).

After downloading, extracting, and renaming the panorama images, your data directory should be arranged as follows:
```
OpenView_code/
├── ...
├── annotations/                              # download from HuggingFace
│   ├── OpenView_dataset.json
│   └── OpenView_bench.json
└── dataset/
    └── data/
        ├── viewer.py                  # for viewing the benchmark panoramas
        ├── utils/...                  # utilities for viewer.py
        ├── 360loc/
        │   ├── 360loc_atrium_daytime_360_1_F1.jpg
        │   └── ...
        ├── 360x/
        │   ├── 360x_0c94d813-6ffe-4711-8627-4f0f62856bb2_F0.png
        │   └── ...
        ├── 3601m/
        ├── mapillary/
        ├── mp3d/
        └── test/              # OpenView-Bench panoramas
            ├── 360loc_atrium_nighttime_360_1_F322.jpg
            └── ...
```

## 2. Preparing the training set with the OpenView pipeline

After running the OpenView pipeline on your prepared panorama sources, the generated proposal files will be saved in the `inference/output` directory.
Next, use the `convert_to_format.py` script to convert proposals into the desired training converstation format and extract the question views.

```bash
python convert_to_format.py --input_file ../inference/output/xxx_output.json
```

*Note: Extraction is only required for preparing the training set. During inference, frames are computed on the fly, so manual extraction is unnecessary.*