#!/usr/bin/env bash

# Convert a folder of PyTorch models (pt, pth) to ONNX format.

INPUT_DIR="/data/ResNet50_ImageNet"
NUM_CLASSES=1000
MODEL_CLASSES="ResNet_ImageNet"

for input_file in $INPUT_DIR/*.pt; do
    onnx_file="${input_file%.pt}.onnx"
    if [ -e $onnx_file ]
    then
        echo "$onnx_file exists: Skip"
    else
        echo "$onnx_file doesn't exist: Exporting..."
        python3 to_onnx.py $input_file -s 1 3 224 224 --model_class $MODEL_CLASSES || exit 1
    fi
done
