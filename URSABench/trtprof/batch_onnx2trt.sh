#!/usr/bin/env bash

# MODEL_SUFFIX="trt32"
# TRT_FLAGS="--explicitBatch"
MODEL_SUFFIX="trt"
TRT_FLAGS="--explicitBatch --fp16"

# Convert an ONNX file to TensorRT engine
do_convert() {
    echo "Converting $1"
    time trtexec --onnx="$1" --saveEngine="${1%.onnx}.$MODEL_SUFFIX" $TRT_FLAGS
}

# Convert a folder of ONNX models to TensorRT engines; Skip if output file already exists.
convert_folder() {
    echo "Coverting models in folder $1"
    for onnx_file in $1/*.onnx; do
        trt_file="${onnx_file%.onnx}.$MODEL_SUFFIX"
        if [ -e $trt_file ]
        then
            echo "$trt_file exists: Skip"
        else
            echo "$trt_file doesn't exist: Exporting..."
            do_convert $onnx_file
        fi
    done
}

# do_convert /data/model_dense.onnx
convert_folder "/data/ResNet50_ImageNet"
