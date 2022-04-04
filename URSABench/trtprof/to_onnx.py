"""
Load a PyTorch model from `pth` file and export it to ONNX format.

To convert ONNX to TensorRT:

    $ /usr/src/tensorrt/bin/trtexec --onnx=/data/rn50_ensemble_2.onnx --saveEngine=/data/rn50_ensemble_2.trt --explicitBatch

where `/data/rn50_ensemble_2.onnx` is the input model and `/data/rn50_ensemble_2.trt` is the output path.
"""

import argparse
import pathlib
import time
from collections import OrderedDict

import torch
import torchvision
from URSABench import models


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    else:
        return text


def run(pickle_file, input_shape, num_classes, model_class):

    output_dir = pickle_file.parent
    model_name = pickle_file.stem
    onnx_filename = output_dir / f"{model_name}.onnx"

    print(f"Loading {args.pickle_file} ... ", end="", flush=True)
    t0 = time.perf_counter()
    if model_class == "ResNet_ImageNet":
        model = torchvision.models.resnet50().to("cuda")
        # model = torch.nn.DataParallel(model)
        checkpoint = torch.load(pickle_file, map_location="cuda")
        checkpoint = OrderedDict(
            [(remove_prefix(k, "module."), v) for k, v in checkpoint.items()]
        )
        model.load_state_dict(checkpoint)
    elif model_class:
        model = getattr(models, model_class)
        model = model.base(*model.args, num_classes=num_classes, **model.kwargs).to(
            "cuda"
        )
        checkpoint = torch.load(pickle_file, map_location="cuda")
        model.load_state_dict(checkpoint)
    else:
        model = torch.load(pickle_file, map_location=torch.device("cuda"))
    t1 = time.perf_counter()
    print(f"done in {t1-t0:.2f}s")

    dummy_input = torch.randn(*input_shape).cuda()

    print("Exporting to ONNX...", end="", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            onnx_filename,
            # store the trained parameter weights inside the model file
            export_params=True,
            # the ONNX version to export the model to
            opset_version=11,
            # whether to execute constant folding for optimization
            do_constant_folding=True,
            # the model's input names
            input_names=["input"],
            # the model's output names
            output_names=["output"],
            verbose=False,
            # dynamic_axes={
            #     # variable length axes
            #     "input": {0: "batch_size"},
            #     "output": {0: "batch_size"},
            # },
        )
    t1 = time.perf_counter()
    print(f"done in {t1-t0:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pth -> onnx")
    parser.add_argument(
        "pickle_file",
        type=pathlib.Path,
        help="Path to PyTorch pickle file.",
    )
    parser.add_argument(
        "-s",
        "--input_shape",
        type=int,
        nargs="+",
        help="Shape of input. The first int is batch size. For conv2d, NCHW.",
    )
    parser.add_argument(
        "--model_class",
        nargs="?",
        type=str,
        help="Class of the model",
    )
    parser.add_argument(
        "--num_classes", type=int, help="Number of classification classes."
    )
    parser.set_defaults(model_class=None, num_classes=10)
    args = parser.parse_args()
    print(args)
    run(args.pickle_file, args.input_shape, args.num_classes, args.model_class)
