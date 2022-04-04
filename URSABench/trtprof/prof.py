"""
TensorRT 7.1.3 Python API:
    https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/api/python_api/index.html
"""

import argparse
import pathlib
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
from torch.utils.data import DataLoader
from URSABench.trtprof.dataset import DummyDataset
from URSABench.trtprof.utils import *


class TensorRTModel:
    """Create a TensorRT model for inference from the given `trt` file.

    See: https://github.com/NVIDIA/TensorRT/blob/eb8442dba3c9e85ffb77e0d870d2e29adcb0a4aa/quickstart/IntroNotebooks/onnx_helper.py"""

    def __init__(self, file: Path, num_classes: int, target_dtype=np.float16):

        self.target_dtype = target_dtype
        self.num_classes = num_classes
        self._load(file)
        self._allocate()
        # self.stream = None

    def _load(self, file: Path):
        logger.debug(f"Loading {file}")
        t0 = time.perf_counter()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(file, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        t1 = time.perf_counter()
        logger.debug(f"{file} loaded in {t1-t0:.4f}s")

    def _allocate(self):
        bindings = []

        # allocate for input
        input_size = trt.volume(self.engine.get_binding_shape("input"))
        input_size *= self.engine.max_batch_size
        input_dtype = trt.nptype(self.engine.get_binding_dtype("input"))
        input_host_mem = cuda.pagelocked_empty(input_size, input_dtype)
        input_device_mem = cuda.mem_alloc(input_host_mem.nbytes)
        bindings.append(int(input_device_mem))

        # allocate for output
        output_size = trt.volume(self.engine.get_binding_shape("output"))
        output_size *= self.engine.max_batch_size
        output_dtype = trt.nptype(self.engine.get_binding_dtype("output"))
        output_host_mem = cuda.pagelocked_empty(output_size, output_dtype)
        output_device_mem = cuda.mem_alloc(output_host_mem.nbytes)
        bindings.append(int(output_device_mem))

        self.input_host_mem = input_host_mem
        self.input_device_mem = input_device_mem
        self.output_host_mem = output_host_mem
        self.output_device_mem = output_device_mem
        self.bindings = bindings
        self.stream = cuda.Stream()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.input_host_mem = X
        cuda.memcpy_htod_async(self.input_device_mem, self.input_host_mem, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.output_host_mem, self.output_device_mem, self.stream
        )
        self.stream.synchronize()
        return self.output_host_mem


class TensorRTEnsemble(TensorRTModel):
    """Create an ensemble model by duplicating a TensorRTModel for several times."""

    def __init__(
        self,
        file: Path,
        num_classes: int,
        target_dtype=np.float32,
        n_ensembles: int = 30,
    ):
        self.models = [
            TensorRTModel(file, num_classes, target_dtype) for _ in range(n_ensembles)
        ]

    def __call__(self, X: np.ndarray) -> np.ndarray:

        output_list = [mlp(X) for mlp in self.models]
        output = np.stack(output_list).mean()
        return output


class PyTorchModel:
    def __init__(self, file: Path):
        self.model = self._load(file)

    def _load(self, file):
        print(f"Loading {file} ... ", end="", flush=True)
        t0 = time.perf_counter()
        model = torch.load(file, map_location=torch.device("cuda"))
        t1 = time.perf_counter()
        print(f"\r{file} loaded in {t1-t0:.4f}s")
        model.eval()
        return model

    def __call__(self, X):
        X = X.to("cuda")
        with torch.no_grad():
            pred = self.model(X)
            torch.cuda.synchronize()
        pred = pred.cpu()
        return pred


class PyTorchEnsemble(PyTorchModel):
    def __init__(self, file: Path, n_ensemble: int = 30):
        m = super()._load(file)
        self.model = [deepcopy(m) for _ in range(n_ensemble)]

    def __call__(self, X):
        X = X.to("cuda")
        with torch.no_grad():
            output_list = [mlp(X) for mlp in self.model]
            output = torch.stack(output_list).mean(dim=0)
            torch.cuda.synchronize()
        output = output.cpu()
        return output


def warm_up(model, dataloader, n):
    print(f"Warming up ... ", end="", flush=True)
    for i, (batch_x, _) in enumerate(dataloader):
        if i > n:
            break
        if isinstance(model, TensorRTModel):
            batch_x = batch_x.numpy()
            model.input_to_gpu(batch_x)
        model(batch_x)
    print(f"done: {n} runs")


def time_batch(model, dataloader, repetition=10):
    print(f"Profiling ... ", end="", flush=True)
    results = []
    for i, (batch_x, _) in enumerate(dataloader):
        latency = []
        if isinstance(model, TensorRTModel):
            batch_x = batch_x.numpy()
            model.input_to_gpu(batch_x)
        for j in range(repetition):
            print(f"\rProfiling ... batch {i}, rep {j}", end="", flush=True)
            t0 = time.perf_counter()
            pred = model(batch_x)
            t1 = time.perf_counter()
            latency.append(t1 - t0)
        results.append(
            {"batch": i, "latency": np.mean(latency), "latency_std": np.std(latency)}
        )
    print(f"\rProfiling ... done: {i+1} batches.")
    return pd.DataFrame(results)


def print_stats(df, model_name):
    print(f"{model_name}: {df.latency.mean():.4f} +/- {df.latency.std():.4f} s")


def run(model_file, is_ensemble, target_dtype):

    batch_size = 1
    n_samples = 32
    n_classes = 10
    n_ensembles = 3
    x = DummyDataset(32, n_samples, np.float32)
    dataloader = DataLoader(x, batch_size=batch_size, shuffle=False)
    # x = MLPDataset(n_samples, n_feats)
    # dataloader = DataLoader(x, batch_size=batch_size, shuffle=False)

    if target_dtype == 32:
        target_dtype = np.float32
    elif target_dtype == 16:
        target_dtype = np.float16

    if model_file.suffix == ".pth" and not is_ensemble:
        model = PyTorchModel(model_file)
    elif model_file.suffix == ".pth" and is_ensemble:
        model = PyTorchEnsemble(model_file, n_ensemble=n_ensembles)
    elif model_file.suffix == ".trt" and not is_ensemble:
        model = TensorRTModel(model_file, n_classes, target_dtype=target_dtype)
    elif model_file.suffix == ".trt" and is_ensemble:
        model = TensorRTEnsemble(
            model_file, n_classes, n_ensembles=n_ensembles, target_dtype=target_dtype
        )
    else:
        raise

    print(f"CPU ensemble is {is_ensemble}, so use {type(model).__name__}")

    warm_up(model, dataloader, 30)
    lat = time_batch(model, dataloader)
    if is_ensemble:
        print_name = model_file.name + ".cpu_ensemble"
    else:
        print_name = model_file.name
    print_stats(lat, print_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profiling")
    parser.add_argument(
        "model_file",
        type=pathlib.Path,
        help="Path to pth or trt file.",
    )
    parser.add_argument("--cpu-ensemble", dest="is_ensemble", action="store_true")
    parser.add_argument(
        "--target-dtype", dest="target_dtype", type=int, action="store_true"
    )
    parser.set_defaults(is_ensemble=False, target_dtype=32)
    args = parser.parse_args()
    run(args.model_file, args.is_ensemble, args.target_dtype)
