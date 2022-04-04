import argparse
import datetime
import glob
import json
import pathlib
import re
import sys
import time
from collections import defaultdict

# Caused by the symbolic link issue, e.g.: "OSError: libcublasLt.so.10: cannot
# open shared object file: Too many levels of symbolic links"
try:
    import URSABench
except (OSError, ImportError):
    sys.exit(3)

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from URSABench import datasets
from URSABench import models
from URSABench.trtprof.dataset import DummyDataset
from URSABench.trtprof.dataset import MLPDataset
from URSABench.trtprof.prof import TensorRTModel
from URSABench.trtprof.utils import logger

# experiment parameters
# IMG_HW = 32
IMG_HW = 224
BATCH_SIZE = 1
LATENCY_MODE_SAMPLE_SIZE = 100
SHUFFLE = False
DEVICE = torch.device("cuda")


def load_results(json_file):
    try:
        with open(json_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def dump_results(json_file, results):
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)


def pytorch_from_stat_dict(checkpoint_file, model_cfg, n_classes):
    model = model_cfg.base(
        *model_cfg.args, num_classes=n_classes, **model_cfg.kwargs
    ).to("cuda")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint)
    return model


def load_model(model_file, model_suffix, model_cfg, num_classes):
    if model_suffix == "pt":
        model = pytorch_from_stat_dict(model_file, model_cfg, num_classes)
    elif model_suffix in ["trt", "trt32"]:
        model = TensorRTModel(model_file, num_classes)
    else:
        raise ValueError
    return model


def get_latency(latencies, burn_in=10):
    by_batch = defaultdict(int)
    for x in latencies:
        if x["batch_idx"] > burn_in:
            by_batch[x["batch_idx"]] += x["latency"]
    lat = np.array(list(by_batch.values()))
    lat_mean = np.mean(lat)
    lat_avg = np.std(lat)
    return lat_mean, lat_avg


def chunks(long_list, n):
    for i in range(0, len(long_list), n):
        yield long_list[i : i + n]


def parse_model_number(fname):
    return int(fname.split(".")[0].split("_")[-1])


def batch_ensemble(file_list, n_ensemble):
    file_list = sorted(file_list, key=parse_model_number)
    ensembles = list(chunks(file_list, n_ensemble))
    ensembles = [sorted(l, key=parse_model_number) for l in ensembles]
    ensembles = {",".join(l): l for l in ensembles}
    return ensembles


def run(
    model_dir: pathlib.Path,
    model_suffix: str,
    profile_mode: str,
    output_suffix: str,
    n_ensemble: int,
):
    model_class_name, dataset_name = tuple(model_dir.name.split("_"))
    assert model_class_name in [
        "WideResNet28x10",
        "INResNet50",
        "MLP200MNIST",
        "ResNet50",
    ]
    assert dataset_name in ["CIFAR10", "CIFAR100", "MNIST", "ImageNet"]
    assert model_suffix in ["pt", "trt", "trt32"]
    assert profile_mode in ["latency", "metrics"]

    dataset_path = f"/data/{dataset_name.lower()}"
    output_json_path = (
        f"{model_dir}.{model_suffix}.{profile_mode}.{output_suffix}{n_ensemble}.json"
    )

    logger.debug(f"model_class_name: {model_class_name}")
    logger.debug(f"model_suffix: {model_suffix}")
    logger.debug(f"profile_mode: {profile_mode}")
    logger.debug(f"dataset_path: {dataset_path}")
    logger.debug(f"output_json_path: {output_json_path}")

    # load cache
    results = load_results(output_json_path)

    # glob model files
    model_files = glob.glob(f"{model_dir}/*.{model_suffix}")
    ensemble_batches = batch_ensemble(model_files, n_ensemble)
    n_total_models = len(ensemble_batches)
    n_processed = len(results)
    logger.debug(f"{n_total_models} in total, {n_processed} already processed")
    # to_be_profiled = [x for x in model_files if x not in results]
    to_be_profiled = {k: v for k, v in ensemble_batches.items() if k not in results}

    if len(to_be_profiled) == 0:
        logger.info("All done.")
        sys.exit(0)

    # model config
    try:
        model_cfg = getattr(models, model_class_name)
    except AttributeError:
        model_cfg = None

    # setup dataloader
    if profile_mode == "latency" and dataset_name in [
        "CIFAR10",
        "CIFAR100",
        "ImageNet",
    ]:
        dummy_dataset = DummyDataset(IMG_HW, LATENCY_MODE_SAMPLE_SIZE, np.float32)
        test_loader = DataLoader(dummy_dataset, batch_size=BATCH_SIZE)
        if dataset_name == "ImageNet":
            num_classes = 1000
        else:
            num_classes = int(re.sub("\D", "", dataset_name))
        logger.debug("Using DummyDataset")
    elif profile_mode == "latency" and dataset_name in ["MNIST"]:
        dummy_dataset = MLPDataset(100, 10)
        test_loader = DataLoader(dummy_dataset, batch_size=BATCH_SIZE)
        num_classes = 10
        logger.debug("Using MLPDataset")
    else:
        URSABench.set_random_seed(0)
        loaders, num_classes = datasets.loaders(
            dataset_name,
            dataset_path,
            BATCH_SIZE,
            0,
            transform_train=model_cfg.transform_train,
            transform_test=model_cfg.transform_test,
            shuffle_train=True,
            use_validation=False,
        )
        test_loader = loaders["test"]
        logger.debug("Using real dataset")
    dataloader = {"in_distribution_test": test_loader}
    logger.debug(
        f"dataloader: {num_classes} classes, {test_loader.batch_size} batch_size"
    )
    logger.debug(
        f"dataloader: {len(test_loader)} batches, {len(test_loader.dataset)} samples"
    )

    # prediction instance
    if profile_mode == "latency":
        latency_mode = True
        metrics = ["ll"]
    else:
        latency_mode = False
        metrics = "ALL"

    t0 = time.perf_counter()
    predict = URSABench.tasks.Prediction(
        dataloader=dataloader,
        metric_list=metrics,
        num_classes=num_classes,
        device=DEVICE,
        latency_mode=latency_mode,
    )
    t1 = time.perf_counter()
    logger.debug(f"Prediction instance created in {t1-t0:.4f}s.")

    # for i, model_file in enumerate(to_be_profiled):
    # for ensemble_name, ensemble_files in to_be_profiled.items():

    # only run one batch of models a time; repeat in bash script which calls
    # this script to finish the whole task, so that avoids fluctuations.
    ensemble_name, ensemble_files = to_be_profiled.popitem()

    logger.debug(f"{ensemble_files}")

    t0 = time.perf_counter()
    model_list = [
        load_model(model_file, model_suffix, model_cfg, num_classes)
        for model_file in ensemble_files
    ]
    t1 = time.perf_counter()
    model_load_time = t1 - t0
    logger.debug(f"{len(model_list)} models loaded in {t1-t0:.4f}s")

    t0 = time.perf_counter()
    predict.update_statistics(model_list, output_performance=False)
    t1 = time.perf_counter()
    logger.debug(
        f"Availabel mem: {psutil.virtual_memory().available / (1024 * 1024)} MB"
    )

    val_dict = predict.get_performance_metrics()
    lat_mean, lat_std = get_latency(predict.latencies)
    val_dict["latency_mean"] = lat_mean
    val_dict["latency_std"] = lat_std
    # include model load time in the result
    val_dict["model_load_time"] = model_load_time
    val_dict["added_time"] = datetime.datetime.now().isoformat()
    results[ensemble_name] = val_dict

    dump_results(output_json_path, results)
    logger.info(f"Ensemble profiled in {t1-t0:.4f} s.")

    logger.info(
        f"{len(results)}/{n_total_models}: {n_total_models - len(results)} left."
    )

    # only finished one batch
    sys.exit(4)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Run models.Prediction")
        parser.add_argument(
            "model_dir",
            type=pathlib.Path,
            help="Path to the directory of the `pt` files or `trt` files.",
        )
        parser.add_argument(
            "model_suffix", type=str, help="Type of models: `pt` or `trt`."
        )
        parser.add_argument(
            "profile_mode", type=str, help="Profiling mode: `latency` or `metrics`."
        )
        parser.add_argument("output_suffix", type=str, help="Suffix to output json")
        parser.add_argument(
            "n_ensemble", type=int, help="Number of ensemble component."
        )
        args = parser.parse_args()
        run(
            args.model_dir,
            args.model_suffix,
            args.profile_mode,
            args.output_suffix,
            args.n_ensemble,
        )
    except KeyboardInterrupt:
        sys.exit(0)
