import glob
import json
import pathlib

import numpy as np
import pandas as pd


def get_results_files():
    return glob.glob("/data/*latency.ensemble*.json")


def parse_file_name(file_name):
    model_dataset, precision, _, ensemble, _ = tuple(
        pathlib.Path(file_name).name.split(".")
    )
    model, dataset = tuple(model_dataset.split("_"))
    if precision == "trt":
        precision = "FP32+FP16"
    else:
        precision = "FP32"
    ensemble = int(ensemble.lstrip("ensemble"))
    return model, dataset, precision, ensemble


def get_average_latency(file_name, n_ensemble):
    with open(file_name) as f:
        results = json.load(f)
    results = {k: v for k, v in results.items() if len(k.split(",")) == n_ensemble}
    lat_mean = np.mean([v["latency_mean"] for v in results.values()])
    std_mean = np.mean([v["latency_std"] for v in results.values()])
    return lat_mean, std_mean


if __name__ == "__main__":
    dataset_keyword = "ImageNet"
    #dataset_keyword = "MNIST"
    #dataset_keyword = "CIFAR"
    files = get_results_files()
    files = [x for x in files if dataset_keyword in x]
    results = []
    for file_name in files:
        model, dataset, precision, ensemble = parse_file_name(file_name)
        lat_mean, std_mean = get_average_latency(file_name, ensemble)
        results.append(
            {
                "dataset": dataset,
                "model": model,
                "precision": precision,
                "1_latency_mean": lat_mean,
                "2_latency_std": std_mean,
                "3_ensemble": ensemble,
            }
        )
    results = pd.DataFrame(results)
    # only keep the largest ensemble
    indices = results.groupby(["dataset", "model", "precision"], sort=True)[
        "3_ensemble"
    ].idxmax()
    results = results.loc[indices]

    results["1_latency_mean"] = results["1_latency_mean"].apply(lambda x: f"{x:.4f}s")
    results["2_latency_std"] = results["2_latency_std"].apply(
        lambda x: f"$\\pm$ {x:.4f}s"
    )
    results["3_ensemble"] = results["3_ensemble"].apply(lambda x: f"({x} models)")

    # format for latex table
    results = pd.melt(results, id_vars=["dataset", "model", "precision"])
    results = results.pivot(
        index=["precision", "variable"],
        columns=["dataset", "model"],
        values="value",
    )
    results.index = results.index.droplevel(1)
    results.to_latex(
        f"/data/result_latency_{dataset_keyword}.tex",
        caption="Caption here.",
        # index=False,
        escape=False,
        column_format="rccccc",
        multicolumn_format="c",
    )
