import json
import os
import shutil

import pandas as pd
from tqdm import tqdm, trange

from storage.storage import Store
from util.run import run_generate, run_eval


def batch_generate(builder, num: int, store_path: str) -> list[tuple[Store, dict]]:
    """
    Generate a batch of data sets with the given builder.
    Will overwrite the previous contents of the store_path, or create the directory if it doesn't exist.
    :param builder: Dataset builder, from datagen.covshift or datagen.conceptshift
    :param num: amount of data sets to generate
    :param store_path: path to the directory containing runs
    :returns for each data set, the store referencing it, and basic set statistics
    """

    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    else:
        os.makedirs(store_path)

    res = []
    padding = len(str(num - 1))
    pbar = trange(num)
    for i in pbar:
        pbar.set_description("Generating dataset...")
        store, stats = run_generate(builder, f"{i:0{padding}}", store_path)
        res.append((store, stats))
        pbar.set_description("Finished generating dataset")
    return res


def batch_eval(store_path: str, model, model_params: dict, fit_params: dict, identifier: str = None) -> list[dict]:
    """Load all dataset stores in a directory and evaluate a model's performance on it, then store results.
    :param store_path: path to the directory containing runs
    :param model: class of the adaptation model, such as adapt DANN, ADDA, MDD etc.
    :param model_params: parameters for the model class's __init__
    :param fit_params: parameters for the model class's fit
    :param identifier: appended to the results file name if not None
    Use to prevent overwriting when evaluating multiple models on the same dataset.
    :returns: resulting dictionaries from evaluation
    """

    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"Directory with stores for batch evaluation was not found in {os.path.abspath(store_path)}")

    names = []
    for entry in os.scandir(store_path):
        if entry.is_dir():
            names.append(entry.name)

    res = []
    pbar = tqdm(names)
    for name in pbar:
        pbar.set_description(f"Evaluating on dataset {name}")
        metrics = run_eval(name, model, model_params, fit_params, identifier, store_path)
        res.append(metrics)
        pbar.set_description("Finished evaluating model")

    return res


def batch_load_eval(store_path: str) -> pd.DataFrame:
    """
    Load *all* evaluation results from all runs in a path with stores into a pandas frame.:param store_path:
    :return: pandas dataframe, one row for each eval file, for each data set.
    """

    # collect list of json dictionaries
    data = []

    for directory in os.scandir(store_path):
        if directory.is_dir():
            for file in os.scandir(directory):
                if file.name.startswith("eval"):
                    eval_json_path = file.path

                    # Read eval.json file
                    with open(eval_json_path, "r") as eval_file:
                        eval_data = json.load(eval_file)
                        eval_data["dataset"] = directory.name  # Add directory name to the data
                        eval_data["identifier"] = file.name.split("_")[1].split(".")[
                            0] if "_" in file.name else ""  # Extract the identifier
                        data.append(eval_data)

    # return the flattened JSON structure as dataframe
    return pd.json_normalize(data)
