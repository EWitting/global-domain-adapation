"""Module for handling folders with data, metadata, and evaluation results."""

import os
import shutil
import time
import numpy as np

# names of files inside the store folder
DATA_FILE = 'data.npz'
CFG_FILE = 'config.json'


class Store:
    """Reference to a directory on disk, for storing a dataset for later retrieval, along with metadata."""

    def __init__(self, name: str, store_path: str):
        """
        Create an object referencing the storage folder on disk of a dataset, including metadata and other files.
        Does not create directory on disk. Use `Store.new` instead.

        :param name: Name of this dataset folder.
        :param store_path: Path to folder containing all stores. If None, uses '<root>/results'.
        """

        self.name = name
        self.path_full = os.path.join(store_path, name)
        if not os.path.exists(self.path_full) or not os.path.isdir(self.path_full):
            raise FileNotFoundError(
                f"Store object references non-existent path. Expected directory in '{self.path_full}'")

    @classmethod
    def new(cls, name: str = None, store_path: str = None, overwrite: bool = False):
        """
        Create a storage folder on disk for a dataset, including metadata and other files.

        :param name: Name of this dataset folder. If None, uses timestamp, adding a postfix for duplicate timestamps
        :param store_path: Path to folder containing all stores. If None, uses '<cwd>/results'.
        :param overwrite: if False, raises exception if directory already exists, if True, deletes existing
        :returns: Store object, after creating directory.
        """

        if store_path is None:
            store_path = os.path.join(os.getcwd(), 'results')

        if name is None:
            name = time.strftime("%Y-%m-%d_%H-%M-%S")
            postfix = 1
            folder_name = name
            while os.path.exists(os.path.join(store_path, folder_name)):
                folder_name = f"{name}_{postfix}"
                postfix += 1
        else:
            folder_name = name

        path_full = os.path.join(store_path, folder_name)
        if os.path.exists(path_full):
            if overwrite:
                shutil.rmtree(path_full)
            else:
                raise FileExistsError(
                    f"Attempted to create new Store, but given path already exists in '{path_full}'.\n"
                    + "Use overwrite=True if intended.")
        os.makedirs(path_full)
        print(f"Created {path_full}")

        return Store(name, store_path)

    def save_data(self, xg, yg, xs, ys, xt, yt) -> None:
        """Store three sets of features and labels in this store"""
        np.savez(os.path.join(self.path_full, DATA_FILE),
                 xg=xg, yg=yg, xs=xs, ys=ys, xt=xt, yt=yt)

    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load a dataset from this store, as the tuple (xg, yg, xs, ys, xt, yt),
        where g=global, s=source, t=target, x=features, y=label."""
        loaded = np.load(os.path.join(self.path_full, DATA_FILE))
        return (loaded['xg'], loaded['yg'],
                loaded['xs'], loaded['ys'],
                loaded['xt'], loaded['yt'])
