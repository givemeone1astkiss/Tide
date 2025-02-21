# Here are the functions and classes that are used for data preprocessing

import pandas as pd
from numpy import ndarray
from torch import tensor
from torch.nn import functional as F
import torch.utils.data as data
from typing import Union, Tuple, Any
from tqdm import tqdm
from config import *
import pytorch_lightning as pl

def load_data(path: str, target: str) -> Tuple[pd.Series, pd.Series]:
    """
    Load data from an Excel file and return x and y
    Args:
        path (str): path to the Excel file
        target (str): target column name
    """
    peptide_data = pd.read_excel(path)
    x = peptide_data["sequence"]
    y = peptide_data[target]
    return x, y

def shuffle_data(x: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Shuffle the data.
    Args:
        x (pd.Series): input data
        y (pd.Series): target data
    """
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    return x.iloc[idx], y.iloc[idx]

def to_one_hot(seqs: pd.Series) -> pd.Series:
    """
    Convert sequences to one-hot encoding.
    Args:
        seqs (pd.Series): sequences
    """

    for i, seq in enumerate(seqs):
        one_hot_seq = seq.upper()
        one_hot_seq = [AAS.index(aa) for aa in one_hot_seq]
        one_hot_seq = np.array(F.one_hot(tensor(one_hot_seq), NUM_AA_TYPE)).astype(np.float32)
        seqs[i] = one_hot_seq
    return seqs

def split_set(x: pd.Series, y: pd.Series, ratio: Union[float, Tuple[float, float]]) -> tuple[Any, Any, Any, Any]:
    """
    Split the dataset into training and testing sets.
    Args:
        x (pd.Series): input data
        y (pd.Series): target data
        ratio (Union[float, Tuple[float, float]]): ratio of the training set
    """
    if isinstance(ratio, float):
        ratio = (ratio, 1 - ratio)
    x_train, x_test = x[:int(len(x) * ratio[0])], x[int(len(x) * ratio[0]):]
    y_train, y_test = y[:int(len(y) * ratio[0])], y[int(len(y) * ratio[0]):]
    return x_train, x_test, y_train, y_test

def k_fold(x: pd.Series, y: pd.Series, k: int) -> list[tuple[tuple[Any, Any], tuple[Any, Any]]]:
    """
    Split the dataset into k folds.
    Args:
        x (pd.Series): input data
        y (pd.Series): target data
        k (int): number of folds
    """
    n = len(x)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size
        x_train = pd.concat([x[:start], x[end:]])
        y_train = pd.concat([y[:start], y[end:]])
        x_test, y_test = x[start:end], y[start:end]
        folds.append(((x_train, y_train), (x_test, y_test)))
    return folds

def load_array(x: np.ndarray, y: np.ndarray, is_train: bool, batch_size: int) -> data.DataLoader:
    """
    Load data as a DataLoader.
    Args:
        x (pd.DataFrame): input data
        y (pd.DataFrame): target data
        is_train (bool): whether it is training or not
        batch_size (int): batch size
    """
    dataset = data.TensorDataset(torch.tensor(x).float(), torch.tensor(y).float())
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


class PeptideDataset(pl.LightningDataModule):
    """
    Dataset for peptide data.
    Args:
        path (str): path to the Excel file
        ratio (float): ratio of the training set
        batch_size (int): batch size
    """
    def __init__(self, path: str, ratio: float, batch_size: int):
        super().__init__()
        x, y = load_data(path, "property")
        x, y = shuffle_data(to_one_hot(x), y)
        self.train_x, self.test_x, self.train_y, self.test_y = split_set(x, y, ratio)
        self.train_x = np.stack(self.train_x.to_list()).astype(np.float32)
        self.test_x = np.stack(self.test_x.to_list()).astype(np.float32)
        self.train_y = self.train_y.values.astype(np.float32)
        self.test_y = self.test_y.values.astype(np.float32)
        self.batch_size = batch_size
        self.ml_train_datasets = list(tqdm(self.train_x.reshape(self.train_x.shape[0],-1), desc="Training Data"))
        self.ml_val_datasets = list(tqdm(self.test_x.reshape(self.test_x.shape[0],-1), desc="Validation Data"))

    def __len__(self) -> int:
        return len(self.train_x)

    def __getitem__(self, idx) -> Tuple[ndarray, ndarray]:
        return self.train_x[idx], self.train_y[idx]

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_x, self.test_x, self.train_y, self.test_y = split_set(self.train_x, self.train_y, 0.8)

    def train_dataloader(self) -> data.DataLoader:
        return load_array(self.train_x, self.train_y, True, batch_size=self.batch_size)

    def val_dataloader(self) -> data.DataLoader:
        return load_array(self.test_x, self.test_y, False, batch_size=self.batch_size)

if __name__ == "__main__":
    pass