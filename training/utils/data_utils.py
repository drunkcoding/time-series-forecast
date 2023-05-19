import pandas as pd
import numpy as np
from tqdm import trange, tqdm
import torch
from sklearn.model_selection import KFold
import scipy.stats as stats
import multiprocessing as mp

def read_dataset(filepath: str)->pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.drop(columns=["time"])

    columns = df.columns.values

    df = df / 1000
    df = df.replace(0, np.nan)

    # array = df.to_numpy()
    # array[array == 0] = np.nan
    # array /= 1000

    return df

def train_test_split(filepath: str, train_split: float = 0.6, clip=True):
    df = pd.read_csv(filepath)
    df = df.drop(columns=["time"])

    # df = df.loc[:, (df != 0).any(axis=0)]
    columns = df.columns.values
    # print(columns)

    df_len = len(df.index.values)
    train_len = int(df_len * train_split)

    array = df.to_numpy()
    array[array == 0] = np.nan
    array /= 1000

    # run z-score outlier test
    z_score = np.abs(stats.zscore(array, nan_policy="omit"))
    z_score_rate = np.sum(z_score > 2.326) / np.prod(z_score.shape)
    print(filepath, z_score_rate)
   
    if clip:
        clip_percentile = 100 - z_score_rate * 1000
        clip_threshold = np.percentile(array, clip_percentile)
        array[array > clip_threshold] = np.nan
        # array = np.clip(array, 0, np.percentile(array, 100 - z_score_rate * 1000))
    # if z_score_rate >= 0.01:
    #     array = np.clip(array, 0, np.percentile(array, 90))

    training_set = array[:train_len]
    test_set = array[train_len:]

    # mean = np.nanmean(training_set, axis=0)
    # std = np.nanstd(training_set, axis=0)

    # mean[np.isnan(mean)] = 0
    # std[std == 0] = 1
    # std[np.isnan(std)] = 1

    # mean =0
    # std = 1

    # training_set_scaled = (training_set - mean) / std
    # test_set_scaled = (test_set - mean) / std

    min = np.nanmin(training_set, axis=0)
    max = np.nanmax(training_set, axis=0)

    training_set_scaled = (training_set - min) / (max - min)
    test_set_scaled = (test_set - min) / (max - min)

    training_set_scaled[np.isnan(training_set_scaled)] = 0
    test_set_scaled[np.isnan(test_set_scaled)] = 0

    # return training_set_scaled, test_set_scaled, {"mean": mean, "std": std}
    return training_set_scaled, test_set_scaled, {"min": min, "max": max}


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in trange(len(dataset) - look_back):
        dataX.append(dataset[i : i + look_back])
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def create_torch_dataloader(
    dataX, dataY, batch_size=1, shuffle=False
):
    if isinstance(dataX, np.ndarray):
        dataX = torch.from_numpy(dataX).float()
    if isinstance(dataY, np.ndarray):
        dataY = torch.from_numpy(dataY).float()

    dataset = torch.utils.data.TensorDataset(dataX, dataY)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0 # mp.cpu_count()
    )


def create_kfold_dataset(dataX, dataY, n_splits=10):
    kf = KFold(n_splits=n_splits)
    # fold_len = int(len(dataX) / n_splits)

    X_train_fold_list = []
    y_train_fold_list = []
    X_test_fold_list = []
    y_test_fold_list = []
    for train_index, test_index in kf.split(dataX):
        X_train_fold = dataX[train_index]
        y_train_fold = dataY[train_index]
        X_test_fold = dataX[test_index]
        y_test_fold = dataY[test_index]

        if (
            np.all(X_train_fold == 0)
            or np.all(y_train_fold == 0)
            or np.all(X_test_fold == 0)
            or np.all(y_test_fold == 0)
        ):
            continue

        # print(X_train_fold.shape)
        # print(y_train_fold.shape)
        # print(X_test_fold.shape)
        # print(y_test_fold.shape)

        X_train_fold_list.append(X_train_fold)
        y_train_fold_list.append(y_train_fold)
        X_test_fold_list.append(X_test_fold)
        y_test_fold_list.append(y_test_fold)

    # find the min length of the folds
    train_min_len = 1e10
    test_min_len = 1e10
    for i in range(len(X_train_fold_list)):
        if len(X_train_fold_list[i]) < train_min_len:
            train_min_len = len(X_train_fold_list[i])
        if len(X_test_fold_list[i]) < test_min_len:
            test_min_len = len(X_test_fold_list[i])

    # truncate the folds to the min length
    for i in range(len(X_train_fold_list)):
        X_train_fold_list[i] = X_train_fold_list[i][:train_min_len]
        y_train_fold_list[i] = y_train_fold_list[i][:train_min_len]
        X_test_fold_list[i] = X_test_fold_list[i][:test_min_len]
        y_test_fold_list[i] = y_test_fold_list[i][:test_min_len]

        # print(X_train_fold_list[i].shape)
        # print(y_train_fold_list[i].shape)
        # print(X_test_fold_list[i].shape)
        # print(y_test_fold_list[i].shape)

    X_train_fold_list = np.stack(X_train_fold_list)
    y_train_fold_list = np.stack(y_train_fold_list)
    X_test_fold_list = np.stack(X_test_fold_list)
    y_test_fold_list = np.stack(y_test_fold_list)

    # print(X_train_fold_list.shape)
    # print(y_train_fold_list.shape)
    # print(X_test_fold_list.shape)
    # print(y_test_fold_list.shape)

    return X_train_fold_list, y_train_fold_list, X_test_fold_list, y_test_fold_list
