from dataclasses import dataclass, field
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import HfArgumentParser
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm, trange
from numpy.lib.stride_tricks import sliding_window_view
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from sklearn.preprocessing import MinMaxScaler
import torch.distributed as dist

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        # out[out < 0] = 0
        # out[out > 0] = 1
        return out

@dataclass
class ModelConfig:
    data: str = field(metadata={"help": "path to csv data"})

if __name__ == "__main__":
    parser = HfArgumentParser(ModelConfig)
    args = parser.parse_args_into_dataclasses()[0]

    # dist.init_process_group("nccl")
    # rank = dist.get_rank()
    # device_id = rank % torch.cuda.device_count()

    # load data
    df = pd.read_csv(args.data)
    df = df.drop(columns=["time"])
    # drop columns that are all 0
    df = df.loc[:, (df != 0).any(axis=0)]

    # get columns
    columns = df.columns.values
    print(columns)
    # print(df.loc[0, :].to_numpy().tolist())

    df_len = len(df.index.values)
    train_len = int(df_len * 0.6)

    array = df.to_numpy()
    kf = KFold(n_splits=10)

    # print(np.min(array[array > 0]))
    
    # array = array + 1
    # array[array == 0] = np.nan

    # clip dataset with 99% percentile
    # array = np.clip(array, 0, np.percentile(array, 99))

    training_set = array[:train_len]
    test_set = array[train_len:]

    # sc = MinMaxScaler(feature_range=(0, 1))
    # training_set_scaled = sc.fit_transform(training_set)
    # test_set_scaled = sc.transform(test_set)

    # z-score normalization
    outlier_threshold = np.percentile(training_set, 90)
    training_set[training_set < outlier_threshold] = 0
    test_set[test_set < outlier_threshold] = 0

    training_set_scaled = training_set
    test_set_scaled = test_set
    training_set_scaled[training_set > 0] = 1
    test_set_scaled[test_set_scaled > 0] = 1

    print(np.sum(test_set_scaled) / np.multiply(*test_set_scaled.shape))

    # training_set_trimmed = training_set[training_set > 0]
    # mean = np.mean(training_set_trimmed)
    # std = np.std(training_set_trimmed)
    # training_set_scaled = (training_set - mean) / std
    # test_set_scaled = (test_set - mean) / std

    # training_set_scaled[np.isnan(training_set_scaled)] = 0
    # test_set_scaled[np.isnan(test_set_scaled)] = 0

    # array_scaled = np.log(array)

    X_train = []
    y_train = []
    for i in trange(60, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - 60 : i, :].tolist())
        y_train.append(training_set_scaled[i, :].tolist())
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test = []
    y_test = []
    for i in trange(60, len(test_set_scaled)):
        X_test.append(test_set_scaled[i - 60 : i, :].tolist())
        y_test.append(test_set_scaled[i, :].tolist())
    X_test, y_test = np.array(X_test), np.array(y_test)

    X_test[np.isnan(X_test)] = 0
    # test_nan = np.isnan(test_set_scaled)
    # test_set_scaled[np.isnan(test_set_scaled)] = 0

    model = LSTMModel(144, 128, 2, 144)
    model = model.to("cuda:0")
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)

    for epoch in trange(100, position=0, leave=True):
        for train_index, test_index in kf.split(X_train):
            X_train_fold = X_train[train_index]
            y_train_fold = y_train[train_index]
            X_test_fold = X_train[test_index]
            y_test_fold = y_train[test_index]
            
            # # add one dimension
            # y_train_fold = y_train_fold[:, :, np.newaxis]
            # y_test_fold = y_test_fold[:, :, np.newaxis]

            X_train_fold = torch.from_numpy(X_train_fold).float()
            y_train_fold = torch.from_numpy(y_train_fold).float()
            X_test_fold = torch.from_numpy(X_test_fold).float()
            y_test_fold = torch.from_numpy(y_test_fold).float()

            X_train_fold = X_train_fold.to("cuda:0")
            y_train_fold = y_train_fold.to("cuda:0")
            X_test_fold = X_test_fold.to("cuda:0")
            y_test_fold = y_test_fold.to("cuda:0")

            print(X_train_fold.shape, y_train_fold.shape, X_test_fold.shape, y_test_fold.shape)

            # create data loader batch size 1024
            train_dataset = torch.utils.data.TensorDataset(X_train_fold, y_train_fold)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)

            model.train()

            
            for i, (X_train_fold, y_train_fold) in enumerate(train_loader):
                outputs = model(X_train_fold)
                loss = criterion(outputs, y_train_fold)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

            model.eval()

            with torch.no_grad():
                test_predict = model(X_test_fold)

            test_predict = test_predict.data.cpu().numpy()
            test_y = y_test_fold.data.cpu().numpy()

            print(test_predict.shape, test_y.shape)
            
            # test_predict = sc.inverse_transform(test_predict)
            # test_y = sc.inverse_transform(test_y)

            # # inverse z-score normalization
            # test_predict = test_predict * std + mean
            # test_y = test_y * std + mean

            # print(test_predict.shape, test_y.shape)

            # # print(test_predict[0, :])
            # # print(test_y[0, :])

            # print("MAE: %.5f" % np.mean(np.abs(test_predict - test_y)))

        model.eval()
        with torch.no_grad():
            test_predict = model(torch.from_numpy(X_test).float().to("cuda:0"))

        test_predict = test_predict.data.cpu().numpy()
        test_y = y_test

        test_predict -= 0.5
        test_predict[test_predict < 0] = 0
        test_predict[test_predict > 0] = 1

        print(test_predict, test_y)

        # accuracy
        print("Accuracy: %.5f" % np.mean((test_predict == test_y)[test_y > 0]))

        # test_predict = sc.inverse_transform(test_predict)
        # test_y = sc.inverse_transform(test_y)

        # # inverse z-score normalization
        # test_predict = test_predict * std + mean
        # test_y = test_y * std + mean

        # print("MAE: %.5f" % np.nanmean(np.abs(test_predict - test_y)))
        