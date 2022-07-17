from datetime import datetime
import glob
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from forecast.utils.timefeatures import time_features

import numpy as np
import torch
import os
import logging
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger('DeepAR.Data')

class TrainDataset(Dataset):
    def __init__(self, data_path, data_name, num_class):
        self.data = np.load(os.path.join(data_path, f'train_data_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, f'train_label_{data_name}.npy'))
        self.train_len = self.data.shape[0]
        logger.info(f'train_len: {self.train_len}')
        logger.info(f'building datasets from {data_path}...')

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]), self.label[index])

class TestDataset(Dataset):
    def __init__(self, data_path, data_name, num_class):
        self.data = np.load(os.path.join(data_path, f'test_data_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, f'test_v_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, f'test_label_{data_name}.npy'))
        self.test_len = self.data.shape[0]
        logger.info(f'test_len: {self.test_len}')
        logger.info(f'building datasets from {data_path}...')

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]),self.v[index],self.label[index])

class WeightedSampler(Sampler):
    def __init__(self, data_path, data_name, replacement=True):
        v = np.load(os.path.join(data_path, f'train_v_{data_name}.npy'))
        self.weights = torch.as_tensor(np.abs(v[:,0])/np.sum(np.abs(v[:,0])), dtype=torch.double)
        logger.info(f'weights: {self.weights}')
        self.num_samples = self.weights.shape[0]
        logger.info(f'num samples: {self.num_samples}')
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples


class SNDLibDatasetBase(Dataset):
    def __init__(self, path) -> None:
        super().__init__()

        if os.path.isdir(path):
            data_folder = os.path.join(path, "*.xml")
            self.paths = glob.glob(data_folder)
        else:
            self.paths = [path]

        self.__parse_xml()

    def __parse_xml(self):
        # Get node list
        pattern = r'<node id="([A-Za-z0-9\.]+)">'
        example_file = self.paths[0]
        with open(example_file, "r") as f:
            data = f.read()
            all_groups = re.findall(pattern, data)
            nodes = all_groups

        records = {f"{i}_{j}": list() for i in nodes for j in nodes if i != j}
        timestamps = []

        pattern = r'<demand id="([_A-Za-z0-9\.]+)">\n   <source>([A-Za-z0-9\.]+)</source>\n   <target>([A-Za-z0-9\.]+)</target>\n   <demandValue> (\d+.\d+) </demandValue>'

        for name in tqdm(self.paths, desc="parse xml"):
            timestamp = name.split(".")[0]
            timestamp = timestamp.split("-")[-2:]
            timestamp = " ".join(timestamp)

            datetime_obj = datetime.strptime(timestamp, "%Y%m%d %H%M")
            # datetime_str = datetime.strftime(datetime_obj, "%Y-%m-%d %H:%M:%S")

            timestamps.append(datetime_obj)

            with open(
                name,
                "r",
            ) as f:
                data = f.read()

            for id in records:
                records[id].append(None)

            all_groups = re.findall(pattern, data)
            for group in all_groups:
                id, source, target, value = group
                records[id][-1] = float(value)

        self.df_raw = pd.DataFrame(records)
        # node_pairs = df.columns
        self.df_raw["date"] = timestamps
        # self.df_raw = self.df_raw.fillna(0)

    def save_data(self):  
        path = os.path.dirname(self.paths[0])
        filepath = os.path.join(path, "all.csv")
        self.df_raw.to_csv(filepath, index=False)


class SNDLibDataset(SNDLibDatasetBase):
    def __init__(self, path, pair=None, size=None, flag="train") -> None:
        super().__init__(path)

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.pair = pair
        # self.df_raw = pd.read_csv(path)

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.cols_data = self.df_raw.columns[1:] if self.pair is None else [self.pair]

        df_stamp = self.df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=0, freq='M')

        self.records = {}
        for col in self.cols_data:
            df_data = self.df_raw[[col]]
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)

            # print(data.shape)

            self.records[col] = {
                'data_x': data[border1:border2],
                'data_y': data[border1:border2],
                'data_stamp': data_stamp,
            }

        self.record_index = 0
        self.record_keys = list(self.records.keys())    

    def __getitem__(self, index):

        key = self.record_keys[self.record_index]
        self.record_index = self.record_index % len(self.record_keys)

        data_x = self.records[key]['data_x']
        data_y = self.records[key]['data_y']
        data_stamp = self.records[key]['data_stamp']

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data_x[s_begin:s_end]
        # if self.inverse:
        #     seq_y = np.concatenate(
        #         [
        #             data_x[r_begin : r_begin + self.label_len],
        #             data_y[r_begin + self.label_len : r_end],
        #         ],
        #         0,
        #     )
        # else:
        #     seq_y = data_y[r_begin:r_end]
        seq_y = data_y[r_begin:r_end]
        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        data_x = self.records[self.record_keys[0]]['data_x']
        return (len(data_x) - self.seq_len - self.pred_len + 1) * len(self.record_keys)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_SNDLib(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="h",
        cols=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [
                    self.data_x[r_begin : r_begin + self.label_len],
                    self.data_y[r_begin + self.label_len : r_end],
                ],
                0,
            )
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_SNDLib_Multi(Dataset):
    def __init__(
        self,
        data_path,
        flag="train",
        size=None,
        features="S",
        target="OT",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="h",
        cols=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index * self.seq_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [
                    self.data_x[r_begin : r_begin + self.label_len],
                    self.data_y[r_begin + self.label_len : r_end],
                ],
                0,
            )
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return int((len(self.data_x) - self.seq_len - self.pred_len + 1) / self.seq_len)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
