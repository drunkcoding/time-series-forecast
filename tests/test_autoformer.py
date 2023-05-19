from dataclasses import dataclass, field
from datetime import datetime
import glob
import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader

from forecast.data.dataset import SNDLibDataset
from forecast.model.autoformer import Autoformer
from forecast.utils.cmdparser import HfArgumentParser, ModelArguments, DataArguments

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm, trange

from forecast.utils.timefeatures import time_features


@dataclass
class ModelConfig:
    folder: str = field(metadata={"help": "folder for xml data"})
    checkpoint: str = field(metadata={"help": "path to checkpoints"})
    fill: str = field(
        metadata={
            "help": "fill with values ['zero': fill with zeros, 'ffill': forward fill, 'backfill': backward fill]"
        }
    )
    debug: bool = field(default=False)

    def __post_init__(self):
        try:
            os.mkdir(self.checkpoint)
        except:
            pass


device = "cuda:0"

parser = HfArgumentParser((ModelArguments, ModelConfig))
model_args, data_args = parser.parse_args_into_dataclasses()


class SNDLibDatasetBase(Dataset):
    def __init__(self, args) -> None:
        super().__init__()

        path = args.folder
        self.fill = args.fill

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

            with open(name, "r",) as f:
                data = f.read()

            for id in records:
                records[id].append(None)

            all_groups = re.findall(pattern, data)
            for group in all_groups:
                id, source, target, value = group
                records[id][-1] = float(value)

        df = pd.DataFrame(records)
        df = df.fillna(0) if self.fill == "zero" else df.fillna(method=self.fill)
        if df.isnull().values.any():
            df = df.fillna(0)
        self.df_raw = df
        # node_pairs = df.columns
        self.df_raw["date"] = timestamps

        self.data_len = len(self.df_raw.index.values)
        self.train_len = train_len = int(self.data_len * 0.6)
        self.test_len = self.data_len - self.train_len
        # self.df_raw = self.df_raw.fillna(0)

    def save_data(self, tag: str):
        path = os.path.dirname(self.paths[0])
        filepath = os.path.join(path, f"{tag}.csv")
        self.df_raw.to_csv(filepath, index=False)


class SNDLibDataset(SNDLibDatasetBase):
    def __init__(self, model_args, data_args, flag="train") -> None:
        super().__init__(data_args)

        size = (model_args.seq_len, model_args.label_len, model_args.pred_len)

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = model_args.features

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            0,
            self.train_len
            # 12 * 30 * 24 - self.seq_len,
            # 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            # 12 * 30 * 24,
            self.train_len,
            0,
            self.data_len
            # 12 * 30 * 24 + 4 * 30 * 24,
            # 12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_stamp = self.df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=0, freq="t")

        cols_data = list(self.df_raw.columns)
        cols_data.remove("date")
        if self.features == "M" or self.features == "MS":
            df_data = self.df_raw[cols_data]
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_stamp = data_stamp

            print(self.data_x.shape, self.data_y.shape, self.data_stamp.shape)
        elif self.features == "S":
            self.records = {}
            for col in cols_data:
                df_data = self.df_raw[[col]]
                train_data = df_data[border1s[0] : border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)

                self.records[col] = {
                    "data_x": data[border1:border2],
                    "data_y": data[border1:border2],
                    "data_stamp": data_stamp,
                }

            self.record_index = 0
            self.record_keys = list(self.records.keys())

        # if self.features == "M" or self.features == "MS":
        #     self.data_x = data[border1:border2]
        #     self.data_y = data[border1:border2]
        #     self.data_stamp = data_stamp
        # elif self.features == "S":
        #     pass

        # self.cols_data = self.df_raw.columns[1:] if self.pair is None else [self.pair]

        # self.records = {}
        # for col in self.cols_data:
        #     df_data = self.df_raw[[col]]
        #     train_data = df_data[border1s[0] : border2s[0]]
        #     self.scaler.fit(train_data.values)
        #     data = self.scaler.transform(df_data.values)

        #     # print(data.shape)

        #     self.records[col] = {
        #         "data_x": data[border1:border2],
        #         "data_y": data[border1:border2],
        #         "data_stamp": data_stamp,
        #     }

        # self.record_index = 0
        # self.record_keys = list(self.records.keys())

    def __getitem__(self, index):

        s_begin = index % self.__unit_len()
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.features == "M" or self.features == "MS":
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
        elif self.features == "S":
            key = self.record_keys[self.record_index]
            self.record_index = self.record_index % len(self.record_keys)

            data_x = self.records[key]["data_x"]
            data_y = self.records[key]["data_y"]
            data_stamp = self.records[key]["data_stamp"]

            seq_x = data_x[s_begin:s_end]
            seq_y = data_y[r_begin:r_end]
            seq_x_mark = data_stamp[s_begin:s_end]
            seq_y_mark = data_stamp[r_begin:r_end]

        # seq_x = data_x[s_begin:s_end]
        # # if self.inverse:
        # #     seq_y = np.concatenate(
        # #         [
        # #             data_x[r_begin : r_begin + self.label_len],
        # #             data_y[r_begin + self.label_len : r_end],
        # #         ],
        # #         0,
        # #     )
        # # else:
        # #     seq_y = data_y[r_begin:r_end]
        # seq_y = data_y[r_begin:r_end]
        # seq_x_mark = data_stamp[s_begin:s_end]
        # seq_y_mark = data_stamp[r_begin:r_end]

        # print(seq_x.shape, seq_y.shape, seq_x_mark.shape, seq_y_mark.shape)
        # print(index, s_begin, s_end, r_begin, r_end)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __unit_len(self):
        if self.features == "M" or self.features == "MS":
            return len(self.data_x) - self.seq_len - self.pred_len + 1
        elif self.features == "S":
            data_x = self.records[self.record_keys[0]]["data_x"]
            return len(data_x) - self.seq_len - self.pred_len + 1

    def __len__(self):
        if self.features == "M" or self.features == "MS":
            return len(self.data_x) - self.seq_len - self.pred_len + 1
        elif self.features == "S":
            data_x = self.records[self.record_keys[0]]["data_x"]
            return (len(data_x) - self.seq_len - self.pred_len + 1) * len(
                self.record_keys
            )

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def forward_pass(model, batch):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch

    batch_x = batch_x.float().to(device)

    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    dec_inp = torch.zeros_like(batch_y[:, -model_args.pred_len :, :]).float()
    dec_inp = (
        torch.cat([batch_y[:, : model_args.label_len, :], dec_inp], dim=1)
        .float()
        .to(device)
    )

    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    f_dim = -1 if model_args.features == "MS" else 0
    outputs = outputs[:, -model_args.pred_len :, f_dim:]
    batch_y = batch_y[:, -model_args.pred_len :, f_dim:].to(device)

    return outputs, batch_y


def train_loop(train_loader, test_loader, model_args, tag: str):
    model = Autoformer(model_args).float().to(device)
    optim = Adam(model.parameters(), lr=model_args.learning_rate)
    loss_func = nn.MSELoss()

    for epoch in trange(model_args.train_epochs):
        model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
            train_loader
        ):
            if i == 0:
                print(
                    batch_x.size(),
                    batch_x_mark.size(),
                    batch_y.shape,
                    batch_y_mark.size(),
                )
            optim.zero_grad()

            outputs, batch_y = forward_pass(
                model, (batch_x, batch_y, batch_x_mark, batch_y_mark)
            )

            loss = loss_func(outputs, batch_y)

            loss.backward()
            optim.step()

        print("Epoch: ", epoch, "train_loss: ", loss.item())

    torch.save(
        model.state_dict(), os.path.join(data_args.checkpoint, f"model_{tag}.pt")
    )

    model.eval()

    test_set = []
    forecast_list = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            outputs, batch_y = forward_pass(
                model, (batch_x, batch_y, batch_x_mark, batch_y_mark)
            )

            forecast_list.append(outputs)
            test_set.append(batch_y)

    test_set = torch.concat(test_set).detach().cpu().numpy()
    forecast_list = torch.concat(forecast_list).detach().cpu().numpy()

    return forecast_list, test_set


train_data = SNDLibDataset(model_args, data_args, flag="train")
test_data = SNDLibDataset(model_args, data_args, flag="test")
train_data.save_data("train")
test_data.save_data("test")

train_loader = DataLoader(train_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

pred_real, test_set = train_loop(train_loader, test_loader, model_args, "real")
pred_oracle, test_set = train_loop(test_loader, test_loader, model_args, "oracle")

columns = list(train_data.df_raw.columns)
columns.remove("date")

print(len(train_data.df_raw.index.values))
print(pred_real.shape)
print(pred_oracle.shape)
print(test_set.shape)

with open(os.path.join(data_args.checkpoint, "columns.json"), "w") as fp:
    json.dump(columns, fp)
np.save(
    os.path.join(data_args.checkpoint, "test.npy"),
    test_set.reshape(-1, len(columns)),
    allow_pickle=False,
)
np.save(
    os.path.join(data_args.checkpoint, "pred_real.npy"), 
    pred_real.reshape(-1, len(columns)), 
    allow_pickle=False,
)
np.save(
    os.path.join(data_args.checkpoint, "pred_oracle.npy"),
    pred_oracle.reshape(-1, len(columns)),
    allow_pickle=False,
)

exit()

model = Autoformer(model_args).float().to(device)

optim = Adam(model.parameters(), lr=model_args.learning_rate)
loss_func = nn.MSELoss()

train_steps = len(train_loader)


for epoch in range(model_args.train_epochs):
    iter_count = 0
    train_loss = []

    time_now = time.time()

    model.train()
    epoch_time = time.time()
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        iter_count += 1
        # print(batch_x.size(), batch_x_mark.size(), batch_y.shape, batch_y_mark.size())
        optim.zero_grad()

        batch_x = batch_x.float().to(device)

        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -model_args.pred_len :, :]).float()
        dec_inp = (
            torch.cat([batch_y[:, : model_args.label_len, :], dec_inp], dim=1)
            .float()
            .to(device)
        )

        # print(dec_inp.shape)

        # pred, true = _process_one_batch(
        #     train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
        # )

        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if model_args.features == "MS" else 0
        outputs = outputs[:, -model_args.pred_len :, f_dim:]
        batch_y = batch_y[:, -model_args.pred_len :, f_dim:].to(device)

        loss = loss_func(outputs, batch_y)
        train_loss.append(loss.item())

        if (i + 1) % 100 == 0:
            print(
                "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                    i + 1, epoch + 1, loss.item()
                )
            )
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((model_args.train_epochs - epoch) * train_steps - i)
            print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
            iter_count = 0
            time_now = time.time()

        loss.backward()
        optim.step()

test_set = []
forecast_list = []

model.eval()
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    batch_x = batch_x.float().to(device)

    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    # decoder input
    dec_inp = torch.zeros_like(batch_y[:, -model_args.pred_len :, :]).float()
    dec_inp = (
        torch.cat([batch_y[:, : model_args.label_len, :], dec_inp], dim=1)
        .float()
        .to(device)
    )
    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    f_dim = -1 if model_args.features == "MS" else 0
    outputs = outputs[:, -model_args.pred_len :, f_dim:]
    batch_y = batch_y[:, -model_args.pred_len :, f_dim:].to(device)

    forecast_list.append(outputs.detach().cpu().numpy().tolist())
    test_set.append(batch_y.detach().cpu().numpy().tolist())

test_set = np.array(test_set)
forecast_list = np.array(forecast_list)

np.save(os.path.join(data_args.checkpoint, "test.npy"), test_set, allow_pickle=False)
np.save(
    os.path.join(data_args.checkpoint, "pred.npy"),
    np.array(forecast_list),
    allow_pickle=False,
)
torch.save(model.state_dict(), os.path.join(data_args.checkpoint, "model.pt"))

