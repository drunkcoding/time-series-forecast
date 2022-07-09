import argparse
from dataclasses import dataclass, field
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch import optim
from torch import nn
from torch.utils.data import Dataset, DataLoader

from forecast.data.prophet import ProphetDataParser
from forecast.model.informer import Informer
from forecast.utils.timefeatures import time_features

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


parser = argparse.ArgumentParser(description="[Informer] Long Sequences Forecasting")

parser.add_argument(
    "--seq_len", type=int, default=96, help="input sequence length of Informer encoder"
)
parser.add_argument(
    "--label_len", type=int, default=48, help="start token length of Informer decoder"
)
parser.add_argument(
    "--pred_len", type=int, default=24, help="prediction sequence length"
)
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
parser.add_argument("--c_out", type=int, default=7, help="output size")
parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
parser.add_argument(
    "--s_layers", type=str, default="3,2,1", help="num of stack encoder layers"
)
parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
parser.add_argument("--factor", type=int, default=5, help="probsparse attn factor")
parser.add_argument("--padding", type=int, default=0, help="padding type")
parser.add_argument(
    "--distil",
    action="store_false",
    help="whether to use distilling in encoder, using this argument means not using distilling",
    default=True,
)
parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
parser.add_argument(
    "--attn",
    type=str,
    default="prob",
    help="attention used in encoder, options:[prob, full]",
)
parser.add_argument(
    "--embed",
    type=str,
    default="timeF",
    help="time features encoding, options:[timeF, fixed, learned]",
)
parser.add_argument("--activation", type=str, default="gelu", help="activation")
parser.add_argument(
    "--output_attention",
    action="store_true",
    help="whether to output attention in ecoder",
)
parser.add_argument(
    "--do_predict", action="store_true", help="whether to predict unseen future data"
)
parser.add_argument(
    "--mix",
    action="store_false",
    help="use mix attention in generative decoder",
    default=True,
)
parser.add_argument(
    "--cols",
    type=str,
    nargs="+",
    help="certain cols from the data files as the input features",
)
parser.add_argument(
    "--num_workers", type=int, default=0, help="data loader num workers"
)
parser.add_argument("--itr", type=int, default=2, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=6, help="train epochs")
parser.add_argument(
    "--batch_size", type=int, default=32, help="batch size of train input data"
)
parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument(
    "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
)
parser.add_argument("--des", type=str, default="test", help="exp description")
parser.add_argument("--loss", type=str, default="mse", help="loss function")
parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
parser.add_argument(
    "--use_amp",
    action="store_true",
    help="use automatic mixed precision training",
    default=False,
)
parser.add_argument(
    "--inverse", action="store_true", help="inverse output data", default=False
)

args = parser.parse_args()


# @dataclass
# class ModelArgs():
#     enc_in: int = field(default=7, metadata={
#             "help": "encoder input size = input time series length"
#         })
#     dec_in: int = field(default=7, metadata={
#             "help": "decoder input size = ?"
#         })
#     c_out: int = field(default=7, metadata={
#             "help": "output size"
#         })
#     d_model: int = field(default=512, metadata={
#             "help": "dimension of model = hidden size"
#         })
#     n_heads: int = field(default=8, metadata={
#             "help": "number of heads"
#         })
#     e_layers: int = field(default=8, metadata={
#             "help": "number of heads"
#         })


# folder = "directed-abilene-zhang-5min-over-6months-ALL"
folder = "directed-geant-uhlig-15min-over-4months-ALL"

parser = ProphetDataParser()
df_dict = parser.format_sndlib_xml(folder)

figure_dir = "figures/prophet"

model = Informer(
    args.enc_in,
    args.dec_in,
    args.c_out,
    args.seq_len,
    args.label_len,
    args.pred_len,
    args.factor,
    args.d_model,
    args.n_heads,
    args.e_layers,
    args.d_layers,
    args.d_ff,
    args.dropout,
    args.attn,
    args.embed,
    args.freq,
    args.activation,
    args.output_attention,
    args.distil,
    args.mix,
).float()

optim = optim.Adam(model.parameters(), lr=args.learning_rate)
loss_func = nn.MSELoss()

train_steps = len(train_loader)

for epoch in range(args.train_epochs):
    iter_count = 0
    train_loss = []

    model.train()
    epoch_time = time.time()
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        iter_count += 1

        optim.zero_grad()
        pred, true = self._process_one_batch(
            train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
        )
        loss = loss_func(pred, true)
        train_loss.append(loss.item())

        if (i + 1) % 100 == 0:
            print(
                "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                    i + 1, epoch + 1, loss.item()
                )
            )
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
            print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
            iter_count = 0
            time_now = time.time()

        loss.backward()
        optim.step()

