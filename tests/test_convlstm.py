from dataclasses import dataclass, field
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import ticker

import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

import os
import torch
import torchvision
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from multiprocessing import Process
from torch.utils.data import TensorDataset, DataLoader
import math

from transformers import HfArgumentParser


@dataclass
class ModelConfig:
    input_folder: str = field(metadata={"help": "folder for xml data"})
    model: str = field(default="convlstm", metadata={"help": "model to use"})
    multivariate: bool = field(default=True, metadata={"help": "multivariate or not"})
    normalize: bool = field(default=True, metadata={"help": "nomralize or not"})
    remove_outliers: bool = field(default=False, metadata={"help": "outlier remove"})

    def __post_init__(self):
        self.dataset = os.path.basename(self.input_folder).split("-")[0].lower()
        self.output_folder = os.path.join(
            "outputs",
            "-".join(
                [
                    self.dataset,
                    self.model.lower(),
                    "M" if self.multivariate else "U",
                    "N" if self.normalize else "R",
                    "O" if self.remove_outliers else "F"
                ]
            ),
        )
        self.model = self.model.lower()

        self.freq_map = {
            "abilene": "5T",
            "geant": "15T",
            "cernet": "5T",
        }
        self.freq = self.freq_map[self.dataset]

        try:
            os.mkdir(self.output_folder)
        except:
            pass


parser = HfArgumentParser((ModelConfig))
args = parser.parse_args_into_dataclasses()[0]

file_name = args.input_folder  # the input filename
if args.dataset == "abilene":
    nodes = 12
elif args.dataset == "geant":
    nodes = 23
elif args.dataset == "cernet":
    nodes = 14

history = 10  # input historical time steps
horizon = 1  # output predicted time steps
test_ratio = 0.2  # testing data ratio
max_evals = 100  # maximal trials for hyper parameter tuning

model_name = "ConvLSTM"
y_true_fn = "%s_true_TM-minmax-%d-%d.pkl" % (model_name, history, horizon)
y_pred_fn = "%s_pred_TM-minmax-%d-%d.pkl" % (model_name, history, horizon)

RMSE_fn = model_name + "_TM-minmax_RMSE-%d-%d-step-%d.pdf"
MAE_fn = model_name + "_TM-minmax_MAE-%d-%d-step-%d.pdf"


df = pd.read_csv(args.input_folder)
df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d-%H-%M")  # Parse the time
# df['time'] = df.time.dt.strftime('%Y-%m-%d %H:%M:%S')  # Convert to string


def df_cleanup(df: pd.DataFrame):
    cols = df.columns
    cols.drop("time")
    num_rows = df.shape[0]
    df = df.dropna(how="all", subset=cols)
    # df = df.dropna(axis=1, thresh=int(num_rows / 2))
    # df["time"] = pd.date_range(df["time"][0], periods=df.shape[0], freq=args.freq)
    return df


df = df_cleanup(df)

data_columns = list(df.columns.values)
data_columns.remove("time")
data = df[data_columns].values
# data = np.clip(
#     data, 0.0, np.percentile(data.flatten(), 99)
# )  # we use 99% as the threshold
# df[data_columns] = data

history = 10  # input historical time steps
horizon = 1  # output predicted time steps
test_ratio = 0.2  # testing data ratio
max_evals = 1  # maximal trials for hyper parameter tuning

test_length = int(df.shape[0] * test_ratio)
train_length = df.shape[0] - test_length
valid_length = int(train_length * 0.2)

if args.remove_outliers:
    data[:train_length] = np.clip(
        data[:train_length], 0.0, np.percentile(data[:train_length].flatten(), 99)
    )  # we use 99% as the threshold

df[data_columns] = data

print("train_length", train_length)
print("valid_length", valid_length)
print("test_length", test_length)

max_list = np.max(data[:train_length], axis=0)
min_list = np.min(data[:train_length], axis=0)

data = (data - min_list) / (max_list - min_list)
data[np.isnan(data)] = 0  # fill the abnormal data with 0
data[np.isinf(data)] = 0

x_data = []
y_data = []
length = data.shape[0]

for i in range(length - history - horizon + 1):
    x = data[i : i + history, :]  # input historical time steps
    y = data[i + history : i + history + horizon :, :]  # output predicted time steps
    x_data.append(x)
    y_data.append(y)
x_data = np.array(x_data)
y_data = np.array(y_data)

x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1, nodes, nodes)
y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1, nodes, nodes)

test_length = test_length - horizon + 1
train_valid_length = x_data.shape[0] - test_length
train_length = int(train_valid_length * 0.8)
valid_length = train_valid_length - train_length

X_train = x_data[:train_length]
y_train = y_data[:train_length]
X_valid = x_data[train_length:train_valid_length]
y_valid = y_data[train_length:train_valid_length]
X_test = x_data[train_valid_length:]
y_test = y_data[train_valid_length:]


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(
            input_dim=in_chan, hidden_dim=nf, kernel_size=(3, 3), bias=True
        )

        self.encoder_2_convlstm = ConvLSTMCell(
            input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True
        )

        self.decoder_1_convlstm = ConvLSTMCell(
            input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True  # nf + 1
        )

        self.decoder_2_convlstm = ConvLSTMCell(
            input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True
        )

        self.decoder_CNN = nn.Conv3d(
            in_channels=nf, out_channels=1, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )

    def autoencoder(
        self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4
    ):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(
                input_tensor=x[:, t, :, :], cur_state=[h_t, c_t]
            )  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(
                input_tensor=h_t, cur_state=[h_t2, c_t2]
            )  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(
                input_tensor=encoder_vector, cur_state=[h_t3, c_t3]
            )  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(
                input_tensor=h_t3, cur_state=[h_t4, c_t4]
            )  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(
            batch_size=b, image_size=(h, w)
        )
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(
            batch_size=b, image_size=(h, w)
        )
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(
            batch_size=b, image_size=(h, w)
        )

        # autoencoder forward
        outputs = self.autoencoder(
            x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4
        )

        return outputs


if args.dataset == "abilene":
    params = {
        "batch_size": 256,
        "epochs": 100,
        "lr": 0.001,
        "n_hidden_dim": 64,
        "patience": 10,
    }
elif args.dataset == "geant":
    params = {
        "batch_size": 256,
        "epochs": 100,
        "lr": 0.001,
        "n_hidden_dim": 64,
        "patience": 5,
    }
elif args.dataset == "cernet":
    params = {
        "batch_size": 256,
        "epochs": 100,
        "lr": 0.001,
        "n_hidden_dim": 32,
        "patience": 10,
    }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TMLightning(pl.LightningModule):
    def __init__(self, hparams=None, model=None):
        super(TMLightning, self).__init__()

        # default config
        self.normalize = False
        self.model = model

        # Training config
        self.criterion = torch.nn.MSELoss()
        self.batch_size = params["batch_size"]
        self.n_steps_past = history
        self.n_steps_ahead = horizon

    def forward(self, x):
        x = x.to(device)

        output = self.model(x, future_seq=self.n_steps_ahead)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y)

        # save learning_rate
        lr_saved = self.trainer.optimizers[0].param_groups[-1]["lr"]
        lr_saved = torch.scalar_tensor(lr_saved).cuda()

        tensorboard_logs = {"train_mse_loss": loss, "learning_rate": lr_saved}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return {"val_loss": loss, "log": {"val_loss": loss}}

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o["val_loss"] for o in outputs]) / len(outputs)
        # show val_acc in progress bar but only log val_loss
        results = {
            "progress_bar": {"val_loss": val_loss_mean.item()},
            "log": {"val_loss": val_loss_mean.item()},
            "val_loss": val_loss_mean.item(),
        }
        return results

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {"test_loss": self.criterion(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss}
        return {"avg_test_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=params["lr"], betas=(0.9, 0.98))

    def train_dataloader(self):
        tensor_x_train = torch.Tensor(X_train)  # transform to torch tensor
        tensor_y_train = torch.Tensor(y_train)

        train_dataset = TensorDataset(tensor_x_train, tensor_y_train)  # create datset
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )  # create dataloader

        return train_loader

    def val_dataloader(self):
        tensor_x_valid = torch.Tensor(X_valid)  # transform to torch tensor
        tensor_y_valid = torch.Tensor(y_valid)

        valid_dataset = TensorDataset(tensor_x_valid, tensor_y_valid)  # create datset
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=True
        )  # create dataloader

        return valid_loader

    def test_dataloader(self):
        tensor_x_test = torch.Tensor(X_test)  # transform to torch tensor
        tensor_y_test = torch.Tensor(y_test)

        test_dataset = TensorDataset(tensor_x_test, tensor_y_test)  # create datset
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )  # create dataloader

        return test_loader


conv_lstm_model = EncoderDecoderConvLSTM(nf=params["n_hidden_dim"], in_chan=1)
model = TMLightning(model=conv_lstm_model)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=params["patience"],
    verbose=True,
    mode="min",
)
trainer = Trainer(max_epochs=params["epochs"], gpus=1, callbacks=[early_stop_callback])
start = time.time()
trainer.fit(model)
training_time = time.time() - start

val_loss = float(trainer.logged_metrics["val_loss"].cpu().numpy())

tensor_x_test = torch.Tensor(X_test)  # transform to torch tensor
tensor_y_test = torch.Tensor(y_test)

test_dataset = TensorDataset(tensor_x_test, tensor_y_test)  # create datset
test_loader = DataLoader(
    test_dataset, batch_size=params["batch_size"], shuffle=True
)  # create dataloader

model.to(device)

prediction_list = []
true_value_list = []
start = time.time()
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = model(inputs)
        prediction_list.append(outputs.cpu().numpy())
        true_value_list.append(labels.cpu().numpy())
prediction_time = time.time() - start

y_true = np.concatenate(true_value_list)
y_pred = np.concatenate(prediction_list)
y_true = y_true.reshape(y_true.shape[0], y_true.shape[1], -1)
y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1)


def inverse_normalization(prediction, y, max_list, min_list):
    inverse_prediction = prediction * (max_list - min_list) + min_list
    inverse_y = y * (max_list - min_list) + min_list

    return inverse_prediction, inverse_y


y_true_list = []
y_pred_list = []
for i in range(y_true.shape[0]):
    y_true_tmp = []
    y_pred_tmp = []
    for step in range(y_true.shape[1]):
        inverse_prediction, inverse_y = inverse_normalization(
            y_pred[i, step, :], y_true[i, step, :], max_list, min_list
        )

        y_true_tmp.append(inverse_y)
        y_pred_tmp.append(inverse_prediction)
    y_true_list.append(y_true_tmp)
    y_pred_list.append(y_pred_tmp)


y_true = np.array(y_true_list)
y_pred = np.array(y_pred_list)

for step in range(horizon):
    print(
        "Test MAE: ",
        mean_absolute_error(y_true[:, step, :].flatten(), y_pred[:, step, :].flatten())
        / 1000, 
        " NAIVE MAE: ", mean_absolute_error(y_true[:-1, step, :].flatten(), y_true[1:, step, :].flatten())
        / 1000
    )

y_true = y_true.squeeze()
y_pred = y_pred.squeeze()

np.save(os.path.join(args.output_folder, "targets.npy"), y_true, allow_pickle=False)
np.save(os.path.join(args.output_folder, "predictions.npy"), y_pred, allow_pickle=False)

with open(os.path.join(args.output_folder, "columns.json"), "w") as fp:
    columns = df.columns.tolist()
    columns.remove("time")
    json.dump(columns, fp)


# def plot_MAE(y_true, y_pred, fn, xlim=60):
#     MAE_list = []

#     for i in range(y_true.shape[0]):
#         mae = mean_absolute_error(y_true[i].flatten(), y_pred[i].flatten()) / 1000
#         MAE_list.append(mae)
    
#     data = MAE_list
#     data_size = len(data)
#     data_set = sorted(set(data))
#     bins = np.append(data_set, data_set[-1] + 1)

#     # Use the histogram function to bin the data
#     counts, bin_edges = np.histogram(data, bins=bins, density=False)
#     counts = counts.astype(float) / data_size
    
#     plt.figure(figsize=[12, 9])
#     plt.xlabel("MAE (Mbps)")
#     plt.xlim((0, xlim))
#     plt.ylabel("CDF")
#     plt.ylim((0, 1))

#     # Format the x tick labels
#     ax = plt.gca()
#     formatter = ticker.ScalarFormatter(useMathText=True)
#     formatter.set_scientific(True)
#     formatter.set_powerlimits((-1, 1))
#     ax.xaxis.set_major_formatter(formatter)

#     cdf = np.cumsum(counts)
#     plt.plot(bin_edges[0:-1], cdf)
    
#     plt.savefig(fn, bbox_inches = 'tight', pad_inches = 0.1)

# for step in range(horizon):
#     # fn = RMSE_fn % (history, horizon, step + 1)
#     # print(fn)
    
#     # plot_RMSE(y_true[:, step, :], y_pred[:, step, :], fn)
    
#     fn = MAE_fn % (history, horizon, step + 1)
#     print(fn)
    
#     plot_MAE(y_true[:, step, :], y_pred[:, step, :], fn)
