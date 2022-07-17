from dataclasses import dataclass, field
import os

import numpy as np
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.preprocessing import MinMaxScaler
from pmdarima.arima import auto_arima
import pandas as pd
from scipy import stats
import torch
from tqdm import tqdm, trange
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler

from sklearnex import patch_sklearn

patch_sklearn()
from forecast.data.loader import DataParser
from forecast.data.dataset import TrainDataset, TestDataset, WeightedSampler
from forecast.data.prophet import ProphetDataParser
from forecast.utils.cmdparser import HfArgumentParser
from forecast.model.deepar import DeepAR, loss_fn

window_size = 192
stride_size = 24
num_covariates = 5


@dataclass
class ModelConfig:
    folder: str = field(metadata={"help": "folder for xml data"})
    checkpoint: str = field(metadata={"help": "path to checkpoints"})

    def __post_init__(self):
        try:
            os.mkdir(self.checkpoint)
        except:
            pass

        if "abilene" in self.checkpoint:
            self.save_name = "abilene"
        else:
            self.save_name = "geant"


@dataclass
class ModelParameters(object):
    learning_rate: float = field(default=1e-3)
    lstm_dropout: float = field(default=0.1)
    batch_size: int = field(default=64)
    lstm_layers: int = field(default=3)
    num_epochs: int = field(default=20)
    train_window: int = field(default=192)
    test_window: int = field(default=192)
    embedding_dim: int = field(default=20)
    lstm_hidden_dim: int = field(default=40)
    cov_dim: int = field(default=5)
    test_window: int = field(default=192)
    num_class: int = field(default=-1)
    test_predict_start: int = field(default=168)
    predict_steps: int = field(default=24)
    predict_start: int = field(default=168)
    
    device: str = field(default="cuda")


# params = {
#     "learning_rate": 1e-3,
#     "batch_size": 64,
#     "lstm_layers": 3,
#     "num_epochs": 20,
#     "train_window": 192,
#     "test_window": 192,
#     "predict_start": 168,
#     "test_predict_start": 168,
#     "predict_steps": 24,
#     "num_class": 370,
#     "cov_dim": 4,
#     "lstm_hidden_dim": 40,
#     "embedding_dim": 20,
#     "sample_times": 200,
#     "lstm_dropout": 0.1,
#     "predict_batch": 256
# }


class SimpleDataset(Dataset):
    def __init__(self, x_input, v_input, label) -> None:
        super().__init__()

        self.x_input = x_input
        self.v_input = v_input
        self.label = label

    def __getitem__(self, index):
        return (
            self.x_input[index],
            self.v_input[index],
            self.label[index],
        )

    def __len__(self):
        return len(self.x_input)


def gen_covariates(times):
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 1] = input_time.weekday()
        covariates[i, 2] = input_time.hour
        covariates[i, 3] = input_time.month
        covariates[i, 4] = input_time.minute
    for i in range(1, num_covariates):
        covariates[:, i] = stats.zscore(covariates[:, i])
    return covariates[:, :num_covariates]


def prepare_data(data, times, save_path, save_name, train):
    time_len, num_series = data.shape
    input_size = window_size - stride_size

    windows_per_series = np.full((num_series), (time_len - input_size) // stride_size)

    # if train: windows_per_series -= (data_start+stride_size-1) // stride_size

    total_windows = np.sum(windows_per_series)
    covariates = gen_covariates(times)
    x_input = np.zeros(
        (total_windows, window_size, 1 + num_covariates + 1), dtype="float32"
    )
    label = np.zeros((total_windows, window_size), dtype="float32")
    v_input = np.zeros((total_windows, 2), dtype="float32")

    count = 0

    for series in trange(num_series):
        cov_age = stats.zscore(np.arange(time_len))  # total_time -> time_len
        covariates[:, 0] = cov_age
        for i in range(windows_per_series[series]):
            window_start = stride_size * i
            window_end = window_start + window_size
            """
            print("x: ", x_input[count, 1:, 0].shape)
            print("window start: ", window_start)
            print("window end: ", window_end)
            print("data: ", data.shape)
            print("d: ", data[window_start:window_end-1, series].shape)
            """
            x_input[count, 1:, 0] = data[window_start : window_end - 1, series]
            x_input[count, :, 1 : 1 + num_covariates] = covariates[
                window_start:window_end, :
            ]
            x_input[count, :, -1] = series
            label[count, :] = data[window_start:window_end, series]
            nonzero_sum = (x_input[count, 1:input_size, 0] != 0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = (
                    np.true_divide(x_input[count, 1:input_size, 0].sum(), nonzero_sum)
                    + 1
                )
                x_input[count, :, 0] = x_input[count, :, 0] / v_input[count, 0]
                label[count, :] = label[count, :] / v_input[count, 0]
            count += 1

    prefix = os.path.join(save_path, "train_" if train else "test_")
    np.save(prefix + "data_" + save_name, x_input)
    np.save(prefix + "v_" + save_name, v_input)
    np.save(prefix + "label_" + save_name, label)

    print(x_input.shape, v_input.shape, label.shape)

    return x_input, v_input, label



@torch.no_grad()
def evaluate(model, loss_fn, test_loader, params, plot_num, sample=True):
    """Evaluate the model on the test set.
    Args:
        model: (torch.nn.Module) the Deep AR model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
        plot_num: (-1): evaluation from evaluate.py; else (epoch): evaluation on epoch
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    """
    model.eval()
    # with torch.no_grad():
    # plot_batch = np.random.randint(len(test_loader) - 1)

    # summary_metric = {}
    # raw_metrics = init_metrics(sample=sample)

    # Test_loader:
    # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
    # id_batch ([batch_size]): one integer denoting the time series id;
    # v ([batch_size, 2]): scaling factor for each window;
    # labels ([batch_size, train_window]): z_{1:T}.

    forecast_list = []
    lb_list = []
    ub_list = []

    for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(test_loader)):
        test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
        id_batch = id_batch.unsqueeze(0).to(params.device)
        v_batch = v.to(torch.float32).to(params.device)
        labels = labels.to(torch.float32).to(params.device)
        batch_size = test_batch.shape[1]
        input_mu = torch.zeros(
            batch_size, params.test_predict_start, device=params.device
        )  # scaled
        input_sigma = torch.zeros(
            batch_size, params.test_predict_start, device=params.device
        )  # scaled
        hidden = model.init_hidden(batch_size)
        cell = model.init_cell(batch_size)

        for t in range(params.test_predict_start):
            # if z_t is missing, replace it by output mu from the last time step
            zero_index = test_batch[t, :, 0] == 0
            if t > 0 and torch.sum(zero_index) > 0:
                test_batch[t, zero_index, 0] = mu[zero_index]

            mu, sigma, hidden, cell = model(
                test_batch[t].unsqueeze(0), id_batch, hidden, cell
            )
            input_mu[:, t] = v_batch[:, 0] * mu + v_batch[:, 1]
            input_sigma[:, t] = v_batch[:, 0] * sigma

        sample_mu, sample_sigma = model.test(
            test_batch, v_batch, id_batch, hidden, cell, sampling=False
        )

        sample_mu = sample_mu.flatten()
        sample_sigma = sample_sigma.flatten()

        # print(sample_mu.shape, sample_sigma.shape, type(sample_mu))

        forecast_list.append(sample_mu.detach().cpu().numpy().tolist())
        lb_list.append((sample_mu - 1.96 * sample_sigma).detach().cpu().numpy().tolist())
        ub_list.append((sample_mu + 1.96 * sample_sigma).detach().cpu().numpy().tolist())

    np.save(os.path.join(args.checkpoint, "test.npy"), test_set, allow_pickle=False)
    np.save(
        os.path.join(args.checkpoint, "pred.npy"),
        np.array(forecast_list),
        allow_pickle=False,
    )

    np.save(
        os.path.join(args.checkpoint, "pred_lb.npy"),
        np.array(lb_list),
        allow_pickle=False,
    )
    np.save(
        os.path.join(args.checkpoint, "pred_ub.npy"),
        np.array(ub_list),
        allow_pickle=False,
    )

    model.train()

parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

parser = DataParser()
df = parser.parse_sndlib_xml(args.folder)
df = df.fillna(0).set_index("timestamps")

print(df.head())

df_len = len(df.index.values)
train_len = int(df_len * 0.6)

dataset = df.values
training_set = dataset[:train_len]
test_set = dataset[train_len:]

print(training_set.shape, test_set.shape)

# sc = MinMaxScaler(feature_range=(0, 1))
# training_set_scaled = sc.fit_transform(training_set)
# test_set_scaled = sc.transform(test_set)

prepare_data(
    training_set, df.index[:train_len], args.folder, args.save_name, train=True
)
prepare_data(
    test_set, df.index[train_len:], args.folder, args.save_name, train=False
)


# print(train_x_input.shape, train_v_input.shape, train_label.shape)

params = ModelParameters(num_class=dataset.shape[1])
model = DeepAR(params)
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
model.train()
model = model.to(params.device)
loss_epoch = np.zeros(len(training_set))

train_dataset = TrainDataset(args.folder, args.save_name, params.num_class)
test_dataset = TestDataset(args.folder, args.save_name, params.num_class)
sampler = WeightedSampler(
    args.folder, args.save_name
)  # Use weighted sampler instead of random sampler
train_loader = DataLoader(
    train_dataset, batch_size=params.batch_size, sampler=sampler, num_workers=4
)
test_loader = DataLoader(
    test_dataset,
    batch_size=params.batch_size,
    sampler=RandomSampler(test_dataset),
    num_workers=4,
)
# logger.info('Loading complete.')

# train_dataset = SimpleDataset(train_x_input, train_v_input, train_label)
# train_loader = DataLoader(train_dataset, batch_size=params.batch_size)
# test_dataset = SimpleDataset(test_x_input, test_v_input, test_label)
# test_loader = DataLoader(test_dataset, batch_size=params.batch_size)

for i, (train_batch, idx, labels_batch) in enumerate(tqdm(train_loader)):
    optimizer.zero_grad()
    batch_size = train_batch.shape[0]

    train_batch = (
        train_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
    )  # not scaled
    labels_batch = (
        labels_batch.permute(1, 0).to(torch.float32).to(params.device)
    )  # not scaled
    idx = idx.unsqueeze(0).to(params.device)

    loss = torch.zeros(1, device=params.device)
    hidden = model.init_hidden(batch_size)
    cell = model.init_cell(batch_size)

    for t in range(params.train_window):
        # if z_t is missing, replace it by output mu from the last time step
        zero_index = train_batch[t, :, 0] == 0
        if t > 0 and torch.sum(zero_index) > 0:
            train_batch[t, zero_index, 0] = mu[zero_index]
        mu, sigma, hidden, cell = model(
            train_batch[t].unsqueeze_(0).clone(), idx, hidden, cell
        )
        loss += loss_fn(mu, sigma, labels_batch[t])

    loss.backward()
    optimizer.step()
    loss = loss.item() / params.train_window  # loss per timestep
    loss_epoch[i] = loss
    if i % 1000 == 0:
        evaluate(model, loss_fn, test_loader, params, None)
        print(f"train_loss: {loss}")
    # if i == 0:
    #     print(f'train_loss: {loss}')




