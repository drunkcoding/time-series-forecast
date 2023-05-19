import functools
import gc
import json
import os
import pickle
import torch
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from transformers import HfArgumentParser
from darts import TimeSeries

import darts.models as models
from sklearn.preprocessing import StandardScaler as Scaler
# from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import GaussianLikelihood
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import hyperopt
from hyperopt import Trials, STATUS_OK, STATUS_FAIL, tpe, fmin, hp
from hyperopt import space_eval

stopper = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min",
)

@dataclass
class ModelConfig:
    input_folder: str = field(metadata={"help": "folder for xml data"})
    model: str = field(metadata={"help": "model to use"})
    multivariate: bool = field(default=False, metadata={"help": "multivariate or not"})
    # normalize: bool = field(default=False, metadata={"help": "nomralize or not"})
    # lag: int = field(default=1, metadata={"help": "lag for the model"})

    def __post_init__(self):

        self.dataset = os.path.basename(self.input_folder).split("-")[0].lower()
        self.freq_map = {
            "abilene": "5min",
            "geant": "15min",
            "cernet": "5min",
        }
        self.freq = self.freq_map[self.dataset]
        self.output_folder = os.path.join(
            "outputs",
            "-".join(
                [
                    self.dataset,
                    self.model.lower(),
                    "multivariate" if self.multivariate else "univariate",
                ]
            ),
        )
        self.model = self.model.lower()

        try:
            os.mkdir(self.output_folder)
        except:
            pass

        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "devices": -1,
            "callbacks": [stopper],
            "strategy": "ddp",
        }

        optimizer_kwargs = {
            "lr": 3e-4,
            "weight_decay": 0.0001,
        }

        self.model_map = {
            "naive": models.NaiveSeasonal,
            "arima": models.ARIMA,
            "kalman": models.KalmanForecaster,
            "prophet": models.Prophet,
            "rnn": models.RNNModel,
            "deepar": models.RNNModel,
            "tcn": models.TCNModel,
            "tft": models.TFTModel,
        }

        self.search_space = {
            "naive": {},
            "arima": {},
            "kalman": {},
            "prophet": {},
            "rnn": {
                "model": hp.choice("model", ["RNN", "LSTM", "GRU"]),
                "hidden_dim": hp.choice("hidden_dim", [32, 64, 128, 256]),
                "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
                "n_rnn_layers": hp.choice("n_rnn_layers", [1, 2, 3, 4, 5]),
                "dropout": hp.choice('dropout', [0.0, 0.1, 0.2, 0.5]),
                # "lr": hp.choice('lr', [0.1, 0.01, 0.001]),
            },
            "deepar": {
                "model": hp.choice("model", ["RNN", "LSTM", "GRU"]),
                "hidden_dim": hp.choice("hidden_dim", [32, 64, 128, 256]),
                "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
                "n_rnn_layers": hp.choice("n_rnn_layers", [1, 2, 3, 4, 5]),
                "dropout": hp.choice('dropout', [0.0, 0.1, 0.2, 0.5]),
                # "lr": hp.choice('lr', [0.1, 0.01, 0.001]),
            },
            "tft": {
                "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
                "hidden_size": hp.choice("hidden_size", [32, 64, 128, 256]),
                "lstm_layers": hp.choice("lstm_layers", [1, 2, 3, 4, 5]),
                "num_attention_heads": hp.choice("num_attention_heads", [1, 2, 4, 8]),
                # "lr": hp.choice('lr', [0.1, 0.01, 0.001]),
            },
            "tcn": {
                "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
                "num_layers": hp.choice("num_layers", [2, 3, 4, 5]),
                "kernel_size": hp.choice("kernel_size", [2, 3, 4, 5]),
                "num_filters": hp.choice("num_filters", [2, 3, 4, 5]),
                # "lr": hp.choice('lr', [0.1, 0.01, 0.001]),
            },
        }

        self.model_params = {
            "naive": {"K": 1},
            "arima": {
                "p": 12,
                "d": 1,
                "q": 0,
            },
            "kalman": {
                "dim_x": 10,
            },
            "prophet": {
                "seasonality_mode": "multiplicative",
                "seasonality_prior_scale": 0.1,
                "changepoint_prior_scale": 0.05,
                "changepoint_range": 0.8,
                "yearly_seasonality": "auto",
                "weekly_seasonality": "auto",
                "daily_seasonality": "auto",
            },
            "rnn": {
                "input_chunk_length": 10,
                "output_chunk_length": 1,
                "n_epochs": 1,
                # "batch_size": 256,
                # "n_rnn_layers": 5,
                # "model": "LSTM",
                # "hidden_dim": 128,
                # "dropout": 0.1,
                "pl_trainer_kwargs": pl_trainer_kwargs,
                "optimizer_kwargs": optimizer_kwargs,
            },
            "deepar": {
                "input_chunk_length": 10,
                "output_chunk_length": 1,
                "n_epochs": 100,
                # "batch_size": 256,
                # "n_rnn_layers": 5,
                # "model": "LSTM",
                # "hidden_dim": 128,
                "likelihood": GaussianLikelihood(),
                "pl_trainer_kwargs": pl_trainer_kwargs,
                "optimizer_kwargs": optimizer_kwargs,
            },
            "tcn": {
                "input_chunk_length": 10,
                "output_chunk_length": 1,
                "n_epochs": 100,
                # "batch_size": 256,
                "pl_trainer_kwargs": pl_trainer_kwargs,
                "optimizer_kwargs": optimizer_kwargs,
            },
            "tft": {
                "input_chunk_length": 10,
                "output_chunk_length": 1,
                "n_epochs": 100,
                # "batch_size": 256,
                "add_relative_index": True,
                "pl_trainer_kwargs": pl_trainer_kwargs,
                "optimizer_kwargs": optimizer_kwargs,
            },
        }

parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

df = pd.read_csv(args.input_folder)
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d-%H-%M')  # Parse the time

data_columns = list(df.columns.values)
data_columns.remove('time')
data = df[data_columns].values

data[np.isnan(data)] = 0  # fill the abnormal data with 0
data[np.isinf(data)] = 0
data = np.clip(data, 0.0, np.percentile(data.flatten(), 99))  # we use 99% as the threshold
df[data_columns] = data

# aggregated_time_series = np.sum(data, axis=1)
# df_ts = pd.DataFrame()
# df_ts['date'] = df['time']
# df_ts['data'] = aggregated_time_series / 1000  # Plot in Mbps

history = 10  # input historical time steps
horizon = 1  # output predicted time steps
test_ratio = 0.2  # testing data ratio
max_evals = 1  # maximal trials for hyper parameter tuning

model_name = args.model
y_true_fn = '%s_true_TM-minmax-%d-%d.pkl' % (model_name, history, horizon)
y_pred_fn = '%s_pred_TM-minmax-%d-%d.pkl' % (model_name, history, horizon)

test_length = int(df.shape[0] * test_ratio)
train_length = df.shape[0] - test_length
valid_length = int(train_length * 0.2)

transformer = Scaler()
transformer.fit(data[:train_length])

data_scaled = transformer.transform(data)
df_scaled = df
df_scaled[data_columns] = data_scaled

df_train_scaled = df_scaled[:train_length]
df_valid_scaled = df_train_scaled[-valid_length:]
df_train_scaled = df_train_scaled[:-valid_length]
df_test = df[-test_length:]

PROBLISTIC_MODEL = ["deepar", "tft"]
DNN_MODEL = ["rnn", "tcn", "deepar", "tft"]

def MAE(y_true, y_pred):
    return np.nanmean(np.abs(y_true - y_pred))

def create_model_hypopt(params, train, val): 
    try:
        # clear memory 
        gc.collect()
        print("Trying params:", params)
        
        # Create model
        model_cls = args.model_map[args.model]
        model = model_cls(**args.model_params[args.model], **params)

        # Train model
        model.fit(train, val_series=val)
        
        # get val_loss
        val_loss = float(model.trainer.logged_metrics['val_loss'].cpu().numpy())
        
        # clear memory
        del model
        gc.collect()

        return {'loss': val_loss, 'status': STATUS_OK} # if accuracy use '-' sign, model is optional
    except Exception as e:
        import traceback
        print(traceback.print_exc())
        return {'loss': None, 'status': STATUS_FAIL} 

def train_model(args, model, train, val, test, col):

    if model is None:
        model_cls = args.model_map[args.model]
        model = model_cls(**args.model_params[args.model])

        trials = Trials()
        best = fmin(functools.partial(create_model_hypopt, train=train, val=val),
            space=args.search_space[args.model],
            algo=tpe.suggest,
            max_evals=max_evals,  # test trials
            trials=trials)

        # save best hyper parameters
        with open(os.path.join(args.output_folder, 'best_params_%s_%s.pkl' % (args.model, col)), 'wb') as f:
            pickle.dump(best, f)
        print(best)

    
    model.fit(train, val_series=val if args.model in DNN_MODEL else None)

    if args.multivariate:
        prediction = model.predict(
            len(val), test, num_samples=100 if args.model in PROBLISTIC_MODEL else 1
        )
        if args.model in PROBLISTIC_MODEL:
            prediction = prediction.quantile_timeseries(0.5)
        # if args.normalize:
        prediction = transformer.inverse_transform(prediction)
        prediction = prediction.pd_dataframe().iloc[:, 1:].values
        target = val.pd_dataframe().iloc[:, 1:].values
        model.save(os.path.join(args.output_folder, "all.ckpt"))
    else:
        prediction = model.predict(
            len(val), num_samples=100 if args.model in PROBLISTIC_MODEL else 1
        )
        if args.model in PROBLISTIC_MODEL:
            prediction = prediction.quantile_timeseries(0.5).with_columns_renamed(col+"_quantiles", col)
        # if args.normalize:
        prediction = transformer.inverse_transform(prediction)
        print(prediction.pd_dataframe())
        prediction = prediction.pd_dataframe()[col].values
        target = val.pd_dataframe()[col].values
        model.save(os.path.join(args.output_folder, col + ".ckpt"))
    # print(prediction.shape, target.shape)

    print(col, MAE(prediction, target))
    return prediction, target

def prepare_data(columns):
    train = TimeSeries.from_dataframe(
        df_train_scaled,
        "time",
        columns,
        fill_missing_dates=True,
        freq=args.freq,
        fillna_value=0,
    )
    val = TimeSeries.from_dataframe(
        df_valid_scaled,
        "time",
        columns,
        fill_missing_dates=True,
        freq=args.freq,
        fillna_value=0,
    )
    test = TimeSeries.from_dataframe(
        df_test,
        "time",
        columns,
        fill_missing_dates=True,
        freq=args.freq,
    )

    return train, val, test

predictions = []
oracles = []
targets = []
nan_idx = []

model_cls = args.model_map[args.model]
model = model_cls(**args.model_params[args.model])

if args.multivariate:
    assert args.model not in ["naive", "arima", "exponentialsmoothing", "kalman"]
    # get df columns without timestamps
    columns = df.columns.tolist()
    columns.remove("time")

    # create a list of dataframes, each containing one column
    train, val, test = prepare_data(columns)
    predictions, targets = train_model(args, None, train, val, test, "all")
else:
    for col in df.columns[:-1]:

        train, val, test = prepare_data(col)

        prediction, target = train_model(args, model, train, val, test, col)

        predictions.append(prediction)
        targets.append(target)

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

print("total", MAE(predictions, targets))
# df = df.drop(columns=["timestamps"])

# save predictions to npy
np.save(
    os.path.join(args.output_folder, "predictions.npy"), predictions, allow_pickle=False
)
# save targets to npy
np.save(os.path.join(args.output_folder, "targets.npy"), targets, allow_pickle=False)

# save df.columns to json
with open(os.path.join(args.output_folder, "columns.json"), "w") as fp:
    columns = df.columns[:-1].tolist()
    columns.remove("time")
    json.dump(columns, fp)

