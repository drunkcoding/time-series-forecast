from dataclasses import dataclass, field
import json
import os
from transformers import HfArgumentParser

# from forecast.data.loader import DataParser
# from forecast.utils.evaluation import *

from darts import TimeSeries
import pandas as pd
import numpy as np

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from darts.models import AutoARIMA, ARIMA
import darts.models as models
from darts.dataprocessing.transformers import Scaler

# from darts.metrics import mape, mase, mse, smape, mae
from darts.utils.likelihood_models import GaussianLikelihood

stopper = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min",
)


def prepare_timeseries(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    freq: str,
    train_size: int,
    vaild_size: int,
):

    # series = TimeSeries.from_dataframe(
    #     df,
    #     time_col,
    #     target_col,
    #     fill_missing_dates=True,
    #     freq=freq,
    #     fillna_value=0,
    # )
    # df = series.pd_dataframe().reset_index()
    # # train_size = int(len(df.index.values) * 0.6)

    df_train = df[:train_size]
    df_test = df[train_size:]
    df_val = df_train[-vaild_size:]
    df_train = df_train[:-vaild_size]

    # print(df_train.shape)
    # print(df_val.shape)

    # nan_idx = np.isnan(df_val[target_col].values)

    train = TimeSeries.from_dataframe(
        df_train,
        time_col,
        target_col,
        # fill_missing_dates=True,
        # freq=freq,
        # fillna_value=0,
    )

    val = TimeSeries.from_dataframe(
        df_val,
        time_col,
        target_col,
        # fill_missing_dates=True,
        # freq=freq,
        # fillna_value=0,
    )

    test = TimeSeries.from_dataframe(
        df_test,
        time_col,
        target_col,
        # fill_missing_dates=True,
        # freq=freq,
    )

    return train, val, test


@dataclass
class ModelConfig:
    input_folder: str = field(metadata={"help": "folder for xml data"})
    model: str = field(metadata={"help": "model to use"})
    multivariate: bool = field(default=False, metadata={"help": "multivariate or not"})
    normalize: bool = field(default=False, metadata={"help": "nomralize or not"})
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

        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "devices": -1,
            "callbacks": [stopper],
            "auto_lr_find": True,
        }

        optimizer_kwargs = {
            "lr": 6e-4,
            # "weight_decay": 0.0001,
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
            "tf": models.TransformerModel,
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
                "n_epochs": 100,
                "batch_size": 16,
                "n_rnn_layers": 1,
                "model": "LSTM",
                "hidden_dim": 200,
                "dropout": 0.0,
                "pl_trainer_kwargs": pl_trainer_kwargs,
                "optimizer_kwargs": optimizer_kwargs,
            },
            "deepar": {
                "input_chunk_length": 120,
                "output_chunk_length": 1,
                "n_epochs": 100,
                "batch_size": 128,
                "n_rnn_layers": 2,
                "model": "LSTM",
                "hidden_dim": 200,
                "dropout": 0.1,
                "likelihood": GaussianLikelihood(),
                "pl_trainer_kwargs": pl_trainer_kwargs,
                "optimizer_kwargs": optimizer_kwargs,
            },
            "tcn": {
                "input_chunk_length": 10,
                "output_chunk_length": 1,
                "n_epochs": 100,
                "batch_size": 128,
                "dropout": 0.1,
                "pl_trainer_kwargs": pl_trainer_kwargs,
                "optimizer_kwargs": optimizer_kwargs,
            },
            "tft": {
                "input_chunk_length": 10,
                "output_chunk_length": 1,
                "n_epochs": 100,
                "batch_size": 128,
                "hidden_size": 256,
                "add_relative_index": True,
                "pl_trainer_kwargs": pl_trainer_kwargs,
                "optimizer_kwargs": optimizer_kwargs,
            },
            "tf": {
                "input_chunk_length": 10,
                "output_chunk_length": 1,
                "n_epochs": 100,
                "batch_size": 128,
                "d_model": 128,
                "pl_trainer_kwargs": pl_trainer_kwargs,
                "optimizer_kwargs": optimizer_kwargs,
            },
        }


def MAE(y_true, y_pred):
    return np.nanmean(np.abs(y_true - y_pred)) / 1000

PROBLISTIC_MODEL = ["deepar", "tft"]

parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]


def df_cleanup(df: pd.DataFrame):
    cols = df.columns
    cols.drop("time")
    num_rows = df.shape[0]
    df = df.dropna(how="all", subset=cols)
    # df = df.dropna(axis=1, thresh=int(num_rows / 2))
    df["time"] = pd.date_range(df["time"][0], periods=df.shape[0], freq=args.freq)
    return df


df = pd.read_csv(args.input_folder)
df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d-%H-%M")  # Parse the time
# df['time'] = df.time.dt.strftime('%Y-%m-%d %H:%M:%S')  # Convert to string

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

# parser = DataParser()
# df = parser.parse_sndlib_xml(args.input_folder)
# df = df_cleanup(df)

# print df rows with nan
# print(df[df.isna().any(axis=1)])

# total_size = len(df.index.values)
# train_size = int(total_size * 0.6)
# val_size = total_size - train_size

print(df)


def train_model(args, model, train, val, test, col):

    total = pd.concat([train.pd_dataframe(), val.pd_dataframe()], axis=0)
    total.reset_index(inplace=True)
    total = TimeSeries.from_dataframe(
        total,
        "time",
        total.columns.tolist()[1:] if col == "all" else col,
    )

    test_df = test.pd_dataframe().reset_index()
    val_test = val.slice(val.time_index[-9], val.time_index[-1])
    val_test = val_test.pd_dataframe()
    test = test.pd_dataframe()
    test = pd.concat([val_test, test], axis=0)
    test.reset_index(inplace=True)
    # print(test)
    test = TimeSeries.from_dataframe(
        test,
        "time",
        test.columns.tolist()[1:] if col == "all" else col,
        # fill_missing_dates=True,
        # freq=freq,
    )
    # test = val_test.stack(test)

    if args.normalize:
        transformer = Scaler()
        transformer.fit(total)

        train = transformer.transform(train)
        val = transformer.transform(val)
        test = transformer.transform(test)

        # val_test = transformer.transform(val_test)

    # slice test in unit of 10 components
    test_slices = []
    for i in test.time_index[:-9]:
        slice = test.slice(i, i + 9 * test.freq)
        test_slices.append(slice)

    if model is None:
        model_cls = args.model_map[args.model]
        model = model_cls(**args.model_params[args.model])

    if args.multivariate:
        model.fit(train, val_series=val, verbose=True)
    else:
        model.fit(train)

    def post_processing(x):
        if args.model in PROBLISTIC_MODEL:
            x = x.quantile_timeseries(0.5)
        if args.normalize:
            x = transformer.inverse_transform(x)
        return x.pd_dataframe()

    num_samples = 100 if args.model in PROBLISTIC_MODEL else 1
    # predictions = model.predict(test_slices, num_samples=num_samples, verbose=True)

    if args.model in ["rnn", "deepar", "tft", "tf", "tcn"]:
        prediction = model.predict(1, test_slices, num_samples=num_samples, verbose=True)
        prediction = list(
            map(
                post_processing,
                prediction,
            )
        )
        prediction = pd.concat(prediction, axis=0).reset_index()
    # elif args.model == "arima":
    #     prediction_list = []
    #     for slice in test_slices:
    #         prediction = model.predict(1, slice)
    #         prediction_list.append(prediction)
    #     prediction = list(
    #         map(
    #             post_processing,
    #             prediction,
    #         )
    #     )
    #     prediction = pd.concat(prediction, axis=0)
    else:
        prediction = model.predict(len(test_df.index.values))
        prediction = prediction.pd_dataframe().reset_index()
        # print(prediction, test_df)
    # prediction = model.predict(1, test_slices, num_samples=num_samples)
    
    
    # prediction_element = prediction[0]
    # for pred in prediction[1:]:
    #     prediction_element = prediction_element.stack(pred)
    # prediction = prediction_element
    # if args.model in PROBLISTIC_MODEL:
    #     prediction = prediction.quantile_timeseries(0.5)
    # if args.normalize:
    #     prediction = transformer.inverse_transform(prediction)
    # print(prediction)

    # prediction = prediction.pd_dataframe().iloc[:, 1:].values
    # print(prediction)
    # print(test_df)

    prediction = prediction.iloc[:, 1:].values
    target = test_df.iloc[:, 1:].values
    model.save(os.path.join(args.output_folder, col + ".ckpt"))

    # if args.multivariate:
    #     # prediction = model.predict(
    #     #     len(test), num_samples=100 if args.model in PROBLISTIC_MODEL else 1
    #     # )
    #     # if args.model in PROBLISTIC_MODEL:
    #     #     prediction = prediction.quantile_timeseries(0.5)
    #     # if args.normalize:
    #     #     prediction = transformer.inverse_transform(prediction)
    #     print(prediction.pd_dataframe())
    #     prediction = prediction.pd_dataframe().iloc[:, 1:].values
    #     target = test.pd_dataframe().iloc[:, 1:].values
    #     model.save(os.path.join(args.output_folder, "all.ckpt"))
    # else:
    #     # prediction = model.predict(
    #     #     len(test), num_samples=100 if args.model in PROBLISTIC_MODEL else 1
    #     # )
    #     # if args.model in PROBLISTIC_MODEL:
    #     #     prediction = prediction.quantile_timeseries(0.5).with_columns_renamed(
    #     #         col + "_quantiles", col
    #     #     )
    #     # if args.normalize:
    #     #     prediction = transformer.inverse_transform(prediction)
    #     print(prediction.pd_dataframe())
    #     prediction = prediction.pd_dataframe()[col].values
    #     target = test.pd_dataframe()[col].values
    #     model.save(os.path.join(args.output_folder, col + ".ckpt"))
    # print(prediction.shape, target.shape)
    # print(prediction, target)
    idx = target != 0
    print(col, MAE(prediction[idx], target[idx]))
    print("naive", MAE(target[1:, ...], target[:-1, ...]))
    return prediction, target


predictions = []
oracles = []
targets = []
nan_idx = []

if args.multivariate:
    assert args.model not in ["naive", "arima", "prophet", "kalman"]
    # get df columns without time
    columns = df.columns
    columns = columns.drop("time")
    train, val, test = prepare_timeseries(
        df,
        "time",
        columns,
        args.freq,
        train_length,
        valid_length,
    )
    predictions, targets = train_model(args, None, train, val, test, "all")
else:
    model_cls = args.model_map[args.model]
    model = model_cls(**args.model_params[args.model])

    columns = df.columns
    columns = columns.drop("time")
    for col in columns:
        train, val, test = prepare_timeseries(
            df,
            "time",
            col,
            args.freq,
            train_length,
            valid_length,
        )

        prediction, target = train_model(args, model, train, val, test, col)

        predictions.append(prediction)
        targets.append(target)

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

idx = targets != 0
print("total", MAE(predictions[idx], targets[idx]))
# df = df.drop(columns=["time"])

# save predictions to npy
np.save(
    os.path.join(args.output_folder, "predictions.npy"), predictions, allow_pickle=False
)
# save targets to npy
np.save(os.path.join(args.output_folder, "targets.npy"), targets, allow_pickle=False)

# save df.columns to json
with open(os.path.join(args.output_folder, "columns.json"), "w") as fp:
    columns = df.columns.tolist()
    columns = columns.remove("time")
    json.dump(columns, fp)

# from dataclasses import dataclass, field
# import json
# import os

# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# from pmdarima.arima import auto_arima, ARIMA
# import pandas as pd
# from tqdm import tqdm, trange
# # from cuml.tsa.arima import ARIMA

# from forecast.data.loader import DataParser

# from sklearnex import patch_sklearn

# patch_sklearn()

# from forecast.data.loader import DataParser
# from forecast.data.prophet import ProphetDataParser
# from forecast.utils.cmdparser import HfArgumentParser


# import multiprocessing as mp
# from functools import partial


# @dataclass
# class ModelConfig:
#     folder: str = field(metadata={"help": "folder for xml data"})
#     checkpoint: str = field(metadata={"help": "path to checkpoints"})
#     fill: str = field(
#         metadata={
#             "help": "fill with values ['zero': fill with zeros, 'ffill': forward fill, 'backfill': backward fill]"
#         }
#     )
#     features: str = field(
#         default="M",
#         metadata={
#             "help": "forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate"
#         },
#     )
#     debug: bool = field(default=False)

#     def __post_init__(self):
#         try:
#             os.mkdir(self.checkpoint)
#         except:
#             pass

#         assert self.features == "MS" or self.features == "S"


# def parameter_gen(max_p, max_d, max_q, max_P, max_D, max_Q, max_s):
#     for p in range(max_p):
#         for d in range(max_d):
#             for q in range(max_q):
#                 for P in range(max_P):
#                     for D in range(max_D):
#                         for Q in range(max_Q):
#                             for s in range(max_s):
#                                 yield (p, d, q), (P, D, Q, s)

# # for i in tqdm(parameter_gen(5,5,5,2,2,2,2)):
# #     pass

# # exit()

# def fit_predict_model(data, predict_len: int, train_X=None, test_X=None):
#     sc = MinMaxScaler(feature_range=(0, 1))
#     if len(data.shape) == 1:
#         data = data.reshape(-1, 1)
#     data_scaled = sc.fit_transform(data)

#     # model = auto_arima(data_scaled, X=train_X)
#     model = ARIMA((1,1,2), max_iter=500)
#     model.fit(data_scaled, train_X)
#     # model = ARIMA(data_scaled, order=(1,1,2), exog=train_X)
#     # model.forecast(predict_len, level=0.95, exog=test_X)
#     # model.predict(level=0.95, exog=train_X)
#     # print(model.summary())
#     # print(model.params())
#     f_p, c_p = model.predict(predict_len, X=test_X, return_conf_int=True)
#     f_pi, c_pi = model.predict_in_sample(X=train_X, return_conf_int=True)

#     f_p = sc.inverse_transform(f_p.reshape(-1, 1))
#     f_pi = sc.inverse_transform(f_pi.reshape(-1, 1))
#     c_p = sc.inverse_transform(c_p)
#     c_pi = sc.inverse_transform(c_pi)

#     return f_p, c_p, f_pi, c_pi


# def prepare_model(i, train_data, test_data, features: str):
#     os.sched_setaffinity(0, {i % (mp.cpu_count() // 4)})

#     train_X = (
#         None
#         if features == "S"
#         else train_data[:, [k for k in range(train_data.shape[1]) if i != k]]
#     )
#     test_X = (
#         None
#         if features == "S"
#         else test_data[:, [k for k in range(test_data.shape[1]) if i != k]]
#     )
#     # train_data = train_data[:, i]

#     result_real = fit_predict_model(train_data[:, i], len(test_data), train_X, test_X)
#     result_oracle = fit_predict_model(test_data[:, i], len(test_data), test_X, test_X)

#     return (
#         result_real[0].tolist(),
#         result_real[1].tolist(),
#         result_oracle[2].tolist(),
#         result_oracle[3].tolist(),
#     )


# if __name__ == "__main__":
#     pool = mp.Pool(mp.cpu_count() // 4)

#     parser = HfArgumentParser(ModelConfig)
#     args = parser.parse_args_into_dataclasses()[0]

#     parser = DataParser()
#     df = parser.parse_sndlib_xml(args.folder)
#     df = df.fillna(0) if args.fill == "zero" else df.fillna(method=args.fill)

#     if args.debug:
#         df = df.loc[:100]

#     df_len = len(df.index.values)
#     train_len = int(df_len * 0.6)

#     dataset = df.drop(columns=["time"]).values
#     train_set = dataset[:train_len]
#     test_set = dataset[train_len:]
#     np.save(os.path.join(args.checkpoint, "test.npy"), test_set, allow_pickle=False)

#     columns = list(df.columns)
#     columns.remove("time")

#     if args.debug:
#         columns = columns[:2]

#     results = pool.map(
#         partial(
#             prepare_model,
#             train_data=train_set,
#             test_data=test_set,
#             features=args.features,
#         ),
#         trange(len(columns)),
#     )

#     forecast_real = []
#     conf_real = []
#     forecast_oracle = []
#     conf_oracle = []

#     for result in results:
#         f_real, c_real, f_oracle, c_oracle = result
#         forecast_real.append(f_real)
#         forecast_oracle.append(f_oracle)
#         conf_real.append(c_real)
#         conf_oracle.append(c_oracle)

#     forecast_real = np.array(forecast_real)
#     forecast_oracle = np.array(forecast_oracle)
#     conf_real = np.array(conf_real)
#     conf_oracle = np.array(conf_oracle)

#     with open(os.path.join(args.checkpoint, "columns.json"), "w") as fp:
#         json.dump(columns, fp)
#     np.save(
#         os.path.join(args.checkpoint, "pred_real.npy"),
#         forecast_real,
#         allow_pickle=False,
#     )
#     np.save(
#         os.path.join(args.checkpoint, "pred_oracle.npy"),
#         forecast_oracle,
#         allow_pickle=False,
#     )

#     np.save(
#         os.path.join(args.checkpoint, "conf_real.npy"), conf_real, allow_pickle=False,
#     )
#     np.save(
#         os.path.join(args.checkpoint, "conf_oracle.npy"),
#         conf_oracle,
#         allow_pickle=False,
#     )

# # columns = list(df.columns)
# # columns.remove("time")
# # forecast_list = []
# # lb_list = []
# # ub_list = []
# # for i, col in tqdm(enumerate(columns)):
# #     data = train_set[:, i]
# #     sc = MinMaxScaler(feature_range=(0, 1))
# #     train_set_scaled = sc.fit_transform(data.reshape(-1, 1))
# #     test_set_scaled = sc.transform(test_set[:, i].reshape(-1, 1))

# #     model_real = auto_arima(train_set_scaled)
# #     model_oracle = auto_arima(test_set_scaled)
# #     # print(model_real.summary())
# #     # print(model_oracle.summary())

# #     # break

# #     forecasts_real, conf_int_real = model_real.predict(df_len - train_len, return_conf_int=True,)
# #     forecasts_oracle, conf_int_oracle = model_oracle.predict_in_sample()


# #     forecasts = sc.inverse_transform(forecasts.reshape(-1, 1))
# #     lower_bound = sc.inverse_transform(conf_int[0, :].reshape(-1, 1))
# #     upper_bound = sc.inverse_transform(conf_int[1, :].reshape(-1, 1))
# #     forecast_list.append(forecasts.tolist())
# #     lb_list.append(lower_bound.tolist())
# #     ub_list.append(upper_bound.tolist())

# #     np.save(
# #         os.path.join(args.checkpoint, "pred.npy"),
# #         np.array(forecast_list),
# #         allow_pickle=False,
# #     )

# #     np.save(
# #         os.path.join(args.checkpoint, "pred_lb.npy"),
# #         np.array(lb_list),
# #         allow_pickle=False,
# #     )
# #     np.save(
# #         os.path.join(args.checkpoint, "pred_ub.npy"),
# #         np.array(ub_list),
# #         allow_pickle=False,
# #     )
