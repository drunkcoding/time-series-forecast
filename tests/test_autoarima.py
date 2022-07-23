from dataclasses import dataclass, field
import json
import os
from pyexpat import features

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pmdarima.arima import auto_arima
import pandas as pd
from tqdm import tqdm, trange

from forecast.data.loader import DataParser

from sklearnex import patch_sklearn

patch_sklearn()

from forecast.data.loader import DataParser
from forecast.data.prophet import ProphetDataParser
from forecast.utils.cmdparser import HfArgumentParser


import multiprocessing as mp
from functools import partial


@dataclass
class ModelConfig:
    folder: str = field(metadata={"help": "folder for xml data"})
    checkpoint: str = field(metadata={"help": "path to checkpoints"})
    fill: str = field(
        metadata={
            "help": "fill with values ['zero': fill with zeros, 'ffill': forward fill, 'backfill': backward fill]"
        }
    )
    features: str = field(
        default="M",
        metadata={
            "help": "forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate"
        },
    )

    def __post_init__(self):
        try:
            os.mkdir(self.checkpoint)
        except:
            pass

        assert self.features == "MS" or self.features == "S"


def fit_predict_model(data, predict_len: int, train_X=None, test_X=None):
    sc = MinMaxScaler(feature_range=(0, 1))
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    data_scaled = sc.fit_transform(data)

    model = auto_arima(data_scaled, X=train_X)
    print(model.summary())
    f_p, c_p = model.predict(predict_len, X=test_X, return_conf_int=True)
    f_pi, c_pi = model.predict_in_sample(X=train_X, return_conf_int=True)

    return f_p, c_p, f_pi, c_pi


def prepare_model(i, train_data, test_data, features: str):
    os.sched_setaffinity(0, {i % mp.cpu_count()})

    train_X = (
        None
        if features == "S"
        else train_data[:, [k for k in range(train_data.shape[1]) if i != k]]
    )
    test_X = (
        None
        if features == "S"
        else test_data[:, [k for k in range(test_data.shape[1]) if i != k]]
    )
    train_data = train_data[:, i]

    result_real = fit_predict_model(train_data, len(test_data), train_X, test_X)
    result_oracle = fit_predict_model(test_data, len(test_data), test_X, test_X)

    return (
        result_real[0].tolist(),
        result_real[1].tolist(),
        result_oracle[2].tolist(),
        result_oracle[3].tolist(),
    )

if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count())

    parser = HfArgumentParser(ModelConfig)
    args = parser.parse_args_into_dataclasses()[0]

    parser = DataParser()
    df = parser.parse_sndlib_xml(args.folder)
    df = df.fillna(0) if args.fill == "zero" else df.fillna(method=args.fill)

    df_len = len(df.index.values)
    train_len = int(df_len * 0.6)

    dataset = df.drop(columns=["timestamps"]).values
    train_set = dataset[:train_len]
    test_set = dataset[train_len:]
    np.save(os.path.join(args.checkpoint, "test.npy"), test_set, allow_pickle=False)

    columns = list(df.columns)
    columns.remove("timestamps")

    results = pool.map(
        partial(
            prepare_model,
            train_data=train_set,
            test_data=test_set,
            features=args.features,
        ),
        trange(len(columns)),
    )

    forecast_real = []
    conf_real = []
    forecast_oracle = []
    conf_oracle = []

    for result in results:
        f_real, c_real, f_oracle, c_oracle = result
        forecast_real.append(f_real)
        forecast_oracle.append(f_oracle)
        conf_real.append(c_real)
        conf_oracle.append(c_oracle)

    forecast_real = np.array(forecast_real)
    forecast_oracle = np.array(forecast_oracle)
    conf_real = np.array(conf_real)
    conf_oracle = np.array(conf_oracle)

    with open(os.path.join(args.checkpoint, "columns.json"), 'w') as fp:
        json.dump(columns, fp)
    np.save(
        os.path.join(args.checkpoint, "pred_real.npy"),
        forecast_real,
        allow_pickle=False,
    )
    np.save(
        os.path.join(args.checkpoint, "pred_oracle.npy"),
        forecast_oracle,
        allow_pickle=False,
    )

    np.save(
        os.path.join(args.checkpoint, "conf_real.npy"),
        conf_real,
        allow_pickle=False,
    )
    np.save(
        os.path.join(args.checkpoint, "conf_oracle.npy"),
        conf_oracle,
        allow_pickle=False,
    )

# columns = list(df.columns)
# columns.remove("timestamps")
# forecast_list = []
# lb_list = []
# ub_list = []
# for i, col in tqdm(enumerate(columns)):
#     data = train_set[:, i]
#     sc = MinMaxScaler(feature_range=(0, 1))
#     train_set_scaled = sc.fit_transform(data.reshape(-1, 1))
#     test_set_scaled = sc.transform(test_set[:, i].reshape(-1, 1))

#     model_real = auto_arima(train_set_scaled)
#     model_oracle = auto_arima(test_set_scaled)
#     # print(model_real.summary())
#     # print(model_oracle.summary())

#     # break

#     forecasts_real, conf_int_real = model_real.predict(df_len - train_len, return_conf_int=True,)
#     forecasts_oracle, conf_int_oracle = model_oracle.predict_in_sample()


#     forecasts = sc.inverse_transform(forecasts.reshape(-1, 1))
#     lower_bound = sc.inverse_transform(conf_int[0, :].reshape(-1, 1))
#     upper_bound = sc.inverse_transform(conf_int[1, :].reshape(-1, 1))
#     forecast_list.append(forecasts.tolist())
#     lb_list.append(lower_bound.tolist())
#     ub_list.append(upper_bound.tolist())

#     np.save(
#         os.path.join(args.checkpoint, "pred.npy"),
#         np.array(forecast_list),
#         allow_pickle=False,
#     )

#     np.save(
#         os.path.join(args.checkpoint, "pred_lb.npy"),
#         np.array(lb_list),
#         allow_pickle=False,
#     )
#     np.save(
#         os.path.join(args.checkpoint, "pred_ub.npy"),
#         np.array(ub_list),
#         allow_pickle=False,
#     )
