from dataclasses import dataclass, field
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pmdarima.arima import auto_arima
import pandas as pd

from forecast.data.loader import DataParser

from sklearnex import patch_sklearn

patch_sklearn()
from forecast.data.loader import DataParser
from forecast.data.prophet import ProphetDataParser
from forecast.utils.cmdparser import HfArgumentParser


@dataclass
class ModelConfig:
    folder: str = field(metadata={"help": "folder for xml data"})
    checkpoint: str = field(metadata={"help": "path to checkpoints"})

    def __post_init__(self):
        try:
            os.mkdir(self.checkpoint)
        except:
            pass


parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

parser = DataParser()
df = parser.parse_sndlib_xml(args.folder)
df = df.fillna(0)

print(df.head())

df_len = len(df.index.values)
train_len = int(df_len * 0.6)

dataset = df.drop(columns=["timestamps"]).values
training_set = dataset[:train_len]
test_set = dataset[train_len:]


columns = list(df.columns)[:-1]
forecast_list = []
lb_list = []
ub_list = []
for i, col in enumerate(columns):
    data = training_set[:, i]
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(data.reshape(-1, 1))

    model = auto_arima(training_set_scaled)
    forecasts, conf_int = model.predict(
        df_len-train_len,
        return_conf_int=True,
    )

    forecasts = sc.inverse_transform(forecasts.reshape(-1, 1))
    lower_bound = sc.inverse_transform(conf_int[0,:].reshape(-1, 1))
    upper_bound = sc.inverse_transform(conf_int[1,:].reshape(-1, 1))
    forecast_list.append(forecasts.tolist())
    lb_list.append(lower_bound.tolist())
    ub_list.append(upper_bound.tolist())

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
