# Importing the libraries
from dataclasses import dataclass, field
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

from forecast.data.loader import DataParser
from forecast.utils.cmdparser import HfArgumentParser

from sklearnex import patch_sklearn

patch_sklearn()

# folder = "directed-abilene-zhang-5min-over-6months-ALL"
# folder = "directed-geant-uhlig-15min-over-4months-ALL"


@dataclass
class ModelConfig:
    folder: str = field(metadata={"help": "folder for xml data"})
    checkpoint: str = field(metadata={"help": "path to checkpoints"})
    layer: str = field(metadata={"help": "gru or lstm"})
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
    win_size: int = field(default=60, metadata={"help": "historical dta to use"})
    batch_size: int = field(default=32, metadata={"help": "batch_size"})
    epoch: int = field(default=50, metadata={"help": "batch_size"})

    def __post_init__(self):
        try:
            os.mkdir(self.checkpoint)
        except:
            pass
        self.checkpoint_path = os.path.join(self.checkpoint, "cp-{epoch:04d}.ckpt")

        self.layer_cls = GRU if self.layer.lower() == "GRU" else LSTM


def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))


def train_loop(X_train, y_train, X_test, sc):
    # The LSTM architecture
    regressor = Sequential()
    # First LSTM layer with Dropout regularisation
    regressor.add(
        args.layer_cls(
            units=50,
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2]),
        )
    )
    regressor.add(Dropout(0.2))
    # Second LSTM layer
    regressor.add(args.layer_cls(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    # Third LSTM layer
    regressor.add(args.layer_cls(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    # Fourth LSTM layer
    regressor.add(args.layer_cls(units=50))
    regressor.add(Dropout(0.2))
    # The output layer
    regressor.add(Dense(units=X_train.shape[2]))

    regressor.summary()
    # exit()

    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=args.checkpoint_path, verbose=1, save_weights_only=True
    )

    # Compiling the RNN
    regressor.compile(optimizer="rmsprop", loss="mean_squared_error")
    
    
    # Fitting to the training set
    # Use cross validation for the training set

    regressor.fit(
        X_train,
        y_train,
        epochs=args.epoch,
        batch_size=args.batch_size,
        callbacks=[cp_callback],
    )

    predicted = regressor.predict(X_test)
    predicted = sc.inverse_transform(predicted)

    return predicted


parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

parser = DataParser()
df = parser.parse_sndlib_xml(args.folder)
df = df.drop(columns=["timestamps"])
df = df.fillna(0) if args.fill == "zero" else df.fillna(method=args.fill)
if df.isnull().values.any():
    df = df.fillna(0)

columns = list(df.columns)

print(df.head())

df_len = len(df.index.values)
train_len = int(df_len * 0.6)

dataset = df.values
training_set = df.loc[:train_len].values
test_set = df.loc[train_len:].values

np.save(os.path.join(args.checkpoint, "test.npy"), test_set, allow_pickle=False)

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.transform(test_set)


X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i - 60 : i, :])
    y_train.append(training_set_scaled[i, :])
X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train.shape, y_train.shape)

inputs = dataset[len(dataset) - len(test_set) - args.win_size :]
inputs = sc.transform(inputs)

X_test = []
y_test = []
for i in range(args.win_size, len(inputs)):
    X_test.append(inputs[i - args.win_size : i, :])
    y_test.append(inputs[i, :])
X_test, y_test = np.array(X_test), np.array(y_test)

forecast_real = train_loop(X_train, y_train, X_test, sc)
forecast_oracle = train_loop(X_test, y_test, X_test, sc)

with open(os.path.join(args.checkpoint, "columns.json"), "w") as fp:
    json.dump(columns, fp)

np.save(
    os.path.join(args.checkpoint, "pred_real.npy"), forecast_real, allow_pickle=False,
)
np.save(
    os.path.join(args.checkpoint, "pred_oracle.npy"),
    forecast_oracle,
    allow_pickle=False,
)

