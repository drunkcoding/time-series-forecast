# Importing the libraries
from dataclasses import dataclass, field
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
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
    checkpoint_path: str = field(metadata={"help": "path to checkpoints"})
    win_size: int = field(default=60, metadata={"help": "historical dta to use"})

    def __post_init__(self):
        try:
            os.mkdir(self.checkpoint_path)
        except:
            pass
        self.checkpoint_path = os.path.join(self.checkpoint_path, "cp-{epoch:04d}.ckpt")

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))

parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

parser = DataParser()
df = parser.parse_sndlib_xml(args.folder)
df = df.fillna(0).drop(columns=["timestamps"])

print(df.head())

df_len = len(df.index.values)
train_len = int(df_len * 0.6)

dataset = df.values
training_set = df.loc[:train_len].values
test_set  = df.loc[train_len:].values

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)


X_train = []
y_train = []
for i in range(60,len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i,:])
    y_train.append(training_set_scaled[i,:])
X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train.shape, y_train.shape)

# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))

cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=args.checkpoint_path, 
    verbose=1, 
    save_weights_only=True)

# Compiling the RNN
regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
# Fitting to the training set
regressor.fit(X_train,y_train,epochs=20,batch_size=32,callbacks=[cp_callback])


inputs = dataset[len(dataset)-len(test_set) - args.win_size:]
inputs  = sc.transform(inputs)

X_test = []
for i in range(args.win_size, len(inputs)):
    X_test.append(inputs[i-args.win_size:i,:])
X_test = np.array(X_test)

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

return_rmse(test_set,predicted_stock_price)