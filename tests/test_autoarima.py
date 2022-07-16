import os
from pmdarima.arima import auto_arima
import pandas as pd

from forecast.data.loader import DataParser

from sklearnex import patch_sklearn
patch_sklearn()

# folder = "directed-abilene-zhang-5min-over-6months-ALL"
folder = "directed-geant-uhlig-15min-over-4months-ALL"

parser = DataParser()
df = parser.parse_sndlib_xml(folder)

# df = pd.read_csv(os.path.join(folder, filename))
df = df.fillna(0)
df_len = len(df.index.values)
train_len = int(df_len * 0.6)

df_train = df.loc[:train_len]
df_test = df.loc[train_len:]

node_pairs = df_train.columns[:-1]
print(node_pairs)

model = auto_arima(df_train[node_pairs[0]], df_train[node_pairs[1:]].to_numpy())
# model.fit(df_train[node_pairs[3]])
forecasts, conf_int = model.predict(len(df_test.index.values), X=df_test[node_pairs[1:]].to_numpy(), return_conf_int=True)

print(forecasts)
print(forecasts - df_test.index.values)
print(conf_int)