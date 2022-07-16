from dataclasses import dataclass, field
import os
import numpy as np
from sklearn.model_selection import train_test_split
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

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

dataset = df.drop(columns=['timestamps']).values
training_set = dataset[:train_len]
test_set = dataset[train_len:]

print(training_set.shape, test_set.shape)

train_time = df['timestamps'][:train_len]
test_time = df['timestamps'][train_len:]

columns = list(df.columns)[:-1]
forecast_list = []
lb_list = []
ub_list = []
for i, col in enumerate(columns):
    m = Prophet()
    data = training_set[:, i]
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(data.reshape(-1,1))

    train = pd.DataFrame(
        {
            "ds": train_time,
            'y': training_set_scaled.flatten()
        }
    )
    try:
        m.fit(train)
    except:
        print(col, train)

    future = m.make_future_dataframe(periods=len(test_time))
    forecast = m.predict(future)
    value = sc.inverse_transform(forecast['yhat'].values.reshape(-1,1))
    lower_bound = sc.inverse_transform(forecast['yhat_lower'].values.reshape(-1,1))
    upper_bound = sc.inverse_transform(forecast['yhat_upper'].values.reshape(-1,1))
    forecast_list.append(value.tolist())
    lb_list.append(lower_bound.tolist())
    ub_list.append(upper_bound.tolist())

np.save(os.path.join(args.checkpoint, "test.npy"), test_set, allow_pickle=False)
np.save(
    os.path.join(args.checkpoint, "pred.npy"), np.array(forecast_list), allow_pickle=False
)

np.save(
    os.path.join(args.checkpoint, "pred_lb.npy"), np.array(lb_list), allow_pickle=False
)
np.save(
    os.path.join(args.checkpoint, "pred_ub.npy"), np.array(ub_list), allow_pickle=False
)

# folder = "directed-abilene-zhang-5min-over-6months-ALL"
# # folder = "directed-geant-uhlig-15min-over-4months-ALL"

# parser = ProphetDataParser()
# df_dict = parser.format_sndlib_xml(folder)

# figure_dir = "figures/prophet"

# print(df_dict)

# def reject_outliers(data, m = 2.):
#     d = np.abs(data - np.median(data))
#     mdev = np.median(d)
#     s = d/mdev if mdev else 0.
#     return data[s<m]

# for pair, df in tqdm(df_dict.items()):
#     index = df.index
#     train_index, test_index = train_test_split(index, test_size=0.2)
#     train, test = df.drop(test_index), df.drop(train_index)
#     # print(train)
#     # print(test)
#     m = Prophet()

#     try:
#         m.fit(train)
#     except:
#         print(pair, train)

#     forecast = m.predict(test[["ds"]])
    
#     error = (forecast.yhat - test.y).abs()
#     # print(error)
#     error = (error / test.y).dropna().to_numpy()
#     error = reject_outliers(error)
#     # print(error)
#     # getting data of the histogram
#     count, bins_count = np.histogram(error, bins=100)
    
#     # finding the PDF of the histogram using count values
#     pdf = count / sum(count)
    
#     # using numpy np.cumsum to calculate the CDF
#     # We can also find using the PDF values by looping and adding
#     cdf = np.cumsum(pdf)
    
#     # plotting PDF and CDF
#     plt.plot(bins_count[1:], cdf)
#     # plt.hist(error / test.y, cumulative=1, histtype='step', bins=100, color='tab:orange')
#     plt.savefig(os.path.join(figure_dir, folder, f"{pair}_cdf.png"), bbox_inches='tight')

#     # fig1 = m.plot(forecast)
#     # plt.savefig(os.path.join(figure_dir, folder, f"{pair}.png"), bbox_inches='tight')

#     # test = pd.merge(left=test, right=forecast, left_on='ds', right_on='ds')
#     # test.to_csv(os.path.join(figure_dir, folder, f"{pair}.csv"), index=False)

#     # fig2 = m.plot_components(forecast)
#     # plt.show()

#     # break
