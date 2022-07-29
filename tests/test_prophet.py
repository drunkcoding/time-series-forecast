from dataclasses import dataclass, field
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange

from forecast.data.loader import DataParser
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
    debug: bool = field(default=False)

    def __post_init__(self):
        try:
            os.mkdir(self.checkpoint)
        except:
            pass

def fit_predict_model(i, train_set, train_time, test_set, test_time):
    os.sched_setaffinity(0, {i % (mp.cpu_count() - 1)})
    
    model_real = Prophet()
    model_oracle = Prophet()

    predict_len = len(test_set)

    train_set = train_set[:, i]
    test_set = test_set[:, i]

    # print(train_set.shape, test_set.shape, predict_len)

    sc = MinMaxScaler(feature_range=(0, 1))
    train_set_scaled = sc.fit_transform(train_set.reshape(-1,1))
    test_set_scaled = sc.fit_transform(test_set.reshape(-1,1))

    # print(train_time.shape, train_set_scaled.shape)

    train = pd.DataFrame(
        {
            "ds": train_time,
            'y': train_set_scaled.flatten()
        }
    )
    test = pd.DataFrame(
        {
            "ds": test_time,
            'y': test_set_scaled.flatten()
        }
    )
    model_real.fit(train)
    model_oracle.fit(test)

    # future = model_real.make_future_dataframe(periods=predict_len)
    # pred_r = model_real.predict(future)
    pred_r = model_real.predict(test[['ds']])
    pred_o = model_oracle.predict(test[['ds']])

    f_r = pred_r['yhat'].values.reshape(-1,1)
    c_r = pred_r[['yhat_lower', 'yhat_upper']].values

    f_o = pred_o['yhat'].values.reshape(-1,1)
    c_o = pred_o[['yhat_lower', 'yhat_upper']].values

    # print(f_r.shape, c_r.shape, f_o.shape, c_o.shape)

    f_r = sc.inverse_transform(f_r)
    c_r = sc.inverse_transform(c_r)
    f_o = sc.inverse_transform(f_o)
    c_o = sc.inverse_transform(c_o)

    return f_r.tolist(), c_r.tolist(), f_o.tolist(), c_o.tolist()


if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count() - 1)

    parser = HfArgumentParser(ModelConfig)
    args = parser.parse_args_into_dataclasses()[0]

    parser = DataParser()
    df = parser.parse_sndlib_xml(args.folder)
    df = df.fillna(0) if args.fill == "zero" else df.fillna(method=args.fill)
    if df.isnull().values.any():
        df = df.fillna(0)

    print(df.head())

    if args.debug:
        df = df.loc[:100]

    df_len = len(df.index.values)
    train_len = int(df_len * 0.6)

    dataset = df.drop(columns=['timestamps']).values
    train_set = dataset[:train_len]
    test_set = dataset[train_len:]
    np.save(os.path.join(args.checkpoint, "test.npy"), test_set, allow_pickle=False)

    # exit()
    # print(train_set.shape, test_set.shape)

    train_time = df['timestamps'][:train_len]
    test_time = df['timestamps'][train_len:]

    columns = list(df.columns)
    columns.remove("timestamps")

    if args.debug:
        columns = columns[:2]

    results = pool.map(
        partial(
            fit_predict_model,
            train_set=train_set, 
            train_time=train_time, 
            test_set=test_set, 
            test_time=test_time,
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

    with open(os.path.join(args.checkpoint, "columns.json"), "w") as fp:
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
        os.path.join(args.checkpoint, "conf_real.npy"), conf_real, allow_pickle=False,
    )
    np.save(
        os.path.join(args.checkpoint, "conf_oracle.npy"),
        conf_oracle,
        allow_pickle=False,
    )

    exit()

    forecast_list = []
    lb_list = []
    ub_list = []
    for i, col in enumerate(columns):
        m = Prophet()
        data = train_set[:, i]
        sc = MinMaxScaler(feature_range=(0, 1))
        train_set_scaled = sc.fit_transform(data.reshape(-1,1))

        train = pd.DataFrame(
            {
                "ds": train_time,
                'y': train_set_scaled.flatten()
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

        break

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
