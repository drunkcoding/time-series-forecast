from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from forecast.data.prophet import ProphetDataParser


def get_stationarity(timeseries):

    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    # rolling statistics plot
    original = plt.plot(timeseries, color="blue", label="Original")
    mean = plt.plot(rolling_mean, color="red", label="Rolling Mean")
    std = plt.plot(rolling_std, color="black", label="Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean & Standard Deviation")
    plt.show(block=False)

    # Dickey–Fuller test:
    result = adfuller(timeseries["Passengers"])
    print("ADF Statistic: {}".format(result[0]))
    print("p-value: {}".format(result[1]))
    print("Critical Values:")
    for key, value in result[4].items():
        print("\t{}: {}".format(key, value))


folder = "directed-abilene-zhang-5min-over-6months-ALL"
# folder = "directed-geant-uhlig-15min-over-4months-ALL"

parser = ProphetDataParser()
df_dict = parser.format_sndlib_xml(folder)

figure_dir = "figures/prophet"

print(df_dict)

for pair, df in tqdm(df_dict.items()):
    index = df.index
    train_index, test_index = train_test_split(index, test_size=0.2)
    train, test = df.drop(test_index), df.drop(train_index)

    df = df.dropna()

    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df.y); axes[0, 0].set_title('Original Series')
    plot_acf(df.y, ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(df.y.diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df.y.diff().dropna(), ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(df.y.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df.y.diff().diff().dropna(), ax=axes[2, 1])

    plt.show()

    # PACF plot of 1st differenced series
    plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(df.y.diff()); axes[0].set_title('1st Differencing')
    # axes[1].set(ylim=(0,5))
    plot_pacf(df.y.diff().dropna(), ax=axes[1])

    plt.show()

    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(df.y.diff()); axes[0].set_title('1st Differencing')
    # axes[1].set(ylim=(0,1.2))
    plot_acf(df.y.diff().dropna(), ax=axes[1])

    plt.show()

    model = ARIMA(df.y.fillna(0), order=(1,1,1))
    model_fit = model.fit()
    print(model_fit.summary())

    # fc, se, conf = model_fit.forecast(len(test.index), alpha=0.05)  # 95% conf
    res = model_fit.forecast(len(test.index), alpha=0.05)  # 95% conf
    print(res)

    # Make as pandas series
    # fc_series = pd.Series(fc, index=test.index)
    # lower_series = pd.Series(conf[:, 0], index=test.index)
    # upper_series = pd.Series(conf[:, 1], index=test.index)

    # Plot
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train.y, label='training')
    plt.plot(test.y, label='actual')
    plt.plot(res, label='forecast')
    # plt.fill_between(lower_series.index, lower_series, upper_series, 
    #                 color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

    break
