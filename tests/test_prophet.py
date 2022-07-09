import os
import numpy as np
from sklearn.model_selection import train_test_split
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from forecast.data.prophet import ProphetDataParser

folder = "directed-abilene-zhang-5min-over-6months-ALL"
# folder = "directed-geant-uhlig-15min-over-4months-ALL"

parser = ProphetDataParser()
df_dict = parser.format_sndlib_xml(folder)

figure_dir = "figures/prophet"

print(df_dict)

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

for pair, df in tqdm(df_dict.items()):
    index = df.index
    train_index, test_index = train_test_split(index, test_size=0.2)
    train, test = df.drop(test_index), df.drop(train_index)
    # print(train)
    # print(test)
    m = Prophet()

    try:
        m.fit(train)
    except:
        print(pair, train)

    forecast = m.predict(test[["ds"]])
    
    error = (forecast.yhat - test.y).abs()
    # print(error)
    error = (error / test.y).dropna().to_numpy()
    error = reject_outliers(error)
    # print(error)
    # getting data of the histogram
    count, bins_count = np.histogram(error, bins=100)
    
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
    
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
    
    # plotting PDF and CDF
    plt.plot(bins_count[1:], cdf)
    # plt.hist(error / test.y, cumulative=1, histtype='step', bins=100, color='tab:orange')
    plt.savefig(os.path.join(figure_dir, folder, f"{pair}_cdf.png"), bbox_inches='tight')

    # fig1 = m.plot(forecast)
    # plt.savefig(os.path.join(figure_dir, folder, f"{pair}.png"), bbox_inches='tight')

    # test = pd.merge(left=test, right=forecast, left_on='ds', right_on='ds')
    # test.to_csv(os.path.join(figure_dir, folder, f"{pair}.csv"), index=False)

    # fig2 = m.plot_components(forecast)
    # plt.show()

    # break
