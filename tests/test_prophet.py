import os
from sklearn.model_selection import train_test_split
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from forecast.data.prophet import ProphetDataParser

# folder = "directed-abilene-zhang-5min-over-6months-ALL"
folder = "directed-geant-uhlig-15min-over-4months-ALL"

parser = ProphetDataParser()
df_dict = parser.format_sndlib_xml(folder)

figure_dir = "figures/prophet"

print(df_dict)

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
    # print(forecast)
    fig1 = m.plot(forecast)
    plt.savefig(os.path.join(figure_dir, folder, f"{pair}.png"), bbox_inches='tight')

    test = pd.merge(left=test, right=forecast, left_on='ds', right_on='ds')
    test.to_csv(os.path.join(figure_dir, folder, f"{pair}.csv"), index=False)

    # fig2 = m.plot_components(forecast)
    # plt.show()

    # break
