import os
import numpy as np
from sklearn.model_selection import train_test_split
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from forecast.data.prophet import ProphetDataParser, MultivariateDataParser

# folder = "directed-abilene-zhang-5min-over-6months-ALL"
folder = "directed-geant-uhlig-15min-over-4months-ALL"

parser = MultivariateDataParser()
df = parser.format_sndlib_xml(folder)
df.to_csv(os.path.join("datasets", folder, f"all.csv"), index=False)

# parser = ProphetDataParser()
# df_dict = parser.format_sndlib_xml(folder)

# figure_dir = "figures/prophet"

# for pair, df in tqdm(df_dict.items()):
#     df = df.rename(columns={"ds": "date", "y": "OT"})
#     df.to_csv(os.path.join("datasets", folder, f"{pair}.csv"), index=False)