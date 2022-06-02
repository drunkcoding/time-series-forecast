import os
from sklearn.model_selection import train_test_split
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from forecast.data.prophet import ProphetDataParser
from forecast.model.informer import Informer

# folder = "directed-abilene-zhang-5min-over-6months-ALL"
folder = "directed-geant-uhlig-15min-over-4months-ALL"

parser = ProphetDataParser()
df_dict = parser.format_sndlib_xml(folder)

figure_dir = "figures/prophet"


