from dataclasses import dataclass, field
from itertools import count
import json
from multiprocessing import freeze_support
import numpy as np
import os
import warnings
from tqdm import tqdm

from forecast.data.loader import DataParser

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import torch

from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import SMAPE, MultivariateNormalDistributionLoss

from forecast.data.prophet import ProphetDataParser
from forecast.utils.cmdparser import HfArgumentParser


@dataclass
class ModelConfig:
    folder: str = field(metadata={"help": "folder for xml data"})
    # checkpoint: str = field(metadata={"help": "path to checkpoints"})
    debug: bool = field(default=False)

    def __post_init__(self):
        if "abilene" in self.folder:
            self.dataset = "abilene"
        elif "geant" in self.folder:
            self.dataset = "geant"
        else:
            raise ValueError("dataset not supported")
        self.checkpoint = os.path.join("checkpoints", self.dataset, "none")
        try:
            os.mkdir(self.checkpoint)
        except:
            pass

        # assert self.features == "S"


parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

# folder = "directed-abilene-zhang-5min-over-6months-ALL"
# folder = "directed-geant-uhlig-15min-over-4months-ALL"

parser = DataParser()
df = parser.parse_sndlib_xml(args.folder)
# df = df.fillna(0) if args.fill == "zero" else df.fillna(method=args.fill)
# df = df.fillna(0)
df = df.drop(columns=["timestamps"])
columns = list(df.columns)

raw_data = df.values
raw_indicator = np.isnan(raw_data)

df_len = len(df.index.values)
train_len = int(df_len * 0.6)

train_indicator = raw_indicator[:train_len]
test_indicator = raw_indicator[train_len:]

train_set = raw_data[:train_len]
test_set = raw_data[train_len:]

with open(os.path.join(args.checkpoint, "columns.json"), "w") as fp:
    json.dump(columns, fp)
np.save(os.path.join(args.checkpoint, "test.npy"), test_set, allow_pickle=False)
np.save(os.path.join(args.checkpoint, "test_indicator.npy"), test_indicator, allow_pickle=False)

pred = np.zeros_like(test_set)

for i in range(test_set.shape[1]):
    pred[: ,i] = np.nanmean(train_set[:, i])

np.save(
    os.path.join(args.checkpoint, "pred_real.npy"),
    pred,
    allow_pickle=False,
)
np.save(
    os.path.join(args.checkpoint, "pred_oracle.npy"),
    pred,
    allow_pickle=False,
)

