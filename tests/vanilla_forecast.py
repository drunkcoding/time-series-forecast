from dataclasses import dataclass, field
from multiprocessing import freeze_support
import numpy as np
import os
import warnings
from tqdm import tqdm
from forecast.data.imputer import train_test_split

from forecast.data.loader import DataParser
from pyfc_utils import (
    create_dataloaders,
    create_pl_trainer,
    get_best_model,
    model_predict,
)

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl

import torch

from pytorch_forecasting import Baseline, TimeSeriesDataSet, get_rnn
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE, QuantileLoss

from forecast.data.prophet import ProphetDataParser
from forecast.utils.cmdparser import HfArgumentParser


@dataclass
class ModelConfig:
    folder: str = field(metadata={"help": "folder for xml data"})
    checkpoint: str = field(metadata={"help": "path to checkpoints"})
    layer: str = field(metadata={"help": "gru or lstm"})
    fill: str = field(
        metadata={
            "help": "fill with values ['zero': fill with zeros, 'ffill': forward fill, 'backfill': backward fill]"
        }
    )
    strategy: str = field(
        default="mean",
        metadata={"help": "fill with values [mean, median, most_frequent]"},
    )
    features: str = field(
        default="M",
        metadata={
            "help": "forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate"
        },
    )
    debug: bool = field(default=False)

    def __post_init__(self):
        try:
            os.mkdir(self.checkpoint)
        except:
            pass

        # assert self.features == "MS" or self.features == "S"


parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

parser = ProphetDataParser()
df_dict = parser.format_sndlib_xml(args.folder)
raw_columns = list(df_dict.keys())

df = pd.concat(df_dict.values(), ignore_index=True)

(
    columns,
    train_set,
    test_set,
    train_time,
    test_time,
    train_indocator,
    test_indicator,
) = train_test_split(df, args)
dataset = np.concatenate((train_set, test_set), axis=0)

df = pd.DataFrame(dataset, columns=columns)
df["time_idx"] = [x for x in range(len(df.index.values))]

# create dataset and dataloaders
max_encoder_length = 60
max_prediction_length = 1

training_cutoff = df["time_idx"].max() // 2
validation_cutoff = training_cutoff + df["time_idx"].max() * 0.1

dataset_params = dict(
    # df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="y",
    group_ids=["gid"],
    categorical_encoders={"gid": NaNLabelEncoder().fit(df.gid)},
    static_categoricals=[
        "gid",
        "missing",
    ],  # as we plan to forecast correlations, it is important to use series characteristics (e.g. a series identifier)
    time_varying_unknown_reals=["y"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff], **dataset_params
)
validation = TimeSeriesDataSet(
    df[lambda x: (x.time_idx > training_cutoff) & (x.time_idx <= validation_cutoff)],
    **dataset_params,
)
testing = TimeSeriesDataSet(
    df[lambda x: x.time_idx > validation_cutoff], **dataset_params
)

pl.seed_everything(42)
import pytorch_forecasting as ptf


def tune_and_train(train, val, test, prefix):

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train, val, test
    )
    trainer = create_pl_trainer(args.debug)

    net = get_rnn(args.layer).from_dataset(
        train,
        learning_rate=1e-3,
        log_interval=10,
        log_val_interval=1,
        hidden_size=256,
        dropout=0.1,
        output_size=1,
        loss=SMAPE(),
        # reduce learning rate if no improvement in validation loss after x epochs
        reduce_on_plateau_patience=4,
    )

    trainer.fit(
        net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )

    best_model = get_best_model(trainer, net)

    actuals, predictions, decoder_cat = model_predict(
        best_model, test_dataloader, columns
    )

    np.save(
        os.path.join(args.checkpoint, f"pred_{prefix}.npy"),
        predictions,
        allow_pickle=False,
    )
    np.save(
        os.path.join(args.checkpoint, f"test.npy"), actuals, allow_pickle=False,
    )
    np.save(
        os.path.join(args.checkpoint, f"test_indicator.npy"),
        decoder_cat,
        allow_pickle=False,
    )


if __name__ == "__main__":
    freeze_support()
    tune_and_train(training, validation, testing, "real")
    tune_and_train(testing, validation, testing, "oracle")
