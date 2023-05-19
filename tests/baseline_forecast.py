from dataclasses import dataclass, field
from itertools import count
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
        self.checkpoint = os.path.join("checkpoints", self.dataset, "baseline")
        try:
            os.mkdir(self.checkpoint)
        except:
            pass

        # assert self.features == "S"


parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

# folder = "directed-abilene-zhang-5min-over-6months-ALL"
# folder = "directed-geant-uhlig-15min-over-4months-ALL"

parser = ProphetDataParser()
df_dict = parser.format_sndlib_xml(args.folder)
columns = list(df_dict.keys())

for key in df_dict.keys():
    df_dict[key]["gid"] = key
    df_dict[key]["time_idx"] = np.arange(len(df_dict[key].index.values))
    df_dict[key]["missing"] = np.where(np.isnan(df_dict[key]["y"].values), 0, 1).astype(
        str
    )

    if args.debug:
        df_dict[key] = df_dict[key].iloc[:1000]


df = pd.concat(df_dict.values(), ignore_index=True)
# df = df.fillna(0) if args.fill == "zero" else df.fillna(method=args.fill)
df = df.fillna(0)


# print(df)

# create dataset and dataloaders
max_encoder_length = 60
max_prediction_length = 1

training_cutoff = df["time_idx"].max() // 2
validation_cutoff = int(training_cutoff + df["time_idx"].max() * 0.1)

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

print(len(training), len(validation), len(testing))

# actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
# baseline_predictions = Baseline().predict(val_dataloader)
# print("baseline_predictions", SMAPE()(baseline_predictions, actuals))

pl.seed_everything(42)
import pytorch_forecasting as ptf


def tune_and_train(
    train: TimeSeriesDataSet,
    val: TimeSeriesDataSet,
    test: TimeSeriesDataSet,
    prefix: str,
):

    # validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training_cutoff + 1)
    batch_size = 512
    # synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
    # train_dataloader = train.to_dataloader(
    #     train=True, batch_size=batch_size * 10, num_workers=torch.cuda.device_count(), pin_memory=True
    # )
    # val_dataloader = val.to_dataloader(
    #     train=False, batch_size=batch_size * 10, num_workers=torch.cuda.device_count()
    # )
    test_dataloader = test.to_dataloader(
        train=False,
        batch_size=batch_size * 50,
        num_workers=torch.cuda.device_count(),
        batch_sampler="synchronized",
    )

    best_model = Baseline()

    decoder_cat = []
    actuals = []

    # count = 0
    for x, y in tqdm(iter(test_dataloader)):
        decoder_cat.append(x["decoder_cat"].squeeze())
        actuals.append(y[0].squeeze())
    decoder_cat = torch.stack(decoder_cat)
    actuals = torch.stack(actuals)
    predictions = best_model.predict(test_dataloader, show_progress_bar=True,).squeeze()

    # actuals = actuals.reshape(-1, len(columns))
    predictions = predictions.reshape(-1, len(columns))
    # predictions = predictions.reshape(len(columns), -1).T
    # decoder_cat = decoder_cat.squeeze()

    actuals = actuals.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    decoder_cat = decoder_cat[..., 1].detach().cpu().numpy().astype(bool).squeeze()

    print("actuals", actuals.shape)
    print("predictions", predictions.shape)
    print("decoder_cat", decoder_cat.shape)

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

    parser = DataParser()
    df = parser.parse_sndlib_xml(args.folder)
    df = df.drop(columns=["timestamps"])
    df = df.fillna(0)

    if args.debug:
        df = df.iloc[:1000]

    print(actuals.shape)
    print(df.values[-actuals.shape[0]:].shape)

    print(actuals[:, 0])
    print(df.values[-actuals.shape[0] :, 0])
    print(predictions[:, 0])
    print(decoder_cat[:, 0])

    assert np.isclose(
        actuals[:, 0], df.values[-actuals.shape[0] :, 0], atol=1e-5
    ).all()


if __name__ == "__main__":
    freeze_support()
    tune_and_train(training, validation, testing, "real")
    # tune_and_train(testing, validation, testing, "oracle")
