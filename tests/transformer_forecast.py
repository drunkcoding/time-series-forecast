from dataclasses import dataclass, field
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasZting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import SMAPE, QuantileLoss

from forecast.data.prophet import ProphetDataParser
from forecast.utils.cmdparser import HfArgumentParser


@dataclass
class ModelConfig:
    folder: str = field(metadata={"help": "folder for xml data"})
    checkpoint: str = field(metadata={"help": "path to checkpoints"})
    fill: str = field(
        metadata={
            "help": "fill with values ['zero': fill with zeros, 'ffill': forward fill, 'backfill': backward fill]"
        }
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

        assert self.features == "MS" or self.features == "S"


parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

# folder = "directed-abilene-zhang-5min-over-6months-ALL"
# folder = "directed-geant-uhlig-15min-over-4months-ALL"

parser = ProphetDataParser()
df_dict = parser.format_sndlib_xml(args.folder)

for key in df_dict.keys():
    df_dict[key]["gid"] = key
    df_dict[key]["time_idx"] = np.arange(len(df_dict[key].index.values))

    if args.debug:
        df_dict[key] = df_dict[key].loc[:1000]

df = pd.concat(df_dict.values(), ignore_index=True)
df = df.fillna(0) if args.fill == "zero" else df.fillna(method=args.fill)

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
    time_varying_unknown_reals=["y"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
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

# actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
# baseline_predictions = Baseline().predict(val_dataloader)
# print("baseline_predictions", SMAPE()(baseline_predictions, actuals))

pl.seed_everything(42)
import pytorch_forecasting as ptf


def tune_and_train(train, val, test, prefix):

    # validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training_cutoff + 1)
    batch_size = 128
    # synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
    train_dataloader = train.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )
    val_dataloader = val.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )
    test_dataloader = test.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )

    trainer = pl.Trainer(gpus=1, gradient_clip_val=1e-1)
    net = TemporalFusionTransformer.from_dataset(
        train,
        learning_rate=3e-2,
        hidden_size=16,  # most important hyperparameter apart from learning rate
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=1,
        dropout=0.1,  # between 0.1 and 0.3 are good values
        hidden_continuous_size=8,  # set to <= hidden_size
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        # reduce learning rate if no improvement in validation loss after x epochs
        reduce_on_plateau_patience=4,
    )

    res = trainer.tuner.lr_find(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        min_lr=1e-5,
        max_lr=1e0,
        early_stop_threshold=100,
    )
    print(f"{prefix} suggested learning rate: {res.suggestion()}")

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=30,
        gpus=1,
        weights_summary="top",
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        limit_train_batches=50,
        enable_checkpointing=True,
    )

    net = DeepAR.from_dataset(
        train,
        learning_rate=res.suggestion(),
        log_interval=10,
        log_val_interval=1,
        hidden_size=16,  # most important hyperparameter apart from learning rate
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=1,
        dropout=0.1,  # between 0.1 and 0.3 are good values
        hidden_continuous_size=8,  # set to <= hidden_size
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        # reduce learning rate if no improvement in validation loss after x epochs
        reduce_on_plateau_patience=4,
    )

    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])
    predictions = best_model.predict(test_dataloader)
    print("model_predictions", (actuals - predictions).abs().mean())

    np.save(
        os.path.join(args.checkpoint, f"pred_{prefix}.npy"),
        predictions.detach().numpy(),
        allow_pickle=False,
    )
    np.save(
        os.path.join(args.checkpoint, f"test.npy"),
        actuals.detach().numpy(),
        allow_pickle=False,
    )


    raw_predictions, x = net.predict(test_dataloader, mode="raw", return_x=True, n_samples=100)

    series = validation.x_to_index(x)["gid"]
    for idx in range(20):  # plot 10 examples
        best_model.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
        plt.suptitle(f"Series: {series.iloc[idx]}")
        plt.savefig(os.path.join(args.checkpoint, f"series_sample_{idx}.png"), bbox_inches="tight")
