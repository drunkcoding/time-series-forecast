from dataclasses import dataclass, field
import json
from multiprocessing import freeze_support
from pyexpat import model
import numpy as np
import os
import warnings
import scipy.stats as stats
from tqdm import tqdm
from forecast.data.imputer import train_test_split

from forecast.data.loader import DataParser

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl

import torch

from pytorch_forecasting import TimeSeriesDataSet, RecurrentNetwork
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, QuantileLoss, MultivariateNormalDistributionLoss
from pytorch_lightning import loggers as pl_loggers
from forecast.data.prophet import ProphetDataParser
from forecast.utils.cmdparser import HfArgumentParser

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


@dataclass
class ModelConfig:
    folder: str = field(metadata={"help": "folder for xml data"})
    # checkpoint: str = field(metadata={"help": "path to checkpoints"})
    layer: str = field(metadata={"help": "gru, lstm, transformer, deepar"})
    debug: bool = field(default=False)

    def __post_init__(self):
        if "abilene" in self.folder:
            self.dataset = "abilene"
        elif "geant" in self.folder:
            self.dataset = "geant"
        else:
            raise ValueError("dataset not supported")
        self.checkpoint = os.path.join("checkpoints", self.dataset, self.layer)
        try:
            os.mkdir(self.checkpoint)
        except:
            pass


parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

parser = ProphetDataParser()
df_dict = parser.format_sndlib_xml(args.folder)
columns = list(df_dict.keys())
del_keys = []
for idx, key in enumerate(columns):
    df_dict[key]["gid"] = str(idx)
    df_dict[key]["time_idx"] = np.arange(len(df_dict[key].index.values))
    df_dict[key]["missing"] = np.where(np.isnan(df_dict[key]["y"].values), 0, 1).astype(
        str
    )

    y_diff = np.abs(np.diff(df_dict[key]["y"].values.flatten()))
    print(key, stats.describe(y_diff, nan_policy="omit"))
    if np.nanmax(y_diff) < 1:
        del_keys.append(key)
        continue

    if args.debug:
        df_dict[key] = df_dict[key].iloc[:1000]

print(len(columns))
print(len(del_keys))

exit()

df = pd.concat(df_dict.values(), ignore_index=True)
df = df.fillna(0)

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

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
import torch
from tqdm import tqdm


def create_pl_trainer(model_name, distributed=False):
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-3, patience=8, verbose=False, mode="min"
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"{model_name}_logs/")
    
    if distributed:
        trainer = pl.Trainer(
            logger=tb_logger,
            max_epochs=1 if args.debug else 50,
            accelerator="gpu",
            devices=-1,
            # weights_summary="top",
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, lr_monitor_callback],
            # limit_train_batches=50,
            enable_progress_bar=True,
            enable_checkpointing=True,
            replace_sampler_ddp=True,
            strategy=DDPStrategy(find_unused_parameters=False),
        )
    else:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            gradient_clip_val=0.1,
        )

    return trainer


def create_dataloaders(train, val, test):
    batch_size = 512
    train_dataloader = train.to_dataloader(
        train=True,
        batch_size=batch_size * 10,
        num_workers=torch.cuda.device_count(),
        pin_memory=False,
    )
    val_dataloader = val.to_dataloader(
        train=False,
        batch_size=batch_size * 10,
        num_workers=torch.cuda.device_count(),
        pin_memory=False,
    )
    test_dataloader = test.to_dataloader(
        train=False,
        batch_size=batch_size * 10,
        num_workers=torch.cuda.device_count(),
        pin_memory=True,
        batch_sampler="synchronized",
    )

    return train_dataloader, val_dataloader, test_dataloader


def get_best_model(trainer, net):
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = net.load_from_checkpoint(best_model_path, map_location="cuda")
    best_model.eval()
    best_model.freeze()
    best_model = best_model.to("cuda")
    return best_model


def model_predict(model, test_dataloader, columns):
    decoder_cat = []
    actuals = []

    # count = 0
    for x, y in tqdm(iter(test_dataloader)):
        decoder_cat.append(x["decoder_cat"])
        actuals.append(y[0])
    decoder_cat = torch.stack(decoder_cat).squeeze()
    actuals = torch.stack(actuals).squeeze()

    predictions = model.predict(test_dataloader, show_progress_bar=True).squeeze()

    predictions = predictions.reshape(-1, len(columns))

    actuals = actuals.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    decoder_cat = decoder_cat[..., 1].detach().cpu().numpy().astype(bool).squeeze()

    return actuals, predictions, decoder_cat


def get_model_class(model_name):
    if model_name == "DeepAR".lower():
        model_class = ptf.DeepAR
    elif model_name == "Transformer".lower():
        model_class = ptf.TemporalFusionTransformer
    elif model_name == "GRU".lower():
        model_class = ptf.RecurrentNetwork
    elif model_name == "LSTM".lower():
        model_class = ptf.RecurrentNetwork
    else:
        raise ValueError("Model name not found")

    return model_class


def get_model_parameters(model_name):
    if model_name == "DeepAR".lower():
        model_parameters = dict(
            rnn_layers=4, loss=MultivariateNormalDistributionLoss(rank=30),
        )
    elif model_name == "Transformer".lower():
        model_parameters = dict(
            attention_head_size=4,
            dropout=0.1,  # between 0.1 and 0.3 are good values
            hidden_continuous_size=128,  # set to <= hidden_size
            output_size=7,  # 7 quantiles by default
            loss=QuantileLoss(),
        )
    elif model_name == "GRU".lower():
        model_parameters = dict(cell_type="GRU", rnn_layers=10, dropout=0.1, output_size=1, loss=MAE(),)
    elif model_name == "LSTM".lower():
        model_parameters = dict(cell_type="LSTM", rnn_layers=10, dropout=0.1, output_size=1, loss=MAE(),)
    else:
        raise ValueError("Model name not found")

    return model_parameters

def create_model(model_class, train, learning_rate=1e-3):
    net = model_class.from_dataset(
        train,
        learning_rate=learning_rate,
        log_interval=10,
        log_val_interval=1,
        hidden_size=256,
        **get_model_parameters(args.layer.lower()),
        # reduce learning rate if no improvement in validation loss after x epochs
        reduce_on_plateau_patience=4,
    )
    return net

def tune_and_train(train, val, test, prefix):

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train, val, test
    )
    
    model_class = get_model_class(args.layer.lower())

    # trainer = create_pl_trainer("_".join([args.dataset, args.layer]), distributed=False)
    # net = create_model(model_class, train)

    # res = trainer.tuner.lr_find(
    #     net,
    #     train_dataloaders=train_dataloader,
    #     val_dataloaders=val_dataloader,
    #     min_lr=1e-5,
    #     max_lr=1e0,
    #     early_stop_threshold=100,
    # )
    # print(f"{prefix} suggested learning rate: {res.suggestion()}")

    net = create_model(model_class, train, learning_rate=1e-3)
    trainer = create_pl_trainer("_".join([args.dataset, args.layer]), distributed=True)
    trainer.fit(
        net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )

    best_model = get_best_model(trainer, model_class)

    actuals, predictions, decoder_cat = model_predict(
        best_model, test_dataloader, columns
    )

    with open(os.path.join(args.checkpoint, "columns.json"), "w") as fp:
        json.dump(columns, fp)
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
