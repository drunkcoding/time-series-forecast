import copy
from dataclasses import dataclass, field
import os
import time
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
import torch.multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
from transformers import HfArgumentParser
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm, trange
from numpy.lib.stride_tricks import sliding_window_view
import os
from sklearn.preprocessing import MinMaxScaler
import torch.distributed as dist
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from models import EncoderDecoderConvLSTM
from utils import TMLightning, ModelConfig, create_torch_dataloader, save_all_results

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=10,
    verbose=True,
    mode="min",
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints/",
    filename="convlstm-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = HfArgumentParser(ModelConfig)
    args = parser.parse_args_into_dataclasses()[0]

    model_name = "convlstm"
    ckpt_name = (
        f"{args.dataset}_outputs_{model_name}_clip[{args.clip}]_norm[{args.norm}].npy"
    )

    model = EncoderDecoderConvLSTM(128, 1)
    model = model.to("cuda:0")
    model = TMLightning(model, args.data, clip=args.clip, image=True, norm=args.norm)

    trainer = Trainer(
        max_epochs=100, callbacks=[early_stop_callback, checkpoint_callback]
    )
    torch.cuda.empty_cache()
    train_start = time.time()
    trainer.fit(model)
    train_end = time.time()
    print("Training time: %d" % (train_end - train_start))

    best_mdoel_path = checkpoint_callback.best_model_path
    print(best_mdoel_path)
    model.load_state_dict(torch.load(best_mdoel_path)["state_dict"])
    torch.save(
        model.state_dict(),
        f"training/model_ckpts/{ckpt_name}.pth",
    )
    trainer.predict(model)

    save_all_results(model, ckpt_name)

    # X_train = model.X_train
    # y_train = model.y_train
    # X_test = model.X_test
    # y_test = model.y_test

    # X_train = model._reshape2image(X_train, model.image_size)
    # y_train = model._reshape2image(y_train, model.image_size)

    # X = torch.cat([X_train, X_test], dim=0)
    # y = torch.cat([y_train, y_test], dim=0)

    # dataloader = create_torch_dataloader(X, y, 64, shuffle=False)

    # model.model = model.model.to("cpu")

    # outputs = []
    # targets = []
    # for X, y in tqdm(dataloader):
    #     yhat = model.model(X, 1)
    #     outputs.append(yhat.detach().cpu())
    #     targets.append(y.detach().cpu())
    #     torch.cuda.empty_cache()

    # outputs = torch.cat(outputs, dim=0).numpy().squeeze()
    # targets = torch.cat(targets, dim=0).numpy().squeeze()

    # outputs = outputs.reshape(outputs.shape[0], -1)
    # targets = targets.reshape(targets.shape[0], -1)

    # np.save(
    #     f"training/outputs/{args.dataset}_outputs_convlstm_clip[{args.clip}].npy",
    #     outputs,
    #     allow_pickle=False,
    # )
    # np.save(
    #     f"training/outputs/{args.dataset}_targets_convlstm_clip[{args.clip}].npy",
    #     targets,
    #     allow_pickle=False,
    # )
