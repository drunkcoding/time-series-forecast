import time

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from transformers import HfArgumentParser
import pandas as pd
from tqdm import tqdm, trange
import torch.multiprocessing as mp
import os

from models import Crossformer
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
    filename="cross-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = HfArgumentParser(ModelConfig)
    args = parser.parse_args_into_dataclasses()[0]

    model_name = "cross"
    ckpt_name = (
        f"{args.dataset}_outputs_{model_name}_clip[{args.clip}]_norm[{args.norm}].npy"
    )

    model = Crossformer(
        args.n_features,
        in_len=60,
        out_len=1,
        seg_len=10,
        d_model=128,
        d_ff=256,
    )
    model = model.to("cuda:0")
    model = TMLightning(model, args.data, clip=args.clip)

    trainer = Trainer(
        max_epochs=100, callbacks=[early_stop_callback, checkpoint_callback]
    )
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

    # X = torch.cat([X_train, X_test], dim=0)
    # y = torch.cat([y_train, y_test], dim=0)

    # dataloader = create_torch_dataloader(X, y, 64, shuffle=False)

    # model.model = model.model.to("cpu")

    # outputs = []
    # targets = []
    # for X, y in tqdm(dataloader):
    #     yhat = model.model(X).squeeze()
    #     outputs.append(yhat.detach().cpu())
    #     targets.append(y.detach().cpu())

    # outputs = torch.cat(outputs, dim=0).numpy()
    # targets = torch.cat(targets, dim=0).numpy()

    # np.save(
    #     f"training/outputs/{args.dataset}_outputs_cross_clip[{args.clip}].npy",
    #     outputs,
    #     allow_pickle=False,
    # )
    # np.save(
    #     f"training/outputs/{args.dataset}_targets_cross_clip[{args.clip}].npy",
    #     targets,
    #     allow_pickle=False,
    # )
