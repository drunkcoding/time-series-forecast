import copy
from dataclasses import dataclass, field
import os
import time

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from transformers import HfArgumentParser
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm, trange
from numpy.lib.stride_tricks import sliding_window_view
import torch.multiprocessing as mp
import os

from models import LSTMModel
from utils import (
    train_test_split,
    create_dataset,
    create_torch_dataloader,
    create_kfold_dataset,
    TMLightning,
    ModelConfig,
    save_all_results,
)


def init_weights(m):
    for name, param in m.named_parameters():
        if "bias" in name:
            nn.init.zeros_(param)
        else:
            nn.init.xavier_uniform_(param)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    filename="lstm-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = HfArgumentParser(ModelConfig)
    args = parser.parse_args_into_dataclasses()[0]

    model_name = "lstm"
    ckpt_name = (
        f"{args.dataset}_outputs_{model_name}_clip[{args.clip}]_norm[{args.norm}].npy"
    )

    model = LSTMModel(args.n_features, 128, 5, args.n_features)
    model = model.to("cuda:0")
    model.apply(init_weights)
    model = TMLightning(model, args.data, clip=args.clip, norm=args.norm)

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
        f"training/model_ckpts/{ckpt_name}",
    )
    trainer.predict(model)

    save_all_results(model, ckpt_name)

exit()

if __name__ == "__main__":
    parser = HfArgumentParser(ModelConfig)
    args = parser.parse_args_into_dataclasses()[0]

    mp.set_start_method("spawn")

    model_log = open(
        "training/model_ckpts/%s_lstm_model_zclip.log" % (args.dataset), "w"
    )

    train_set, test_set, norm_meta = train_test_split(args.data, 0.6, clip=True)

    mean = norm_meta["mean"]
    std = norm_meta["std"]

    X_train, y_train = create_dataset(train_set, 60)
    X_test, y_test = create_dataset(test_set, 60)

    num_features = X_train.shape[-1]
    model = LSTMModel(num_features, 64, 5, num_features)
    model = model.to("cuda:0")
    model.apply(init_weights)
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0001)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)

    X_train_fold, y_train_fold, X_test_fold, y_test_fold = create_kfold_dataset(
        X_train, y_train, 10
    )

    X_test_fold = torch.from_numpy(X_test_fold).float().to("cuda:0")
    y_test_fold = torch.from_numpy(y_test_fold).float().to("cuda:0")

    min_mae = 1e10
    min_mae_model = None
    for epoch in trange(100, position=0, leave=True):
        mae_fold = []
        for fold_idx in range(len(X_train_fold)):
            train_fold_loader = create_torch_dataloader(
                X_train_fold[fold_idx],
                y_train_fold[fold_idx],
                batch_size=128,
                shuffle=True,
            )

            # model.train()
            for X_train_fold_sample, y_train_fold_sample in train_fold_loader:
                outputs = model(X_train_fold_sample)
                loss = criterion(outputs, y_train_fold_sample)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # model.eval()

            with torch.no_grad():
                test_predict = model(X_test_fold[fold_idx])

            test_predict = test_predict.data.clone().detach().cpu().numpy()
            test_y = y_test_fold[fold_idx].data.clone().detach().cpu().numpy()

            # print(test_predict.shape, test_y.shape)

            # print(test_predict.shape, test_y.shape)
            test_mae = np.mean((np.abs(test_predict - test_y) * std + mean)[test_y > 0])
            mae_fold.append(test_mae)
            print("Epoch: %d, loss: %1.5f, MAE: %.5f" % (epoch, loss.item(), test_mae))
            break

        kfold_mae = np.mean(mae_fold)
        # print("Epoch: %d, K Fold mean MAE: %.5f" % (epoch, kfold_mae))

        # model.eval()

        if kfold_mae < min_mae:
            min_mae = kfold_mae
            min_mae_model = copy.deepcopy(model)

        with torch.no_grad():
            test_predict = model(torch.from_numpy(X_test).float().to("cuda:0"))

        test_predict = test_predict.data.clone().detach().cpu().numpy()
        test_y = y_test.copy()

        test_mae = np.mean((np.abs(test_predict - test_y) * std + mean)[test_y > 0])
        print("MAE: %.5f" % test_mae)

        # save model
        model_log.write(
            "Epoch: %d, MAE: %.5f, K-Fold MAE: %.5f\n" % (epoch, test_mae, kfold_mae)
        )
        torch.save(
            min_mae_model.state_dict(),
            "training/model_ckpts/%s_lstm_model_zclip.pth" % args.dataset,
        )
