from .trainer_utils import TMLightning
import numpy as np
import torch
from .cmdline_utils import ModelConfig
from .data_utils import create_torch_dataloader
from tqdm import tqdm
import gc

def save_all_results(model: TMLightning, ckpt_name: str):
    X_train = model.X_train
    y_train = model.y_train
    X_test = model.X_test
    y_test = model.y_test
    train_rolling_mean = model.train_rolling_mean
    train_rolling_std = model.train_rolling_std
    test_rolling_mean = model.test_rolling_mean
    test_rolling_std = model.test_rolling_std

    if model.image:
        X_train = model._reshape2image(X_train, model.image_size)
        y_train = model._reshape2image(y_train, model.image_size)
        X_test = model._reshape2image(X_test, model.image_size)
        y_test = model._reshape2image(y_test, model.image_size)

    X = torch.cat([X_train, X_test], dim=0).cpu()
    y = torch.cat([y_train, y_test], dim=0).cpu()
    rolling_mean = np.concatenate([train_rolling_mean, test_rolling_mean], axis=0)
    rolling_std = np.concatenate([train_rolling_std, test_rolling_std], axis=0)

    dataloader = create_torch_dataloader(X, y, 128, shuffle=False)

    model.model = model.model.to("cuda")
    model.model.eval()

    outputs = []
    targets = []

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X = X.to("cuda")
            if model.image:
                yhat = model.model(X, 1)
            else:
                yhat = model.model(X)
            outputs.append(yhat.detach().cpu().squeeze())
            targets.append(y.detach().cpu().squeeze())
            torch.cuda.empty_cache()

    outputs = torch.cat(outputs, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()

    if model.image:
        outputs = outputs.reshape(outputs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        rolling_mean = rolling_mean.reshape(rolling_mean.shape[0], -1)
        rolling_std = rolling_std.reshape(rolling_std.shape[0], -1)

    outputs[targets == 0] = np.nan
    targets[targets == 0] = np.nan

    outputs = outputs * rolling_std + rolling_mean
    targets = targets * rolling_std + rolling_mean

    outputs[np.isnan(outputs)] = 0
    targets[np.isnan(targets)] = 0

    print(outputs.shape)
    print(targets.shape)

    np.save(f"training/outputs/{ckpt_name}.npy", outputs, allow_pickle=False)
    np.save(f"training/outputs/{ckpt_name}.npy", targets, allow_pickle=False)
