import numpy as np
import torch
import pytorch_lightning as pl
import pandas as pd

from .data_utils import (
    create_dataset,
    read_dataset,
    create_kfold_dataset,
    create_torch_dataloader,
)
from sklearn.model_selection import train_test_split


class TMLightning(pl.LightningModule):
    def __init__(
        self, model, filepath: str, train_split: float = 0.6, clip=True, image=False, online=False, norm="roll"
    ):
        super(TMLightning, self).__init__()

        assert norm in ["roll", "glob", "indv"]

        # default config
        self.normalize = False
        self.online = online
        self.model = model

        # Training config
        self.criterion = torch.nn.MSELoss()
        self.batch_size = 64
        self.n_steps_past = 60
        self.n_steps_ahead = 1
        self.n_fold = 10
        self.image = image

        dataset = read_dataset(filepath)
        self.train_set, self.test_set = train_set, test_set = train_test_split(dataset, test_size=0.4, shuffle=False)

        if norm == "roll":
            train_rolling = train_set.rolling(window=self.n_steps_past, closed="left")
            test_rolling = test_set.rolling(window=self.n_steps_past, closed="left")        
            train_rolling_mean = train_rolling.apply(
                lambda x: np.nanmean(x), engine="numba", raw=True
            ).fillna(0)
            train_rolling_std = train_rolling.apply(
                lambda x: np.nanstd(x), engine="numba", raw=True
            ).fillna(1)
            test_rolling_mean = test_rolling.apply(
                lambda x: np.nanmean(x), engine="numba", raw=True
            ).fillna(0)
            test_rolling_std = test_rolling.apply(
                lambda x: np.nanstd(x), engine="numba", raw=True
            ).fillna(1)

            self.train_rolling_mean = train_rolling_mean = train_rolling_mean.iloc[
                self.n_steps_past :
            ].to_numpy()
            self.train_rolling_std = train_rolling_std = train_rolling_std.iloc[
                self.n_steps_past :
            ].to_numpy()
            self.test_rolling_mean = test_rolling_mean = test_rolling_mean.iloc[
                self.n_steps_past :
            ].to_numpy()
            self.test_rolling_std = test_rolling_std = test_rolling_std.iloc[
                self.n_steps_past :
            ].to_numpy()

            print(train_rolling_mean.shape, train_rolling_mean.shape, train_set.shape)
            print(test_rolling_mean.shape, test_rolling_mean.shape, test_set.shape)
            # print(rolling_mean.head(), rolling_std.head(), train_set.head())
            # exit()

        X_train, y_train = create_dataset(train_set.to_numpy(), self.n_steps_past)
        X_test, y_test = create_dataset(test_set.to_numpy(), self.n_steps_past)

        if norm == "glob":
            mean = np.nanmean(train_set.to_numpy())
            std = np.nanstd(train_set.to_numpy())
            self.train_rolling_mean = train_rolling_mean = np.ones_like(y_train) * mean
            self.train_rolling_std = train_rolling_std = np.ones_like(y_train) * std
            self.test_rolling_mean = test_rolling_mean = np.ones_like(y_test) * mean
            self.test_rolling_std = test_rolling_std = np.ones_like(y_test) * std

        if norm == "indv":
            mean = np.nanmean(train_set.to_numpy(), axis=0)
            std = np.nanstd(train_set.to_numpy(), axis=0)
            self.train_rolling_mean = train_rolling_mean = np.ones_like(y_train) * mean
            self.train_rolling_std =  train_rolling_std = np.ones_like(y_train) * std
            self.test_rolling_mean = test_rolling_mean = np.ones_like(y_test) * mean
            self.test_rolling_std =  test_rolling_std = np.ones_like(y_test) * std

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert train_rolling_mean.shape == y_train.shape
        assert test_rolling_mean.shape == y_test.shape

        train_rolling_mean_extend = train_rolling_mean[:, None, :]
        train_rolling_std_extend = train_rolling_std[:, None, :]
        test_rolling_mean_extend = test_rolling_mean[:, None, :]
        test_rolling_std_extend = test_rolling_std[:, None, :]
        # use rolling mean to clip the data
        if clip:
            z_score = 2.326
            train_z_score_threshold = train_rolling_mean + z_score * train_rolling_std
            test_z_score_threshold = test_rolling_mean + z_score * test_rolling_std
            train_z_score_threshold_extend = train_z_score_threshold[:, None, :]
            test_z_score_threshold_extend = test_z_score_threshold[:, None, :]
            # X_train = np.clip(X_train, np.nan, train_z_score_threshold)
            X_train[X_train > train_z_score_threshold_extend] = np.nan
            y_train[y_train > train_z_score_threshold] = np.nan
            X_test[X_test > test_z_score_threshold_extend] = np.nan
            y_test[y_test > test_z_score_threshold] = np.nan

            # # undo sliding window for X_train and y_train
            # train_set_clipped = np.concatenate(
            #     [X_train[: self.n_steps_past, 0, :], y_train]
            # )
            # test_set_clipped = np.concatenate(
            #     [X_test[: self.n_steps_past, 0, :], y_test]
            # )

            # # need to recompute rolling mean and std
            # train_rolling = pd.DataFrame(train_set_clipped).rolling(
            #     window=self.n_steps_past, closed="left"
            # )
            # test_rolling = pd.DataFrame(test_set_clipped).rolling(
            #     window=self.n_steps_past, closed="left"
            # )
            # train_rolling_mean = train_rolling.apply(
            #     lambda x: np.nanmean(x), engine="numba", raw=True
            # ).fillna(0)
            # train_rolling_std = train_rolling.apply(
            #     lambda x: np.nanstd(x), engine="numba", raw=True
            # ).fillna(1)
            # test_rolling_mean = test_rolling.apply(
            #     lambda x: np.nanmean(x), engine="numba", raw=True
            # ).fillna(0)
            # test_rolling_std = test_rolling.apply(
            #     lambda x: np.nanstd(x), engine="numba", raw=True
            # ).fillna(1)

            # self.train_rolling_mean = train_rolling_mean = train_rolling_mean.iloc[
            #     self.n_steps_past :
            # ].to_numpy()
            # self.train_rolling_std = train_rolling_std = train_rolling_std.iloc[
            #     self.n_steps_past :
            # ].to_numpy()
            # self.test_rolling_mean = test_rolling_mean = test_rolling_mean.iloc[
            #     self.n_steps_past :
            # ].to_numpy()
            # self.test_rolling_std = test_rolling_std = test_rolling_std.iloc[
            #     self.n_steps_past :
            # ].to_numpy()

            # train_rolling_mean_extend = train_rolling_mean[:, None, :]
            # train_rolling_std_extend = train_rolling_std[:, None, :]
            # test_rolling_mean_extend = test_rolling_mean[:, None, :]
            # test_rolling_std_extend = test_rolling_std[:, None, :]

        X_train = (X_train - train_rolling_mean_extend) / train_rolling_std_extend
        y_train = (y_train - train_rolling_mean) / train_rolling_std
        X_test = (X_test - test_rolling_mean_extend) / test_rolling_std_extend
        y_test = (y_test - test_rolling_mean) / test_rolling_std
        X_train[np.isnan(X_train)] = 0
        y_train[np.isnan(y_train)] = 0
        X_test[np.isnan(X_test)] = 0
        y_test[np.isnan(y_test)] = 0

        self.X_train = X_train
        self.y_train = y_train

        if image:
            self.image_size = image_size = int(np.sqrt(X_train.shape[-1]))
            X_train = self._reshape2image(X_train, image_size)
            X_test = self._reshape2image(X_test, image_size)
            y_train = self._reshape2image(y_train, image_size)
            y_test = self._reshape2image(y_test, image_size)

            self.train_rolling_mean = self._reshape2image(
                train_rolling_mean, image_size
            )
            self.train_rolling_std = self._reshape2image(train_rolling_std, image_size)
            self.test_rolling_mean = self._reshape2image(test_rolling_mean, image_size)
            self.test_rolling_std = self._reshape2image(test_rolling_std, image_size)

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        X_train_fold, y_train_fold, X_test_fold, y_test_fold = create_kfold_dataset(
            X_train, y_train, self.n_fold
        )
        self.X_train_fold = self._convert2tensor(X_train_fold)
        self.y_train_fold = self._convert2tensor(y_train_fold)
        self.X_test_fold = self._convert2tensor(X_test_fold)
        self.y_test_fold = self._convert2tensor(y_test_fold)
        self.X_train = self._convert2tensor(self.X_train)
        self.y_train = self._convert2tensor(self.y_train)
        self.X_test = self._convert2tensor(X_test)
        self.y_test = self._convert2tensor(y_test)

        self.fold_idx = 5

        return
        # train_set, test_set, norm_meta = train_test_split(filepath, train_split, clip)

        # self.mean = norm_meta["mean"]
        # self.std = norm_meta["std"]

        self.min = norm_meta["min"]
        self.max = norm_meta["max"]

        X_train, y_train = create_dataset(train_set, self.n_steps_past)
        X_test, y_test = create_dataset(test_set, self.n_steps_past)

        # # rolling mean and std
        # self.X_train_mean = X_train_mean = np.mean(X_train, axis=1)
        # self.X_train_std = X_train_std = np.std(X_train, axis=1)
        # self.X_test_mean = X_test_mean = np.mean(X_test, axis=1)
        # self.X_test_std = X_test_std = np.std(X_test, axis=1)

        # X_train_mean[np.isnan(X_train_mean)] = 0
        # X_test_mean[np.isnan(X_test_mean)] = 1
        # X_train_std[X_train_std == 0] = 1
        # X_test_std[X_test_std == 0] = 1

        # X_train = (X_train - X_train_mean[:, None, :]) / X_train_std[:, None, :]
        # X_test = (X_test - X_test_mean[:, None, :]) / X_test_std[:, None, :]
        # y_train = (y_train - X_train_mean) / X_train_std
        # y_test = (y_test - X_test_mean) / X_test_std

        # convert last dimention d to (1, sqrt(d), sqrt(d))
        if image:
            image_size = int(np.sqrt(X_train.shape[-1]))
            X_train = np.reshape(
                X_train, (X_train.shape[0], X_train.shape[1], 1, image_size, image_size)
            )
            X_test = np.reshape(
                X_test, (X_test.shape[0], X_test.shape[1], 1, image_size, image_size)
            )

            y_train = np.reshape(
                y_train, (y_train.shape[0], 1, 1, image_size, image_size)
            )
            y_test = np.reshape(y_test, (y_test.shape[0], 1, 1, image_size, image_size))

            self.mean = np.reshape(self.mean, (1, 1, image_size, image_size))
            self.std = np.reshape(self.std, (1, 1, image_size, image_size))

        X_train_fold, y_train_fold, X_test_fold, y_test_fold = create_kfold_dataset(
            X_train, y_train, self.n_fold
        )
        self.X_train_fold = torch.from_numpy(X_train_fold).float().to("cuda")
        self.y_train_fold = torch.from_numpy(y_train_fold).float().to("cuda")
        self.X_test_fold = torch.from_numpy(X_test_fold).float().to("cuda")
        self.y_test_fold = torch.from_numpy(y_test_fold).float().to("cuda")
        self.X_test = torch.from_numpy(X_test).float().to("cuda")
        self.y_test = torch.from_numpy(y_test).float().to("cuda")

        self.fold_idx = 5

        self.val_outputs = []

    def _convert2tensor(self, x):
        return torch.from_numpy(x).float().pin_memory()

    def _reshape2image(self, x, image_size):
        if len(x.shape) > 2:
            return x.reshape((x.shape[0], x.shape[1], 1, image_size, image_size))
        else:
            return x.reshape((x.shape[0], 1, 1, image_size, image_size))

    def forward(self, x, y):
        x = x.to("cuda")
        y = y.to("cuda")
        if self.image:
            output = self.model(x, self.n_steps_ahead)
        elif self.online:
            output = self.model(x, y)
        else:
            output = self.model(x)

        if not self.image:
            output = output.squeeze(1)
        torch.cuda.empty_cache()
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x, y)

        loss = self.mse_loss(y_hat, y)

        # save learning_rate
        lr_saved = self.trainer.optimizers[0].param_groups[-1]["lr"]
        lr_saved = torch.scalar_tensor(lr_saved).cuda()

        tensorboard_logs = {"train_mse_loss": loss, "learning_rate": lr_saved}

        return {"loss": loss, "log": tensorboard_logs}

    def mse_loss(self, y_hat: torch.Tensor, y: torch.Tensor):
        # print(y_hat.shape, y.shape)
        return torch.mean((y_hat - y)[y != 0] ** 2)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x, y)
        loss = self.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return {"val_loss": loss, "log": {"val_loss": loss}}

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     val_loss_mean = sum([o["val_loss"] for o in outputs]) / len(outputs)
    #     # show val_acc in progress bar but only log val_loss
    #     results = {
    #         "progress_bar": {"val_loss": val_loss_mean.item()},
    #         "log": {"val_loss": val_loss_mean.item()},
    #         "val_loss": val_loss_mean.item(),
    #     }
    #     return results

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x, y)
        self.predictions.append(y_hat.detach().cpu())
        self.targets.append(y.detach().cpu())
        return {"test_loss": self.mse_loss(y_hat, y)}

    def on_predict_start(self) -> None:
        self.predictions = []
        self.targets = []

    def on_predict_end(self) -> None:
        self.predictions = torch.cat(self.predictions)
        self.targets = torch.cat(self.targets)

        if self.image:
            self.predictions = self.predictions.reshape((self.predictions.shape[0], -1))
            self.targets = self.targets.reshape((self.targets.shape[0], -1))
            test_rolling_std = self.test_rolling_std.reshape(
                (self.test_rolling_std.shape[0], -1)
            )
        else:
            test_rolling_std = self.test_rolling_std

        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(25, 10))
        # plt.plot(self.predictions[:, 0], label="predictions")
        # plt.plot(self.targets[:, 0], label="targets")
        # plt.legend()
        # plt.savefig("predictions.png")

        # print(self.predictions.shape)
        # print(self.targets.shape)

        # print(self.std.shape)
        # print(self.mean.shape)

        # compute MAE
        # mae = np.mean(
        #     np.abs((self.predictions - self.targets) * self.std)[self.targets != 0]
        # )
        # mae = np.mean(
        #     np.abs((self.predictions - self.targets) * self.X_test_std)[
        #         self.targets != 0
        #     ]
        # )
        # mae = np.nanmean(
        #     np.abs((self.predictions - self.targets) * (self.max - self.min))[
        #         self.targets != 0
        #     ]
        # )
        print(self.predictions.shape, self.targets.shape, test_rolling_std.shape)
        test_rolling_std = torch.from_numpy(test_rolling_std).float()
        mae = torch.nanmean(
            torch.abs((self.predictions - self.targets) * test_rolling_std)[
                self.targets != 0
            ]
        )
        print("test_mae", mae)

    # def test_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    #     tensorboard_logs = {"test_loss": avg_loss}
    #     return {"avg_test_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.98))

    def train_dataloader(self):
        print(
            "Fold: ",
            self.fold_idx,
            " of ",
            self.n_fold,
            " folds",
            "Shape:",
            self.X_train_fold[self.fold_idx].shape,
        )
        train_fold_loader = create_torch_dataloader(
            self.X_train_fold[self.fold_idx],
            self.y_train_fold[self.fold_idx],
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.fold_idx += 1

        return train_fold_loader

    def val_dataloader(self):
        val_fold_loader = create_torch_dataloader(
            self.X_test_fold[self.fold_idx],
            self.y_test_fold[self.fold_idx],
            batch_size=self.batch_size,
            shuffle=False,
        )

        return val_fold_loader

    def predict_dataloader(self):
        test_loader = create_torch_dataloader(
            self.X_test, self.y_test, batch_size=self.batch_size, shuffle=False
        )
        return test_loader
