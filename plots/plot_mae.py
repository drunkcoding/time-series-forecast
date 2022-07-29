import os
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from forecast.utils.cmdparser import HfArgumentParser

sns.set_style("whitegrid")


@dataclass
class ModelConfig:
    checkpoint: str = field(metadata={"help": "path to checkpoints"})

    def __post_init__(self):
        self.col_name_path = os.path.join(self.checkpoint, "columns.json")
        with open(self.col_name_path, "r") as fp:
            self.col_name = json.load(fp)

        self.pred_real_path = os.path.join(self.checkpoint, "pred_real.npy")
        self.pred_oracle_path = os.path.join(self.checkpoint, "pred_oracle.npy")
        self.conf_real_path = os.path.join(self.checkpoint, "conf_real.npy")
        self.conf_oracle_path = os.path.join(self.checkpoint, "conf_oracle.npy")
        self.test_path = os.path.join(self.checkpoint, "test.npy")

        self.save_path = os.path.join(self.checkpoint, "all_ae.png")
        self.fig_raw_path = os.path.join(self.checkpoint, "raw.png")


def reject_outliers(data, m=2.581):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def plot_distribution(data_oracle, data_real, save_path: str, xlabel=None, ylabel=None):

    data_oracle = reject_outliers(data_oracle)
    data_real = reject_outliers(data_real)

    sns.set(font_scale=5)
    sns.set_style("white")
    fig, ax_pdf = plt.subplots(figsize=(20, 15), dpi=300)
    ax_cdf = ax_pdf.twinx()
    sns.distplot(data_oracle, ax=ax_pdf, bins=200, label="Oracle PDF", color="black")
    sns.distplot(
        data_real,
        ax=ax_pdf,
        bins=200,
        label="Real PDF",
        color="blue",
        hist_kws=dict(alpha=0.4),
    )
    sns.ecdfplot(x=data_oracle, ax=ax_cdf, color="red", linewidth=4, label="Oracle CDF")
    sns.ecdfplot(
        x=data_real,
        ax=ax_cdf,
        color="red",
        linestyle="--",
        linewidth=4,
        label="Real CDF",
    )
    ax_pdf.set_xlabel(xlabel)
    ax_pdf.set_ylabel(ylabel)
    ax_cdf.set_ylabel("CDF")

    ax_pdf.legend(loc="upper right")
    ax_cdf.legend(loc="lower right")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_raw(data_oracle, data_real, ref, save_path: str, xlabel=None, ylabel=None):
    pred_oracle, conf = data_oracle
    pred_real, conf = data_real
    x = np.arange(len(pred_real))
    sns.set(font_scale=5)
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
    sns.lineplot(x=x, y=pred_oracle, label="oracle", color="black", linewidth=4)
    sns.lineplot(x=x, y=pred_real, label="pred", color="blue", linewidth=4)
    sns.lineplot(
        x=x, y=ref, label="ref", color="red", linewidth=4, linestyle="--", alpha=0.4
    )
    if conf is not None:
        ax.fill_between(x, conf[:, 0], conf[:, 1], color="blue", alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def absolute_error(pred, ref):
    err = np.abs(pred.squeeze() - ref.squeeze()).flatten()
    # print(np.isnan(err).any())
    return err[~np.isnan(err)]


parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

test_set = np.load(args.test_path, allow_pickle=False)
pred_real = np.load(args.pred_real_path, allow_pickle=False)
pred_oracle = np.load(args.pred_oracle_path, allow_pickle=False)

conf_real = (
    np.load(args.conf_real_path, allow_pickle=False)
    if os.path.exists(args.conf_real_path)
    else np.array([None] * len(pred_real)).reshape((-1, 1))
)
conf_oracle = (
    np.load(args.conf_oracle_path, allow_pickle=False)
    if os.path.exists(args.conf_oracle_path)
    else np.array([None] * len(pred_oracle)).reshape((-1, 1))
)

print(np.isnan(pred_oracle).any())
print(np.isnan(pred_real).any())
print(np.isnan(test_set).any())


def reshape_array(arr: np.ndarray):
    print(arr.shape)
    shape = [i for i in range(len(arr.shape))]
    tmp = shape[0]
    shape[0] = shape[1]
    shape[1] = tmp
    if arr.shape[0] < arr.shape[1]:
        return arr.transpose(shape)

    return arr


test_set = reshape_array(test_set)
pred_real = reshape_array(pred_real)
pred_oracle = reshape_array(pred_oracle)
conf_real = reshape_array(conf_real)
conf_oracle = reshape_array(conf_oracle)

print(test_set.shape)
print(pred_real.shape)
print(pred_oracle.shape)
print(conf_real.shape)
print(conf_oracle.shape)

plot_distribution(
    absolute_error(test_set, pred_oracle),
    absolute_error(test_set, pred_real),
    os.path.join(args.checkpoint, "all_ae.png"),
    "AbsoluteError",
    "PDF",
)

import multiprocessing as mp
from functools import partial


def plot_graph(params):
    i, c = params
    os.sched_setaffinity(0, {i % mp.cpu_count()})
    
    try:
        plot_distribution(
            absolute_error(test_set[:, i], pred_oracle[:, i]),
            absolute_error(test_set[:, i], pred_real[:, i]),
            os.path.join(args.checkpoint, f"{c}_ae.png"),
            "AbsoluteError",
            "PDF",
        )
    except:
        print("======================================================")
    plot_raw(
        (pred_oracle[:, i].flatten(), None),
        (pred_real[:, i].flatten(), None),
        test_set[:, i].flatten(),
        os.path.join(args.checkpoint, f"{c}_raw.png"),
        "Timestamp",
        "DemandValue",
    )

    return 0


if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count())
    # for params in tqdm(enumerate(args.col_name)):
    #     print(params)
    pool.map(
        plot_graph, enumerate(args.col_name),
    )


# for i, c in tqdm(enumerate(args.col_name)):
#     plot_distribution(
#         absolute_error(test_set[:, i], pred_oracle[:, i]),
#         absolute_error(test_set[:, i], pred_real[:, i]),
#         os.path.join(args.checkpoint, f"{c}_ae.png"),
#         "AbsoluteError",
#         "PDF",
#     )
#     plot_raw(
#         (pred_oracle[:, i].flatten(), None),
#         (pred_real[:, i].flatten(), None),
#         test_set[:, i].flatten(),
#         os.path.join(args.checkpoint, f"{c}_raw.png"),
#         "Timestamp",
#         "DemandValue",
#     )


# plot_raw((pred_real[0].flatten(), conf_real[0]), test_set[:, 0].flatten(), args.fig_raw_path, "Timestamp", "DemandValue")

