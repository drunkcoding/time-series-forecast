import os
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from forecast.utils.cmdparser import HfArgumentParser

sns.set_style("whitegrid")

@dataclass
class ModelConfig:
    checkpoint: str = field(metadata={"help": "path to checkpoints"})

    def __post_init__(self):
        self.col_name_path = os.path.join(self.checkpoint, "columns.json")
        with open(self.col_name_path, 'r') as fp:
            self.col_name = json.load(fp)

        self.pred_real_path = os.path.join(self.checkpoint, "pred_real.npy")
        self.pred_oracle_path = os.path.join(self.checkpoint, "pred_oracle.npy")
        self.conf_real_path = os.path.join(self.checkpoint, "conf_real.npy")
        self.conf_oracle_path = os.path.join(self.checkpoint, "conf_oracle.npy")
        self.test_path = os.path.join(self.checkpoint, "test.npy")

        self.save_path = os.path.join(self.checkpoint, "test.png")
        self.fig_raw_path = os.path.join(self.checkpoint, "raw.png")

def plot_distribution(x, save_path: str, xlabel=None, ylabel=None):
    sns.set(font_scale=5)
    sns.set_style("white")
    fig, ax_pdf = plt.subplots(figsize=(20, 15), dpi=300)
    ax_cdf = ax_pdf.twinx()
    sns.distplot(x, ax=ax_pdf, bins=200, label="PDF")
    sns.ecdfplot(x=x, ax=ax_cdf, color="red", label="CDF")
    ax_pdf.set_xlabel(xlabel)
    ax_pdf.set_ylabel(ylabel)
    ax_cdf.set_ylabel("CDF")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_raw(data, ref, save_path: str, xlabel=None, ylabel=None):
    pred, conf = data
    x = np.arange(len(pred))
    sns.set(font_scale=5)
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
    sns.lineplot(x=x, y=pred, label="pred", color="black", linewidth=5)
    sns.lineplot(x=x, y=ref, label="ref", color="red", linewidth=5, linestyle="--")
    ax.fill_between(x, conf[:, 0], conf[:, 1], color="blue", alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def absolute_error(pred, ref):
    return np.abs(pred - ref).flatten()

parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

test_set = np.load(args.test_path, allow_pickle=False)
pred_real = np.load(args.pred_real_path, allow_pickle=False)
pred_oracle = np.load(args.pred_oracle_path, allow_pickle=False)
conf_real = np.load(args.conf_real_path, allow_pickle=False)
conf_oracle = np.load(args.conf_oracle_path, allow_pickle=False)

print(test_set.shape)
print(pred_real.shape)
print(pred_oracle.shape)
print(conf_real.shape)
print(conf_oracle.shape)

plot_distribution(absolute_error(test_set[:, 0].flatten(), pred_real[0].flatten()), args.save_path, "AbsoluteError", "PDF")
# plot_raw((pred_real[0].flatten(), conf_real[0]), test_set[:, 0].flatten(), args.fig_raw_path, "Timestamp", "DemandValue")
plot_raw((pred_oracle[0].flatten(), conf_oracle[0]), test_set[:, 0].flatten(), args.fig_raw_path, "Timestamp", "DemandValue")
