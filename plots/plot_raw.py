

from cProfile import label
from dataclasses import dataclass, field
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from forecast.data.loader import DataParser

matplotlib.rcParams.update({'font.size': 36})

from forecast.utils.cmdparser import HfArgumentParser

@dataclass
class ModelConfig:
    folder: str = field(metadata={"help": "folder for xml data"})

    def __post_init__(self):
        self.checkpoint = os.path.join("plots", os.path.basename(self.folder))
        try:
            os.mkdir(self.checkpoint)
        except:
            pass


parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

parser = DataParser()
df = parser.parse_sndlib_xml(args.folder)
df = df.drop(columns=["timestamps"])
data = df.values

def plot_cdf(data, save_path: str, xlabel=None, log_x=False, log_y=False):
    base_quantiles = np.linspace(0, 1, 100, endpoint=True)
    quantiles = np.quantile(data, base_quantiles)
    plt.figure(figsize=(20, 15), dpi=300)
    plt.plot(quantiles, base_quantiles, linewidth=4)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    if log_x:
        plt.xscale("log")
    # plt.yscale("log")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_distribution(data, save_path: str, xlabel=None, ylabel=None):
    sns.set(font_scale=5)
    sns.set_style("white")
    fig, ax_pdf = plt.subplots(figsize=(20, 15), dpi=300)
    ax_cdf = ax_pdf.twinx()
    sns.histplot(data, ax=ax_pdf, color="blue", alpha = 0.5, label="Histogram")
    sns.ecdfplot(x=data, ax=ax_cdf, color="red", linewidth=4, label="CDF")
    ax_pdf.set_xlabel(xlabel)
    ax_pdf.set_ylabel(ylabel)
    ax_cdf.set_ylabel("CDF")

    ax_pdf.set_xscale("log")
    ax_cdf.set_xscale("log")

    # ax_pdf.set_xlim(-1e3, 1e3)
    # ax_cdf.set_xlim(-1e3, 1e3)

    # ax_pdf.set_ylim(1e-1, 1e5)

    plt.xticks(rotation=30)

    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

nan_ind = np.isnan(data)
data = data[~nan_ind]

# data = data.flatten() if "abliene" in args.checkpoint else data.flatten() / 1024 ** 3 * 8
print(data.shape, nan_ind.shape)
# exit()
plot_cdf(data.flatten(), os.path.join(args.checkpoint, "raw_value.png"), "Demand", log_x=True)
plot_cdf(np.sum(nan_ind, axis=0) / len(df.index.values), os.path.join(args.checkpoint, "raw_missing.png"), "Percentage of Missing Values")
