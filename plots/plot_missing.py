import os
import json
from tkinter import font
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass, field
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from forecast.data.loader import DataParser

from forecast.utils.cmdparser import HfArgumentParser

sns.set_style("whitegrid")
matplotlib.rcParams.update({"font.size": 36})

def plt_scatter(x, y, xlabel, ylabel, filename, log_x=False, log_y=False):
    plt.figure(figsize=(15, 10), dpi=300)
    plt.scatter(x, y, s=48)
    plt.xlabel(xlabel, fontsize=36)
    plt.ylabel(ylabel, fontsize=36)
    if log_x: 
        plt.xscale("log")
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 101, 10))
    if log_y: 
        plt.yscale("log")
        plt.xlim(0, 100)
        plt.xticks(np.arange(0, 101, 10))
    
    plt.savefig(os.path.join(args.checkpoint, filename), bbox_inches="tight")
    plt.close()

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
# df = df.iloc[:int(len(df.index.values) * 0.6), :]
columns = df.columns

dataset = df.values
nan_ind = np.isnan(dataset)
nan_percent = np.sum(nan_ind, axis=0) / len(df.index.values)
data_mean = np.nanmean(dataset, axis=0)
data_std = np.nanstd(dataset, axis=0)

has_nan = (nan_percent > 0) & (nan_percent < 0.95)

nan_percent = nan_percent[has_nan] * 100
data_mean = data_mean[has_nan]
data_std = data_std[has_nan]

print(dataset.shape)
print(data_mean.shape)

plt_scatter(data_mean, nan_percent, "Mean Demand", "NaN %", "mean_nan.png", log_x=True)
plt_scatter(data_std, nan_percent, "Std Demand", "NaN %", "std_nan.png", log_x=True)

col_avg_missing = []
for i in range(nan_ind.shape[1]):
    col_nan_ind = (~nan_ind[:, i]).astype(int).astype(str).tolist()
    col_nan_ind = "".join(col_nan_ind)
    col_nan_ind = col_nan_ind.split("1")
    col_nan_ind = [len(x) for x in col_nan_ind]
    col_nan_ind = [x for x in col_nan_ind if x > 0]
    col_avg_missing.append(np.mean(col_nan_ind))
col_avg_missing = np.array(col_avg_missing)
col_avg_missing = col_avg_missing[has_nan]

# print(col_avg_missing)

plt_scatter(nan_percent, col_avg_missing, "NaN %", "Mean Consecutive Missing", "period_nan.png", log_y=True)

