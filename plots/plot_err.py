from functools import partial
import os
import json
import re
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from forecast.data.loader import DataParser

from forecast.utils.cmdparser import HfArgumentParser

sns.set_style("whitegrid")


@dataclass
class ModelConfig:
    folder: str = field(metadata={"help": "folder for xml data"})
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

        


def reject_outliers(data, m=2.581):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def export_legend(legend, filename="legend.png", expand=[-5, -5, 5, 5]):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def absolute_error(pred, ref):
    err = np.abs(pred.squeeze() - ref.squeeze()).flatten()
    # print(np.isnan(err).any())
    return err[~np.isnan(err)]

def relative_error(pred, ref):
    pred = pred.squeeze()
    ref = ref.squeeze()
    err = (np.abs(pred - ref) / (np.abs(pred) + np.abs(ref)) * 2 ).flatten()
    # print(np.isnan(err).any())
    return err[~np.isnan(err)]

def abs_diff(pred, ref):
    err = (pred.squeeze() - ref.squeeze()).flatten()
    return err[~np.isnan(err)]

parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

parser = DataParser()
df = parser.parse_sndlib_xml(args.folder)
df = df.drop(columns=["timestamps"])
df = df[args.col_name]

# test_set = np.load(args.test_path, allow_pickle=False)
pred_real = np.load(args.pred_real_path, allow_pickle=False)
pred_oracle = np.load(args.pred_oracle_path, allow_pickle=False)

# print(pred_real.shape)

test_set = df.values[-max(pred_real.shape):]
test_indicator = np.isnan(test_set)

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

# print(np.isnan(pred_oracle).any())
# print(np.isnan(pred_real).any())
# print(np.isnan(test_set).any())


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

if "deepar" in args.checkpoint:
    pred_real = pred_real.flatten("F").reshape((-1, test_set.shape[1]))
    pred_oracle = pred_oracle.flatten("F").reshape((-1, test_set.shape[1]))

if "autoformer" in args.checkpoint:
    test_set = np.load(args.test_path, allow_pickle=False)
    test_indicator = np.isclose(test_set, 0)
    test_set[test_indicator] = np.nan
if "baseline" in args.checkpoint:
    pred_real = pred_oracle = df.iloc[-max(pred_real.shape):].shift(1).fillna(0).values
    
# print(test_set.shape)
# print(pred_real.shape)
# print(pred_oracle.shape)
# print(conf_real.shape)
# print(conf_oracle.shape)

idx = test_set > 0

import multiprocessing as mp


def plot_graph(params):
    i, c = params
    os.sched_setaffinity(0, {i % mp.cpu_count()})

    return (
        np.nanmean(absolute_error(pred_oracle[:, i], test_set[:, i])),
        np.nanmean(absolute_error(pred_real[:, i], test_set[:, i],)),
        np.nanmean(relative_error(pred_oracle[:, i], test_set[:, i])),
        np.nanmean(relative_error(pred_real[:, i], test_set[:, i],)),
        np.nanstd(test_set[:, i]),
        np.nanmean(test_set[:, i]),
        np.nanquantile(
            test_set[:, i],
            q=[0.05, 0.95],
            # np.arange(0.1, 1.0, 0.1).tolist() + [0.99, 0.999, 0.9999, 0.99999],
        ),
    )


def plt_scatter(x, y, xlabel, ylabel, filename):
    sns.scatterplot(x=x, y=y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale("log")
    # plt.yscale("log")

    fit_x = np.log(x)
    fit_y = y

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(fit_x, fit_y)

    p1 = (1, slope* 1 + intercept)
    p2 = (10, slope* 10 + intercept)

    def distance(p0, p1,p2):
        return np.abs((p2[0]-p1[0])*(p1[1]-p0[1]) - (p1[0]-p0[0])*(p2[1]-p1[1])) / np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    
    fit_x = list(fit_x)
    fit_y = list(fit_y)
    points = list(zip(fit_x, fit_y))
    distances = [distance(p, p1, p2) for p in points]

    mean_dist = np.mean(distances)
    std_dist = np.std(distances)


    columns = args.col_name
    indicator = np.abs(distances - mean_dist) > 2 * std_dist
    outliers = {
        columns[i]: [distances[i], mean_dist, std_dist] for i in range(len(indicator)) if indicator[i]
    }
    
    name = filename.split("_")[1]
    with open(os.path.join(args.checkpoint, f"outliers_{name}.json"), "w") as f:
        json.dump(outliers, f)

    #         plt.text(x=x[i],
    #             y=y[i],
    #             s=columns[i],
    #             fontsize=12)

    print("r square", args.checkpoint, filename, r_value**2)

    plt.savefig(os.path.join(args.checkpoint, filename), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count())
    # for params in tqdm(enumerate(args.col_name)):
    #     print(params)
    results = pool.map(plot_graph, enumerate(args.col_name),)
    mae_oracle, mae_real, rpe_oracle, rpe_real, std, mean, quantiles = zip(*results)

    quantiles = np.array(quantiles)
    quantiles = quantiles[:, 1] - quantiles[:, 0]

    sns.set(font_scale=2)
    sns.set_style("white")

    plt_scatter(mean, rpe_oracle, "Mean TD", "RPE Oracle", "rpe_oracle_mean.png")
    plt_scatter(std, rpe_oracle, "Std TD", "RPE Oracle", "rpe_oracle_std.png")
    plt_scatter(mean, rpe_real, "Mean TD", "RPE Real", "rpe_real_mean.png")
    plt_scatter(std, rpe_real, "Std TD", "RPE Real", "rpe_real_std.png")
    plt_scatter(quantiles, rpe_real, "Q95-Q05 TD", "RPE Real", "rpe_real_quantile.png")
    plt_scatter(quantiles, rpe_oracle, "Q95-Q05 TD", "RPE Oracle", "rpe_oracle_quantile.png")

    plt_scatter(mean, mae_oracle, "Mean TD", "MAE Oracle", "mae_oracle_mean.png")
    plt_scatter(std, mae_oracle, "Std TD", "MAE Oracle", "mae_oracle_std.png")
    plt_scatter(mean, mae_real, "Mean TD", "MAE Real", "mae_real_mean.png")
    plt_scatter(std, mae_real, "Std TD", "MAE Real", "mae_real_std.png")
    plt_scatter(quantiles, mae_real, "Q95-Q05 TD", "MAE Real", "mae_real_quantile.png")
    plt_scatter(quantiles, mae_oracle, "Q95-Q05 TD", "MAE Oracle", "mae_oracle_quantile.png")

