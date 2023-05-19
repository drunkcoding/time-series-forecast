from dataclasses import dataclass, field
import os
import json
import pandas as pd
from transformers import HfArgumentParser
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple

mpl.rcParams["font.size"] = 48
cmap = mpl.colormaps["cividis"]


@dataclass
class Arguments:
    input: str = field(metadata={"help": "csv file with original data"})
    models: str = field(metadata={"help": "(model,m/u):..."})
    remove_outliers: bool = field(default=False, metadata={"help": "outlier remove"})

    def __post_init__(self):
        self.output_folder = "plots"
        self.dataset = os.path.basename(self.input).split("-")[0].lower()
        self.models = self.models.split(":")
        print(self.models)
        self.models = list(map(lambda x: make_tuple(x.strip()), self.models))


COLOR = {
    "naive": "black",
    "arima": "orange",
    "prophet": "orange",
    "kalman": "orange",
    "rnn": "red",
    "deepar": "orange",
    "tcn": "red",
    "tft": "orange",
    "tf": "orange",
    "convlstm": "black",
}

LINE = {
    "naive": "--",
    "arima": "--",
    "prophet": "-",
    "kalman": "-",
    "rnn": "--",
    "deepar": "--",
    "tcn": "-",
    "tft": "-",
    "tf": "-.",
    "convlstm": "-",
}


def export_legend(legend, filename="legend.png", expand=[-5, -5, 5, 5]):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def df_cleanup(df: pd.DataFrame):
    cols = df.columns
    cols.drop("time")
    num_rows = df.shape[0]
    df = df.dropna(how="all", subset=cols)
    # df["time"] = pd.date_range(df["time"][0], periods=df.shape[0], freq=args.freq)
    return df

if __name__ == "__main__":
    parser = HfArgumentParser((Arguments))
    args = parser.parse_args_into_dataclasses()[0]
    print(args)

    df = pd.read_csv(args.input)
    df = df_cleanup(df)

    data_columns = list(df.columns.values)
    data_columns.remove("time")
    data = df[data_columns].values

    if args.remove_outliers:
        data = np.clip(
            data, 0.0, np.percentile(data.flatten(), 99)
        )  # we use 99% as the threshold

    test_length = int(df.shape[0] * 0.2)
    test = data[-test_length:]

    plt.figure(figsize=(20, 15), dpi=300)
    for model in args.models:

        variate = "univariate" if model[1] == "u" else "multivariate"
        folder_name = os.path.join("outputs", f"{args.dataset}-{model[0]}-{variate}")
        model_name = model[0]
        print(folder_name)

        if model_name == "naive":
            predictions = test[:-1, ...]
            targets = test[1:, ...]
        else:
            with open(os.path.join(folder_name, "columns.json"), "r") as fp:
                col_name = json.load(fp)

            predictions = np.load(os.path.join(folder_name, "predictions.npy")).squeeze()
            # targets = np.load(os.path.join(folder_name, "targets.npy")).squeeze()
            targets = test.reshape(predictions.shape)

        diff = (predictions - targets) / 1000
        diff = diff[targets > 0].flatten()
        print(model, np.nanmean(np.abs(diff)))
        ax = sns.ecdfplot(
            x=diff,
            label=model_name,
            linewidth=6,
            linestyle=LINE[model_name],
            color=COLOR[model_name],
        )
        # ax.set(xticklabels=[]) 
        # ax.legend_.remove()

    axbox = ax.get_position()
    legend = plt.legend(
        bbox_to_anchor=[0, axbox.y0 + 0.3, 1, 1], loc="upper right", ncol=5, fontsize=48
    )
    export_legend(legend, filename="plots/legend_mae.png")

    legend.remove()

    labels = ax.get_xticklabels()
    print(labels)

    plt.xlabel("Absolute Error")
    plt.xscale("symlog")
    plt.ylabel("CDF")
    plt.xticks(fontsize=48, rotation=45, ha="right", rotation_mode="anchor", x=0.5)
    plt.savefig(
        os.path.join(args.output_folder, f"{args.dataset}_mae_cdf_{args.remove_outliers}.png"),
        bbox_inches="tight",
    )
