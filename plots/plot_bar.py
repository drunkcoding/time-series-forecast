import numpy as np
import itertools
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


font = {"size": 36}
matplotlib.rc("font", **font)
matplotlib.rcParams['hatch.linewidth'] = 4
sns.set(palette="colorblind")
sns.set_style("whitegrid")

# colors = ["#afd5ff", "#006dc1", "#003671"]
colors = ["#000000", "#003671", "#006dc1", "#0725e6", "#afd5ff", "#ffffff"]
palette = sns.color_palette(colors)

# df = pd.DataFrame(
#     [
#         {"model": "oracle", "impute": "zero", "MAE": 31.005407053924845},
#         {"model": "oracle", "impute": "forward", "MAE": 28.76670098118061},
#         {"model": "oracle", "impute": "mean", "MAE": 28.987454465228907},
#         {"model": "oracle", "impute": "median", "MAE": 29.042614270938998},
#         {"model": "oracle", "impute": "most frequent", "MAE": 29.952352388490684},
#         {"model": "oracle", "impute": "knn", "MAE": 28.852138086528164},
#         {"model": "real", "impute": "zero", "MAE": 92.75076499188653},
#         {"model": "real", "impute": "forward", "MAE": 86.62484988273866},
#         {"model": "real", "impute": "mean", "MAE": 84.29596518797841},
#         {"model": "real", "impute": "median", "MAE": 83.11207413948537},
#         {"model": "real", "impute": "most frequent", "MAE": 82.34016648573761},
#         {"model": "real", "impute": "knn", "MAE": 84.35886002184981},
#     ]
# )

# df = pd.DataFrame(
#     [
#         {"model": "oracle", "impute": "zero", "MAE": 7.099167887206158},
#         {"model": "oracle", "impute": "forward", "MAE": 7.105343851460341},
#         {"model": "oracle", "impute": "mean", "MAE": 7.0980794923659065},
#         {"model": "oracle", "impute": "median", "MAE": 7.092209580888721},
#         {"model": "oracle", "impute": "most frequent", "MAE": 7.098220568044828},
#         {"model": "oracle", "impute": "knn", "MAE": 7.08957818403817},
#         {"model": "real", "impute": "zero", "MAE": 15.11237045274301},
#         {"model": "real", "impute": "forward", "MAE": 15.06514918414725},
#         {"model": "real", "impute": "mean", "MAE": 14.986384698004946},
#         {"model": "real", "impute": "median", "MAE": 15.017667976623017},
#         {"model": "real", "impute": "most frequent", "MAE": 15.068923578704396},
#         {"model": "real", "impute": "knn", "MAE": 14.947918664990377},
#     ]
# )

df = pd.DataFrame(
    [
        {"model": "oracle", "impute": "Uni-mean", "MAE": 2.5864776148115403},
        {"model": "oracle", "impute": "Multi-mean", "MAE": 2.545130863505307},
        {"model": "real", "impute": "Uni-mean", "MAE": 41.15882775894535},
        {"model": "real", "impute": "Multi-mean", "MAE": 21.51345629044489},
        {"model": "oracle", "impute": "Uni-median", "MAE": 2.5864776148115403},
        {"model": "oracle", "impute": "Multi-median", "MAE": 2.545130863505307},
        {"model": "real", "impute": "Uni-median", "MAE": 41.15882775894535},
        {"model": "real", "impute": "Multi-median", "MAE": 21.51345629044489},
    ]
)

df = pd.DataFrame(
    [
        {"model": "oracle", "impute": "Uni-mean", "MAE": 12.004580800222767},
        {"model": "oracle", "impute": "Multi-mean", "MAE": 13.207976483607546},
        {"model": "real", "impute": "Uni-mean", "MAE": 2915.0828326965534},
        {"model": "real", "impute": "Multi-mean", "MAE": 195.21639807994183},
    ]
)

num_locations = len(df.model.unique())
hatches = itertools.cycle(['/', '//', '+', '-', 'x', '\\', '*', '.'])

plt.figure(figsize=(10, 8), dpi=300)
ax = sns.barplot(x="model", y="MAE", hue="impute", data=df, palette=palette)
ax.legend_.remove()

for i,thisbar in enumerate(ax.patches):
    # thisbar.set_hatch(hatches[i])
    thisbar.set_edgecolor('k')

axbox = ax.get_position()

plt.yscale("log")
# plt.yticks(np.arange(0,1000, 10))
plt.xticks(fontsize=48)
plt.yticks(fontsize=48)
# plt.ylim(100, 2e5)
plt.xlabel("", fontsize=48)
plt.ylabel("Demand MAE", fontsize=48)
legend = plt.legend(bbox_to_anchor=[0, axbox.y0+0.3,1,1], loc='upper right', ncol=3, fontsize=48)

def export_legend(legend, filename="plots/legend_var_bar.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)
legend.remove()

plt.savefig("plots/geant_arima_var_mae_bar.png", bbox_inches="tight")