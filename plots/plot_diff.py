from functools import partial
import os
import json
import re
from statistics import quantiles
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from forecast.data.loader import DataParser

from forecast.utils.cmdparser import HfArgumentParser

import matplotlib

matplotlib.rcParams.update({"font.size": 36})

@dataclass
class ModelConfig:
    folder: str = field(metadata={"help": "folder for xml data"})

    def __post_init__(self):
        if "abilene" in self.folder:
            self.dataset = "abilene"
        elif "geant" in self.folder:
            self.dataset = "geant"
        else:
            raise ValueError("dataset not supported")
parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

parser = DataParser()
df = parser.parse_sndlib_xml(args.folder)
df = df.drop(columns=["timestamps"])

df = df.diff()
data = df.values.flatten()
data = data[~np.isnan(data)]


quantiles = np.arange(0.01, 1, 0.01)
values = np.quantile(data, quantiles)

plt.figure(figsize=(20, 15), dpi=300)
plt.plot(values, quantiles, linewidth=4)
plt.xlabel("Demand Difference from Previous Time Step")
plt.ylabel("CDF")
plt.xscale("log")
plt.savefig(os.path.join("plots", f"{args.dataset}_demand_diff.png"), bbox_inches="tight")
plt.close()
