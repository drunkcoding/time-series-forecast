

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from transformers import HfArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ModelConfig:
    data: str = field(metadata={"help": "path to csv data"})

# pandas unlimited print
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

df = pd.read_csv(args.data)
df = df.drop(columns=["time"])

print(df.max())

columns = df.columns.values

for col in columns:

    data = df[col].to_numpy()

    data_flatten = data.flatten()
    data_flatten = data_flatten[data_flatten > 0]

    sns.ecdfplot(data_flatten, label=col)
    print(col, data_flatten.max(), np.percentile(data_flatten, 99), np.percentile(data_flatten, 80))
plt.xscale("log")
plt.savefig("evaluation/values_cdf.png")
