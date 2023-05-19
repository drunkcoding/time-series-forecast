from dataclasses import dataclass, field
import os

import pandas as pd


@dataclass
class ModelConfig:
    data: str = field(metadata={"help": "path to csv data"})
    clip: bool = field(default=False, metadata={"help": "whether to clip data"})
    norm: str = field(
        default="roll", metadata={"help": "normalization method, [roll, glob, indv]"}
    )

    def __post_init__(self):
        filename = os.path.basename(self.data)
        self.dataset = filename.split("_")[0].lower()

        # read first line of csv to get number of features
        df = pd.read_csv(self.data, nrows=1)
        self.n_features = len(df.columns) - 1
