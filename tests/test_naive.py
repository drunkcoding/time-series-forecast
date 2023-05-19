from dataclasses import dataclass, field
import json
import os
from forecast.utils.cmdparser import HfArgumentParser
from forecast.data.loader import DataParser
from forecast.utils.evaluation import *


@dataclass
class ModelConfig:
    input_folder: str = field(metadata={"help": "folder for xml data"})
    output_folder: str = field(metadata={"help": "path to checkpoints"})

    def __post_init__(self):
        try:
            os.mkdir(self.output_folder)
        except:
            pass


parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

parser = DataParser()
df = parser.parse_sndlib_xml(args.input_folder)

df = df_cleanup(df)

print(df)

total_size = len(df.index.values)
train_size = int(total_size * 0.6)
val_size = total_size - train_size

predictions = []
targets = []
nan_idx = []

# iterate over all df columns as numpy array
for col in df.columns:
    if col == "timestamps":
        continue
    train, val, idx = prepare_timeseries(
        df,
        "timestamps",
        col,
        "5min" if "abilene" in args.input_folder else "15min",
        train_size,
    )

    prediction = np.concatenate(
            [
                np.array([train.pd_dataframe()[col].values[-1]]),
                val.pd_dataframe()[col].values[:-1],
            ]
        )
    target = val.pd_dataframe()[col].values

    predictions.append(
        prediction
    )
    targets.append(target)
    nan_idx.append(idx)

    print(col, mae(prediction, target))

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)
nan_idx = np.concatenate(nan_idx)

print("total", mae(predictions, targets))

# save predictions to npy
np.save(
    os.path.join(args.output_folder, "predictions.npy"), predictions, allow_pickle=False
)
# save targets to npy
np.save(os.path.join(args.output_folder, "targets.npy"), targets, allow_pickle=False)
# save nan_idx to npy
np.save(os.path.join(args.output_folder, "nan_idx.npy"), nan_idx, allow_pickle=False)

# save df.columns to json
with open(os.path.join(args.output_folder, "columns.json"), "w") as fp:
    json.dump(list(df.columns), fp)
