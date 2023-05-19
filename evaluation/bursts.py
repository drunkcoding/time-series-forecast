# Import burst detection functions
from dataclasses import dataclass, field
from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats

# Import simulation code for creating test data
from neurodsp.sim import sim_combined
from neurodsp.utils import set_random_seed, create_times

# Import utilities for loading and plotting data
from neurodsp.utils.download import load_ndsp_data
from neurodsp.plts.time_series import plot_time_series, plot_bursts
import numpy as np

import pandas as pd
from transformers import HfArgumentParser
import matplotlib.pyplot as plt
import scipy.stats as stats


@dataclass
class Arguments:
    data: str = field(metadata={"help": "Path to the data file to load."})


parser = HfArgumentParser((Arguments,))
args = parser.parse_args_into_dataclasses()[0]

# Set a random seed, for consistency simulating data
set_random_seed(0)

df = pd.read_csv(args.data)
columns = df.columns.values
print(columns)
print(df.head())

col_name = columns[2]

print(stats.describe(df[col_name].to_numpy()))

fs = 1
sig = df[col_name].to_numpy()
sig /= 1000  # convert to Mbps

# # clip sig 95 percentile
# tail_threshold = np.percentile(sig, 99)
# sig[sig < tail_threshold] = 0

# times = create_times(len(sig)/fs, fs)

# plot_time_series(
#     times,
#     sig,
#     "Example OD Pair",
# )
plt.figure(figsize=(25, 5), dpi=300)
plt.plot(sig)
plt.xlabel("Epochs (Virtual Time)", fontsize=36)
plt.ylabel("Demand (Mbps)", fontsize=36)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)
plt.xlim(0, len(sig))
plt.savefig("evaluation/simulated_eeg.png", bbox_inches="tight")
plt.close()
exit()
# amp_dual_thresh = (1, 2)
# f_range = (5, 30)

# # Detect bursts using dual threshold algorithm
# bursting = detect_bursts_dual_threshold(sig, fs, amp_dual_thresh, f_range)

# plot_bursts(times, sig, bursting, labels=['Simulated EEG', 'Detected Burst'])
# plt.savefig("evaluation/simulated_eeg.png")
# plt.close()
